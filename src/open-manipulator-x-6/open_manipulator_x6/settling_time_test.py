#!/usr/bin/env python3
"""
Wrist step-response settling-time characterization.

Standalone script (no ROS2 required) - talks directly to the Dynamixel
driver so polling isn't limited by the arm_driver ROS timer (50 Hz) or
message serialization overhead. This measures T_settle: the mechanical
settling time used to size the "consecutive quiet samples" dwell window
(N) in the CUSUM detected -> recovery gating logic.

Key change from v1: every trial starts from a fixed --reference-rad pose
(commanded and confirmed-settled before the step is issued), instead of
chaining off wherever the previous trial ended. This removes cumulative
drift across trials. The settling band is also now derived from a measured
static-noise calibration pass (band = noise_k * sigma_position), instead
of an arbitrary fraction of the step size, so "did not settle" reflects a
real physical issue rather than an unrealistically tight tolerance.

Place this file in the same directory as dynamixel_driver.py and utils.py
(or adjust the imports below to match your package layout).

Usage:
    python3 settling_time_test.py --moving-id 6 --step-rad 0.3 --trials 5
    python3 settling_time_test.py --moving-id 6 --step-rad 0.3 --trials 6 --alternate
"""

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dynamixel_driver import DynamixelHardwareDriver
from utils import (
    rad_to_ticks,
    ticks_to_rad,
    raw_current_to_effort,
    raw_vel_to_rad_s,
)


@dataclass
class Sample:
    t: float
    positions: dict
    currents: dict
    velocities: dict


@dataclass
class TrialResult:
    trial_idx: int
    moving_id: int
    step_rad: float
    start_rad: float
    target_rad: float
    band_rad: float
    settle_time_s: Optional[float]
    final_error_rad: float
    achieved_rate_hz: float
    clamped: bool
    samples: list = field(default_factory=list)


def connect_and_prepare(port: str, baudrate: int, joint_ids: list) -> DynamixelHardwareDriver:
    driver = DynamixelHardwareDriver(port=port, baudrate=baudrate)
    if not driver.connect():
        raise RuntimeError(f"Failed to open {port} @ {baudrate}")

    for jid in joint_ids:
        if not driver.ping(jid):
            raise RuntimeError(f"Failed to ping motor ID {jid} - check wiring/power/ID.")

    driver.set_operating_mode(joint_ids)  # defaults to POSITION_CONTROL_MODE
    if not driver.enable_torque(joint_ids, True):
        raise RuntimeError("Failed to enable torque on one or more joints.")

    return driver


def read_all_positions(driver: DynamixelHardwareDriver, joint_ids: list) -> dict:
    states = driver.read_states(joint_ids)
    return {jid: ticks_to_rad(states[jid]["position"]) for jid in joint_ids if jid in states}


def check_clamped(target_rad: float) -> bool:
    target_ticks_ideal = 2048 + (target_rad * (4096 / (2 * 3.14159265358979)))
    return not (0 <= target_ticks_ideal <= 4095)


def poll_until_settled(
    driver: DynamixelHardwareDriver,
    joint_ids: list,
    moving_id: int,
    target_rad: float,
    band_rad: float,
    dwell_samples: int,
    timeout_s: float,
):
    """
    Generic polling loop: reads all joints as fast as the bus allows until
    moving_id has stayed within band_rad of target_rad for dwell_samples
    consecutive reads, or timeout_s elapses. Returns (samples, settle_time,
    final_error_rad, achieved_rate_hz).
    """
    samples = []
    consecutive_ok = 0
    settle_time = None
    n_reads = 0

    t0 = time.monotonic()
    while True:
        now = time.monotonic() - t0
        states = driver.read_states(joint_ids)
        if not states:
            if now >= timeout_s:
                break
            continue

        n_reads += 1
        positions, currents, velocities = {}, {}, {}
        for jid in joint_ids:
            st = states.get(jid)
            if st is None:
                continue
            positions[jid] = ticks_to_rad(st["position"])
            currents[jid] = raw_current_to_effort(st["current"])
            velocities[jid] = raw_vel_to_rad_s(st["velocity"])

        samples.append(Sample(now, positions, currents, velocities))

        if moving_id in positions:
            err = abs(positions[moving_id] - target_rad)
            if err <= band_rad:
                consecutive_ok += 1
                if consecutive_ok >= dwell_samples and settle_time is None:
                    settle_time = now - (dwell_samples - 1) * (now / max(n_reads, 1))
                    settle_time = max(settle_time, 0.0)
                    break
            else:
                consecutive_ok = 0

        if now >= timeout_s:
            break

    final_error_rad = None
    if samples and moving_id in samples[-1].positions:
        final_error_rad = samples[-1].positions[moving_id] - target_rad

    achieved_rate_hz = n_reads / max(samples[-1].t, 1e-6) if samples else 0.0
    return samples, settle_time, final_error_rad, achieved_rate_hz


def measure_quiescent_noise(
    driver: DynamixelHardwareDriver,
    joint_ids: list,
    moving_id: int,
    reference_rad: float,
    duration_s: float,
) -> float:
    """
    Commands moving_id to reference_rad, waits briefly, then measures the
    standard deviation of its position readings while (nominally) static.
    This is the position-domain analog of the taxel noise calibration used
    for the tactile array - used to set a physically meaningful settling
    band instead of an arbitrary fraction of the step size.
    """
    print(f"Calibrating static position noise at reference={reference_rad:.3f} rad "
          f"over {duration_s:.1f}s ...")

    hold_positions = read_all_positions(driver, joint_ids)
    target_ticks = [
        rad_to_ticks(reference_rad if jid == moving_id else hold_positions.get(jid, 0.0))
        for jid in joint_ids
    ]
    driver.write_positions(joint_ids, target_ticks)
    time.sleep(1.0)  # allow initial transient to die out before measuring "static" noise

    readings = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        states = driver.read_states(joint_ids)
        if moving_id in states:
            readings.append(ticks_to_rad(states[moving_id]["position"]))

    if len(readings) < 2:
        print("  [WARNING] too few readings for noise calibration - defaulting sigma to 1 tick.")
        return 0.00153  # ~1 tick in rad, conservative fallback

    sigma = statistics.pstdev(readings)
    print(f"  sigma_position = {sigma*1000:.3f} mrad over {len(readings)} samples "
          f"(~{sigma/0.00153:.1f} ticks)")
    return sigma


def return_to_reference(
    driver: DynamixelHardwareDriver,
    joint_ids: list,
    moving_id: int,
    reference_rad: float,
    band_rad: float,
    dwell_samples: int,
    timeout_s: float,
) -> bool:
    """
    Commands moving_id back to reference_rad and waits for it to settle
    there before the next trial starts, so trials don't chain drift.
    Returns True if it settled, False if it timed out (trial proceeds
    anyway, but this is worth flagging if it happens repeatedly).
    """
    hold_positions = read_all_positions(driver, joint_ids)
    target_ticks = [
        rad_to_ticks(reference_rad if jid == moving_id else hold_positions.get(jid, 0.0))
        for jid in joint_ids
    ]
    driver.write_positions(joint_ids, target_ticks)
    _, settle_time, final_error, _ = poll_until_settled(
        driver, joint_ids, moving_id, reference_rad, band_rad, dwell_samples, timeout_s
    )
    if settle_time is None:
        err_mrad = (final_error or 0.0) * 1000
        print(f"  [WARNING] did not confirm return-to-reference within {timeout_s}s "
              f"(final error {err_mrad:+.2f} mrad). Proceeding to next trial anyway.")
        return False
    return True


def run_trial(
    driver: DynamixelHardwareDriver,
    joint_ids: list,
    moving_id: int,
    reference_rad: float,
    step_rad: float,
    band_rad: float,
    dwell_samples: int,
    timeout_s: float,
    trial_idx: int,
) -> TrialResult:
    start_rad = reference_rad
    target_rad = start_rad + step_rad
    clamped = check_clamped(target_rad)
    if clamped:
        print(
            f"  [WARNING] trial {trial_idx}: commanded target ({target_rad:.3f} rad) "
            f"falls outside the single-turn tick range and will be clamped by rad_to_ticks(). "
            f"Reduce --step-rad or shift --reference-rad."
        )

    hold_positions = read_all_positions(driver, joint_ids)
    target_ticks = [
        rad_to_ticks(target_rad if jid == moving_id else hold_positions.get(jid, 0.0))
        for jid in joint_ids
    ]
    driver.write_positions(joint_ids, target_ticks)

    samples, settle_time, final_error, achieved_rate_hz = poll_until_settled(
        driver, joint_ids, moving_id, target_rad, band_rad, dwell_samples, timeout_s
    )

    return TrialResult(
        trial_idx=trial_idx,
        moving_id=moving_id,
        step_rad=step_rad,
        start_rad=start_rad,
        target_rad=target_rad,
        band_rad=band_rad,
        settle_time_s=settle_time,
        final_error_rad=final_error if final_error is not None else float("nan"),
        achieved_rate_hz=achieved_rate_hz,
        clamped=clamped,
        samples=samples,
    )


def write_trial_csv(result: TrialResult, joint_ids: list, out_dir: Path):
    path = out_dir / f"trial_{result.trial_idx:02d}_step{result.step_rad:+.3f}.csv"
    fieldnames = ["t"] + [f"pos_{j}" for j in joint_ids] + [f"cur_{j}" for j in joint_ids] + [f"vel_{j}" for j in joint_ids]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in result.samples:
            row = {"t": f"{s.t:.5f}"}
            for j in joint_ids:
                row[f"pos_{j}"] = s.positions.get(j, "")
                row[f"cur_{j}"] = s.currents.get(j, "")
                row[f"vel_{j}"] = s.velocities.get(j, "")
            writer.writerow(row)


def write_summary_csv(results: list, out_dir: Path):
    path = out_dir / "settling_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["trial", "moving_id", "step_rad", "start_rad", "target_rad", "band_rad",
             "settle_time_s", "final_error_rad", "final_error_ticks",
             "achieved_rate_hz", "clamped"]
        )
        for r in results:
            err_ticks = "" if r.final_error_rad != r.final_error_rad else f"{r.final_error_rad/0.00153:.2f}"
            writer.writerow(
                [r.trial_idx, r.moving_id, f"{r.step_rad:.4f}", f"{r.start_rad:.4f}",
                 f"{r.target_rad:.4f}", f"{r.band_rad:.5f}",
                 "" if r.settle_time_s is None else f"{r.settle_time_s:.4f}",
                 f"{r.final_error_rad:.5f}", err_ticks,
                 f"{r.achieved_rate_hz:.1f}", r.clamped]
            )


def main():
    # arguments

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--joint-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
                     help="All joints to hold/monitor (default matches arm_driver.py).")
    ap.add_argument("--moving-id", type=int, required=True,
                     help="Which joint ID to step.")
    ap.add_argument("--reference-rad", type=float, default=0.0,
                     help="Fixed pose to return to before every trial (removes cumulative drift).")
    ap.add_argument("--step-rad", type=float, default=0.3,
                     help="Step size in radians. Sign sets direction.")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--alternate", action="store_true",
                     help="Alternate step direction (+/-) each trial instead of repeating the same sign.")
    ap.add_argument("--noise-calib-duration", type=float, default=1.5,
                     help="Seconds to measure static position noise before any trials.")
    ap.add_argument("--noise-k", type=float, default=4.0,
                     help="Settling band = noise_k * sigma_position (measured at reference pose).")
    ap.add_argument("--band-frac-floor", type=float, default=0.0,
                     help="Optional minimum band as a fraction of step size, in case noise-based "
                          "band comes out unrealistically tight for your application.")
    ap.add_argument("--dwell-samples", type=int, default=10,
                     help="Consecutive in-band samples required to declare settled.")
    ap.add_argument("--timeout", type=float, default=3.0, help="Per-trial timeout, seconds.")
    ap.add_argument("--return-timeout", type=float, default=None,
                     help="Timeout for the return-to-reference phase (defaults to --timeout).")
    ap.add_argument("--kappa", type=float, default=1.75,
                     help="Safety margin multiplier for the recommended N calculation.")
    ap.add_argument("--control-rate-hz", type=float, default=100.0,
                     help="Sample rate the recommended N should be computed against.")
    ap.add_argument("--out-dir", default="./settling_test_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.moving_id not in args.joint_ids:
        raise SystemExit(f"--moving-id {args.moving_id} must be one of --joint-ids {args.joint_ids}")

    return_timeout = args.return_timeout if args.return_timeout is not None else args.timeout

    driver = connect_and_prepare(args.port, args.baudrate, args.joint_ids)
    results = []

    try:
        sigma = measure_quiescent_noise(
            driver, args.joint_ids, args.moving_id, args.reference_rad, args.noise_calib_duration
        )
        band_rad = max(args.noise_k * sigma, args.band_frac_floor * abs(args.step_rad))
        print(f"Using settling band = {band_rad*1000:.3f} mrad "
              f"(~{band_rad/0.00153:.1f} ticks) for all trials.\n")

        for i in range(args.trials):
            step = args.step_rad
            if args.alternate and i % 2 == 1:
                step = -args.step_rad

            return_to_reference(
                driver, args.joint_ids, args.moving_id, args.reference_rad,
                band_rad, args.dwell_samples, return_timeout,
            )

            print(f"Trial {i}: commanding step {step:+.3f} rad from reference "
                  f"({args.reference_rad:.3f} rad) on joint {args.moving_id} ...")
            result = run_trial(
                driver, args.joint_ids, args.moving_id, args.reference_rad, step,
                band_rad, args.dwell_samples, args.timeout, i,
            )
            results.append(result)
            write_trial_csv(result, args.joint_ids, out_dir)

            err_mrad = result.final_error_rad * 1000
            if result.settle_time_s is None:
                print(f"  did NOT settle within {args.timeout}s "
                      f"(final error {err_mrad:+.2f} mrad, band ±{band_rad*1000:.2f} mrad).")
            else:
                print(f"  settled in {result.settle_time_s*1000:.1f} ms "
                      f"(achieved poll rate ~{result.achieved_rate_hz:.0f} Hz, "
                      f"final error {err_mrad:+.2f} mrad)")

    finally:
        write_summary_csv(results, out_dir)
        driver.disconnect()

    settled = [r.settle_time_s for r in results if r.settle_time_s is not None]
    print("\n--- Summary ---")
    print(f"Trials: {len(results)}  |  Settled: {len(settled)}  |  Timed out: {len(results)-len(settled)}")
    if results:
        errs_mrad = [r.final_error_rad*1000 for r in results]
        print(f"Final error across ALL trials (settled or not): "
              f"min={min(errs_mrad):+.2f}  max={max(errs_mrad):+.2f}  mrad")
    if settled:
        t_min, t_max = min(settled), max(settled)
        t_mean = sum(settled) / len(settled)
        print(f"T_settle: min={t_min*1000:.1f} ms  mean={t_mean*1000:.1f} ms  max={t_max*1000:.1f} ms")

        n_recommended = int((args.kappa * t_max * args.control_rate_hz) + 0.999)
        print(
            f"\nRecommended dwell samples N (kappa={args.kappa}, "
            f"control_rate={args.control_rate_hz} Hz, using worst-case T_settle={t_max*1000:.1f} ms):"
        )
        print(f"  N = ceil({args.kappa} * {t_max:.4f}s * {args.control_rate_hz}Hz) = {n_recommended}")
        print("  Note: this is a MECHANICAL lower bound. Re-verify once tactile sensors")
        print("  are mounted, since contact-patch reshaping may add settling lag beyond this.")
    else:
        print("\nNo trials satisfied the strict settle criterion. Check the final-error values above:")
        print("if they're already small (within a few mrad / a handful of ticks), the joint is")
        print("likely fine physically and --dwell-samples is too strict relative to --noise-k's")
        print("band - try raising --noise-k or lowering --dwell-samples rather than assuming")
        print("instability.")

    print(f"\nRaw per-trial CSVs and settling_summary.csv written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()