#!/usr/bin/env python3
"""
Wrist step-response settling-time characterization.

Script that talks directly to the Dynamixel
driver so polling isn't limited by the arm_driver ROS timer (50 Hz) or
message serialization overhead. This measures T_settle: the mechanical
settling time used to size the "consecutive quiet samples" dwell window
(N) in the CUSUM detected -> recovery gating logic.


Usage:
    python3 settling_time_test.py --moving-id 2 --step-rad 0.3 --trials 5
    python3 settling_time_test.py --moving-id 2 --step-rad 0.3 --trials 6 --alternate
    python3 settling_time_test.py --moving-id 1 --step-rad 0.6 --trials 5 --alternate
"""

import argparse
import csv
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
    settle_time_s: Optional[float]
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


def run_trial(
    driver: DynamixelHardwareDriver,
    joint_ids: list,
    moving_id: int,
    step_rad: float,
    band_frac: float,
    dwell_samples: int,
    timeout_s: float,
    trial_idx: int,
) -> TrialResult:
    hold_positions = read_all_positions(driver, joint_ids)
    if moving_id not in hold_positions:
        raise RuntimeError(f"Could not read starting position for moving joint {moving_id}.")

    start_rad = hold_positions[moving_id]
    target_rad = start_rad + step_rad

    # Sanity-check for silent clamping in rad_to_ticks (single-turn range).
    target_ticks_ideal = 2048 + (target_rad * (4096 / (2 * 3.14159265358979)))
    clamped = not (0 <= target_ticks_ideal <= 4095)
    if clamped:
        print(
            f"  [WARNING] trial {trial_idx}: commanded target ({target_rad:.3f} rad) "
            f"falls outside the single-turn tick range and will be clamped by rad_to_ticks(). "
            f"Reduce --step-rad or shift the starting pose."
        )

    # Build the full command: everyone else holds, moving_id gets the new target.
    target_ticks = []
    for jid in joint_ids:
        rad = target_rad if jid == moving_id else hold_positions[jid]
        target_ticks.append(rad_to_ticks(rad))

    band_rad = band_frac * abs(step_rad)

    samples = []
    consecutive_ok = 0
    settle_time = None

    t0 = time.monotonic()
    driver.write_positions(joint_ids, target_ticks)

    n_reads = 0
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
                    # settle_time marks the START of the dwell window, not
                    # the moment the window closes - subtract the window length.
                    settle_time = now - (dwell_samples - 1) * (now / max(n_reads, 1))
                    settle_time = max(settle_time, 0.0)
                    break
            else:
                consecutive_ok = 0

        if now >= timeout_s:
            break

    achieved_rate_hz = n_reads / max(samples[-1].t, 1e-6) if samples else 0.0

    return TrialResult(
        trial_idx=trial_idx,
        moving_id=moving_id,
        step_rad=step_rad,
        start_rad=start_rad,
        target_rad=target_rad,
        settle_time_s=settle_time,
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
            ["trial", "moving_id", "step_rad", "start_rad", "target_rad",
             "settle_time_s", "achieved_rate_hz", "clamped"]
        )
        for r in results:
            writer.writerow(
                [r.trial_idx, r.moving_id, f"{r.step_rad:.4f}", f"{r.start_rad:.4f}",
                 f"{r.target_rad:.4f}",
                 "" if r.settle_time_s is None else f"{r.settle_time_s:.4f}",
                 f"{r.achieved_rate_hz:.1f}", r.clamped]
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--joint-ids", type=int, nargs="+", default=[11, 12, 13, 14, 15, 2, 1],
                     help="All joints to hold/monitor (default matches arm_driver.py).")
    ap.add_argument("--moving-id", type=int, default=1,
                     help="Which joint ID to step (default 2 = wrist_rotate).")
    ap.add_argument("--step-rad", type=float, default=0.3,
                     help="Step size in radians. Sign sets direction.")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--alternate", action="store_true",
                     help="Alternate step direction (+/-) each trial instead of repeating the same sign.")
    ap.add_argument("--band-frac", type=float, default=0.02,
                     help="Settling band as a fraction of the step size (default 2%%).")
    ap.add_argument("--dwell-samples", type=int, default=10,
                     help="Consecutive in-band samples required to declare settled.")
    ap.add_argument("--timeout", type=float, default=6.0, help="Per-trial timeout, seconds.")
    ap.add_argument("--settle-time-warn", type=float, default=None,
                     help="If a trial exceeds this many seconds without settling, warn loudly.")
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

    driver = connect_and_prepare(args.port, args.baudrate, args.joint_ids)
    results = []

    try:
        for i in range(args.trials):
            step = args.step_rad
            if args.alternate and i % 2 == 1:
                step = -args.step_rad

            print(f"Trial {i}: commanding step {step:+.3f} rad on joint {args.moving_id} ...")
            result = run_trial(
                driver, args.joint_ids, args.moving_id, step,
                args.band_frac, args.dwell_samples, args.timeout, i,
            )
            results.append(result)
            write_trial_csv(result, args.joint_ids, out_dir)

            if result.settle_time_s is None:
                print(f"  did NOT settle within {args.timeout}s timeout (band={args.band_frac*100:.1f}%).")
            else:
                print(f"  settled in {result.settle_time_s*1000:.1f} ms "
                      f"(achieved poll rate ~{result.achieved_rate_hz:.0f} Hz)")
                if args.settle_time_warn and result.settle_time_s > args.settle_time_warn:
                    print(f"  [WARNING] settle time exceeds --settle-time-warn ({args.settle_time_warn}s)")

            time.sleep(0.3)  # brief pause between trials so transients don't overlap

    finally:
        write_summary_csv(results, out_dir)
        driver.disconnect()

    settled = [r.settle_time_s for r in results if r.settle_time_s is not None]
    print("\n--- Summary ---")
    print(f"Trials: {len(results)}  |  Settled: {len(settled)}  |  Timed out: {len(results)-len(settled)}")
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
        print("No trials settled - loosen --band-frac, increase --timeout, or check for oscillation.")

    print(f"\nRaw per-trial CSVs and settling_summary.csv written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()