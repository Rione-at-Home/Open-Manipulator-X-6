#!/usr/bin/env python3
"""
Feedforward + PID position control for a joint run in PWM Control Mode.

WHY THIS EXISTS: Coulomb friction creates a deadband that linear feedback
fights poorly. This script runs joints (including XL430 series lacking current sensors)
in PWM Control Mode (Operating Mode 16) and implements software position loop control:

    PWM_cmd = Kp*e + Ki*integral(e) + Kd*de/dt   (feedback)
            + PWM_gravity(theta)                  (from fit model)
            + PWM_friction * tanh(e / eps)        (Coulomb compensation)

*** SAFETY ***
PWM Control Mode is NOT fail-safe like Position Control - if this script stops
updating, the last-written voltage duty cycle continues being applied. This script:
  - always writes Goal PWM = 0 before disconnecting (finally block)
  - aborts and zeroes PWM after --max-comm-failures consecutive failed reads
  - hard-clamps every commanded PWM to --max-pwm-raw before writing

*** DIAGNOSTIC ADDITIONS (2026-10-01) ***
Added to directly answer the "is the integral clamp starving the correction,
or is there a genuine uncaptured bias?" question instead of inferring it from
position error alone:
  - Every sample now logs the running integral value (per-segment CSVs).
  - Each segment reports whether the integral was pinned at +/-integral_max
    at the moment the loop ended ("[INTEGRAL SATURATED]" in the console and
    an integral_saturated column in the summary CSV). If a residual persists
    but the integral is NOT saturated, raising --integral-max will not help -
    the bias is not clamp-limited.
  - New --dir-bias: a fixed additive feedforward term (raw PWM units),
    applied only on segments stepping in the positive direction (i.e. where
    target_rad > 0 relative to the trial's step sign). This is meant to be
    filled in from calibrate_gravityj6.py's forward/reverse split-fit delta
    once you've measured it, so the directional (cable-tension/hysteresis)
    bias is compensated proactively by feedforward instead of being fought
    reactively by the integral term every single step.

Usage:
    python3 current_control_wrist.py --moving-id 6 --step-rad 0.02 --trials 6 --alternate \
        --gravity-model ./gravity_calib_out/gravity_model.json --friction-comp 35 \
        --integral-max 500

    # once a directional bias has been measured (see calibrate_gravityj6.py):
    python3 current_control_wrist.py --moving-id 6 --step-rad 0.02 --trials 6 --alternate \
        --gravity-model ./gravity_calib_out/gravity_model.json --friction-comp 35 \
        --integral-max 150 --dir-bias 22.5
"""

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dynamixel_driver import (
    DynamixelHardwareDriver,
    POSITION_CONTROL_MODE,
    PWM_CONTROL_MODE,
)
from utils import (
    rad_to_ticks,
    ticks_to_rad,
    raw_current_to_effort,
    RAD_PER_TICK,
)


@dataclass
class Sample:
    t: float
    pos: float
    error: float
    integral: float
    u_fb: float
    i_grav: float
    i_fric: float
    u_total: float
    measured_current: float


@dataclass
class SegmentResult:
    label: str
    target_rad: float
    band_rad: float
    settle_time_s: Optional[float]
    final_error_rad: float
    achieved_rate_hz: float
    final_integral: float
    integral_saturated: bool
    samples: list = field(default_factory=list)


class SafetyAbort(Exception):
    pass


def measure_bus_timing(driver: DynamixelHardwareDriver, moving_id: int, n: int = 50):
    t0 = time.monotonic()
    for _ in range(n):
        driver.read_states([moving_id])
    read_ms = (time.monotonic() - t0) / n * 1000

    t0 = time.monotonic()
    for _ in range(n):
        driver.write_goal_pwm(moving_id, 0)
    write_ms = (time.monotonic() - t0) / n * 1000

    combined_hz = 1000.0 / (read_ms + write_ms)
    print(f"Bus timing probe ({n} calls each): read={read_ms:.2f} ms/call, "
          f"write={write_ms:.2f} ms/call -> combined loop ceiling ~{combined_hz:.1f} Hz")
    if read_ms > 5 or write_ms > 5:
        print("  [WARNING] >5ms per call is consistent with the USB-serial adapter's default")
        print("  latency_timer (often 16ms on Linux). Check/set it, e.g.:")
        print("    echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer")
    print()


def connect_and_prepare(port, baudrate, joint_ids, moving_id):
    driver = DynamixelHardwareDriver(port=port, baudrate=baudrate)
    if not driver.connect():
        raise RuntimeError(f"Failed to open {port} @ {baudrate}")

    for jid in joint_ids:
        if not driver.ping(jid):
            raise RuntimeError(f"Failed to ping motor ID {jid} - check wiring/power/ID.")

    other_ids = [j for j in joint_ids if j != moving_id]

    # Set moving_id to PWM Control Mode; others stay in Position Control
    driver.set_operating_mode([moving_id], mode=PWM_CONTROL_MODE)
    if other_ids:
        driver.set_operating_mode(other_ids, mode=POSITION_CONTROL_MODE)

    actual_mode = driver.read_operating_mode(moving_id)
    if actual_mode != PWM_CONTROL_MODE:
        raise RuntimeError(
            f"Joint {moving_id} did NOT switch into PWM Control Mode "
            f"(read back mode={actual_mode}, expected {PWM_CONTROL_MODE})."
        )
    print(f"Confirmed joint {moving_id} is in PWM Control Mode (readback={actual_mode}).")

    # Safety: force zero PWM before torque goes live
    driver.write_goal_pwm(moving_id, 0)

    if not driver.enable_torque(joint_ids, True):
        raise RuntimeError("Failed to enable torque on one or more joints.")

    return driver, other_ids


def hold_others(driver, other_ids, hold_positions: dict):
    if not other_ids:
        return
    ticks = [rad_to_ticks(hold_positions[j]) for j in other_ids]
    driver.write_positions(other_ids, ticks)


def load_gravity_model(path: Path) -> dict:
    with open(path) as f:
        model = json.load(f)
    m = model["sinusoidal_model"]
    return {"A": m["A_cos_amplitude"], "B": m["B_sin_amplitude"], "C": m["C_bias_offset"]}


def run_segment(
    driver: DynamixelHardwareDriver,
    moving_id: int,
    label: str,
    target_rad: float,
    kp: float, ki: float, kd: float,
    integral_max: float,
    grav_model: dict,
    friction_comp: float,
    friction_eps_rad: float,
    max_pwm_raw: float,
    band_rad: float,
    dwell_samples: int,
    timeout_s: float,
    max_comm_failures: int,
    extra_bias: float = 0.0,
    allow_early_exit: bool = True,
) -> SegmentResult:
    samples = []
    integral = 0.0
    prev_error = None
    prev_t = None
    consecutive_ok = 0
    settle_time = None
    comm_failures = 0
    n_reads = 0

    t0 = time.monotonic()
    while True:
        now = time.monotonic() - t0
        states = driver.read_states([moving_id])

        if moving_id not in states:
            comm_failures += 1
            if comm_failures >= max_comm_failures:
                driver.write_goal_pwm(moving_id, 0)
                raise SafetyAbort(
                    f"[{label}] {max_comm_failures} consecutive read failures - "
                    f"zeroed PWM and aborting."
                )
            continue
        comm_failures = 0
        n_reads += 1

        pos = ticks_to_rad(states[moving_id]["position"])
        measured_current = raw_current_to_effort(states[moving_id].get("current", 0))
        error = target_rad - pos

        if prev_t is None:
            dt = 0.0
            derivative = 0.0
        else:
            dt = max(now - prev_t, 1e-4)
            derivative = (error - prev_error) / dt
            integral += error * dt
            integral = max(-integral_max, min(integral_max, integral))

        u_fb = kp * error + ki * integral + kd * derivative

        i_grav = grav_model["A"] * math.cos(pos) + grav_model["B"] * math.sin(pos) + grav_model["C"]
        i_fric = friction_comp * math.tanh(error / max(friction_eps_rad, 1e-6))

        u_total = u_fb + i_grav + i_fric + extra_bias
        u_total_clamped = max(-max_pwm_raw, min(max_pwm_raw, u_total))

        driver.write_goal_pwm(moving_id, int(round(u_total_clamped)))

        samples.append(Sample(now, pos, error, integral, u_fb, i_grav, i_fric, u_total_clamped, measured_current))

        if abs(error) <= band_rad:
            consecutive_ok += 1
            if consecutive_ok >= dwell_samples and settle_time is None:
                settle_time = now - (dwell_samples - 1) * (now / max(n_reads, 1))
                settle_time = max(settle_time, 0.0)
        else:
            consecutive_ok = 0

        prev_error = error
        prev_t = now

        if now >= timeout_s:
            break
        if allow_early_exit and settle_time is not None and now >= timeout_s * 0.15:
            break

    final_error = samples[-1].error if samples else float("nan")
    achieved_rate_hz = n_reads / max(samples[-1].t, 1e-6) if samples else 0.0
    final_integral = integral
    # Treat "saturated" as sitting at (or having been clamped to) the ceiling,
    # not just numerically equal to it, since float accumulation may land a
    # hair under integral_max after the clamp on the last update.
    integral_saturated = integral_max > 0 and abs(final_integral) >= (integral_max - 1e-6)

    return SegmentResult(
        label, target_rad, band_rad, settle_time, final_error, achieved_rate_hz,
        final_integral, integral_saturated, samples,
    )


def measure_quiescent_noise(
    driver, moving_id, reference_rad, kp, ki, kd, integral_max,
    grav_model, friction_comp, friction_eps_rad, max_pwm_raw,
    duration_s, max_comm_failures, presettle_timeout_s: float = 1.0,
) -> float:
    # BUGFIX (2026-10-01): this used to call run_segment with band_rad=1e9,
    # dwell_samples=1 - meaning the very first sample ever read trivially
    # satisfies "settled", which tripped run_segment's early-exit rule
    # (break once settled + 15% of timeout) and silently truncated every
    # noise-calibration window to ~15% of the requested duration_s,
    # regardless of what you passed for --noise-calib-duration. Fixed by
    # passing allow_early_exit=False so this segment actually runs the
    # full window you asked for.
    #
    # Also added a short pre-settle phase: if the joint wasn't already at
    # reference_rad when this is called (e.g. first run after a power
    # cycle, or torque was just re-enabled), the noise window would
    # otherwise include that approach transient, inflating sigma for a
    # reason that has nothing to do with actual quiescent noise.
    print(f"Pre-settling at reference={reference_rad:.3f} rad before measuring noise...")
    presettle = run_segment(
        driver, moving_id, "noise_presettle", reference_rad,
        kp, ki, kd, integral_max, grav_model, friction_comp, friction_eps_rad,
        max_pwm_raw, band_rad=0.01, dwell_samples=5, timeout_s=presettle_timeout_s,
        max_comm_failures=max_comm_failures, allow_early_exit=True,
    )
    if presettle.settle_time_s is None:
        print(f"  [WARNING] did not confirm pre-settle within {presettle_timeout_s:.1f}s "
              f"(final error {presettle.final_error_rad*1000:+.2f} mrad). The noise window "
              f"below may still include some approach transient.")

    print(f"Calibrating static position noise at reference={reference_rad:.3f} rad over {duration_s:.1f}s ...")
    result = run_segment(
        driver, moving_id, "noise_calib", reference_rad,
        kp, ki, kd, integral_max, grav_model, friction_comp, friction_eps_rad,
        max_pwm_raw, band_rad=1e9, dwell_samples=1, timeout_s=duration_s,
        max_comm_failures=max_comm_failures, allow_early_exit=False,
    )
    positions = [reference_rad - s.error for s in result.samples]
    if len(positions) < 2:
        print("  [WARNING] too few readings for noise calibration - defaulting sigma to 1 tick.")
        return RAD_PER_TICK
    sigma = statistics.pstdev(positions)
    print(f"  sigma_position = {sigma*1000:.3f} mrad over {len(positions)} samples "
          f"(~{sigma/RAD_PER_TICK:.1f} ticks)")
    return sigma


def write_segment_csv(result: SegmentResult, idx: int, out_dir: Path):
    path = out_dir / f"segment_{idx:02d}_{result.label}_target{result.target_rad:+.3f}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "pos", "error", "integral", "u_fb", "i_grav", "i_fric", "u_total", "measured_current"])
        for s in result.samples:
            writer.writerow([f"{s.t:.5f}", f"{s.pos:.5f}", f"{s.error:.5f}", f"{s.integral:.3f}",
                              f"{s.u_fb:.3f}", f"{s.i_grav:.3f}", f"{s.i_fric:.3f}",
                              f"{s.u_total:.3f}", f"{s.measured_current:.4f}"])


def write_summary_csv(results: list, out_dir: Path):
    path = out_dir / "pwm_control_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "target_rad", "band_rad", "settle_time_s",
                          "final_error_rad", "final_error_ticks", "achieved_rate_hz",
                          "final_integral", "integral_saturated"])
        for i, r in enumerate(results):
            writer.writerow([i, r.label, f"{r.target_rad:.4f}", f"{r.band_rad:.5f}",
                              "" if r.settle_time_s is None else f"{r.settle_time_s:.4f}",
                              f"{r.final_error_rad:.5f}", f"{r.final_error_rad/RAD_PER_TICK:.2f}",
                              f"{r.achieved_rate_hz:.1f}",
                              f"{r.final_integral:.2f}", r.integral_saturated])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--joint-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--moving-id", type=int, required=True)
    ap.add_argument("--reference-rad", type=float, default=0.0)
    ap.add_argument("--step-rad", type=float, default=0.02)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--alternate", action="store_true")

    ap.add_argument("--gravity-model", type=Path, default=Path("./gravity_calib_out/gravity_model.json"))
    ap.add_argument("--friction-comp", type=float, default=35.0,
                     help="Coulomb friction feedforward magnitude, raw PWM units.")
    ap.add_argument("--friction-eps-rad", type=float, default=0.003)
    ap.add_argument("--dir-bias", type=float, default=0.0,
                     help="Fixed additive feedforward PWM bias applied only on positive-direction "
                          "step segments (target_rad relative step > 0). Intended to compensate a "
                          "known, repeatable directional bias (e.g. cable-tension hysteresis) "
                          "proactively instead of relying on the integral term to fight it every "
                          "step. Source this value from calibrate_gravityj6.py's forward/reverse "
                          "split-fit delta-at-theta0 output. Leave at 0.0 until you've measured it.")

    ap.add_argument("--kp", type=float, default=80.0)
    ap.add_argument("--ki", type=float, default=5.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--integral-max", type=float, default=150.0)

    ap.add_argument("--max-pwm-raw", type=float, default=200.0,
                     help="Hard safety clamp on total commanded PWM (max 885).")
    ap.add_argument("--max-comm-failures", type=int, default=20)

    ap.add_argument("--noise-calib-duration", type=float, default=1.5)
    ap.add_argument("--noise-presettle-timeout", type=float, default=1.0,
                     help="Max time to confirm the joint is actually at --reference-rad before the "
                          "noise-calibration window starts, so an approach transient doesn't get "
                          "counted as quiescent noise.")
    ap.add_argument("--noise-k", type=float, default=4.0)
    ap.add_argument("--min-band-ticks", type=float, default=2.0)
    ap.add_argument("--dwell-samples", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--kappa", type=float, default=1.75)
    ap.add_argument("--control-rate-hz", type=float, default=100.0)
    ap.add_argument("--out-dir", default="./pwm_control_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.moving_id not in args.joint_ids:
        raise SystemExit(f"--moving-id {args.moving_id} must be one of --joint-ids {args.joint_ids}")

    grav_model = load_gravity_model(args.gravity_model)
    print(f"Loaded gravity model: I_ff(theta) = {grav_model['A']:+.3f}*cos(theta) "
          f"+ {grav_model['B']:+.3f}*sin(theta) + {grav_model['C']:+.3f}")
    print(f"Friction feedforward magnitude: {args.friction_comp:.1f} PWM units "
          f"(blend width {args.friction_eps_rad*1000:.1f} mrad)")
    if args.dir_bias != 0.0:
        print(f"Directional bias: {args.dir_bias:+.1f} PWM units, applied on positive-direction steps only")
    print()

    driver, other_ids = connect_and_prepare(args.port, args.baudrate, args.joint_ids, args.moving_id)
    results = []

    measure_bus_timing(driver, args.moving_id)

    ctrl = dict(kp=args.kp, ki=args.ki, kd=args.kd, integral_max=args.integral_max,
                grav_model=grav_model, friction_comp=args.friction_comp,
                friction_eps_rad=args.friction_eps_rad, max_pwm_raw=args.max_pwm_raw,
                max_comm_failures=args.max_comm_failures)

    try:
        other_hold = {j: ticks_to_rad(driver.read_states([j])[j]["position"])
                      for j in other_ids if j in driver.read_states([j])}
        hold_others(driver, other_ids, other_hold)

        sigma = measure_quiescent_noise(
            driver, args.moving_id, args.reference_rad,
            duration_s=args.noise_calib_duration,
            presettle_timeout_s=args.noise_presettle_timeout, **ctrl,
        )
        band_rad = max(args.noise_k * sigma, args.min_band_ticks * RAD_PER_TICK)
        print(f"Using settling band = {band_rad*1000:.3f} mrad "
              f"(~{band_rad/RAD_PER_TICK:.1f} ticks) for all segments.")

        if band_rad >= abs(args.step_rad):
            print(
                f"[WARNING] settling band ({band_rad*1000:.2f} mrad) is >= your step size "
                f"({abs(args.step_rad)*1000:.2f} mrad)! The target will already be 'in band' "
                f"before the joint moves at all, so every trial will report settling almost "
                f"instantly regardless of real controller behavior - these results will NOT be "
                f"usable for diagnosing anything. If this fires, stop and either re-run (noise "
                f"calibration is sensitive to whatever transient was happening when it started), "
                f"increase --noise-calib-duration, or use a larger --step-rad."
            )
        print()

        for i in range(args.trials):
            step = args.step_rad if not (args.alternate and i % 2 == 1) else -args.step_rad
            target = args.reference_rad + step
            extra_bias = args.dir_bias if step > 0 else 0.0

            start_pos = ticks_to_rad(driver.read_states([args.moving_id])[args.moving_id]["position"])
            r_return = run_segment(driver, args.moving_id, "return", args.reference_rad,
                                    band_rad=band_rad, dwell_samples=args.dwell_samples,
                                    timeout_s=args.timeout, extra_bias=0.0, **ctrl)
            write_segment_csv(r_return, 2 * i, out_dir)

            print(f"Trial {i}: target {target:+.3f} rad on joint {args.moving_id} "
                  f"(extra_bias={extra_bias:+.1f}) ...")
            r_step = run_segment(driver, args.moving_id, "step", target,
                                  band_rad=band_rad, dwell_samples=args.dwell_samples,
                                  timeout_s=args.timeout, extra_bias=extra_bias, **ctrl)
            write_segment_csv(r_step, 2 * i + 1, out_dir)
            results.append(r_step)

            err_mrad = r_step.final_error_rad * 1000
            sat_flag = " [INTEGRAL SATURATED]" if r_step.integral_saturated else ""
            if r_step.settle_time_s is None:
                print(f"  did NOT settle within {args.timeout}s (final error {err_mrad:+.2f} mrad, "
                      f"final integral={r_step.final_integral:+.1f}){sat_flag}")
            else:
                print(f"  settled in {r_step.settle_time_s*1000:.1f} ms (final error {err_mrad:+.2f} mrad, "
                      f"final integral={r_step.final_integral:+.1f}){sat_flag}")

    except SafetyAbort as e:
        print(f"\n[SAFETY ABORT] {e}")
    finally:
        try:
            driver.write_goal_pwm(args.moving_id, 0)
        except Exception:
            pass
        write_summary_csv(results, out_dir)
        driver.disconnect()
        print(f"\nGoal PWM zeroed and port closed. Outputs in: {out_dir.resolve()}")

    saturated_positive = [r for r in results if r.target_rad > args.reference_rad and r.integral_saturated]
    if saturated_positive:
        print(f"\n[NOTE] {len(saturated_positive)}/{len(results)} positive-direction trial(s) ended with "
              f"the integral pinned at +/-{args.integral_max:.0f}. If a residual remains, raising "
              f"--integral-max further (or adding --dir-bias) is the next thing to try.")
    elif any(r.target_rad > args.reference_rad for r in results):
        pos_trials = [r for r in results if r.target_rad > args.reference_rad]
        max_pos_err = max(abs(r.final_error_rad) for r in pos_trials) * 1000
        if max_pos_err > 2.0:
            print(f"\n[NOTE] Positive-direction trials did NOT saturate the integral, but still show "
                  f"up to {max_pos_err:.2f} mrad residual. Raising --integral-max will not fix this - "
                  f"it points to an uncaptured bias in the feedforward model (see --dir-bias).")


if __name__ == "__main__":
    main()