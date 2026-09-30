#!/usr/bin/env python3
"""
Joint 6 Static Gravity Calibration Script.

Sweeps Joint 6 through its angular range, records steady-state holding currents
at static poses, and fits both a sinusoidal model:
    I_ff(theta) = A * cos(theta) + B * sin(theta) + C
and a linear lookup table (LUT) array for runtime feedforward compensation.

Usage:
    python3 calibrate_gravity_j6.py --port /dev/ttyUSB0 --moving-id 6 --min-rad -1.2 --max-rad 1.2 --steps 13
"""

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np

from dynamixel_driver import DynamixelHardwareDriver
from utils import (
    rad_to_ticks,
    ticks_to_rad,
    raw_current_to_effort,
    RAD_PER_TICK,
)


def connect_and_prepare(port: str, baudrate: int, joint_ids: list) -> DynamixelHardwareDriver:
    driver = DynamixelHardwareDriver(port=port, baudrate=baudrate)
    if not driver.connect():
        raise RuntimeError(f"Failed to open {port} @ {baudrate}")

    for jid in joint_ids:
        if not driver.ping(jid):
            raise RuntimeError(f"Failed to ping motor ID {jid} - check wiring/power/ID.")

    driver.set_operating_mode(joint_ids)  # Default position control mode
    if not driver.enable_torque(joint_ids, True):
        raise RuntimeError("Failed to enable torque on one or more joints.")

    return driver


def read_all_positions(driver: DynamixelHardwareDriver, joint_ids: list) -> dict:
    states = driver.read_states(joint_ids)
    return {jid: ticks_to_rad(states[jid]["position"]) for jid in joint_ids if jid in states}

def to_signed_int16(val: float) -> float:
    v = int(round(val))
    if v >= 32768:
        return float(v - 65536)
    return float(v)

def hold_and_sample(
    driver: DynamixelHardwareDriver,
    joint_ids: list,
    moving_id: int,
    target_rad: float,
    hold_time_s: float,
    sample_duration_s: float,
) -> tuple[float, float, float]:
    """
    Commands moving_id to target_rad, waits for hold_time_s to let transients decay,
    then samples actual position and current for sample_duration_s.
    Returns (mean_measured_rad, mean_raw_current, mean_effort).
    """
    hold_positions = read_all_positions(driver, joint_ids)
    target_ticks = [
        rad_to_ticks(target_rad if jid == moving_id else hold_positions.get(jid, 0.0))
        for jid in joint_ids
    ]
    driver.write_positions(joint_ids, target_ticks)

    # Allow mechanical settle before sampling static holding current
    time.sleep(hold_time_s)

    positions = []
    raw_currents = []
    efforts = []

    t0 = time.monotonic()
    while time.monotonic() - t0 < sample_duration_s:
        states = driver.read_states(joint_ids)
        if moving_id in states:
            st = states[moving_id]
            positions.append(ticks_to_rad(st["position"]))
            raw_cur = to_signed_int16(st.get("current", 0))
            raw_currents.append(raw_cur)
            efforts.append(raw_current_to_effort(raw_cur))

    if not positions:
        raise RuntimeError(f"Failed to sample state for Joint {moving_id} at {target_rad:.3f} rad.")

    return (float(np.mean(positions)), float(np.mean(raw_currents)), float(np.mean(efforts)))


def fit_sinusoidal_model(angles: np.ndarray, currents: np.ndarray) -> tuple[float, float, float]:
    """
    Fits I(theta) = A * cos(theta) + B * sin(theta) + C using least squares.
    Returns parameters (A, B, C).
    """
    # Construct design matrix [cos(theta), sin(theta), 1]
    M = np.column_stack([np.cos(angles), np.sin(angles), np.ones_like(angles)])
    params, _, _, _ = np.linalg.lstsq(M, currents, rcond=None)
    return float(params[0]), float(params[1]), float(params[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--joint-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
                    help="All joints to monitor/hold stationary.")
    ap.add_argument("--moving-id", type=int, default=6, help="Joint ID to calibrate.")
    ap.add_argument("--min-rad", type=float, default=-1.2, help="Minimum sweep angle in radians (~ -70 deg).")
    ap.add_argument("--max-rad", type=float, default=1.2, help="Maximum sweep angle in radians (~ +70 deg).")
    ap.add_argument("--steps", type=int, default=13, help="Number of evaluation poses.")
    ap.add_argument("--hold-time", type=float, default=1.5, help="Settling delay before sampling (s).")
    ap.add_argument("--sample-duration", type=float, default=1.0, help="Window length for averaging holding current (s).")
    ap.add_argument("--out-dir", default="./gravity_calib_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = connect_and_prepare(args.port, args.baudrate, args.joint_ids)

    # Generate forward and reverse sweep grids to average out direction-dependent friction hysteresis
    grid_forward = np.linspace(args.min_rad, args.max_rad, args.steps)
    grid_reverse = grid_forward[::-1]

    data = []

    try:
        print(f"\n--- Starting Gravity Calibration Sweep for Joint {args.moving_id} ---")
        print(f"Angle Range: [{args.min_rad:.2f}, {args.max_rad:.2f}] rad | Steps: {args.steps}\n")

        # Forward Pass
        print("Executing Forward Sweep (+ direction) ...")
        for target in grid_forward:
            meas_rad, raw_cur, effort = hold_and_sample(
                driver, args.joint_ids, args.moving_id, target, args.hold_time, args.sample_duration
            )
            print(f"  Target: {target:+.3f} rad | Meas: {meas_rad:+.3f} rad | Hold Current: {raw_cur:+.2f} | Effort: {effort:+.3f}")
            data.append({"direction": "forward", "target_rad": target, "meas_rad": meas_rad, "raw_current": raw_cur, "effort": effort})

        # Return smoothly to start before reverse pass
        time.sleep(0.5)

        # Reverse Pass
        print("\nExecuting Reverse Sweep (- direction) ...")
        for target in grid_reverse:
            meas_rad, raw_cur, effort = hold_and_sample(
                driver, args.joint_ids, args.moving_id, target, args.hold_time, args.sample_duration
            )
            print(f"  Target: {target:+.3f} rad | Meas: {meas_rad:+.3f} rad | Hold Current: {raw_cur:+.2f} | Effort: {effort:+.3f}")
            data.append({"direction": "reverse", "target_rad": target, "meas_rad": meas_rad, "raw_current": raw_cur, "effort": effort})

    finally:
        # Safely return to 0.0 rad pose before shutting down torque
        print("\nReturning Joint 6 to 0.0 rad reference pose...")
        hold_and_sample(driver, args.joint_ids, args.moving_id, 0.0, 1.0, 0.5)
        driver.disconnect()

    # Extract arrays for analysis
    angles = np.array([d["meas_rad"] for d in data])
    raw_currents = np.array([d["raw_current"] for d in data])
    efforts = np.array([d["effort"] for d in data])

    # Fit Sinusoidal Model for raw motor current units
    A, B, C = fit_sinusoidal_model(angles, raw_currents)

    # Compute mean curve per unique angle for LUT generation
    unique_targets = sorted(list(set(d["target_rad"] for d in data)))
    lut_angles = []
    lut_currents = []
    for tgt in unique_targets:
        matched_cur = [d["raw_current"] for d in data if abs(d["target_rad"] - tgt) < 1e-4]
        lut_angles.append(tgt)
        lut_currents.append(float(np.mean(matched_cur)))

    # Save raw CSV
    csv_path = out_dir / "gravity_calibration_raw.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["direction", "target_rad", "meas_rad", "raw_current", "effort"])
        writer.writeheader()
        writer.writerows(data)

    # Save Model Summary JSON
    summary = {
        "joint_id": args.moving_id,
        "sinusoidal_model": {
            "equation": "I_ff(theta) = A * cos(theta) + B * sin(theta) + C",
            "A_cos_amplitude": A,
            "B_sin_amplitude": B,
            "C_bias_offset": C,
        },
        "lookup_table": {
            "angles_rad": lut_angles,
            "holding_currents": lut_currents,
        }
    }

    summary_path = out_dir / "gravity_model.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("CALIBRATION COMPLETE")
    print(f"Fitted Gravity Equation (Raw Current Units):")
    print(f"  I_ff(theta) = ({A:+.3f}) * cos(theta) + ({B:+.3f}) * sin(theta) + ({C:+.3f})")
    print(f"\nOutputs written to: {out_dir.resolve()}")
    print("  - Raw Data CSV: gravity_calibration_raw.csv")
    print("  - Fitted Model & LUT: gravity_model.json")

    # Optional Plotting
    try:
        import matplotlib.pyplot as plt
        theta_grid = np.linspace(min(angles), max(angles), 200)
        i_fit = A * np.cos(theta_grid) + B * np.sin(theta_grid) + C

        plt.figure(figsize=(8, 5))
        plt.plot(angles[:len(grid_forward)], raw_currents[:len(grid_forward)], "ro", label="Forward Sweep (+)")
        plt.plot(angles[len(grid_forward):], raw_currents[len(grid_forward):], "bo", label="Reverse Sweep (-)")
        plt.plot(lut_angles, lut_currents, "k--", alpha=0.5, label="Averaged LUT")
        plt.plot(theta_grid, i_fit, "g-", linewidth=2, label=f"Fit: {A:.1f}cos({B:+.1f}sin){C:+.1f}")
        plt.title(f"Joint {args.moving_id} Gravity Holding Current vs. Position")
        plt.xlabel("Joint Angle (rad)")
        plt.ylabel("Raw Motor Holding Current")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "gravity_holding_curve.png", dpi=150)
        print("  - Visualization Plot: gravity_holding_curve.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()