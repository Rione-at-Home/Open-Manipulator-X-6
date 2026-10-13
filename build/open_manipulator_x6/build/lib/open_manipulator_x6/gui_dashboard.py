#!/usr/bin/env python3
"""
PWM Feedforward+PID Controller — Test Dashboard.

*** UPDATED 2026-10-01 ***
Previously wrapped settling_time_test.py (the Dynamixel built-in position
PID, in Position Control Mode). That script is NOT what's been used for
the directional-residual investigation - current_control_wrist.py (the
software PWM feedforward+PID loop) is. This dashboard now drives that
script instead, with fields matching its actual CLI flags, and defaults
pre-filled with tonight's last-known values plus the next planned test
(--integral-max 500) so opening this tomorrow and clicking "Run Test"
reproduces the next diagnostic step with no retyping.

The plot also now reads the new segment/summary CSV schema
(pwm_control_summary.csv + segment_XX_step_targetYYY.csv) and flags any
trial whose integral term ended pinned at the clamp ([SAT] in the legend)
so you can see saturation at a glance instead of cross-referencing CSVs.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import csv
import glob
from pathlib import Path
import matplotlib.pyplot as plt

class PwmControlDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("PWM Feedforward+PID Controller — Test Dashboard")
        self.root.geometry("900x760")

        # Defaults match current_control_wrist.py's CLI flags. Values below
        # reflect the last confirmed-good configuration plus the next
        # planned diagnostic step (integral-max 150 -> 500), not the
        # script's own stock argparse defaults.
        self.args_config = {
            "Connection & Hardware": {
                "--port": ("/dev/ttyUSB0", str),
                "--baudrate": ("1000000", int),
                "--joint-ids": ("1 2 3 4 5 6 7", str),
                "--moving-id": ("6", int),
            },
            "Test Parameters": {
                "--reference-rad": ("0.0", float),
                "--step-rad": ("0.02", float),
                "--trials": ("6", int),
                "--alternate": (True, bool),
            },
            "Control Gains": {
                "--kp": ("2000", float),
                "--ki": ("80", float),
                "--kd": ("30", float),
                "--integral-max": ("500", float),
                "--dir-bias": ("0", float),
            },
            "Feedforward": {
                "--gravity-model": ("./gravity_calib_out/gravity_model.json", str),
                "--friction-comp": ("40", float),
                "--friction-eps-rad": ("0.003", float),
            },
            "Safety & Outputs": {
                "--max-pwm-raw": ("350", float),
                "--max-comm-failures": ("20", int),
                "--out-dir": ("./pwm_control_out", str),
            },
            "Settling Criteria": {
                "--noise-calib-duration": ("1.5", float),
                "--noise-k": ("4.0", float),
                "--min-band-ticks": ("2.0", float),
                "--dwell-samples": ("10", int),
                "--timeout": ("3.0", float),
            },
        }

        self.vars = {}
        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        note = ttk.Label(
            main_frame,
            text="Driving current_control_wrist.py (PWM feedforward+PID). "
                 "integral-max defaults to 500 for tomorrow's clamp test.",
            foreground="#555555",
        )
        note.pack(fill=tk.X, pady=(0, 6))

        params_frame = ttk.Frame(main_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        col = 0
        for group_name, params in self.args_config.items():
            group = ttk.LabelFrame(params_frame, text=group_name, padding="10")
            group.grid(row=0, column=col, sticky="nsew", padx=5)
            params_frame.columnconfigure(col, weight=1)

            row = 0
            for arg, (default, typ) in params.items():
                ttk.Label(group, text=arg).grid(row=row, column=0, sticky="w", pady=2)

                if typ == bool:
                    var = tk.BooleanVar(value=default)
                    cb = ttk.Checkbutton(group, variable=var)
                    cb.grid(row=row, column=1, sticky="w", pady=2)
                    self.vars[arg] = var
                else:
                    var = tk.StringVar(value=str(default))
                    entry = ttk.Entry(group, textvariable=var, width=16)
                    entry.grid(row=row, column=1, sticky="ew", pady=2)
                    self.vars[arg] = var
                row += 1
            col += 1

        self.run_btn = ttk.Button(main_frame, text="Run Test", command=self.start_test)
        self.run_btn.pack(pady=10)

        self.console = scrolledtext.ScrolledText(main_frame, height=15, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.console.pack(fill=tk.BOTH, expand=True)

    def start_test(self):
        self.run_btn.config(state=tk.DISABLED)

        self.log("\n" + "-" * 60 + "\n")
        self.log("STARTING NEW TEST RUN\n")
        self.log("-" * 60 + "\n")

        cmd = ["python3", "-u", "current_control_wrist.py"]
        for arg, var in self.vars.items():
            val = var.get()
            if isinstance(var, tk.BooleanVar):
                if val:
                    cmd.append(arg)
            elif val.strip() != "":
                if arg == "--joint-ids":
                    cmd.append(arg)
                    cmd.extend(val.split())
                else:
                    cmd.extend([arg, val.strip()])

        self.log(f"Executing: {' '.join(cmd)}\n")

        threading.Thread(target=self.run_process, args=(cmd,), daemon=True).start()

    def run_process(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                self.root.after(0, self.log, line)
        self.root.after(0, self.test_finished)

    def log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def test_finished(self):
        self.run_btn.config(state=tk.NORMAL)
        self.log("\nTest complete. Generating visualization...\n")
        self.plot_results()

    def plot_results(self):
        out_dir = Path(self.vars["--out-dir"].get())
        summary_file = out_dir / "pwm_control_summary.csv"

        if not summary_file.exists():
            self.log(f"Error: Could not find {summary_file} to plot.\n")
            return

        trials_meta = []
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trials_meta.append({
                    "idx": int(row["idx"]),
                    "label": row["label"],
                    "target_rad": float(row["target_rad"]),
                    "band_rad": float(row["band_rad"]),
                    "settle_time_s": float(row["settle_time_s"]) if row["settle_time_s"] else None,
                    "final_integral": float(row["final_integral"]) if row.get("final_integral") else None,
                    "integral_saturated": str(row.get("integral_saturated", "")).strip().lower() in ("true", "1"),
                })

        if not trials_meta:
            self.log("No trials found in summary.\n")
            return

        plt.figure(figsize=(10, 6))

        for meta in trials_meta:
            # current_control_wrist.py's results[] only holds "step" segments.
            # Those are written with file idx = 2*trial_i + 1 (return segments
            # are the even-numbered files and aren't plotted here).
            file_idx = 2 * meta["idx"] + 1
            matches = glob.glob(str(out_dir / f"segment_{file_idx:02d}_step_target*.csv"))
            if not matches:
                continue
            file = matches[0]

            times, positions = [], []
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(float(row["t"]))
                    positions.append(float(row["pos"]))

            sat_tag = " [SAT]" if meta["integral_saturated"] else ""
            line, = plt.plot(times, positions, label=f'Trial {meta["idx"]} ({meta["target_rad"]:+.3f} rad){sat_tag}')

            plt.axhline(meta["target_rad"], color=line.get_color(), linestyle='--', alpha=0.5)
            plt.fill_between(times,
                             meta["target_rad"] - meta["band_rad"],
                             meta["target_rad"] + meta["band_rad"],
                             color=line.get_color(), alpha=0.1)

            if meta["settle_time_s"] is not None:
                plt.axvline(meta["settle_time_s"], color=line.get_color(), linestyle=':', alpha=0.8)

        moving_id = self.vars["--moving-id"].get()
        plt.title(f"PWM Feedforward+PID Step Response (Joint {moving_id})\n"
                  f"[SAT] = integral pinned at --integral-max when the segment ended")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (rad)")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = PwmControlDashboard(root)
    root.mainloop()
