#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import csv
import glob
from pathlib import Path
import matplotlib.pyplot as plt

class SettlingTimeDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Settling Time Characterization Dashboard")
        self.root.geometry("850x700")
        
        # Define the script arguments, their types, and default values matching the script[cite: 2]
        self.args_config = {
            "Connection & Hardware": {
                "--port": ("/dev/ttyUSB0", str),
                "--baudrate": ("1000000", int),
                "--joint-ids": ("11 12 13 14 15 2 6", str),
                "--moving-id": ("11", int)
            },
            "Test Parameters": {
                "--reference-rad": ("0.0", float),
                "--step-rad": ("0.3", float),
                "--trials": ("5", int),
                "--alternate": (False, bool)
            },
            "Settling Criteria": {
                "--noise-calib-duration": ("1.5", float),
                "--noise-k": ("4.0", float),
                "--band-frac-floor": ("0.0", float),
                "--min-band-ticks": ("2.0", float),
                "--dwell-samples": ("10", int),
                "--timeout": ("3.0", float),
                "--return-timeout": ("", str)  # Empty means None[cite: 2]
            },
            "Gains & Outputs": {
                "--kappa": ("1.75", float),
                "--control-rate-hz": ("100.0", float),
                "--set-p": ("", str),
                "--set-i": ("", str),
                "--set-d": ("", str),
                "--out-dir": ("./settling_test_out", str)
            }
        }
        
        self.vars = {}
        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dashboard Parameters (Grid Layout)[cite: 2]
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
                    entry = ttk.Entry(group, textvariable=var, width=15)
                    entry.grid(row=row, column=1, sticky="ew", pady=2)
                    self.vars[arg] = var
                row += 1
            col += 1

        # Run Button[cite: 2]
        self.run_btn = ttk.Button(main_frame, text="Run Test", command=self.start_test)
        self.run_btn.pack(pady=10)
        
        # Console Output[cite: 2]
        self.console = scrolledtext.ScrolledText(main_frame, height=15, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.console.pack(fill=tk.BOTH, expand=True)

    def start_test(self):
        self.run_btn.config(state=tk.DISABLED)
        
        # Use a dashed line to indicate a new test instead of clearing the console[cite: 2]
        self.log("\n" + "-" * 60 + "\n")
        self.log("STARTING NEW TEST RUN\n")
        self.log("-" * 60 + "\n")
        
        # Construct command with the -u flag for unbuffered output to ensure real-time logs[cite: 2]
        cmd = ["python3", "-u", "settling_time_test.py"]
        for arg, var in self.vars.items():
            val = var.get()
            if isinstance(var, tk.BooleanVar):
                if val: cmd.append(arg)
            elif val.strip() != "":
                if arg == "--joint-ids":
                    cmd.append(arg)
                    cmd.extend(val.split())
                else:
                    cmd.extend([arg, val.strip()])
                    
        self.log(f"Executing: {' '.join(cmd)}\n")
        
        # Run in thread to keep GUI responsive[cite: 2]
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
        summary_file = out_dir / "settling_summary.csv"
        
        if not summary_file.exists():
            self.log(f"Error: Could not find {summary_file} to plot.\n")
            return
            
        # Read summary data[cite: 2]
        trials_meta = {}
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trials_meta[int(row["trial"])] = {
                    "moving_id": row["moving_id"],
                    "target_rad": float(row["target_rad"]),
                    "band_rad": float(row["band_rad"]),
                    "settle_time_s": float(row["settle_time_s"]) if row["settle_time_s"] else None
                }

        # Plot all trials[cite: 2]
        trial_files = sorted(glob.glob(str(out_dir / "trial_*.csv")))
        if not trial_files:
            return

        plt.figure(figsize=(10, 6))
        
        for file in trial_files:
            # Extract trial index from filename (e.g. trial_00_step+0.300.csv)[cite: 2]
            trial_idx = int(Path(file).stem.split('_')[1])
            if trial_idx not in trials_meta:
                continue
                
            meta = trials_meta[trial_idx]
            moving_id = meta["moving_id"]
            
            times = []
            positions = []
            
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(float(row["t"]))
                    pos_key = f"pos_{moving_id}"
                    if row.get(pos_key):
                        positions.append(float(row[pos_key]))
                        
            # Plot the response curve[cite: 2]
            line, = plt.plot(times[:len(positions)], positions, label=f'Trial {trial_idx}')
            
            # Draw settling bands and target for each trial[cite: 2]
            plt.axhline(meta["target_rad"], color=line.get_color(), linestyle='--', alpha=0.5)
            plt.fill_between(times[:len(positions)], 
                             meta["target_rad"] - meta["band_rad"], 
                             meta["target_rad"] + meta["band_rad"], 
                             color=line.get_color(), alpha=0.1)
            
            # Mark settle time if achieved[cite: 2]
            if meta["settle_time_s"] is not None:
                plt.axvline(meta["settle_time_s"], color=line.get_color(), linestyle=':', alpha=0.8)

        plt.title(f"Step Response Settling Time Characterization (Joint {moving_id})")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (rad)")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SettlingTimeDashboard(root)
    root.mainloop()