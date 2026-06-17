#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tkinter as tk
from tkinter import ttk
import threading
import time

class ArmGUI(Node):
    def __init__(self):
        super().__init__('arm_gui')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory')

    def send_arm(self, positions, duration_sec=3):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'joint_1','joint_2','joint_3','joint_4','joint_5']
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [pt]
        self.arm_client.wait_for_server()
        future = self.arm_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.05)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.05)

    def send_gripper(self, left, right, duration_sec=2):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'left_finger_joint', 'right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [left, right]
        pt.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [pt]
        self.gripper_client.wait_for_server()
        future = self.gripper_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.05)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.05)


def ros_spin(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = ArmGUI()
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    joint_limits = [
        ('joint_1', -3.14, 3.14),
        ('joint_2', -1.57, 1.57),
        ('joint_3', -1.57, 1.57),
        ('joint_4', -1.00, 2.00),
        ('joint_5', -3.14, 3.14),
    ]

    RAISED_POSITION = [0.2, -0.5, 0.5, 1.0, 0.0]
    HOME_POSITION   = [0.0,  0.0, 0.0, 0.0, 0.0]

    # Pick and place waypoints — tuned for bag at x=0.15, y=0.0
    # Adjust these if the bag position changes
    PREGRASP  = [0.0,  0.3, 0.3,  0.8,  0.0]  # hover above bag
    GRASP     = [0.0,  0.5, 0.4,  1.2,  0.0]  # descend to bag height
    LIFT      = [0.0,  0.2, 0.3,  0.8,  0.0]  # lift bag up
    CARRY     = [1.2,  0.2, 0.3,  0.8,  0.0]  # rotate to place position
    PLACE     = [1.2,  0.5, 0.4,  1.2,  0.0]  # descend to place height
    RETRACT   = [1.2,  0.2, 0.3,  0.8,  0.0]  # lift back up after release

    root = tk.Tk()
    root.title('Arm Controller')
    root.geometry('520x680')
    root.configure(bg='#2b2b2b')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TScale', background='#2b2b2b')
    style.configure('TLabel', background='#2b2b2b', foreground='white')
    style.configure('TFrame', background='#2b2b2b')

    sliders = {}
    abort_flag = threading.Event()

    tk.Label(root, text='Robot Arm Controller', font=('Helvetica', 16, 'bold'),
             bg='#2b2b2b', fg='white').pack(pady=10)

    # Status bar
    status_var = tk.StringVar(value='Ready')
    status_label = tk.Label(root, textvariable=status_var, font=('Helvetica', 10),
                            bg='#2b2b2b', fg='#ffcc00')
    status_label.pack()

    # Progress bar
    progress = ttk.Progressbar(root, orient='horizontal', length=460, mode='determinate')
    progress.pack(pady=4)

    def set_status(msg, pct=None):
        status_var.set(msg)
        if pct is not None:
            progress['value'] = pct

    # --- Pick and Place ---
    tk.Label(root, text='── Autonomous Pick & Place ──', font=('Helvetica', 11),
             bg='#2b2b2b', fg='#aaaaaa').pack(pady=(8, 0))

    pp_frame = ttk.Frame(root)
    pp_frame.pack(pady=6)

    def pick_and_place():
        abort_flag.clear()

        def run():
            steps = [
                ('Opening gripper',          5,  lambda: node.send_gripper(0.0, 0.0, 1)),
                ('Moving to pre-grasp',      15, lambda: node.send_arm(PREGRASP, 3)),
                ('Descending to bag',        30, lambda: node.send_arm(GRASP, 2)),
                ('Closing gripper on bag',   45, lambda: node.send_gripper(0.019, -0.019, 2)),
                ('Lifting bag',              55, lambda: node.send_arm(LIFT, 2)),
                ('Carrying to place pos',    70, lambda: node.send_arm(CARRY, 3)),
                ('Placing bag down',         82, lambda: node.send_arm(PLACE, 2)),
                ('Releasing bag',            90, lambda: node.send_gripper(0.0, 0.0, 2)),
                ('Retracting arm',           95, lambda: node.send_arm(RETRACT, 2)),
                ('Returning home',          100, lambda: node.send_arm(HOME_POSITION, 3)),
            ]
            for msg, pct, action in steps:
                if abort_flag.is_set():
                    set_status('Aborted', 0)
                    return
                set_status(msg, pct)
                action()
                time.sleep(0.3)
            set_status('Pick & place complete!', 100)

        threading.Thread(target=run, daemon=True).start()

    def abort():
        abort_flag.set()
        set_status('Aborting...', 0)

    tk.Button(pp_frame, text='▶  Pick & Place', command=pick_and_place,
              bg='#336600', fg='white', font=('Helvetica', 12, 'bold'),
              relief='flat', padx=18, pady=8).pack(side='left', padx=8)

    tk.Button(pp_frame, text='■  Abort', command=abort,
              bg='#880000', fg='white', font=('Helvetica', 12, 'bold'),
              relief='flat', padx=18, pady=8).pack(side='left', padx=8)

    # --- Presets ---
    tk.Label(root, text='── Presets ──', font=('Helvetica', 11),
             bg='#2b2b2b', fg='#aaaaaa').pack(pady=(10, 0))

    preset_frame = ttk.Frame(root)
    preset_frame.pack(pady=5)

    def raised_position():
        for i, (j, _, _) in enumerate(joint_limits):
            sliders[j].set(RAISED_POSITION[i])
        def run():
            set_status('Moving to raised position...', 50)
            node.send_arm(RAISED_POSITION, duration_sec=3)
            set_status('Raised position reached', 100)
        threading.Thread(target=run, daemon=True).start()

    def home_position():
        for i, (j, _, _) in enumerate(joint_limits):
            sliders[j].set(HOME_POSITION[i])
        def run():
            set_status('Returning to home...', 50)
            node.send_arm(HOME_POSITION, duration_sec=3)
            set_status('Home position reached', 100)
        threading.Thread(target=run, daemon=True).start()

    tk.Button(preset_frame, text='Raised Position', command=raised_position,
              bg='#225588', fg='white', font=('Helvetica', 10, 'bold'),
              relief='flat', padx=12, pady=6).pack(side='left', padx=8)

    tk.Button(preset_frame, text='Home Position', command=home_position,
              bg='#555555', fg='white', font=('Helvetica', 10, 'bold'),
              relief='flat', padx=12, pady=6).pack(side='left', padx=8)

    # --- Arm sliders ---
    tk.Label(root, text='── Arm Joints ──', font=('Helvetica', 11),
             bg='#2b2b2b', fg='#aaaaaa').pack(pady=(10, 0))

    arm_frame = ttk.Frame(root)
    arm_frame.pack(fill='x', padx=20, pady=5)

    for name, lo, hi in joint_limits:
        row = ttk.Frame(arm_frame)
        row.pack(fill='x', pady=3)
        ttk.Label(row, text=f'{name}', width=10).pack(side='left')
        var = tk.DoubleVar(value=0.0)
        val_label = tk.Label(row, text='0.000', width=6, bg='#2b2b2b', fg='#00ff88',
                             font=('Courier', 10))
        val_label.pack(side='right')
        ttk.Scale(row, from_=lo, to=hi, variable=var,
                  orient='horizontal', length=280).pack(side='left', padx=5)
        var.trace_add('write', lambda *a, lbl=val_label,
                      vr=var: lbl.config(text=f'{vr.get():.3f}'))
        sliders[name] = var

    def send_arm_cmd():
        positions = [sliders[j].get() for j, _, _ in joint_limits]
        def run():
            set_status('Sending arm command...', 50)
            node.send_arm(positions, duration_sec=3)
            set_status('Arm command reached', 100)
        threading.Thread(target=run, daemon=True).start()

    tk.Button(root, text='Send Arm Command', command=send_arm_cmd,
              bg='#0055cc', fg='white', font=('Helvetica', 11, 'bold'),
              relief='flat', padx=10, pady=5).pack(pady=8)

    # --- Gripper ---
    tk.Label(root, text='── Gripper ──', font=('Helvetica', 11),
             bg='#2b2b2b', fg='#aaaaaa').pack(pady=(4, 0))

    gripper_frame = ttk.Frame(root)
    gripper_frame.pack(pady=5)

    tk.Button(gripper_frame, text='Open Gripper',
              command=lambda: [
                  threading.Thread(target=node.send_gripper,
                                   args=(0.0, 0.0), daemon=True).start(),
                  set_status('Opening gripper...')],
              bg='#007755', fg='white', font=('Helvetica', 11, 'bold'),
              relief='flat', padx=15, pady=8).pack(side='left', padx=10)

    tk.Button(gripper_frame, text='Close Gripper',
              command=lambda: [
                  threading.Thread(target=node.send_gripper,
                                   args=(0.019, -0.019), daemon=True).start(),
                  set_status('Closing gripper...')],
              bg='#aa3300', fg='white', font=('Helvetica', 11, 'bold'),
              relief='flat', padx=15, pady=8).pack(side='left', padx=10)

    def on_close():
        node.destroy_node()
        rclpy.shutdown()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
