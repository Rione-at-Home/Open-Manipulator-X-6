#
## armgui.py
#
#
# This code is part of the complete_arm_assembly package, which provides a GUI for controlling a robotic arm and its gripper using ROS 2. The GUI allows users to adjust joint angles and control the gripper's open/close state. It uses Tkinter for the graphical interface and communicates with ROS 2 action servers to send commands to the arm and gripper controllers.
#
#

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tkinter as tk
from tkinter import ttk
import threading

class ArmGUI(Node):
    def __init__(self):
        super().__init__('arm_gui')
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

    def send_arm(self, positions, duration_sec=2):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5']
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [pt]
        self.arm_client.wait_for_server()
        self.arm_client.send_goal_async(goal)

    def send_gripper(self, left, right, duration_sec=2):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['left_finger_joint', 'right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [left, right]
        pt.time_from_start = Duration(sec=duration_sec)
        goal.trajectory.points = [pt]
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)


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

    root = tk.Tk()
    root.title('Arm Controller')
    root.geometry('500x520')
    root.configure(bg='#2b2b2b')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TScale', background='#2b2b2b')
    style.configure('TLabel', background='#2b2b2b', foreground='white')
    style.configure('TFrame', background='#2b2b2b')

    sliders = {}

    tk.Label(root, text='Robot Arm Controller', font=('Helvetica', 16, 'bold'),
             bg='#2b2b2b', fg='white').pack(pady=10)

    tk.Label(root, text='── Arm Joints ──', font=('Helvetica', 11),
             bg='#2b2b2b', fg='#aaaaaa').pack()

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
        sl = ttk.Scale(row, from_=lo, to=hi, variable=var, orient='horizontal', length=280)
        sl.pack(side='left', padx=5)
        def update_label(v, lbl=val_label, variable=var):
            lbl.config(text=f'{variable.get():.3f}')
        var.trace_add('write', lambda *a, fn=update_label, vr=var: fn(vr.get()))
        sliders[name] = var

    def send_arm_cmd():
        positions = [sliders[j].get() for j, _, _ in joint_limits]
        threading.Thread(target=node.send_arm, args=(positions,), daemon=True).start()

    def home():
        for j, _, _ in joint_limits:
            sliders[j].set(0.0)
        threading.Thread(target=node.send_arm, args=([0.0]*5,), daemon=True).start()

    tk.Button(root, text='Send Arm Command', command=send_arm_cmd,
              bg='#0055cc', fg='white', font=('Helvetica', 11, 'bold'),
              relief='flat', padx=10, pady=5).pack(pady=8)

    tk.Button(root, text='Home Position', command=home,
              bg='#555555', fg='white', font=('Helvetica', 10),
              relief='flat', padx=10, pady=4).pack(pady=2)

    tk.Label(root, text='── Gripper ──', font=('Helvetica', 11),
             bg='#2b2b2b', fg='#aaaaaa').pack(pady=(12, 0))

    gripper_frame = ttk.Frame(root)
    gripper_frame.pack(pady=5)

    # Open = [0.0, 0.0], Close = [0.019, -0.019]
    def open_gripper():
        threading.Thread(target=node.send_gripper, args=(0.0, 0.0), daemon=True).start()

    def close_gripper():
        threading.Thread(target=node.send_gripper, args=(0.019, -0.019), daemon=True).start()

    btn_frame = ttk.Frame(gripper_frame)
    btn_frame.pack()
    tk.Button(btn_frame, text='Open Gripper', command=open_gripper,
              bg='#007755', fg='white', font=('Helvetica', 11, 'bold'),
              relief='flat', padx=15, pady=8).pack(side='left', padx=10)
    tk.Button(btn_frame, text='Close Gripper', command=close_gripper,
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
