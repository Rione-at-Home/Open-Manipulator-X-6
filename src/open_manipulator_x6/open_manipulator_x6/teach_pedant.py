#!/usr/bin/env python3
"""
X6 teach pendant: gamepad teleop + PyQt5 window for saving named poses.

Run (arm_driver and the FlexiGrasp pad node already running):
    ros2 launch open_manipulator_x6 teach.launch.py
or by hand:
    ros2 run joy joy_node
    ros2 run open_manipulator_x6 teach_pendant --ros-args --params-file config/joy_ps4.yaml

Gamepad: L1/R1 pick joint, circle/cross move it (gripper: close/open), triangle fine/coarse,
square save pose (name from the window), options STOP.

Poses store arm joints only (measured from /joint_states). The gripper is open/close actions.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Empty, Int16, Int32

from .joy_teleop_core import ButtonMap, JoyTeleopCore
from .pose_store import PoseError, PoseStore

try:
    from flexigrasp_msgs.msg import TactileFrame
except ImportError:  # pendant still works without the pad package; pad display shows "--"
    TactileFrame = None

DEFAULT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "wrist_rotate", "gripper"]


class PendantNode(Node):
    def __init__(self):
        super().__init__("teach_pendant")
        P = self.declare_parameter
        P("joint_order", DEFAULT_JOINTS)
        P("gripper_name", "gripper")
        P("poses_file", "~/Open-Manipulator-X-6/poses.yaml")
        P("coarse_step", 0.02)          # rad per tick while a button is held (20 Hz -> 0.4 rad/s)
        P("fine_step", 0.004)
        P("gripper_step", 0.01)
        P("close_sign", -1.0)           # gripper position decreases when closing
        P("contact_stop", True)
        P("idle_resync_s", 1.0)
        P("joint_limits", [0.0])        # flat [lo, hi] per joint in joint_order; [0.0] = +/- pi
        P("rate_hz", 20.0)
        P("gripper_current", 100)       # must match arm_driver's gripper_goal_current at start
        P("cap_step", 10)
        P("cap_max", 200)
        P("profile_velocity", 40)       # must match arm_driver's profile_velocity at start
        P("velocity_step", 5)
        P("velocity_max", 100)
        P("fingers", ["left", "right"])
        P("topic_prefix", "flexigrasp")
        for k, v in ButtonMap().__dict__.items():
            P(f"buttons.{k}", v)
        g = lambda n: self.get_parameter(n).value

        order = list(g("joint_order"))
        lim_flat = list(g("joint_limits"))
        limits = {}
        if len(lim_flat) == 2 * len(order):
            limits = {n: (lim_flat[2 * i], lim_flat[2 * i + 1]) for i, n in enumerate(order)}
        elif lim_flat not in ([0.0], []):
            self.get_logger().warn("joint_limits must have 2 values per joint; using +/- pi")
        buttons = ButtonMap.from_dict({k: g(f"buttons.{k}") for k in ButtonMap().__dict__})
        self.gripper = g("gripper_name")
        self.core = JoyTeleopCore(order, self.gripper, g("coarse_step"), g("fine_step"),
                                  g("gripper_step"), limits, g("close_sign"), g("contact_stop"),
                                  g("idle_resync_s"), buttons)
        self.arm_joints = [n for n in order if n != self.gripper]
        self.store = PoseStore(g("poses_file"))
        self.cap, self.cap_step, self.cap_max = g("gripper_current"), g("cap_step"), g("cap_max")
        self.vel, self.vel_step, self.vel_max = g("profile_velocity"), g("velocity_step"), g("velocity_max")

        self.measured, self.velocity = {}, {}
        self.pads = {}
        self.fingers = list(g("fingers"))
        self.on_save_request = None     # set by the window
        self.joy_seen = False

        self.pub_cmd = self.create_publisher(JointState, "/arm_command", 10)
        self.pub_stop = self.create_publisher(Empty, "/arm_stop", 10)
        self.pub_cap = self.create_publisher(Int16, "/arm_gripper_current", 10)
        self.pub_vel = self.create_publisher(Int32, "/arm_profile_velocity", 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.create_subscription(Joy, "/joy", self.on_joy, 10)
        if TactileFrame is not None:
            for f in self.fingers:
                self.create_subscription(TactileFrame, f"/{g('topic_prefix')}/{f}/frame",
                                         lambda m, f=f: self.on_pad(f, m), qos_profile_sensor_data)
        self.create_timer(1.0 / g("rate_hz"), self.tick)

    # ---- callbacks ------------------------------------------------------------------------------
    def on_joint_states(self, m: JointState):
        self.measured = dict(zip(m.name, m.position))
        self.velocity = dict(zip(m.name, m.velocity))
        if not self.core.ready:
            self.core.sync(m.name, m.position)

    def on_pad(self, f, m):
        self.pads[f] = (time.monotonic(), bool(m.in_contact), float(m.total))

    def both_in_contact(self) -> bool:
        now = time.monotonic()
        return bool(self.fingers) and all(
            f in self.pads and now - self.pads[f][0] < 0.5 and self.pads[f][1] for f in self.fingers)

    def on_joy(self, m: Joy):
        self.joy_seen = True
        events = self.core.on_joy(list(m.buttons))
        if "stop" in events:
            self.stop()
        if "save" in events and self.on_save_request:
            self.on_save_request()

    def tick(self):
        if self.core.tick(time.monotonic(), self.measured, self.both_in_contact()):
            self.send()

    # ---- actions ------------------------------------------------------------------------------
    def send(self):
        names, pos = self.core.command()
        if not names:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = names, pos
        self.pub_cmd.publish(msg)

    def stop(self):
        self.pub_stop.publish(Empty())
        self.core.names = []
        self.core.message = "STOP sent; targets re-sync to the arm"

    def save_pose(self, name: str):
        missing = [j for j in self.arm_joints if j not in self.measured]
        if missing:
            raise PoseError(f"no measured position yet for {missing}")
        self.store.set(name, self.arm_joints, [self.measured[j] for j in self.arm_joints])

    def move_to(self, name: str):
        pose = self.store.get(name)
        if not self.core.ready:
            raise PoseError("no /joint_states yet")
        self.core.set_targets(pose["names"], pose["positions"], time.monotonic())
        self.send()

    def set_gripper_open(self):
        if self.gripper not in self.measured:
            raise PoseError("no gripper position yet")
        self.store.set_gripper_open(self.measured[self.gripper])

    def change_cap(self, sign: int):
        self.cap = max(0, min(self.cap_max, self.cap + sign * self.cap_step))
        self.pub_cap.publish(Int16(data=int(self.cap)))

    def change_velocity(self, sign: int):
        self.vel = max(5, min(self.vel_max, self.vel + sign * self.vel_step))
        self.pub_vel.publish(Int32(data=int(self.vel)))


def build_window(node: PendantNode):  # pragma: no cover - exercised by the offscreen smoke test
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QFont
    from PyQt5.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel,
                                 QLineEdit, QListWidget, QMessageBox, QPushButton,
                                 QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("X6 Teach Pendant")
            self.resize(560, 760)
            lay = QVBoxLayout(self)

            self.sel_label = QLabel()
            self.sel_label.setFont(QFont("Sans", 16, QFont.Bold))
            lay.addWidget(self.sel_label)
            row = QHBoxLayout()
            self.mode_label = QLabel()
            row.addWidget(self.mode_label)
            self.contact_box = QCheckBox("Stop closing on contact")
            self.contact_box.setChecked(node.core.contact_stop)
            self.contact_box.toggled.connect(lambda v: setattr(node.core, "contact_stop", v))
            row.addWidget(self.contact_box)
            stop = QPushButton("STOP")
            stop.setStyleSheet("background:#c62828;color:white;font-weight:bold;padding:6px 18px")
            stop.clicked.connect(node.stop)
            row.addWidget(stop)
            lay.addLayout(row)

            self.table = QTableWidget(len(node.core.joint_order), 3)
            self.table.setHorizontalHeaderLabels(["Joint", "Measured (rad)", "Target (rad)"])
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            for i, n in enumerate(node.core.joint_order):
                self.table.setItem(i, 0, QTableWidgetItem(n))
                self.table.setItem(i, 1, QTableWidgetItem("--"))
                self.table.setItem(i, 2, QTableWidgetItem("--"))
            from PyQt5.QtWidgets import QHeaderView
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.resizeRowsToContents()
            rows_h = sum(self.table.rowHeight(i) for i in range(self.table.rowCount()))
            self.table.setFixedHeight(rows_h + self.table.horizontalHeader().height() + 4)
            self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            lay.addWidget(self.table)

            grid = QGridLayout()
            self.pad_label = QLabel()
            grid.addWidget(self.pad_label, 0, 0, 1, 4)
            self.cap_label, self.vel_label = QLabel(), QLabel()
            grid.addWidget(self.cap_label, 1, 0)
            for col, (txt, fn) in enumerate((("-", lambda: node.change_cap(-1)),
                                             ("+", lambda: node.change_cap(+1))), start=1):
                b = QPushButton(txt); b.clicked.connect(fn); grid.addWidget(b, 1, col)
            grid.addWidget(self.vel_label, 2, 0)
            for col, (txt, fn) in enumerate((("-", lambda: node.change_velocity(-1)),
                                             ("+", lambda: node.change_velocity(+1))), start=1):
                b = QPushButton(txt); b.clicked.connect(fn); grid.addWidget(b, 2, col)
            lay.addLayout(grid)

            lay.addWidget(QLabel("Pose name"))
            self.name = QLineEdit()
            self.name.setPlaceholderText("pre_grasp")
            lay.addWidget(self.name)
            btns = QHBoxLayout()
            for txt, fn in (("Save (square)", self.save), ("Update", self.update_pose),
                            ("Move To", self.move_to), ("Delete", self.delete)):
                b = QPushButton(txt); b.clicked.connect(fn); btns.addWidget(b)
            lay.addLayout(btns)
            gopen = QPushButton("Set gripper open = current position")
            gopen.clicked.connect(self.set_gripper_open)
            lay.addWidget(gopen)

            lay.addWidget(QLabel("Saved poses (arm joints only)"))
            self.poses = QListWidget()
            self.poses.itemClicked.connect(lambda it: self.name.setText(it.data(Qt.UserRole)))
            lay.addWidget(self.poses)
            self.status = QLabel()
            self.status.setWordWrap(True)
            lay.addWidget(self.status)
            self.debug = QLabel()
            self.debug.setStyleSheet("color:#777")
            lay.addWidget(self.debug)

            node.on_save_request = self.save
            self.refresh_poses()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)

        # -- helpers --
        def say(self, text, error=False):
            self.status.setStyleSheet("color:#c62828" if error else "")
            self.status.setText(text)

        def selected_pose(self):
            it = self.poses.currentItem()
            return it.data(Qt.UserRole) if it else None

        def refresh_poses(self):
            self.poses.clear()
            from PyQt5.QtWidgets import QListWidgetItem
            for n in node.store.names():
                it = QListWidgetItem(f"{n}    ({node.store.poses[n].get('saved', '')})")
                it.setData(Qt.UserRole, n)
                self.poses.addItem(it)
            g = node.store.gripper_open
            self.poses.setToolTip(f"gripper_open = {g:.3f} rad" if g is not None else "gripper_open not set")

        # -- actions --
        def save(self):
            name = self.name.text().strip()
            if not name:
                return self.say("Type a pose name first.", True)
            if name in node.store.poses and QMessageBox.question(
                    self, "Overwrite?", f"Pose '{name}' exists. Overwrite?") != QMessageBox.Yes:
                return
            try:
                node.save_pose(name)
                self.refresh_poses()
                self.say(f"Saved '{name}'.")
            except PoseError as e:
                self.say(str(e), True)

        def update_pose(self):
            name = self.selected_pose()
            if not name:
                return self.say("Select a pose to update.", True)
            try:
                node.save_pose(name)
                self.refresh_poses()
                self.say(f"Updated '{name}' to the current position.")
            except PoseError as e:
                self.say(str(e), True)

        def move_to(self):
            name = self.selected_pose()
            if not name:
                return self.say("Select a pose to move to.", True)
            try:
                node.move_to(name)
                self.say(f"Moving to '{name}' at profile velocity {node.vel}.")
            except PoseError as e:
                self.say(str(e), True)

        def delete(self):
            name = self.selected_pose()
            if name and QMessageBox.question(self, "Delete?", f"Delete pose '{name}'?") == QMessageBox.Yes:
                node.store.delete(name)
                self.refresh_poses()
                self.say(f"Deleted '{name}'.")

        def set_gripper_open(self):
            try:
                node.set_gripper_open()
                self.refresh_poses()
                self.say(f"gripper_open = {node.store.gripper_open:.3f} rad")
            except PoseError as e:
                self.say(str(e), True)

        def refresh(self):
            c = node.core
            self.sel_label.setText(f"Selected: {c.selected_joint}")
            self.mode_label.setText(f"Step: {'FINE' if c.fine else 'coarse'}")
            for i, n in enumerate(c.joint_order):
                m = node.measured.get(n)
                t = c.targets.get(n)
                self.table.item(i, 1).setText("--" if m is None else f"{m:+.3f}")
                self.table.item(i, 2).setText("--" if t is None else f"{t:+.3f}")
                colour = QColor("#fff3c4") if i == c.selected else QColor("white")
                for col in range(3):
                    self.table.item(i, col).setBackground(colour)
            pads = []
            for f in node.fingers:
                if f in node.pads and time.monotonic() - node.pads[f][0] < 0.5:
                    _, on, tot = node.pads[f]
                    pads.append(f"{f}: {tot:6.0f}{' CONTACT' if on else ''}")
                else:
                    pads.append(f"{f}: --")
            self.pad_label.setText("Pads   " + "    ".join(pads))
            self.cap_label.setText(f"Gripper cap: {node.cap} (~{node.cap * 2.69:.0f} mA)")
            self.vel_label.setText(f"Speed (profile velocity): {node.vel}")
            if c.message and not self.status.text().startswith(c.message):
                self.say(c.message)
            joy = "gamepad OK" if node.joy_seen else "no /joy yet (is joy_node running?)"
            self.debug.setText(f"{joy}; last buttons pressed: {c.last_pressed}")

    return Window


def main(args=None):  # pragma: no cover
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    rclpy.init(args=args)
    node = PendantNode()
    app = QApplication(sys.argv)
    win = build_window(node)()
    win.show()
    spin = QTimer()
    spin.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    spin.start(5)
    try:
        app.exec_()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()