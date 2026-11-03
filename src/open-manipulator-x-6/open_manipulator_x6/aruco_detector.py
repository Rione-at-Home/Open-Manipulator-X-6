#!/usr/bin/env python3
"""
Workspace marker checker: live view of the ArUco markers from the world camera (RealSense D435
colour stream), their positions in the robot frame, how far the bottle is from its start/place
reference, and a Snapshot button that appends one row to a CSV.

    ros2 launch open_manipulator_x6 workspace_markers.launch.py

Camera frame (for a camera that moves, such as the wrist camera):
    ros2 run open_manipulator_x6 aruco_detector --ros-args -r __node:=wrist_aruco -p frame:=camera \
        -p image_topic:=/wrist_cam/image_raw -p camera_info_topic:=/wrist_cam/camera_info
  Shows each marker's position from the lens (x right in the image, y down, z out of the lens)
  and its straight-line distance. Needs no base marker; the lock and references are hidden.

Buttons (robot frame)
- Lock base: with the camera and base marker still, press once. The base pose is averaged over a
  few seconds and frozen, which steadies every position. Press again (Re-lock) if the camera or
  the base is moved; the window warns when the live base no longer matches the lock.
- Set start reference / Set place reference: with the base locked and the bottle placed correctly
  by hand, press to remember where it sits relative to that marker. Saved to refs_file.
- Snapshot: writes the current marker positions and bottle deviations, with the label you type,
  to log_file (median of the last few frames).
"""
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .aruco_core import (REFERENCE_TARGETS, ArucoTracker, MarkerSpec, load_references,
                         save_references)


def image_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> BGR array for the common USB-camera encodings (no cv_bridge needed)."""
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("mono8", "8uc1"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        img = buf.reshape(msg.height, msg.step // 2, 2)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if enc == "uyvy":
        img = buf.reshape(msg.height, msg.step // 2, 2)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"unsupported image encoding '{msg.encoding}' (use rgb8 or bgr8)")


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("aruco_detector")
        P = self.declare_parameter
        P("image_topic", "/camera/camera/color/image_raw")
        P("camera_info_topic", "/camera/camera/color/camera_info")
        P("dictionary", "DICT_4X4_50")
        P("marker_ids", [0, 1, 2, 3])
        P("marker_names", ["base", "bottle_start", "bottle_place", "bottle"])
        P("marker_sizes_mm", [40.0, 40.0, 40.0, 30.0])   # measured outer edge of the black square
        P("base_offset_mm", [0.0, 0.0, 0.0])             # base marker centre in the robot frame
        P("base_yaw_deg", -90.0)                         # -90 = printed arrow points to robot front
        P("tolerance_mm", 10.0)                          # bottle deviation shown as OK below this
        P("refs_file", "~/Open-Manipulator-X-6/aruco_refs.yaml")
        P("log_file", "")                                # default depends on frame (below)
        P("process_hz", 15.0)
        P("frame", "robot")                              # robot (needs the base marker) or camera
        g = lambda n: self.get_parameter(n).value
        self.frame = str(g("frame")).strip().lower()
        if self.frame not in ("robot", "camera"):
            raise ValueError("frame must be 'robot' or 'camera'")

        ids, names, sizes = list(g("marker_ids")), list(g("marker_names")), list(g("marker_sizes_mm"))
        if not (len(ids) == len(names) == len(sizes)):
            raise ValueError("marker_ids, marker_names and marker_sizes_mm must be the same length")
        self.tracker = ArucoTracker([MarkerSpec(int(i), n, float(s)) for i, n, s in zip(ids, names, sizes)],
                                    g("dictionary"), list(g("base_offset_mm")), float(g("base_yaw_deg")))
        self.tolerance = float(g("tolerance_mm"))
        self.refs_file = g("refs_file")
        default_log = ("~/flexigrasp_logs/aruco_snapshots.csv" if self.frame == "robot"
                       else f"~/flexigrasp_logs/aruco_snapshots_{self.get_name()}.csv")
        self.log_file = os.path.expanduser(g("log_file") or default_log)
        self.tracker.references = load_references(self.refs_file)

        self.latest_msg = None
        self.view = None                 # last annotated BGR frame for the GUI
        self.result = {}
        self.error = ""
        self.create_subscription(Image, g("image_topic"), self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g("camera_info_topic"), self.on_info, qos_profile_sensor_data)
        self.create_timer(1.0 / g("process_hz"), self.process)
        self.get_logger().info(f"Listening on {g('image_topic')} ({self.frame} frame); references: "
                               f"{sorted(self.tracker.references) or 'none yet'}")

    def on_info(self, m: CameraInfo):
        self.tracker.set_camera(m.k, m.d)

    def on_image(self, m: Image):
        self.latest_msg = m              # only the newest frame is processed

    def process(self):
        m, self.latest_msg = self.latest_msg, None
        if m is None:
            return
        try:
            bgr = image_to_bgr(m)
        except ValueError as e:
            self.error = str(e)
            return
        now = time.monotonic()
        self.result = self.tracker.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), now)
        self.view = self.tracker.draw(bgr, self.result)
        self.error = ""

    # ---- actions used by the GUI ----
    def lock_base(self) -> str:
        if not self.tracker.calibrated:
            raise ValueError("camera_info is empty, so the base pose cannot be measured yet")
        lk = self.tracker.lock_base(time.monotonic())
        msg = (f"Base locked from {lk.samples} frames (spread {lk.spread_mm:.1f} mm, "
               f"{lk.spread_deg:.1f} deg)")
        self.get_logger().info(msg)
        return msg

    def set_reference(self, target: str) -> str:
        p = self.tracker.set_reference(target, time.monotonic())
        save_references(self.refs_file, self.tracker.references)
        return f"{target} reference saved ({p[0]:+.1f}, {p[1]:+.1f}) mm"

    def snapshot(self, label: str) -> str:
        now = time.monotonic()
        stamp = datetime.now().isoformat(timespec="seconds")
        row = (self.tracker.snapshot_row(label, stamp, now) if self.frame == "robot"
               else self.tracker.snapshot_row_camera(label, stamp, now))
        path = self.log_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stem, ext = os.path.splitext(self.log_file)
        k = 0
        while os.path.exists(path):                  # never mix column layouts in one file
            with open(path, newline="") as f:
                header = next(csv.reader(f), [])
            if not header or header == list(row):
                break
            k += 1
            path = f"{stem}_{self.frame}{'' if k == 1 else f'_{k}'}{ext}"
        self.log_file = path
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        return f"snapshot '{label}' saved to {path}"


def build_window(node: ArucoDetectorNode):  # pragma: no cover - exercised by the offscreen test
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
                                 QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

    names = list(node.tracker.by_name)
    camera_mode = node.frame == "camera"
    columns = (["Marker", "Seen", "x (mm)", "y (mm)", "z (mm)", "dist (mm)"] if camera_mode
               else ["Marker", "Seen", "x (mm)", "y (mm)", "z (mm)"])

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("X6 Markers (camera frame)" if camera_mode else "X6 Workspace Markers")
            lay = QVBoxLayout(self)
            self.image = QLabel("waiting for camera ...")
            self.image.setAlignment(Qt.AlignCenter)
            self.image.setMinimumSize(640, 360)
            lay.addWidget(self.image)

            if camera_mode:
                note = QLabel("Camera frame: each marker's position from the lens. x = right in "
                              "the image, y = down, z = out of the lens; dist = straight-line "
                              "distance. Check dist with a tape measure from the lens.")
                note.setWordWrap(True)
                lay.addWidget(note)
            self.table = QTableWidget(len(names), len(columns))
            self.table.setHorizontalHeaderLabels(columns)
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for r, n in enumerate(names):
                for c, v in enumerate([n] + [""] * (len(columns) - 1)):
                    self.table.setItem(r, c, QTableWidgetItem(v))
            self.table.resizeRowsToContents()
            h = sum(self.table.rowHeight(i) for i in range(len(names)))
            self.table.setFixedHeight(h + self.table.horizontalHeader().height() + 4)
            lay.addWidget(self.table)

            self.dev = {}
            if not camera_mode:                       # base lock and references: robot frame only
                base = QHBoxLayout()
                self.lock_btn = QPushButton("Lock base")
                self.lock_btn.clicked.connect(self.lock)
                base.addWidget(self.lock_btn)
                self.base_state = QLabel()
                self.base_state.setWordWrap(True)
                base.addWidget(self.base_state, 1)
                lay.addLayout(base)

                self.dev = {t: QLabel() for t in REFERENCE_TARGETS}
                for t in REFERENCE_TARGETS:
                    lay.addWidget(self.dev[t])

                refs = QHBoxLayout()
                for t, txt in (("bottle_start", "Set start reference"), ("bottle_place", "Set place reference")):
                    b = QPushButton(txt)
                    b.clicked.connect(lambda _=False, t=t: self.set_ref(t))
                    refs.addWidget(b)
                lay.addLayout(refs)

            snap = QHBoxLayout()
            self.label = QLineEdit()
            self.label.setPlaceholderText("label, e.g. bottle_1 before")
            snap.addWidget(self.label)
            b = QPushButton("Snapshot")
            b.clicked.connect(self.snapshot)
            snap.addWidget(b)
            lay.addLayout(snap)
            self.status = QLabel()
            self.status.setWordWrap(True)
            lay.addWidget(self.status)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(100)

        def say(self, text, error=False):
            self.status.setStyleSheet("color:#c62828" if error else "")
            self.status.setText(text)

        def lock(self):
            try:
                self.say(node.lock_base())
            except ValueError as e:
                self.say(str(e), True)

        def set_ref(self, target):
            try:
                self.say(node.set_reference(target))
            except ValueError as e:
                self.say(str(e), True)

        def snapshot(self):
            label = self.label.text().strip()
            if not label:
                return self.say("Type a label first.", True)
            self.say(node.snapshot(label))

        def refresh(self):
            if node.view is not None:
                rgb = cv2.cvtColor(node.view, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                self.image.setPixmap(QPixmap.fromImage(qimg).scaled(
                    self.image.width(), self.image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            now = time.monotonic()
            for r, n in enumerate(names):
                seen = node.tracker.seen(n, now) or any(e["name"] == n for e in node.result.values())
                self.table.item(r, 1).setText("yes" if seen else "no")
                if camera_mode:
                    pose = node.tracker.camera_pose(n, now)
                    cells = list(pose[0]) + [pose[1]] if pose else ["--"] * 4
                else:
                    pose = None if n == "base" else node.tracker.robot_pose(n, now)
                    cells = ("ref", "ref", "ref") if n == "base" else (pose[0] if pose else ("--", "--", "--"))
                for c, v in enumerate(cells, start=2):
                    if isinstance(v, str):
                        self.table.item(r, c).setText(v)
                    else:
                        self.table.item(r, c).setText(f"{v:.1f}" if c == 5 else f"{v:+.1f}")
            if node.error:
                self.say(node.error, True)
            elif not node.tracker.calibrated and node.view is not None:
                self.say("Camera not calibrated (camera_info is empty): markers are detected but "
                         "positions need a calibration.", True)
            if camera_mode:
                return
            tr = node.tracker
            if tr.locked:
                self.lock_btn.setText("Re-lock base")
                moved = tr.base_moved(now)
                if moved:
                    self.base_state.setText(f"Base moved? Live base is {moved[0]:.1f} mm / "
                                            f"{moved[1]:.1f} deg from the lock. Re-lock if the "
                                            f"camera or base was bumped.")
                    self.base_state.setStyleSheet("color:#c62828")
                else:
                    self.base_state.setText(f"Base locked ({tr.lock.samples} frames, spread "
                                            f"{tr.lock.spread_mm:.1f} mm / {tr.lock.spread_deg:.1f} deg)")
                    self.base_state.setStyleSheet("color:#2e7d32")
            else:
                self.lock_btn.setText("Lock base")
                self.base_state.setText("Base not locked: positions follow the live base marker "
                                        "and jitter more. Keep everything still and press Lock base.")
                self.base_state.setStyleSheet("color:#777")
            for t in REFERENCE_TARGETS:
                d = node.tracker.deviation(t, now)
                short = t.replace("bottle_", "")
                if not node.tracker.locked:
                    txt, colour = f"Bottle vs {short}: lock the base first", "#777"
                elif t not in node.tracker.references:
                    txt, colour = f"Bottle vs {short}: no reference yet", "#777"
                elif d is None:
                    txt, colour = f"Bottle vs {short}: markers not in view", "#777"
                else:
                    ok = d[2] <= node.tolerance
                    txt = (f"Bottle vs {short}: {d[2]:.1f} mm  (dx {d[0]:+.1f}, dy {d[1]:+.1f})  "
                           f"{'OK' if ok else 'OFF'}")
                    colour = "#2e7d32" if ok else "#c62828"
                self.dev[t].setText(txt)
                self.dev[t].setStyleSheet(f"color:{colour}")

    return Window


def main(args=None):  # pragma: no cover
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    rclpy.init(args=args)
    node = ArucoDetectorNode()
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