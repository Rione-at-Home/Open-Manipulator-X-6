#!/usr/bin/env python3
"""
Checkerboard camera calibration with guidance.

Live (ROS 2):
    ros2 run open_manipulator_x6 camera_calibrator --ros-args \
        -p image_topic:=/wrist_cam/image_raw -p camera_name:=wrist_cam

    Show the board to the camera. Steady, new views are captured automatically; the panel says how
    to move for each kind of view still missing. Press Calibrate, check the results, press Save.
    Save writes ~/.ros/camera_info/<camera_name>.yaml (the old file is kept as .bak) and the views
    to ~/calib_<camera_name>/<time>/. Restart the camera node to load the new calibration.

From saved images (no ROS needed), e.g. the ROS calibrator's /tmp/calibrationdata.tar.gz:
    mkdir -p ~/calib_old && tar -xzf /tmp/calibrationdata.tar.gz -C ~/calib_old
    python3 -m open_manipulator_x6.camera_calibrator --images ~/calib_old
    python3 -m open_manipulator_x6.camera_calibrator --images ~/calib_old --save ~/.ros/camera_info/wrist_cam.yaml

Board: 9 x 7 squares of 25 mm = 8 x 6 inner corners (the defaults).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .calibration_core import (DISTANCES, HOW, POSITIONS, TILTS, Board, Collector, calibrate,
                               describe, detect, load_folder, outer_quad, save_samples,
                               write_camera_info)

COLOURS = {"good": "#2e7d32", "warn": "#b26a00", "bad": "#c62828"}


def report(res, extra=""):
    lines = [f"Used {res.used} views" + (f", dropped {len(res.dropped)} with high error" if res.dropped else "")
             + extra]
    for level, name, text in res.checks():
        lines.append(f"[{level.upper():4}] {name}: {text}")
    lines.append(f"K: fx={res.K[0, 0]:.1f} fy={res.K[1, 1]:.1f} cx={res.K[0, 2]:.1f} cy={res.K[1, 2]:.1f}")
    lines.append("D: " + ", ".join(f"{d:+.4f}" for d in res.D))
    return "\n".join(lines)


# ---- offline -----------------------------------------------------------------------------------
def offline(args):
    cols, rows = (int(x) for x in args.size.lower().split("x"))
    board = Board(cols, rows, args.square)
    corners, size, used, skipped = load_folder(args.images, board)
    print(f"{len(used)} images with the board, {len(skipped)} skipped; image size {size[0]}x{size[1]}")
    col = Collector(board)
    for c in corners:
        col.coverage.add(describe(c, board, *size))
    missing = col.coverage.missing()
    print("Coverage missing: " + (", ".join(missing) if missing else "none"))
    res = calibrate(corners, board, size)
    print(report(res))
    if res.dropped:
        print("Dropped: " + ", ".join(os.path.basename(used[i]) for i in res.dropped))
    if args.save:
        if not res.ok() and not args.force:
            print("Not saved: a check failed (see [BAD] above). Collect better views, or add --force.")
            return 1
        backup = write_camera_info(res, args.camera_name, args.save)
        print(f"Saved {os.path.expanduser(args.save)}" + (f" (old file kept as {backup})" if backup else ""))
    return 0


# ---- live GUI ----------------------------------------------------------------------------------
def build_window(get_frame, board, camera_name, output, samples_dir):  # pragma: no cover - offscreen test
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                                 QPushButton, QVBoxLayout, QWidget)

    GROUP_HELP = {
        "Tilt": "Most important: these fix the focal length. Turn the board 30-45 degrees so one "
                "edge comes toward the camera, one direction at a time.",
        "Distance": "Near: the board fills about half the image width or more. Middle: about a third. "
                    "Far: about a quarter. Keep every corner inside the image.",
        "Position": "Move the board (or the camera) so the board's centre sits in each region of "
                    "the image, edges and corners included. The grid is drawn on the video.",
    }

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Camera calibration: {camera_name}")
            self.col = Collector(board)
            self.result = None
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.job = None
            self.frame = None

            root = QHBoxLayout(self)
            left = QVBoxLayout()
            self.image = QLabel("waiting for camera ...")
            self.image.setAlignment(Qt.AlignCenter)
            self.image.setMinimumSize(640, 480)
            left.addWidget(self.image)
            self.status = QLabel()
            self.status.setWordWrap(True)
            left.addWidget(self.status)
            root.addLayout(left, 3)

            right = QVBoxLayout()
            self.next = QLabel()
            self.next.setWordWrap(True)
            self.next.setStyleSheet("font-weight:600; font-size:14px; padding:6px; "
                                    "border:1px solid #bbb; border-radius:4px;")
            right.addWidget(QLabel("Next:"))
            right.addWidget(self.next)

            self.items = {}
            for title, names, ncols in (("Tilt", TILTS, 2), ("Distance", DISTANCES, 3),
                                        ("Position", POSITIONS, 3)):
                box = QGroupBox(title)
                v = QVBoxLayout(box)
                help_ = QLabel(GROUP_HELP[title])
                help_.setWordWrap(True)
                help_.setStyleSheet("color:#555; font-size:12px;")
                v.addWidget(help_)
                grid = QGridLayout()
                for k, n in enumerate(names):
                    lab = QLabel()
                    lab.setToolTip(HOW[n])
                    lab.setAlignment(Qt.AlignCenter)
                    grid.addWidget(lab, k // ncols, k % ncols)
                    self.items[n] = lab
                v.addLayout(grid)
                right.addWidget(box)

            self.count = QLabel()
            right.addWidget(self.count)
            btns = QHBoxLayout()
            self.calib_btn = QPushButton("Calibrate")
            self.calib_btn.clicked.connect(self.run_calibration)
            self.save_btn = QPushButton("Save")
            self.save_btn.clicked.connect(self.save)
            reset = QPushButton("Reset")
            reset.clicked.connect(self.reset)
            for b in (self.calib_btn, self.save_btn, reset):
                btns.addWidget(b)
            right.addLayout(btns)
            self.undist = QCheckBox("Show undistorted (straight lines should look straight)")
            right.addWidget(self.undist)
            self.result_box = QLabel()
            self.result_box.setWordWrap(True)
            self.result_box.setTextFormat(Qt.RichText)
            right.addWidget(self.result_box)
            right.addStretch(1)
            root.addLayout(right, 2)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(120)
            self.refresh_panel()

        # -- live loop --
        def tick(self):
            if self.job is not None and self.job.done():
                self.finish_calibration()
            bgr = get_frame()
            if bgr is None:
                return
            self.frame = bgr
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            corners = detect(gray, board)
            if self.job is None:
                self.status.setText(self.col.offer(corners, w, h, bgr))
            self.show_frame(bgr, corners)
            self.refresh_panel()

        def show_frame(self, bgr, corners):
            h, w = bgr.shape[:2]
            img = bgr.copy()
            if self.result is not None and self.undist.isChecked():
                img = cv2.undistort(img, self.result.K, self.result.D)
            else:
                over = img.copy()
                for k, name in enumerate(POSITIONS):          # tint covered regions green
                    if self.col.coverage.done(name):
                        r, c = divmod(k, 3)
                        cv2.rectangle(over, (c * w // 3, r * h // 3), ((c + 1) * w // 3, (r + 1) * h // 3),
                                      (0, 180, 0), -1)
                img = cv2.addWeighted(over, 0.18, img, 0.82, 0)
                for i in (1, 2):
                    cv2.line(img, (i * w // 3, 0), (i * w // 3, h), (200, 200, 200), 1)
                    cv2.line(img, (0, i * h // 3), (w, i * h // 3), (200, 200, 200), 1)
                for s in self.col.samples:                    # outlines of captured views
                    q = outer_quad(s.corners, board).astype(np.int32)
                    cv2.polylines(img, [q], True, (255, 170, 0), 1)
                if corners is not None:
                    cv2.drawChessboardCorners(img, (board.cols, board.rows),
                                              corners.reshape(-1, 1, 2), True)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.image.setPixmap(QPixmap.fromImage(q).scaled(
                self.image.width(), self.image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def refresh_panel(self):
            cov = self.col.coverage
            for n, lab in self.items.items():
                have, need = cov.counts.get(n, 0), cov.need(n)
                done = have >= need
                short = n.split("-")[0][0].upper() + n.split("-")[1][0].upper() if n in POSITIONS else n
                lab.setText(f"{'✓ ' if done else ''}{short} {min(have, need)}/{need}")
                lab.setStyleSheet("padding:3px; border-radius:3px; " +
                                  ("background:#d8efd9; color:#1b5e20;" if done else "background:#eee; color:#333;"))
            self.next.setText(cov.next_hint())
            n = len(self.col.samples)
            self.count.setText(f"Views captured: {n} (Calibrate unlocks at 12; aim for 25+ with "
                               f"every box ticked)")
            busy = self.job is not None
            self.calib_btn.setEnabled(n >= 12 and not busy)
            self.calib_btn.setText("Calibrating..." if busy else "Calibrate")
            self.save_btn.setEnabled(self.result is not None and not busy)
            self.undist.setEnabled(self.result is not None)

        # -- actions --
        def run_calibration(self):
            corners = [s.corners for s in self.col.samples]
            size = self.col.size
            self.job = self.pool.submit(calibrate, corners, board, size)
            self.status.setText("Calibrating in the background; the video keeps running.")
            self.refresh_panel()

        def finish_calibration(self):
            job, self.job = self.job, None
            try:
                self.result = job.result()
            except Exception as e:                     # noqa: BLE001 - shown to the user
                self.result = None
                self.result_box.setText(f"<span style='color:{COLOURS['bad']}'>Calibration failed: {e}</span>")
                return
            r = self.result
            missing = self.col.coverage.missing()
            rows = [f"<b>Result</b> ({r.used} views used"
                    + (f", {len(r.dropped)} dropped for high error" if r.dropped else "") + ")"]
            for level, name, text in r.checks():
                rows.append(f"<span style='color:{COLOURS[level]}'>● {name}: {text}</span>")
            if missing:
                rows.append(f"<span style='color:{COLOURS['warn']}'>Coverage still missing: "
                            f"{', '.join(missing)}. More views there will make the result more reliable.</span>")
            rows.append("<span style='color:#555'>Saving is allowed only when no check is red.</span>"
                        if not r.ok() else "Looks good: press Save.")
            self.result_box.setText("<br>".join(rows))
            self.status.setText("Calibration done. Tick 'Show undistorted' to inspect it.")

        def save(self):
            r = self.result
            if r is None:
                return
            if not r.ok():
                self.status.setText("Not saved: fix the red checks first (usually: more tilted views, "
                                    "and views near the image edges).")
                return
            backup = write_camera_info(r, camera_name, output)
            folder = os.path.join(os.path.expanduser(samples_dir), time.strftime("%Y%m%d_%H%M%S"))
            n = save_samples(self.col.samples, folder)
            self.status.setText(f"Saved {os.path.expanduser(output)}"
                                + (f" (old file kept as {os.path.basename(backup)})" if backup else "")
                                + f" and {n} views in {folder}. Restart the camera node to load it.")

        def reset(self):
            self.col.reset()
            self.result = None
            self.result_box.setText("")
            self.undist.setChecked(False)
            self.refresh_panel()

    return Window


def live(argv):  # pragma: no cover
    import rclpy
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    from .aruco_detector import image_to_bgr

    rclpy.init(args=argv)
    node = Node("camera_calibrator")
    P = node.declare_parameter
    P("image_topic", "/wrist_cam/image_raw")
    P("camera_name", "wrist_cam")
    P("board_cols", 8)          # inner corners
    P("board_rows", 6)
    P("square_mm", 25.0)
    P("output", "")             # default ~/.ros/camera_info/<camera_name>.yaml
    P("samples_dir", "")        # default ~/calib_<camera_name>
    g = lambda n: node.get_parameter(n).value
    name = g("camera_name")
    board = Board(int(g("board_cols")), int(g("board_rows")), float(g("square_mm")))
    output = g("output") or f"~/.ros/camera_info/{name}.yaml"
    samples_dir = g("samples_dir") or f"~/calib_{name}"
    latest = {"msg": None}
    node.create_subscription(Image, g("image_topic"), lambda m: latest.__setitem__("msg", m),
                             qos_profile_sensor_data)
    node.get_logger().info(f"Calibrating '{name}' from {g('image_topic')}; board {board.cols}x{board.rows} "
                           f"inner corners, {board.square_mm} mm squares; output {output}")

    def get_frame():
        m, latest["msg"] = latest["msg"], None
        if m is None:
            return None
        try:
            return image_to_bgr(m)
        except ValueError as e:
            node.get_logger().error(str(e), throttle_duration_sec=5.0)
            return None

    app = QApplication(sys.argv[:1])
    win = build_window(get_frame, board, name, output, samples_dir)()
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


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if "--images" in argv:
        ap = argparse.ArgumentParser(description="Calibrate from a folder of checkerboard images.")
        ap.add_argument("--images", required=True)
        ap.add_argument("--size", default="8x6", help="inner corners, e.g. 8x6 for a 9x7-square board")
        ap.add_argument("--square", type=float, default=25.0, help="square size in mm")
        ap.add_argument("--camera-name", default="wrist_cam")
        ap.add_argument("--save", default="", help="write the camera_info YAML here")
        ap.add_argument("--force", action="store_true", help="save even if a check fails")
        sys.exit(offline(ap.parse_args([a for a in argv[1:] if a != "--ros-args"])))
    live(argv)


if __name__ == "__main__":
    main()