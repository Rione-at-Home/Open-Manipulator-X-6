#!/usr/bin/env python3
"""
Guided checkerboard camera calibration.

Live (ROS 2):
    ros2 run open_manipulator_x6 camera_calibrator --ros-args \
        -p image_topic:=/wrist_cam/image_raw -p camera_name:=wrist_cam

    A dashed outline shows where the board should be. Line the board up with it (the hint under
    the video says which way to move); it is captured automatically when it matches and is steady,
    and the next outline appears. Use Skip for a pose you cannot reach. After the targets, press
    Calibrate (the views are saved to ~/calib_<camera_name>/<time>/ at that moment), check the
    results, and press Save to write ~/.ros/camera_info/<camera_name>.yaml (the old file is kept as
    .bak). Restart the camera node to load it.

From saved images (no ROS needed): a folder written by Calibrate, or the ROS calibrator's
/tmp/calibrationdata.tar.gz unpacked:
    python3 -m open_manipulator_x6.camera_calibrator --images ~/calib_wrist_cam/<time>
    python3 -m open_manipulator_x6.camera_calibrator --images <folder> --save ~/.ros/camera_info/wrist_cam.yaml

Board: 9 x 7 squares of 25 mm = 8 x 6 inner corners (the defaults).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .calibration_core import (Board, Coverage, TargetSession, calibrate, describe, detect,
                               fit_target, load_folder, outer_quad, save_samples, write_camera_info)

COLOURS = {"good": "#2e7d32", "warn": "#b26a00", "bad": "#c62828"}


def report(res):
    lines = [f"Used {res.used} views" + (f", dropped {len(res.dropped)} with high error" if res.dropped else "")]
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
    cov = Coverage()
    for c in corners:
        cov.add(describe(c, board, *size))
    missing = cov.missing()
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
    from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                                 QProgressBar, QPushButton, QVBoxLayout, QWidget)

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Camera calibration: {camera_name}")
            self.session = TargetSession(board)
            self.result = None
            self.views_folder = None
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.job = None

            root = QHBoxLayout(self)
            left = QVBoxLayout()
            self.image = QLabel("waiting for camera ...")
            self.image.setAlignment(Qt.AlignCenter)
            self.image.setMinimumSize(640, 400)
            left.addWidget(self.image, 1)
            self.hint = QLabel()
            self.hint.setWordWrap(True)
            self.hint.setStyleSheet("font-size:15px; font-weight:600; padding:4px;")
            left.addWidget(self.hint)
            self.bar = QProgressBar()
            self.bar.setRange(0, 100)
            self.bar.setFormat("match %p%")
            left.addWidget(self.bar)
            root.addLayout(left, 3)

            right = QVBoxLayout()
            self.target_lab = QLabel()
            self.target_lab.setWordWrap(True)
            self.target_lab.setStyleSheet("font-weight:600; font-size:14px; padding:6px; "
                                          "border:1px solid #bbb; border-radius:4px;")
            right.addWidget(self.target_lab)
            how = QLabel("Line the board up with the dashed outline on the video, with its dot on "
                         "the board's top-left. Move the board, or move the camera (the opposite "
                         "way), whichever is easier. It captures by itself when the board matches "
                         "and is steady.")
            how.setWordWrap(True)
            how.setStyleSheet("color:#555; font-size:12px;")
            right.addWidget(how)
            self.list = QListWidget()
            for k, t in enumerate(self.session.targets):
                self.list.addItem(QListWidgetItem(f"{k + 1:2d}. {t.text}"))
            right.addWidget(self.list, 1)

            row1 = QHBoxLayout()
            self.skip_btn = QPushButton("Skip this target")
            self.skip_btn.clicked.connect(self.skip)
            reset = QPushButton("Start over")
            reset.clicked.connect(self.reset)
            row1.addWidget(self.skip_btn)
            row1.addWidget(reset)
            right.addLayout(row1)
            row2 = QHBoxLayout()
            self.calib_btn = QPushButton("Calibrate")
            self.calib_btn.clicked.connect(self.run_calibration)
            self.save_btn = QPushButton("Save")
            self.save_btn.clicked.connect(self.save)
            row2.addWidget(self.calib_btn)
            row2.addWidget(self.save_btn)
            right.addLayout(row2)
            self.undist = QCheckBox("Show undistorted (straight lines should look straight)")
            right.addWidget(self.undist)
            self.result_box = QLabel()
            self.result_box.setWordWrap(True)
            self.result_box.setTextFormat(Qt.RichText)
            right.addWidget(self.result_box)
            self.status = QLabel()
            self.status.setWordWrap(True)
            right.addWidget(self.status)
            root.addLayout(right, 2)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(60)
            self.refresh_panel()

        # -- live loop --
        def tick(self):
            if self.job is not None and self.job.done():
                self.finish_calibration()
            bgr = get_frame()
            if bgr is None:
                return
            h, w = bgr.shape[:2]
            corners = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), board)
            if self.job is None and self.result is None:
                self.session.offer(corners, w, h, bgr)
            self.show_frame(bgr, corners)
            self.refresh_panel()

        def show_frame(self, bgr, corners):
            h, w = bgr.shape[:2]
            img = bgr.copy()
            if self.result is not None and self.undist.isChecked():
                img = cv2.undistort(img, self.result.K, self.result.D)
            else:
                t = self.session.current
                if t is not None and self.result is None:
                    q = fit_target(t, w, h).astype(np.int32)
                    good = self.session.closeness() >= 1.0
                    colour = (0, 200, 0) if good else (0, 165, 255)
                    th = max(2, w // 320)
                    for i in range(4):                       # dashed outline
                        a, b = q[i].astype(float), q[(i + 1) % 4].astype(float)
                        n = int(np.linalg.norm(b - a) // 18) + 1
                        for s in range(0, n, 2):
                            p1 = a + (b - a) * s / n
                            p2 = a + (b - a) * min(s + 1, n) / n
                            cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), colour, th)
                    # mark the outline's top-left corner so the board is not lined up rotated
                    cv2.circle(img, tuple(q[0]), th * 3, colour, -1)
                if corners is not None:
                    cv2.drawChessboardCorners(img, (board.cols, board.rows), corners.reshape(-1, 1, 2), True)
                    dq = outer_quad(corners, board).astype(np.int32)
                    cv2.polylines(img, [dq], True, (255, 255, 255), 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.image.setPixmap(QPixmap.fromImage(qi).scaled(
                self.image.width(), self.image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def refresh_panel(self):
            s = self.session
            for k, st in enumerate(s.status):
                item = self.list.item(k)
                mark = {"done": "✓", "skipped": "–"}.get(st, "  ")
                item.setText(f"{mark} {k + 1:2d}. {s.targets[k].text}")
                item.setForeground(Qt.darkGreen if st == "done" else Qt.gray if st == "skipped" else Qt.black)
                if k == s.index:
                    self.list.setCurrentRow(k)
            done = s.status.count("done")
            if s.finished:
                self.target_lab.setText(f"All targets visited ({done} captured). Press Calibrate.")
            else:
                self.target_lab.setText(f"Target {s.index + 1} of {len(s.targets)}: {s.current.text}")
            if self.result is None and self.job is None:
                self.hint.setText(s.last_hint or "Show the board to the camera.")
            self.bar.setValue(int(100 * s.closeness()))
            busy = self.job is not None
            self.calib_btn.setEnabled(done >= 10 and not busy)
            self.calib_btn.setText("Calibrating..." if busy else f"Calibrate ({done} views)")
            self.save_btn.setEnabled(self.result is not None and not busy)
            self.skip_btn.setEnabled(not s.finished and self.result is None)
            self.undist.setEnabled(self.result is not None)

        # -- actions --
        def skip(self):
            self.session.skip()
            self.refresh_panel()

        def run_calibration(self):
            s = self.session
            self.views_folder = os.path.join(os.path.expanduser(samples_dir), time.strftime("%Y%m%d_%H%M%S"))
            n = save_samples(s.samples, self.views_folder)
            self.job = self.pool.submit(calibrate, [x.corners for x in s.samples], board, s.size)
            self.status.setText(f"{n} views saved in {self.views_folder}. Calibrating in the background...")
            self.hint.setText("Calibrating...")
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
            rows = [f"<b>Result</b> ({r.used} views used"
                    + (f", {len(r.dropped)} dropped for high error" if r.dropped else "") + ")"]
            for level, name, text in r.checks():
                rows.append(f"<span style='color:{COLOURS[level]}'>● {name}: {text}</span>")
            rows.append("Press Save." if r.ok() else
                        "<span style='color:#555'>A red check blocks saving: press Start over and "
                        "collect again.</span>")
            self.result_box.setText("<br>".join(rows))
            self.hint.setText("Calibration done. Tick 'Show undistorted' to inspect it.")

        def save(self):
            r = self.result
            if r is None or not r.ok():
                self.status.setText("Not saved: a check is red.")
                return
            backup = write_camera_info(r, camera_name, output)
            self.status.setText(f"Saved {os.path.expanduser(output)}"
                                + (f" (old file kept as {os.path.basename(backup)})" if backup else "")
                                + ". Restart the camera node to load it.")

        def reset(self):
            self.session.reset()
            self.result = None
            self.result_box.setText("")
            self.status.setText("")
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