"""Synthetic check of the base lock: a camera 0.4 m away, tilted 45 deg, with pixel noise on the
marker corners. Run: python3 -m pytest test/test_base_lock.py -q  (from the package root)."""
import math
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from open_manipulator_x6.aruco_core import (ArucoTracker, MarkerSpec, inv_T, load_references,  # noqa: E402
                                            object_points, rot_z, save_references)

K = np.array([[910.0, 0, 640.0], [0, 910.0, 360.0], [0, 0, 1.0]])   # D435 colour, 1280x720
D = np.zeros(5)
BASE_OFFSET = (100.0, 160.0, 125.0)
SPECS = [MarkerSpec(0, "base", 40.0), MarkerSpec(1, "bottle_start", 40.0),
         MarkerSpec(2, "bottle_place", 40.0), MarkerSpec(3, "bottle", 30.0)]


def pose(R, p_mm):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(p_mm) / 1000.0
    return T


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# Ground truth in the robot frame (x forward, y left, z up), markers flat on the table at z = 125 mm
# except the bottle marker, which is on the bottle's side facing the camera.
T_ROBOT_BASE = pose(rot_z(-90), BASE_OFFSET)
TRUTH = {
    "base": T_ROBOT_BASE,
    "bottle_start": pose(rot_z(-90), (260.0, 60.0, 125.0)),
    "bottle_place": pose(rot_z(-90), (260.0, -80.0, 125.0)),
    "bottle": pose(rot_z(-90) @ rot_x(90), (265.0, 62.0, 185.0)),
}
# Camera in front of the arm, 0.4 m from the start/place markers, looking back at them at 45 deg.
target = np.array([0.26, 0.0, 0.125])
cam_pos = target + 0.40 * np.array([math.cos(math.radians(45)), 0.0, math.sin(math.radians(45))])
z_c = (target - cam_pos) / np.linalg.norm(target - cam_pos)         # optical axis
x_c = np.cross(z_c, [0, 0, 1.0]); x_c /= np.linalg.norm(x_c)       # image right
y_c = np.cross(z_c, x_c)                                            # image down
T_ROBOT_CAM = np.eye(4)
T_ROBOT_CAM[:3, :3] = np.column_stack([x_c, y_c, z_c])
T_ROBOT_CAM[:3, 3] = cam_pos


def corners_for(name, size_mm, rng, noise_px):
    T_cam_m = inv_T(T_ROBOT_CAM) @ TRUTH[name]
    rvec, _ = cv2.Rodrigues(T_cam_m[:3, :3])
    pts, _ = cv2.projectPoints(object_points(size_mm / 1000.0), rvec, T_cam_m[:3, 3], K, D)
    return pts.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))


def run(tracker, frames, rng, noise_px=0.3, t0=0.0, fps=15.0, each=None):
    """Feed synthetic detections straight into the tracker (bypassing image detection).
    `each(t)` is called after every frame; returns the frame times."""
    out = []
    for k in range(frames):
        t = t0 + k / fps
        found = {s.id: corners_for(s.name, s.size_mm, rng, noise_px) for s in SPECS}
        tracker.detect = lambda gray, f=found: f
        tracker.process(None, t)
        if each:
            each(t)
        out.append(t)
    return out


def make_tracker():
    tr = ArucoTracker(SPECS, base_offset_mm=BASE_OFFSET, base_yaw_deg=-90.0)
    tr.set_camera(K.flatten(), D)
    return tr


def series(tr, name, rng, frames=150, t0=10.0, noise_px=0.3):
    pts = []
    run(tr, frames, rng, noise_px=noise_px, t0=t0,
        each=lambda t: pts.append(tr.robot_pose(name, t)[0]))
    return np.array(pts)


@pytest.mark.parametrize("noise_px", [0.3, 1.0])
def test_lock_is_steadier_and_accurate(noise_px):
    rng = np.random.default_rng(1)
    unlocked = make_tracker()
    run(unlocked, 45, rng, noise_px=noise_px)
    a = series(unlocked, "bottle_start", rng, noise_px=noise_px)

    locked = make_tracker()
    t = run(locked, 45, rng, noise_px=noise_px)[-1]
    lk = locked.lock_base(t)
    assert lk.samples >= 30
    b = series(locked, "bottle_start", rng, noise_px=noise_px)

    truth = TRUTH["bottle_start"][:3, 3] * 1000
    jitter_unlocked = np.ptp(a, axis=0).max()
    jitter_locked = np.ptp(b, axis=0).max()
    err_locked = np.abs(b.mean(axis=0) - truth).max()
    print(f"\n{noise_px} px noise, peak-to-peak jitter: unlocked {jitter_unlocked:.2f} mm, locked {jitter_locked:.2f} mm; "
          f"locked mean error {err_locked:.2f} mm")
    assert jitter_locked < jitter_unlocked
    assert err_locked < 3.0


def test_lock_needs_enough_frames():
    rng = np.random.default_rng(2)
    tr = make_tracker()
    t = run(tr, 5, rng)[-1]
    with pytest.raises(ValueError, match="only"):
        tr.lock_base(t)


def test_references_and_deviation_in_robot_frame(tmp_path=None):
    rng = np.random.default_rng(3)
    tr = make_tracker()
    with pytest.raises(ValueError, match="lock the base"):
        tr.set_reference("bottle_start", 0.0)
    t = run(tr, 45, rng)[-1]
    tr.lock_base(t)
    t = run(tr, 15, rng, t0=t + 0.1)[-1]
    tr.set_reference("bottle_start", t)

    # move the bottle 20 mm forward (+x) and 10 mm left (+y)
    TRUTH["bottle"] = pose(rot_z(-90) @ rot_x(90), (285.0, 72.0, 185.0))
    try:
        t = run(tr, 30, rng, t0=t + 0.1)[-1]
        dx, dy, dist = tr.deviation("bottle_start", t)
    finally:
        TRUTH["bottle"] = pose(rot_z(-90) @ rot_x(90), (265.0, 62.0, 185.0))
    print(f"\ndeviation dx {dx:+.1f} dy {dy:+.1f} dist {dist:.1f} (true +20.0 +10.0 22.4)")
    assert abs(dx - 20) < 2.5 and abs(dy - 10) < 2.5

    d = tempfile.mkdtemp()
    path = os.path.join(d, "refs.yaml")
    save_references(path, tr.references)
    assert set(load_references(path)) == {"bottle_start"}
    with open(path, "w") as f:                       # an old-format file is ignored
        f.write("bottle_start: [1.0, 2.0, 3.0]\n")
    assert load_references(path) == {}


def test_base_moved_warning():
    rng = np.random.default_rng(4)
    tr = make_tracker()
    t = run(tr, 45, rng)[-1]
    tr.lock_base(t)
    t = run(tr, 15, rng, t0=t + 0.1)[-1]
    assert tr.base_moved(t) is None
    global T_ROBOT_CAM
    saved = T_ROBOT_CAM.copy()
    T_ROBOT_CAM[:3, 3] += [0.0, 0.02, 0.0]          # camera bumped 20 mm sideways
    try:
        t = run(tr, 30, rng, t0=t + 0.1)[-1]
        moved = tr.base_moved(t)
    finally:
        T_ROBOT_CAM[:] = saved
    assert moved is not None and moved[0] > 10