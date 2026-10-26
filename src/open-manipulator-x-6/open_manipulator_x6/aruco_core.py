"""
ArUco marker detection and pose math for the X6 workspace (ROS-free, so it can be tested
without a camera).

Frames
- Camera frame: OpenCV convention (x right, y down, z forward), from solvePnP.
- Marker frame: x right, y up (toward the top of the printed marker), z out of the marker.
- Robot frame: origin on joint 1's axis at the arm's mounting surface, x forward, y left, z up.
  The base marker (lying flat on the bar) links the camera to the robot frame:
  base_offset_mm = where the base marker's centre is in the robot frame,
  base_yaw_deg   = direction of the marker's x-axis in the robot frame. With the printed arrow
                   (= marker +y) pointing to the robot's front (+x), the marker's x-axis points to
                   the robot's right (-y), i.e. base_yaw_deg = -90 (the default).

Works with both OpenCV ArUco APIs: the old one (4.5, e.g. Ubuntu 22.04 / ROS Humble) and the new
one (4.7+).
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ARUCO = cv2.aruco
NEW_API = hasattr(ARUCO, "ArucoDetector")


# ---- OpenCV compatibility -------------------------------------------------------------------
def get_dictionary(name: str = "DICT_4X4_50"):
    const = getattr(ARUCO, name)
    if hasattr(ARUCO, "getPredefinedDictionary"):
        return ARUCO.getPredefinedDictionary(const)
    return ARUCO.Dictionary_get(const)


def marker_bits(dictionary, marker_id: int, border_bits: int = 1) -> np.ndarray:
    """Marker as a small image with one pixel per cell (e.g. 6x6 for a 4x4 marker): 0 = black."""
    n = dictionary.markerSize + 2 * border_bits
    if hasattr(ARUCO, "generateImageMarker"):
        return ARUCO.generateImageMarker(dictionary, marker_id, n, borderBits=border_bits)
    return ARUCO.drawMarker(dictionary, marker_id, n, borderBits=border_bits)


class _Detector:
    def __init__(self, dictionary):
        if NEW_API:
            p = ARUCO.DetectorParameters()
            p.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
            det = ARUCO.ArucoDetector(dictionary, p)
            self._fn = lambda g: det.detectMarkers(g)
        else:
            p = ARUCO.DetectorParameters_create()
            p.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX
            self._fn = lambda g: ARUCO.detectMarkers(g, dictionary, parameters=p)

    def __call__(self, gray) -> Dict[int, np.ndarray]:
        corners, ids, _ = self._fn(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.flatten(), corners)}


# ---- geometry ---------------------------------------------------------------------------------
def to_T(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def object_points(size_m: float) -> np.ndarray:
    h = size_m / 2.0     # ArUco corner order: top-left, top-right, bottom-right, bottom-left
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def solve_marker(corners: np.ndarray, size_m: float, K, D) -> Tuple[np.ndarray, float]:
    """Marker pose in the camera frame (4x4, metres) and RMS reprojection error (px)."""
    obj = object_points(size_m)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - corners) ** 2, axis=1))))
    return to_T(rvec, tvec), err


def yaw_deg(T: np.ndarray) -> float:
    """Heading of the frame's x-axis in the XY plane of the parent frame (degrees)."""
    return math.degrees(math.atan2(T[1, 0], T[0, 0]))


# ---- tracker ------------------------------------------------------------------------------------
@dataclass
class MarkerSpec:
    id: int
    name: str
    size_mm: float


DEFAULT_MARKERS = [MarkerSpec(0, "base", 40.0), MarkerSpec(1, "bottle_start", 40.0),
                   MarkerSpec(2, "bottle_place", 40.0), MarkerSpec(3, "bottle", 30.0)]

REFERENCE_TARGETS = ("bottle_start", "bottle_place")   # bottle position is checked against these


class ArucoTracker:
    def __init__(self, markers: List[MarkerSpec] = None, dictionary: str = "DICT_4X4_50",
                 base_offset_mm=(0.0, 0.0, 0.0), base_yaw_deg: float = -90.0,
                 history: int = 10, stale_s: float = 0.5, max_reproj_px: float = 3.0):
        self.markers = {m.id: m for m in (markers or DEFAULT_MARKERS)}
        self.by_name = {m.name: m for m in self.markers.values()}
        if "base" not in self.by_name:
            raise ValueError("one marker must be named 'base'")
        self.dictionary = get_dictionary(dictionary)
        self.detect = _Detector(self.dictionary)
        T = np.eye(4)
        T[:3, :3] = rot_z(base_yaw_deg)
        T[:3, 3] = np.asarray(base_offset_mm, dtype=np.float64) / 1000.0
        self.T_robot_base = T
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.stale_s, self.max_reproj_px = stale_s, max_reproj_px
        self.last: Dict[str, Tuple[float, np.ndarray]] = {}           # name -> (t, T_cam_marker)
        self.hist: Dict[str, deque] = {n: deque(maxlen=history) for n in self.by_name}
        self.references: Dict[str, np.ndarray] = {}                    # target -> bottle pos (mm) in target frame
        self.last_corners: Dict[int, np.ndarray] = {}

    # -- camera --
    def set_camera(self, K, D):
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            self.K = self.D = None            # uncalibrated camera_info (all zeros)
            return
        self.K, self.D = K, np.asarray(D, dtype=np.float64).reshape(-1)

    @property
    def calibrated(self) -> bool:
        return self.K is not None

    # -- per frame --
    def process(self, gray: np.ndarray, t: float) -> Dict[int, dict]:
        found = self.detect(gray)
        self.last_corners = {i: c for i, c in found.items() if i in self.markers}
        out = {}
        for i, c in self.last_corners.items():
            spec = self.markers[i]
            entry = {"name": spec.name, "corners": c, "T": None, "reproj": None}
            if self.calibrated:
                T, err = solve_marker(c, spec.size_mm / 1000.0, self.K, self.D)
                entry["T"], entry["reproj"] = T, err
                if err <= self.max_reproj_px:
                    self.last[spec.name] = (t, T)
                    self.hist[spec.name].append((t, T))
            out[i] = entry
        return out

    def seen(self, name: str, now: float) -> bool:
        return name in self.last and now - self.last[name][0] <= self.stale_s

    def _median_T(self, name: str, now: float) -> Optional[np.ndarray]:
        """Latest rotation with the median of recent positions (steadier than one frame)."""
        if not self.seen(name, now):
            return None
        recent = [T for (t, T) in self.hist[name] if now - t <= self.stale_s * 4]
        T = self.last[name][1].copy()
        T[:3, 3] = np.median(np.array([R[:3, 3] for R in recent]), axis=0)
        return T

    # -- outputs --
    def robot_pose(self, name: str, now: float) -> Optional[Tuple[np.ndarray, float]]:
        """(xyz in mm, yaw in degrees) in the robot frame; None if it or the base is not visible."""
        Tb, Tm = self._median_T("base", now), self._median_T(name, now)
        if Tb is None or Tm is None:
            return None
        T = self.T_robot_base @ inv_T(Tb) @ Tm
        return T[:3, 3] * 1000.0, yaw_deg(T)

    def relative_mm(self, frame: str, name: str, now: float) -> Optional[np.ndarray]:
        """Position of marker `name` in marker `frame`'s frame (mm). Needs no base marker."""
        Tf, Tm = self._median_T(frame, now), self._median_T(name, now)
        if Tf is None or Tm is None:
            return None
        return (inv_T(Tf) @ Tm)[:3, 3] * 1000.0

    def set_reference(self, target: str, now: float) -> np.ndarray:
        """Remember where the bottle marker sits relative to `target` right now (bottle placed
        correctly by hand). Later deviations are measured from this."""
        if target not in REFERENCE_TARGETS:
            raise ValueError(f"target must be one of {REFERENCE_TARGETS}")
        p = self.relative_mm(target, "bottle", now)
        if p is None:
            raise ValueError(f"need both '{target}' and 'bottle' markers in view")
        self.references[target] = p
        return p

    def deviation(self, target: str, now: float) -> Optional[Tuple[float, float, float]]:
        """(dx, dy, distance) in mm, in the target marker's plane, from the saved reference."""
        ref = self.references.get(target)
        p = self.relative_mm(target, "bottle", now)
        if ref is None or p is None:
            return None
        dx, dy = float(p[0] - ref[0]), float(p[1] - ref[1])
        return dx, dy, math.hypot(dx, dy)

    def snapshot_row(self, label: str, stamp: str, now: float) -> dict:
        row = {"time": stamp, "label": label, "base_seen": int(self.seen("base", now))}
        for name in self.by_name:
            pose = None if name == "base" else self.robot_pose(name, now)
            row[f"{name}_seen"] = int(self.seen(name, now))
            if name == "base":
                continue
            for k, v in zip(("x_mm", "y_mm", "z_mm"), pose[0] if pose else ("", "", "")):
                row[f"{name}_{k}"] = "" if v == "" else round(float(v), 1)
            row[f"{name}_yaw_deg"] = round(pose[1], 1) if pose else ""
        for target in REFERENCE_TARGETS:
            d = self.deviation(target, now)
            short = target.replace("bottle_", "")
            row[f"{short}_dx_mm"] = round(d[0], 1) if d else ""
            row[f"{short}_dy_mm"] = round(d[1], 1) if d else ""
            row[f"{short}_dev_mm"] = round(d[2], 1) if d else ""
        return row

    # -- drawing --
    def draw(self, bgr: np.ndarray, result: Dict[int, dict]) -> np.ndarray:
        img = bgr.copy()
        for i, e in result.items():
            c = e["corners"].astype(np.int32)
            good = e["reproj"] is None or e["reproj"] <= self.max_reproj_px
            cv2.polylines(img, [c], True, (0, 200, 0) if good else (0, 0, 255), 2)
            cv2.circle(img, tuple(c[0]), 4, (255, 0, 0), -1)          # top-left corner
            if e["T"] is not None and self.calibrated:
                rvec, _ = cv2.Rodrigues(e["T"][:3, :3])
                L = self.markers[i].size_mm / 1000.0 * 0.75
                if hasattr(cv2, "drawFrameAxes"):
                    cv2.drawFrameAxes(img, self.K, self.D, rvec, e["T"][:3, 3], L)
            cv2.putText(img, f"{i}:{e['name']}", tuple(c[0] + np.array([0, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, f"{i}:{e['name']}", tuple(c[0] + np.array([0, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img


# ---- reference file -------------------------------------------------------------------------------
def save_references(path: str, refs: Dict[str, np.ndarray]):
    import os
    import yaml
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump({k: [round(float(x), 2) for x in v] for k, v in refs.items()}, f)
    os.replace(tmp, path)


def load_references(path: str) -> Dict[str, np.ndarray]:
    import os
    import yaml
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {k: np.asarray(v, dtype=np.float64) for k, v in data.items() if k in REFERENCE_TARGETS}