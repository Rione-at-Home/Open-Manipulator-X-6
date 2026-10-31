"""
Checkerboard camera calibration without ROS: board detection, view coverage with plain-language
guidance, calibration with outlier removal, sanity checks, and the camera_info YAML that ROS
camera drivers (v4l2_camera, usb_cam) load through camera_info_url.

Board size is given in INNER corners: a board of 9 x 7 squares has 8 x 6 inner corners.
"""
import glob
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---- detection ---------------------------------------------------------------------------------
@dataclass
class Board:
    cols: int = 8              # inner corners across
    rows: int = 6              # inner corners down
    square_mm: float = 25.0

    def object_points(self) -> np.ndarray:
        g = np.zeros((self.rows * self.cols, 3), np.float32)
        g[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * (self.square_mm / 1000.0)
        return g


def detect(gray: np.ndarray, board: Board) -> Optional[np.ndarray]:
    """Inner corners (N x 2, float32) with sub-pixel accuracy, or None."""
    size = (board.cols, board.rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, c = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            return c.reshape(-1, 2).astype(np.float32)
        return None
    ok, c = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    return c.reshape(-1, 2).astype(np.float32)


# ---- view description and coverage --------------------------------------------------------------
@dataclass
class View:
    cx: float          # board centre, fraction of image width (0 = left)
    cy: float          # board centre, fraction of image height (0 = top)
    size: float        # sqrt(board area / image area)
    yaw: float         # >0: left edge nearer the camera (looks longer); <0: right edge nearer
    pitch: float       # >0: top edge nearer; <0: bottom edge nearer

    def vec(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.size * 1.5, self.yaw * 2.0, self.pitch * 2.0])


def outer_quad(corners: np.ndarray, board: Board) -> np.ndarray:
    """The four outer inner-corners ordered as they appear in the image: TL, TR, BR, BL.
    Ordering by image position (not by grid index) keeps the labels right whichever way the
    detector numbered the grid."""
    c, n = corners, board.cols
    quad = np.array([c[0], c[n - 1], c[-1], c[-n]], dtype=np.float64)
    centre = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])   # image y points down
    quad = quad[np.argsort(ang)]                 # from -pi: TL, TR, BR, BL for a mildly rotated board
    tl = int(np.argmin(quad.sum(axis=1)))        # top-left has the smallest x + y
    return np.roll(quad, -tl, axis=0)


def describe(corners: np.ndarray, board: Board, w: int, h: int) -> View:
    tl, tr, br, bl = outer_quad(corners, board)
    left, right = np.linalg.norm(tl - bl), np.linalg.norm(tr - br)
    top, bottom = np.linalg.norm(tl - tr), np.linalg.norm(bl - br)
    hull = cv2.convexHull(corners.astype(np.float32))
    area = cv2.contourArea(hull)
    centre = corners.mean(axis=0)
    return View(float(centre[0] / w), float(centre[1] / h), float(math.sqrt(area / (w * h))),
                float((left - right) / (left + right)), float((top - bottom) / (top + bottom)))


TILT = 0.06            # |yaw| or |pitch| above this counts as tilted (roughly 15 deg or more)
NEAR, FAR = 0.45, 0.28  # size thresholds

POSITIONS = [f"{v}-{u}" for v in ("top", "middle", "bottom") for u in ("left", "centre", "right")]
DISTANCES = ["near", "middle", "far"]
TILTS = ["left edge nearer", "right edge nearer", "top edge nearer", "bottom edge nearer"]

HOW = {
    # positions
    **{p: f"Move the board (or the camera) so the board sits in the {p.replace('-', ' ')} "
          f"of the image, all corners still inside." for p in POSITIONS},
    "near": "Bring the camera closer until the board fills most of the image (about half the "
            "image width or more), with every corner still inside.",
    "middle": "Hold the board at a medium distance: it should cover roughly a third of the image width.",
    "far": "Move the camera back until the board is small, about a quarter of the image width "
           "(as far as the corners are still detected).",
    "left edge nearer": "Turn the board so its LEFT edge comes toward the camera (or swing the "
                        "camera to the board's left), about 30-45 degrees.",
    "right edge nearer": "Turn the board so its RIGHT edge comes toward the camera (or swing the "
                         "camera to the board's right), about 30-45 degrees.",
    "top edge nearer": "Tip the board so its TOP edge comes toward the camera (or raise the camera "
                       "and look down at it), about 30-45 degrees.",
    "bottom edge nearer": "Tip the board so its BOTTOM edge comes toward the camera (or lower the "
                          "camera and look up at it), about 30-45 degrees.",
}


def position_bin(v: View) -> str:
    u = min(2, int(v.cx * 3))
    r = min(2, int(v.cy * 3))
    return POSITIONS[r * 3 + u]


def distance_bin(v: View) -> str:
    return "near" if v.size >= NEAR else "far" if v.size <= FAR else "middle"


def tilt_bins(v: View) -> List[str]:
    out = []
    if v.yaw > TILT:
        out.append("left edge nearer")
    if v.yaw < -TILT:
        out.append("right edge nearer")
    if v.pitch > TILT:
        out.append("top edge nearer")
    if v.pitch < -TILT:
        out.append("bottom edge nearer")
    return out


@dataclass
class Coverage:
    """How many captured views fall in each bin, and what to do next."""
    need_pos: int = 2       # views wanted per image region
    need_dist: int = 4
    need_tilt: int = 3
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, v: View):
        for b in [position_bin(v), distance_bin(v)] + tilt_bins(v):
            self.counts[b] = self.counts.get(b, 0) + 1

    def need(self, b: str) -> int:
        return self.need_pos if b in POSITIONS else self.need_dist if b in DISTANCES else self.need_tilt

    def done(self, b: str) -> bool:
        return self.counts.get(b, 0) >= self.need(b)

    def missing(self) -> List[str]:
        # tilts first (they fix the focal length), then distances, then image regions
        return [b for b in TILTS + DISTANCES + POSITIONS if not self.done(b)]

    def next_hint(self) -> str:
        m = self.missing()
        return HOW[m[0]] if m else "Coverage complete. Press Calibrate."

    def ready(self) -> bool:
        return not self.missing()


@dataclass
class Sample:
    corners: np.ndarray
    view: View
    image: Optional[np.ndarray] = None     # kept so the views can be saved and re-run


class Collector:
    """Decides which live detections to keep: the board must be steady, and the view must differ
    from the ones already captured."""

    def __init__(self, board: Board, still_px: float = 1.5, still_frames: int = 3,
                 min_novelty: float = 0.12, max_samples: int = 60):
        self.board = board
        self.still_px, self.still_frames = still_px, still_frames
        self.min_novelty, self.max_samples = min_novelty, max_samples
        self.samples: List[Sample] = []
        self.coverage = Coverage()
        self.size: Optional[Tuple[int, int]] = None
        self._prev: Optional[np.ndarray] = None
        self._steady = 0

    def reset(self):
        self.__init__(self.board, self.still_px, self.still_frames, self.min_novelty, self.max_samples)

    def novelty(self, v: View) -> float:
        if not self.samples:
            return float("inf")
        return float(min(np.linalg.norm(v.vec() - s.view.vec()) for s in self.samples))

    def offer(self, corners: Optional[np.ndarray], w: int, h: int,
              image: Optional[np.ndarray] = None) -> str:
        """Feed one detection (None = no board). Returns a short status for the GUI."""
        if self.size is not None and self.size != (w, h):
            self.reset()
        self.size = (w, h)
        if corners is None:
            self._prev, self._steady = None, 0
            return "No board found: show the whole board, flat and well lit."
        moved = np.inf if self._prev is None else float(np.max(np.linalg.norm(corners - self._prev, axis=1)))
        self._prev = corners
        self._steady = self._steady + 1 if moved <= self.still_px else 0
        if self._steady < self.still_frames:
            return "Board found: hold still..."
        if len(self.samples) >= self.max_samples:
            return f"{self.max_samples} views collected (the maximum). Press Calibrate."
        v = describe(corners, self.board, w, h)
        if self.novelty(v) < self.min_novelty:
            return "Already have a view like this: move to a new position, distance or tilt."
        self.samples.append(Sample(corners.copy(), v, None if image is None else image.copy()))
        self.coverage.add(v)
        self._steady = 0
        return f"Captured view {len(self.samples)}."


# ---- calibration ----------------------------------------------------------------------------------
@dataclass
class Result:
    K: np.ndarray
    D: np.ndarray
    size: Tuple[int, int]
    rms: float
    per_view: List[float]
    used: int
    dropped: List[int]

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.size[0] / (2 * self.K[0, 0])))

    def checks(self) -> List[Tuple[str, str, str]]:
        """(level, name, text) with level in good / warn / bad."""
        w, h = self.size
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        out = []
        lvl = "good" if self.rms < 0.5 else "warn" if self.rms < 1.0 else "bad"
        out.append((lvl, "Reprojection error", f"{self.rms:.2f} px (good below 0.5)"))
        dx, dy = abs(cx - w / 2) / w, abs(cy - h / 2) / h
        lvl = "good" if max(dx, dy) < 0.06 else "warn" if max(dx, dy) < 0.12 else "bad"
        out.append((lvl, "Image centre", f"({cx:.0f}, {cy:.0f}); expected near ({w / 2:.0f}, {h / 2:.0f})"))
        r = abs(fx - fy) / fx
        lvl = "good" if r < 0.02 else "warn" if r < 0.05 else "bad"
        out.append((lvl, "fx vs fy", f"{fx:.0f} vs {fy:.0f} (should match within 2%)"))
        out.append(("good" if 40 <= self.hfov_deg <= 100 else "warn", "Horizontal field of view",
                    f"{self.hfov_deg:.0f} deg (ordinary webcams: about 55-80)"))
        return out

    def ok(self) -> bool:
        return all(level != "bad" for level, _, _ in self.checks())


def calibrate(corner_sets: List[np.ndarray], board: Board, size: Tuple[int, int],
              drop_factor: float = 2.5, drop_floor_px: float = 1.0) -> Result:
    """Calibrate (k1, k2, p1, p2; k3 fixed at 0, which suits ordinary webcams), then drop views
    whose error is far above the rest and calibrate once more."""
    if len(corner_sets) < 8:
        raise ValueError(f"only {len(corner_sets)} views; collect at least 8 (20+ recommended)")
    obj = board.object_points()
    flags = cv2.CALIB_FIX_K3

    def run(idx):
        o = [obj] * len(idx)
        img = [corner_sets[i].reshape(-1, 1, 2) for i in idx]
        rms, K, D, rv, tv = cv2.calibrateCamera(o, img, size, None, None, flags=flags)
        errs = []
        for k, i in enumerate(idx):
            proj, _ = cv2.projectPoints(obj, rv[k], tv[k], K, D)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corner_sets[i]) ** 2, axis=1)))))
        return rms, K, D, errs

    idx = list(range(len(corner_sets)))
    rms, K, D, errs = run(idx)
    limit = max(drop_factor * float(np.median(errs)), drop_floor_px)
    keep = [i for i, e in zip(idx, errs) if e <= limit]
    dropped = [i for i in idx if i not in keep]
    if dropped and len(keep) >= 8:
        rms, K, D, errs2 = run(keep)
        per_view = [float("nan")] * len(idx)
        for i, e in zip(keep, errs2):
            per_view[i] = e
        for i, e in zip(idx, errs):
            if i in dropped:
                per_view[i] = e
    else:
        dropped, keep, per_view = [], idx, errs
    return Result(K, D.reshape(-1)[:5], size, float(rms), per_view, len(keep), dropped)


# ---- output ---------------------------------------------------------------------------------------
def camera_info_yaml(res: Result, camera_name: str) -> str:
    w, h = res.size
    P, _ = cv2.getOptimalNewCameraMatrix(res.K, res.D, (w, h), 0.0)
    Pm = np.zeros((3, 4))
    Pm[:3, :3] = P

    def mat(name, rows, cols, data):
        vals = ", ".join(f"{float(x):.6f}" for x in np.asarray(data).reshape(-1))
        return f"{name}:\n  rows: {rows}\n  cols: {cols}\n  data: [{vals}]\n"

    return (f"image_width: {w}\nimage_height: {h}\ncamera_name: {camera_name}\n"
            + mat("camera_matrix", 3, 3, res.K)
            + "distortion_model: plumb_bob\n"
            + mat("distortion_coefficients", 1, 5, res.D)
            + mat("rectification_matrix", 3, 3, np.eye(3))
            + mat("projection_matrix", 3, 4, Pm))


def write_camera_info(res: Result, camera_name: str, path: str) -> Optional[str]:
    """Write the YAML; an existing file is kept as <path>.bak. Returns the backup path, if any."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    backup = None
    if os.path.exists(path):
        backup = path + ".bak"
        os.replace(path, backup)
    with open(path, "w") as f:
        f.write(camera_info_yaml(res, camera_name))
    return backup


def save_samples(samples: List[Sample], folder: str) -> int:
    folder = os.path.expanduser(folder)
    os.makedirs(folder, exist_ok=True)
    n = 0
    for k, s in enumerate(samples):
        if s.image is not None:
            cv2.imwrite(os.path.join(folder, f"view_{k:03d}.png"), s.image)
            n += 1
    return n


def load_folder(folder: str, board: Board) -> Tuple[List[np.ndarray], Tuple[int, int], List[str], List[str]]:
    """Detect the board in every image of a folder (png/jpg, including the left-*.png files in the
    ROS calibrator's calibrationdata.tar.gz). Returns corners, image size, used files, skipped files."""
    files = sorted(sum((glob.glob(os.path.join(os.path.expanduser(folder), e))
                        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")), []))
    corners, used, skipped, size = [], [], [], None
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append(f)
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        if (img.shape[1], img.shape[0]) != size:
            skipped.append(f)
            continue
        c = detect(img, board)
        if c is None:
            skipped.append(f)
        else:
            corners.append(c)
            used.append(f)
    if size is None:
        raise ValueError(f"no images found in {folder}")
    return corners, size, used, skipped