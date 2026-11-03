"""
Checkerboard camera calibration without ROS: fast board detection, guided target poses (an outline
on the video that the board is lined up with), calibration with outlier removal, sanity checks,
and the camera_info YAML that ROS camera drivers (v4l2_camera, usb_cam) load via camera_info_url.

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


_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)


def _find(gray: np.ndarray, board: Board) -> Optional[np.ndarray]:
    size = (board.cols, board.rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, c = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        ok, c = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    return c.reshape(-1, 2).astype(np.float32) if ok else None


def detect(gray: np.ndarray, board: Board, max_side: int = 640) -> Optional[np.ndarray]:
    """Inner corners (N x 2, float32, full-resolution pixels) with sub-pixel accuracy, or None.
    Large images are searched at reduced size (fast), then the corners are refined at full size."""
    h, w = gray.shape[:2]
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        c = _find(small, board)
        if c is None:
            return None
        c = (c / scale).reshape(-1, 1, 2).astype(np.float32)
    else:
        c = _find(gray, board)
        if c is None:
            return None
        c = c.reshape(-1, 1, 2)
    # refinement window about a third of a square, measured at full resolution
    sq = float(np.median(np.linalg.norm(np.diff(c.reshape(board.rows, board.cols, 2), axis=1), axis=2)))
    win = int(max(3, min(11, sq / 3)))
    c = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), _SUBPIX)
    return c.reshape(-1, 2).astype(np.float32)


# ---- describing a board's place in the image ----------------------------------------------------
@dataclass
class View:
    cx: float          # board centre, fraction of image width (0 = left)
    cy: float          # board centre, fraction of image height (0 = top)
    size: float        # sqrt(board area / image area)
    yaw: float         # >0: left edge nearer the camera (looks longer); <0: right edge nearer
    pitch: float       # >0: top edge nearer; <0: bottom edge nearer


def outer_quad(corners: np.ndarray, board: Board) -> np.ndarray:
    """The four outer inner-corners ordered as they appear in the image: TL, TR, BR, BL.
    Ordering by image position (not by grid index) keeps the labels right whichever way the
    detector numbered the grid."""
    c, n = corners, board.cols
    quad = np.array([c[0], c[n - 1], c[-1], c[-n]], dtype=np.float64)
    centre = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))]
    return np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)


def view_of_quad(quad: np.ndarray, w: int, h: int) -> View:
    tl, tr, br, bl = quad
    left, right = np.linalg.norm(tl - bl), np.linalg.norm(tr - br)
    top, bottom = np.linalg.norm(tl - tr), np.linalg.norm(bl - br)
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    c = quad.mean(axis=0)
    return View(float(c[0] / w), float(c[1] / h), float(math.sqrt(area / (w * h))),
                float((left - right) / (left + right)), float((top - bottom) / (top + bottom)))


def describe(corners: np.ndarray, board: Board, w: int, h: int) -> View:
    return view_of_quad(outer_quad(corners, board), w, h)


# ---- coverage bins (used for the report) ---------------------------------------------------------
TILT = 0.06             # |yaw| or |pitch| above this counts as tilted (roughly 15 deg or more)
NEAR, FAR = 0.45, 0.28  # size thresholds

POSITIONS = [f"{v}-{u}" for v in ("top", "middle", "bottom") for u in ("left", "centre", "right")]
DISTANCES = ["near", "middle", "far"]
TILTS = ["left edge nearer", "right edge nearer", "top edge nearer", "bottom edge nearer"]


def position_bin(v: View) -> str:
    return POSITIONS[min(2, int(v.cy * 3)) * 3 + min(2, int(v.cx * 3))]


def distance_bin(v: View) -> str:
    return "near" if v.size >= NEAR else "far" if v.size <= FAR else "middle"


def tilt_bins(v: View, threshold: float = TILT) -> List[str]:
    out = []
    if v.yaw > threshold:
        out.append("left edge nearer")
    if v.yaw < -threshold:
        out.append("right edge nearer")
    if v.pitch > threshold:
        out.append("top edge nearer")
    if v.pitch < -threshold:
        out.append("bottom edge nearer")
    return out


@dataclass
class Coverage:
    need_pos: int = 1
    need_dist: int = 2
    need_tilt: int = 2
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, v: View):
        for b in [position_bin(v), distance_bin(v)] + tilt_bins(v):
            self.counts[b] = self.counts.get(b, 0) + 1

    def need(self, b: str) -> int:
        return self.need_pos if b in POSITIONS else self.need_dist if b in DISTANCES else self.need_tilt

    def missing(self) -> List[str]:
        return [b for b in TILTS + DISTANCES + POSITIONS if self.counts.get(b, 0) < self.need(b)]


# ---- guided targets ------------------------------------------------------------------------------
BOARD_ASPECT = 7.0 / 5.0     # width / height of the inner-corner grid (8 x 6 corners = 7 x 5 squares)


@dataclass
class Target:
    text: str                       # what the view should look like, in words
    cx: float                       # centre, fraction of width / height
    cy: float
    width: float                    # board width as a fraction of image width (flat board)
    tilt: Optional[str] = None      # one of TILTS, or None for facing the camera

    def quad(self, w: int, h: int, board_aspect: float = BOARD_ASPECT) -> np.ndarray:
        """Outline in pixels, TL, TR, BR, BL. A tilted board is drawn as a trapezoid: the nearer
        edge longer, and the board foreshortened along the tilt."""
        bw = self.width * w
        bh = bw / board_aspect
        k, f = 1.16, 0.85                 # near/far edge ratio about 1.35; foreshortening
        hl = hr = bh / 2
        wt = wb = bw / 2
        if self.tilt == "left edge nearer":
            hl, hr, wt, wb = hl * k, hr / k, wt * f, wb * f
        elif self.tilt == "right edge nearer":
            hl, hr, wt, wb = hl / k, hr * k, wt * f, wb * f
        elif self.tilt == "top edge nearer":
            wt, wb, hl, hr = wt * k, wb / k, hl * f, hr * f
        elif self.tilt == "bottom edge nearer":
            wt, wb, hl, hr = wt / k, wb * k, hl * f, hr * f
        x, y = self.cx * w, self.cy * h
        if self.tilt in ("left edge nearer", "right edge nearer"):
            w_half = wt
            return np.array([[x - w_half, y - hl], [x + w_half, y - hr],
                             [x + w_half, y + hr], [x - w_half, y + hl]])
        h_half = hl
        return np.array([[x - wt, y - h_half], [x + wt, y - h_half],
                         [x + wb, y + h_half], [x - wb, y + h_half]])


def default_targets() -> List[Target]:
    """14 poses that between them cover the image, three distances and four tilts."""
    t = [Target("Large, in the centre, facing the camera", 0.50, 0.50, 0.60),
         Target("Small, in the centre, facing the camera", 0.50, 0.50, 0.30),
         Target("Top-left corner of the image, facing the camera", 0.24, 0.27, 0.36),
         Target("Top-right corner of the image, facing the camera", 0.76, 0.27, 0.36),
         Target("Bottom-right corner of the image, facing the camera", 0.76, 0.73, 0.36),
         Target("Bottom-left corner of the image, facing the camera", 0.24, 0.73, 0.36),
         Target("Centre, LEFT edge turned toward the camera", 0.50, 0.50, 0.45, TILTS[0]),
         Target("Centre, RIGHT edge turned toward the camera", 0.50, 0.50, 0.45, TILTS[1]),
         Target("Centre, TOP edge tipped toward the camera", 0.50, 0.50, 0.45, TILTS[2]),
         Target("Centre, BOTTOM edge tipped toward the camera", 0.50, 0.50, 0.45, TILTS[3]),
         Target("Left side, RIGHT edge turned toward the camera", 0.30, 0.50, 0.38, TILTS[1]),
         Target("Right side, LEFT edge turned toward the camera", 0.70, 0.50, 0.38, TILTS[0]),
         Target("Top, BOTTOM edge tipped toward the camera", 0.50, 0.28, 0.38, TILTS[3]),
         Target("Bottom, TOP edge tipped toward the camera", 0.50, 0.72, 0.38, TILTS[2])]
    return t


def fit_target(t: Target, w: int, h: int) -> np.ndarray:
    """Target outline, shrunk if needed so it stays inside the image (for very wide/tall images)."""
    q = t.quad(w, h)
    for _ in range(20):
        if q[:, 0].min() >= 0.02 * w and q[:, 0].max() <= 0.98 * w and \
           q[:, 1].min() >= 0.02 * h and q[:, 1].max() <= 0.98 * h:
            break
        c = q.mean(axis=0)
        q = c + (q - c) * 0.93
    return q


def match_error(det_quad: np.ndarray, tgt_quad: np.ndarray, w: int, h: int) -> float:
    """Mean corner distance between the board's outline and the target, as a fraction of the
    image diagonal."""
    return float(np.mean(np.linalg.norm(det_quad - tgt_quad, axis=1)) / math.hypot(w, h))


TILT_HOW = {
    TILTS[0]: "turn the board so its LEFT edge comes toward the camera",
    TILTS[1]: "turn the board so its RIGHT edge comes toward the camera",
    TILTS[2]: "tip the board so its TOP edge comes toward the camera",
    TILTS[3]: "tip the board so its BOTTOM edge comes toward the camera",
}


def hint(det: View, tgt: View, target: Target, matched_tilt: bool) -> str:
    """The single most useful correction, in words."""
    dx, dy = tgt.cx - det.cx, tgt.cy - det.cy
    ratio = det.size / tgt.size
    if abs(dx) > 0.12 or abs(dy) > 0.12:
        pass  # fall through to position below
    elif ratio < 0.85:
        return "Bring the camera closer to the board (the board should look bigger)."
    elif ratio > 1.18:
        return "Move the camera back from the board (the board should look smaller)."
    elif target.tilt and not matched_tilt:
        return TILT_HOW[target.tilt].capitalize() + ", about 30-40 degrees."
    elif not target.tilt and tilt_bins(det, 0.05):
        return "Face the board straight at the camera (no tilt)."
    parts = []
    if abs(dx) > 0.03:
        parts.append("toward the RIGHT of the image" if dx > 0 else "toward the LEFT of the image")
    if abs(dy) > 0.03:
        parts.append("DOWN in the image" if dy > 0 else "UP in the image")
    if parts:
        return "Shift the board " + " and ".join(parts) + "."
    if ratio < 0.95:
        return "Slightly closer."
    if ratio > 1.07:
        return "Slightly further."
    return "Nearly there: match the outline's shape and hold still."


@dataclass
class Sample:
    corners: np.ndarray
    view: View
    image: Optional[np.ndarray] = None     # kept so the views can be saved and re-run
    target: int = -1


class TargetSession:
    """Walks through the targets: a view is captured when the board matches the current target's
    outline closely and is steady for a moment."""

    def __init__(self, board: Board, targets: Optional[List[Target]] = None,
                 match_tol: float = 0.045, still_frac: float = 0.003, still_frames: int = 2):
        self.board = board
        self.targets = targets or default_targets()
        self.match_tol, self.still_frac, self.still_frames = match_tol, still_frac, still_frames
        self.samples: List[Sample] = []
        self.status: List[str] = ["todo"] * len(self.targets)     # todo / done / skipped
        self.index = 0
        self.size: Optional[Tuple[int, int]] = None
        self._prev: Optional[np.ndarray] = None
        self._steady = 0
        self.last_error: Optional[float] = None
        self.last_hint = ""

    def reset(self):
        self.__init__(self.board, self.targets, self.match_tol, self.still_frac, self.still_frames)

    @property
    def finished(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def current(self) -> Optional[Target]:
        return None if self.finished else self.targets[self.index]

    def _advance(self):
        self.index += 1
        while self.index < len(self.targets) and self.status[self.index] != "todo":
            self.index += 1
        self._steady = 0

    def skip(self):
        if not self.finished:
            self.status[self.index] = "skipped"
            self._advance()

    def closeness(self) -> float:
        """0..1 for the GUI bar: 1 = within tolerance."""
        if self.last_error is None:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - (self.last_error - self.match_tol) / (4 * self.match_tol))))

    def offer(self, corners: Optional[np.ndarray], w: int, h: int,
              image: Optional[np.ndarray] = None) -> str:
        if self.size is not None and self.size != (w, h):
            self.reset()
        self.size = (w, h)
        self.last_error = None
        if self.finished:
            self.last_hint = "All targets done. Press Calibrate."
            return self.last_hint
        if corners is None:
            self._prev, self._steady = None, 0
            self.last_hint = "No board found: show the whole board, flat and well lit."
            return self.last_hint
        t = self.current
        dq = outer_quad(corners, self.board)
        tq = fit_target(t, w, h)
        det, tgt = view_of_quad(dq, w, h), view_of_quad(tq, w, h)
        self.last_error = match_error(dq, tq, w, h)
        tilt_ok = t.tilt is None or t.tilt in tilt_bins(det, TILT * 0.8)
        moved = np.inf if self._prev is None else float(np.max(np.linalg.norm(corners - self._prev, axis=1)))
        self._prev = corners
        steady = moved <= self.still_frac * math.hypot(w, h)
        self._steady = self._steady + 1 if steady else 0
        if self.last_error <= self.match_tol and tilt_ok:
            if self._steady >= self.still_frames:
                self.samples.append(Sample(corners.copy(), det, None if image is None else image.copy(),
                                           self.index))
                self.status[self.index] = "done"
                n = self.index + 1
                self._advance()
                self.last_hint = f"Captured target {n}. " + (
                    "All targets done. Press Calibrate." if self.finished else "Next outline shown.")
                return self.last_hint
            self.last_hint = "Matched: hold still..."
            return self.last_hint
        self.last_hint = hint(det, tgt, t, tilt_ok)
        return self.last_hint


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
        """(level, name, text) with level in good / warn / bad. Only 'bad' blocks saving."""
        w, h = self.size
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        out = []
        lvl = "good" if self.rms < 0.5 else "warn" if self.rms < 1.0 else "bad"
        out.append((lvl, "Reprojection error", f"{self.rms:.2f} px (good below 0.5)"))
        off = max(abs(cx - w / 2) / w, abs(cy - h / 2) / h)
        if off < 0.06:
            out.append(("good", "Image centre", f"({cx:.0f}, {cy:.0f}), near the middle"))
        elif off < 0.25:
            out.append(("warn", "Image centre",
                        f"({cx:.0f}, {cy:.0f}), away from the middle ({w / 2:.0f}, {h / 2:.0f}). "
                        f"Normal for a cropped camera mode; check with the ruler test"))
        else:
            out.append(("bad", "Image centre", f"({cx:.0f}, {cy:.0f}) is implausibly far from the middle"))
        r = abs(fx - fy) / fx
        lvl = "good" if r < 0.02 else "warn" if r < 0.05 else "bad"
        out.append((lvl, "fx vs fy", f"{fx:.0f} vs {fy:.0f} (should match within 2%)"))
        out.append(("good" if 40 <= self.hfov_deg <= 100 else "warn", "Horizontal field of view",
                    f"{self.hfov_deg:.0f} deg (ordinary webcams: about 55-80; narrower in cropped modes)"))
        return out

    def ok(self) -> bool:
        return all(level != "bad" for level, _, _ in self.checks())


def calibrate(corner_sets: List[np.ndarray], board: Board, size: Tuple[int, int],
              drop_factor: float = 2.5, drop_floor_px: float = 1.0) -> Result:
    """Calibrate (k1, k2, p1, p2; k3 fixed at 0, which suits ordinary webcams), then drop views
    whose error is far above the rest and calibrate once more."""
    if len(corner_sets) < 8:
        raise ValueError(f"only {len(corner_sets)} views; collect at least 8 (12+ recommended)")
    obj = board.object_points()
    flags = cv2.CALIB_FIX_K3

    def run(idx):
        rms, K, D, rv, tv = cv2.calibrateCamera([obj] * len(idx),
                                                [corner_sets[i].reshape(-1, 1, 2) for i in idx],
                                                size, None, None, flags=flags)
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
        per_view = list(errs)
        for i, e in zip(keep, errs2):
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