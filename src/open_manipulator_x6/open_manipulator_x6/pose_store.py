"""
Saved-pose file for the X6 teach pendant and sequence runner (ROS-free).

File layout (YAML):

    version: 1
    gripper_open: 1.57            # optional, rad; set with "Set gripper open" in the pendant
    poses:
      home:
        names: [joint1, joint2, joint3, joint4, joint5, wrist_rotate]
        positions: [0.0, -1.2, 0.4, 0.8, 0.0, 0.0]
        saved: '2026-10-20T19:40:00'

Poses hold ARM joints only. The gripper is handled as separate open/close actions, so a grasp
is decided by the pads, not by replaying a saved gripper angle.

Writes go to a temp file that is then renamed over the old one, so a crash mid-write cannot
corrupt the pose file.
"""
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import yaml

FORMAT_VERSION = 1


class PoseError(ValueError):
    pass


class PoseStore:
    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        self.poses: Dict[str, dict] = {}
        self.gripper_open: Optional[float] = None
        self.load()

    # ---- file I/O -------------------------------------------------------------------------
    def load(self):
        self.poses, self.gripper_open = {}, None
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise PoseError(f"{self.path}: expected a mapping at the top level")
        poses = data.get("poses", {}) or {}
        for name, p in poses.items():
            self._check(name, p.get("names"), p.get("positions"))
            self.poses[str(name)] = {
                "names": [str(n) for n in p["names"]],
                "positions": [float(x) for x in p["positions"]],
                "saved": p.get("saved", ""),
            }
        g = data.get("gripper_open")
        self.gripper_open = None if g is None else float(g)

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        data = {"version": FORMAT_VERSION, "poses": self.poses}
        if self.gripper_open is not None:
            data["gripper_open"] = float(self.gripper_open)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, self.path)

    # ---- editing ------------------------------------------------------------------------------
    @staticmethod
    def _check(name, names, positions):
        if not isinstance(name, str) or not name.strip():
            raise PoseError("pose name is empty")
        if not names or positions is None or len(names) != len(positions):
            raise PoseError(f"pose '{name}': names and positions must be the same non-zero length")
        for x in positions:
            if not math.isfinite(float(x)):
                raise PoseError(f"pose '{name}': non-finite position")

    def set(self, name: str, names: List[str], positions: List[float]):
        name = name.strip()
        self._check(name, names, positions)
        self.poses[name] = {
            "names": list(names),
            "positions": [round(float(x), 5) for x in positions],
            "saved": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def delete(self, name: str) -> bool:
        if name in self.poses:
            del self.poses[name]
            self.save()
            return True
        return False

    def set_gripper_open(self, position: float):
        if not math.isfinite(float(position)):
            raise PoseError("gripper_open must be finite")
        self.gripper_open = round(float(position), 5)
        self.save()

    def get(self, name: str) -> dict:
        if name not in self.poses:
            raise PoseError(f"pose '{name}' not found; available: {sorted(self.poses)}")
        return self.poses[name]

    def names(self) -> List[str]:
        return list(self.poses.keys())