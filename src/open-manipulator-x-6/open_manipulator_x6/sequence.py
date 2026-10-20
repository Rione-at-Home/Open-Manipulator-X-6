"""
Sequence file for the X6 sequence runner (ROS-free).

    name: bottle_stability
    poses_file: ~/Open-Manipulator-X-6/poses.yaml
    repeats: 5
    profile_velocity: 30          # sent to /arm_profile_velocity at the start
    gripper_current: 100          # sent to /arm_gripper_current at the start
    setup:                        # run once, before the repeats
      - move: home
      - gripper: open
    steps:                        # run `repeats` times
      - rezero                    # re-zero both pads (gripper should be open, nothing touching)
      - move: pre_grasp
      - move: grasp
      - gripper: close            # close until both pads touch, then squeeze a little past contact
      - wait: 2.0
      - move: lift
      - gripper: open
    teardown:                     # run once, after the repeats
      - move: home

Step types: move: <pose>, gripper: open|close, wait: <s>, speed: <profile velocity>,
cap: <gripper goal current>, rezero, mark: <label>.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

STEP_KINDS = ("move", "gripper", "wait", "speed", "cap", "rezero", "mark")

DEFAULT_OPTIONS = {
    "repeats": 1,
    "profile_velocity": 30,
    "gripper_current": 100,
    "gripper_open": None,           # rad; None = use gripper_open from the pose file
    "move_tolerance_rad": 0.03,
    "gripper_tolerance_rad": 0.03,
    "velocity_tolerance": 0.05,     # rad/s, "stopped" threshold
    "settle_s": 0.3,
    "move_timeout_s": 15.0,
    "close_step_rad": 0.01,
    "close_period_s": 0.05,
    "squeeze_rad": 0.05,
    "max_close_travel_rad": 1.2,
    "close_sign": -1.0,
    "after_contact_s": 1.0,
    "rezero_wait_s": 1.0,
}


class SequenceError(ValueError):
    pass


@dataclass
class Step:
    kind: str
    arg: Any = None
    phase: str = "steps"            # setup | steps | teardown
    repeat: int = 0                 # 1-based for "steps", 0 for setup/teardown
    index: int = 0                  # position within its block

    @property
    def label(self) -> str:
        return self.kind if self.arg is None else f"{self.kind}:{self.arg}"


@dataclass
class Sequence:
    name: str
    poses_file: str
    options: Dict[str, Any]
    setup: List[Step] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    teardown: List[Step] = field(default_factory=list)

    def expanded(self) -> List[Step]:
        out = [Step(s.kind, s.arg, "setup", 0, i) for i, s in enumerate(self.setup)]
        for r in range(1, int(self.options["repeats"]) + 1):
            out += [Step(s.kind, s.arg, "steps", r, i) for i, s in enumerate(self.steps)]
        out += [Step(s.kind, s.arg, "teardown", 0, i) for i, s in enumerate(self.teardown)]
        return out

    def pose_names(self) -> List[str]:
        return [s.arg for s in self.setup + self.steps + self.teardown if s.kind == "move"]


def _parse_step(raw, where: str) -> Step:
    if isinstance(raw, str):
        kind, arg = raw.strip(), None
    elif isinstance(raw, dict) and len(raw) == 1:
        kind, arg = next(iter(raw.items()))
    else:
        raise SequenceError(f"{where}: each step must be a word (e.g. rezero) or one key: value")
    if kind not in STEP_KINDS:
        raise SequenceError(f"{where}: unknown step '{kind}'; use one of {STEP_KINDS}")
    if kind == "move" and (not isinstance(arg, str) or not arg):
        raise SequenceError(f"{where}: move needs a pose name")
    if kind == "gripper" and arg not in ("open", "close"):
        raise SequenceError(f"{where}: gripper must be open or close")
    if kind == "wait":
        if not isinstance(arg, (int, float)) or arg < 0:
            raise SequenceError(f"{where}: wait needs seconds >= 0")
        arg = float(arg)
    if kind in ("speed", "cap"):
        if not isinstance(arg, int) or arg < 0:
            raise SequenceError(f"{where}: {kind} needs a whole number >= 0")
    if kind == "rezero" and arg is not None:
        raise SequenceError(f"{where}: rezero takes no value")
    if kind == "mark":
        arg = str(arg)
    return Step(kind, arg)


def parse(data: dict, default_poses_file: str = "~/Open-Manipulator-X-6/poses.yaml") -> Sequence:
    if not isinstance(data, dict):
        raise SequenceError("sequence file must be a mapping")
    unknown = set(data) - {"name", "poses_file", "setup", "steps", "teardown"} - set(DEFAULT_OPTIONS)
    if unknown:
        raise SequenceError(f"unknown keys: {sorted(unknown)}")
    if not data.get("steps"):
        raise SequenceError("'steps' is empty")
    opts = dict(DEFAULT_OPTIONS)
    opts.update({k: data[k] for k in DEFAULT_OPTIONS if k in data})
    if int(opts["repeats"]) < 1:
        raise SequenceError("repeats must be >= 1")
    blocks = {}
    for block in ("setup", "steps", "teardown"):
        raw = data.get(block) or []
        if not isinstance(raw, list):
            raise SequenceError(f"'{block}' must be a list")
        blocks[block] = [_parse_step(s, f"{block}[{i}]") for i, s in enumerate(raw)]
    return Sequence(str(data.get("name", "sequence")), str(data.get("poses_file", default_poses_file)),
                    opts, **blocks)


def load(path: str, **kw) -> Sequence:
    with open(path) as f:
        return parse(yaml.safe_load(f), **kw)


def check_against_poses(seq: Sequence, pose_names: List[str], gripper_open_in_file: Optional[float]):
    """Raise SequenceError if the sequence uses a pose or gripper-open value that does not exist."""
    missing = sorted(set(seq.pose_names()) - set(pose_names))
    if missing:
        raise SequenceError(f"poses not found in the pose file: {missing}")
    uses_open = any(s.kind == "gripper" and s.arg == "open" for s in seq.setup + seq.steps + seq.teardown)
    if uses_open and seq.options["gripper_open"] is None and gripper_open_in_file is None:
        raise SequenceError("gripper: open is used but no gripper_open is set (pendant 'Set gripper open', "
                            "or gripper_open: in the sequence file)")