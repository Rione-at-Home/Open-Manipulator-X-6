"""
Gamepad teleop logic for the X6 (ROS-free, so it can be tested without a robot).

Controls (indices come from config/joy_ps4.yaml):
    L1 / R1          previous / next joint (joint1 ... wrist_rotate, gripper, then wraps)
    circle / cross   held: move the selected joint forward / reverse
                     (gripper selected: circle closes, cross opens)
    triangle         fine / coarse step
    square           save pose (the pendant supplies the name)
    options          stop: /arm_stop, then targets re-sync to where the arm is

Targets always cover every joint and start from the measured /joint_states, so nothing jumps
at start-up and joints you are not moving hold their pose.
"""
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ButtonMap:
    prev_joint: int = 4     # L1
    next_joint: int = 5     # R1
    forward: int = 1        # circle
    reverse: int = 0        # cross
    fine: int = 2           # triangle
    save: int = 3           # square
    stop: int = 9           # options

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ButtonMap":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown button names: {sorted(unknown)}")
        bm = cls(**{k: int(v) for k, v in d.items()})
        vals = list(bm.__dict__.values())
        if len(set(vals)) != len(vals):
            raise ValueError(f"two actions share a button index: {bm}")
        return bm


class JoyTeleopCore:
    def __init__(self, joint_order: List[str], gripper: str = "gripper",
                 coarse_step: float = 0.02, fine_step: float = 0.004, gripper_step: float = 0.01,
                 limits: Optional[Dict[str, Tuple[float, float]]] = None, close_sign: float = -1.0,
                 contact_stop: bool = True, idle_resync_s: float = 1.0, buttons: ButtonMap = None):
        self.joint_order = list(joint_order)       # selection order shown in the pendant
        self.gripper = gripper
        self.coarse_step, self.fine_step, self.gripper_step = coarse_step, fine_step, gripper_step
        self.limits = dict(limits or {})
        self.close_sign = 1.0 if close_sign >= 0 else -1.0
        self.contact_stop = contact_stop
        self.idle_resync_s = idle_resync_s
        self.buttons = buttons or ButtonMap()

        self.names: List[str] = []
        self.targets: Dict[str, float] = {}
        self.selected = 0
        self.fine = False
        self.held = [0] * 16
        self._prev = [0] * 16
        self.last_cmd_t = -1e9
        self.prev_contact = False
        self.message = ""
        self.last_pressed: List[int] = []

    # ---- state ------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return bool(self.names)

    @property
    def selected_joint(self) -> str:
        return self.joint_order[self.selected]

    def sync(self, names: Iterable[str], positions: Iterable[float]):
        self.names = list(names)
        self.targets = {n: float(p) for n, p in zip(self.names, positions)}

    def _clamp(self, name: str, value: float) -> Tuple[float, bool]:
        lo, hi = self.limits.get(name, (-math.pi, math.pi))
        v = min(hi, max(lo, value))
        return v, v != value

    def set_targets(self, names: Iterable[str], positions: Iterable[float], now: float):
        """Move To a saved pose: set only the joints given (arm joints); the gripper is untouched."""
        for n, p in zip(names, positions):
            if n in self.targets:
                self.targets[n], _ = self._clamp(n, float(p))
        self.last_cmd_t = now

    # ---- input ------------------------------------------------------------------------------
    def on_joy(self, buttons: List[int]) -> set:
        """Edge-triggered actions. Returns a subset of {'stop', 'save', 'select', 'fine'}."""
        b = list(buttons) + [0] * max(0, 16 - len(buttons))
        pressed = [i for i, (now, was) in enumerate(zip(b, self._prev + [0] * len(b))) if now and not was]
        self._prev, self.held = b, b
        if pressed:
            self.last_pressed = pressed
        events = set()
        bm = self.buttons
        if bm.stop in pressed:
            events.add("stop")
            self.names = []                      # re-sync from the next /joint_states
            self.message = "STOP sent; targets re-sync to the arm"
            return events
        if bm.prev_joint in pressed:
            self.selected = (self.selected - 1) % len(self.joint_order)
            events.add("select")
        if bm.next_joint in pressed:
            self.selected = (self.selected + 1) % len(self.joint_order)
            events.add("select")
        if bm.fine in pressed:
            self.fine = not self.fine
            events.add("fine")
        if bm.save in pressed:
            events.add("save")
        return events

    def tick(self, now: float, measured: Dict[str, float], contact_both: bool) -> bool:
        """Call at a fixed rate. Returns True when targets changed and a command should go out."""
        changed = False
        if not self.ready:
            self.prev_contact = contact_both
            return False

        # Contact just made while stop-on-contact is on: hold the gripper where it is now,
        # dropping any closing that was queued ahead of the motion.
        # Only while this teleop is in use, so it never overrides a sequence run with stale targets.
        active = now - self.last_cmd_t <= self.idle_resync_s
        if (contact_both and not self.prev_contact and self.contact_stop and active
                and self.gripper in measured):
            self.targets[self.gripper], _ = self._clamp(self.gripper, measured[self.gripper])
            self.message = f"contact: gripper held at {self.targets[self.gripper]:+.3f}"
            self.prev_contact = contact_both
            self.last_cmd_t = now
            return True
        self.prev_contact = contact_both

        bm = self.buttons
        direction = (1 if self.held[bm.forward] else 0) - (1 if self.held[bm.reverse] else 0)
        if direction == 0:
            return changed

        # Coming back after a pause (or after another program moved the arm): start from the arm.
        if now - self.last_cmd_t > self.idle_resync_s and measured:
            for n in self.names:
                if n in measured:
                    self.targets[n] = float(measured[n])

        joint = self.selected_joint
        if joint not in self.targets:
            self.message = f"{joint} not in /joint_states"
            return changed
        if joint == self.gripper:
            closing = direction > 0
            if closing and self.contact_stop and contact_both:
                self.message = "both pads in contact: not closing further"
                return changed
            delta = self.gripper_step * (self.close_sign if closing else -self.close_sign)
        else:
            delta = direction * (self.fine_step if self.fine else self.coarse_step)
        self.targets[joint], hit = self._clamp(joint, self.targets[joint] + delta)
        self.message = f"{joint} at its limit" if hit else ""
        self.last_cmd_t = now
        return True

    def command(self) -> Tuple[List[str], List[float]]:
        return list(self.names), [self.targets[n] for n in self.names]