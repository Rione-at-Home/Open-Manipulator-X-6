"""
Step-by-step executor for a taught sequence (ROS-free; the runner node feeds it and acts on
what it returns).

Each tick gets the measured joint positions/velocities and whether both pads are in contact,
and returns actions:
    ("command", {joint: target})     publish a full /arm_command
    ("velocity", int)                publish /arm_profile_velocity
    ("cap", int)                     publish /arm_gripper_current
    ("rezero",)                      call /flexigrasp/rezero_all
    ("event", dict)                  publish /sequence_step and log a row
    ("done", str)                    finished ("ok" or "aborted: ...")

Motion is point to point: a move publishes the pose once and waits until every arm joint is
within tolerance and nearly still (or times out), letting the driver's profile velocity shape
the motion. A timeout or a close without contact is logged and the sequence carries on.
"""
from typing import Dict, List, Optional, Tuple

from .sequence import Sequence, Step

Action = Tuple


class SequenceExecutor:
    def __init__(self, seq: Sequence, poses: Dict[str, dict], gripper_open: Optional[float],
                 gripper: str = "gripper"):
        self.seq = seq
        self.o = seq.options
        self.poses = poses
        self.gripper = gripper
        self.gripper_open = self.o["gripper_open"] if self.o["gripper_open"] is not None else gripper_open
        self.steps: List[Step] = seq.expanded()
        self.i = -1
        self.targets: Dict[str, float] = {}
        self.finished: Optional[str] = None
        self._st: dict = {}
        self.results: List[dict] = []

    # ---- helpers ------------------------------------------------------------------------------
    @property
    def step(self) -> Optional[Step]:
        return self.steps[self.i] if 0 <= self.i < len(self.steps) else None

    def _event(self, now, status, **extra) -> Action:
        s = self.step
        ev = {"t": round(now, 3), "phase": s.phase, "repeat": s.repeat, "index": s.index,
              "step": s.label, "status": status}
        ev.update(extra)
        return ("event", ev)

    def abort(self, now: float, reason: str) -> List[Action]:
        if self.finished:
            return []
        acts = []
        if self.step is not None:
            acts.append(self._event(now, "aborted", reason=reason))
        self.finished = f"aborted: {reason}"
        acts.append(("done", self.finished))
        return acts

    # ---- main loop ------------------------------------------------------------------------------
    def tick(self, now: float, pos: Dict[str, float], vel: Dict[str, float],
             contact_both: bool) -> List[Action]:
        if self.finished:
            return []
        if not pos:
            return []
        if not self.targets:                       # first tick: hold everything where it is
            self.targets = {n: float(p) for n, p in pos.items()}
            acts = [("velocity", int(self.o["profile_velocity"])),
                    ("cap", int(self.o["gripper_current"]))]
            return acts + self._next(now, pos)
        return self._run(now, pos, vel, contact_both)

    def _next(self, now, pos) -> List[Action]:
        self.i += 1
        if self.i >= len(self.steps):
            self.finished = "ok"
            return [("done", "ok")]
        s = self.step
        self._st = {"t0": now, "ok_since": None}
        acts = [self._event(now, "start")]
        if s.kind == "move":
            pose = self.poses[s.arg]
            for n, p in zip(pose["names"], pose["positions"]):
                self.targets[n] = float(p)
            self._st["joints"] = list(pose["names"])
            acts.append(("command", dict(self.targets)))
        elif s.kind == "gripper" and s.arg == "open":
            self.targets[self.gripper] = float(self.gripper_open)
            acts.append(("command", dict(self.targets)))
        elif s.kind == "gripper" and s.arg == "close":
            self._st.update(start=self.targets.get(self.gripper, pos.get(self.gripper, 0.0)),
                            last_step=now, contact_at=None, hold_until=None)
        elif s.kind == "speed":
            acts.append(("velocity", int(s.arg)))
        elif s.kind == "cap":
            acts.append(("cap", int(s.arg)))
        elif s.kind == "rezero":
            acts.append(("rezero",))
        return acts

    def _finish(self, now, pos, result, **extra) -> List[Action]:
        dur = round(now - self._st["t0"], 3)
        row = {"phase": self.step.phase, "repeat": self.step.repeat, "index": self.step.index,
               "step": self.step.label, "result": result, "duration_s": dur}
        row.update(extra)
        self.results.append(row)
        return [self._event(now, result, duration_s=dur, **extra)] + self._next(now, pos)

    def _settled(self, now, pos, vel, joints, tol) -> bool:
        ok = all(abs(pos.get(j, 1e9) - self.targets[j]) <= tol and
                 abs(vel.get(j, 0.0)) <= self.o["velocity_tolerance"] for j in joints)
        if not ok:
            self._st["ok_since"] = None
            return False
        if self._st["ok_since"] is None:
            self._st["ok_since"] = now
        return now - self._st["ok_since"] >= self.o["settle_s"]

    def _run(self, now, pos, vel, contact_both) -> List[Action]:
        s, st, o = self.step, self._st, self.o
        elapsed = now - st["t0"]

        if s.kind == "move":
            if self._settled(now, pos, vel, st["joints"], o["move_tolerance_rad"]):
                return self._finish(now, pos, "ok")
            if elapsed > o["move_timeout_s"]:
                err = max(abs(pos.get(j, 0.0) - self.targets[j]) for j in st["joints"])
                return self._finish(now, pos, "timeout", max_error_rad=round(err, 4))
            return []

        if s.kind == "gripper" and s.arg == "open":
            if self._settled(now, pos, vel, [self.gripper], o["gripper_tolerance_rad"]):
                return self._finish(now, pos, "ok")
            if elapsed > o["move_timeout_s"]:
                return self._finish(now, pos, "timeout")
            return []

        if s.kind == "gripper" and s.arg == "close":
            g = self.gripper
            if st["contact_at"] is None:
                if contact_both:
                    st["contact_at"] = pos.get(g, self.targets[g])
                    self.targets[g] = st["contact_at"] + o["close_sign"] * o["squeeze_rad"]
                    st["hold_until"] = now + o["after_contact_s"]
                    return [("command", dict(self.targets))]
                travel = abs(self.targets[g] - st["start"])
                if travel >= o["max_close_travel_rad"] or elapsed > o["move_timeout_s"]:
                    # Carry on (logged), holding the gripper where it actually is.
                    self.targets[g] = pos.get(g, self.targets[g])
                    return [("command", dict(self.targets))] + self._finish(
                        now, pos, "no_contact", gripper_rad=round(self.targets[g], 4))
                if now - st["last_step"] >= o["close_period_s"]:
                    st["last_step"] = now
                    self.targets[g] += o["close_sign"] * o["close_step_rad"]
                    return [("command", dict(self.targets))]
                return []
            if now >= st["hold_until"]:
                return self._finish(now, pos, "ok", contact_rad=round(st["contact_at"], 4),
                                    held_rad=round(pos.get(g, float("nan")), 4))
            return []

        if s.kind == "wait":
            return self._finish(now, pos, "ok") if elapsed >= s.arg else []
        if s.kind == "rezero":
            return self._finish(now, pos, "ok") if elapsed >= o["rezero_wait_s"] else []
        # speed, cap, mark: done immediately
        return self._finish(now, pos, "ok")