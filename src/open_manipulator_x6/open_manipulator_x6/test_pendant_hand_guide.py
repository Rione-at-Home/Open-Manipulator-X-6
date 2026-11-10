"""Hand-guide in the teleop core and the pendant node (ROS replaced by stand-ins)."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from open_manipulator_x6.joy_teleop_core import ButtonMap, JoyTeleopCore  # noqa: E402

NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "wrist_rotate", "gripper"]


def core():
    c = JoyTeleopCore(NAMES)
    c.sync(NAMES, [0.0] * 7)
    return c


def press(c, *idx):
    b = [0] * 16
    for i in idx:
        b[i] = 1
    ev = c.on_joy(b)
    c.on_joy([0] * 16)
    return ev


def test_share_button_is_an_event_and_default_map_is_valid():
    assert ButtonMap().hand_guide == 8
    ButtonMap.from_dict({})                                    # no duplicate indices
    assert "hand_guide" in press(core(), 8)


def test_no_commands_while_limp_and_resync_after():
    c = core()
    c.set_hand_guide(True)
    c.on_joy([0, 1] + [0] * 14)                                # hold circle
    assert c.tick(10.0, {n: 0.0 for n in NAMES}, False) is False
    c.set_hand_guide(False)                                    # arm was moved by hand
    assert not c.ready                                         # waits for fresh /joint_states
    c.sync(NAMES, [0.5] * 7)                                    # the pose it was moved to
    assert c.targets["joint1"] == 0.5
    assert c.tick(20.0, {n: 0.5 for n in NAMES}, False) is True
    assert abs(c.targets["joint1"] - 0.52) < 1e-9              # jogs on from the new pose


# ---- node-level, with stand-ins ---------------------------------------------------------------
class Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Pub:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m)


class FakeNode:
    def __init__(self, name):
        self._p, self.pubs = {}, {}

    def declare_parameter(self, n, v):
        self._p[n] = v

    def get_parameter(self, n):
        return types.SimpleNamespace(value=self._p[n])

    def create_publisher(self, t, topic, q):
        self.pubs[topic] = Pub()
        return self.pubs[topic]

    def create_subscription(self, *a): pass
    def create_timer(self, *a): pass
    def get_logger(self): return types.SimpleNamespace(warn=print, info=print)
    def get_clock(self): return types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: 0))


@pytest.fixture
def node(monkeypatch):
    for m in ("rclpy", "rclpy.node", "rclpy.qos", "sensor_msgs", "sensor_msgs.msg", "std_msgs",
              "std_msgs.msg"):
        monkeypatch.setitem(sys.modules, m, types.ModuleType(m))
    sys.modules["rclpy.node"].Node = FakeNode
    sys.modules["rclpy.qos"].qos_profile_sensor_data = None
    sys.modules["sensor_msgs.msg"].JointState = Msg
    sys.modules["sensor_msgs.msg"].Joy = Msg
    for n in ("Bool", "Empty", "Int16", "Int32", "String"):
        setattr(sys.modules["std_msgs.msg"], n, Msg)
    ps = types.ModuleType("open_manipulator_x6.pose_store")
    ps.PoseError = type("PoseError", (Exception,), {})
    ps.PoseStore = lambda path: types.SimpleNamespace(poses={}, names=lambda: [], gripper_open=None)
    monkeypatch.setitem(sys.modules, "open_manipulator_x6.pose_store", ps)
    monkeypatch.delitem(sys.modules, "open_manipulator_x6.teach_pendant", raising=False)
    import open_manipulator_x6.teach_pendant as tp
    n = tp.PendantNode()
    n.on_joint_states(Msg(name=NAMES, position=[0.0] * 7, velocity=[0.0] * 7))
    return tp, n


def test_share_toggles_and_publishes(node):
    tp, n = node
    n.on_joy(Msg(buttons=[0] * 8 + [1] + [0] * 7))
    n.on_joy(Msg(buttons=[0] * 16))
    assert n.core.hand_guiding and n.pubs["/arm_hand_guide"].sent[-1].data is True
    n.on_joy(Msg(buttons=[0] * 8 + [1] + [0] * 7))
    assert not n.core.hand_guiding and n.pubs["/arm_hand_guide"].sent[-1].data is False


def test_status_grace_then_driver_is_the_authority(node, monkeypatch):
    tp, n = node
    t = [100.0]
    monkeypatch.setattr(tp.time, "monotonic", lambda: t[0])
    n.hand_guide(True)
    n.on_status(Msg(data="READY"))                             # driver not switched yet: ignored
    assert n.core.hand_guiding
    n.on_status(Msg(data="HAND_GUIDE"))
    t[0] += 5.0
    n.on_status(Msg(data="READY"))                             # e.g. STOP from the keyboard
    assert not n.core.hand_guiding and not n.core.ready        # resyncs before jogging again


def test_old_driver_is_reported(node, monkeypatch):
    tp, n = node
    t = [100.0]
    monkeypatch.setattr(tp.time, "monotonic", lambda: t[0])
    n.hand_guide(True)
    t[0] += 0.8
    n.on_status(Msg(data="READY"))                             # never became HAND_GUIDE
    assert not n.core.hand_guiding and "did not enter hand-guide" in n.core.message


def test_stop_ends_hand_guide(node):
    tp, n = node
    n.hand_guide(True)
    n.stop()
    assert not n.core.hand_guiding and n.pubs["/arm_stop"].sent