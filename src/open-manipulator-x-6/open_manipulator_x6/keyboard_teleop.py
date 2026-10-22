#!/usr/bin/env python3
"""
Keyboard teleop for joint 1 and the gripper, for the pad-on-arm tests.

Publishes to the topics arm_driver already listens on:
    /arm_command          sensor_msgs/JointState  (ALL joints, so nothing else drifts)
    /arm_gripper_current  std_msgs/Int16          (gripper torque cap, 2.69 mA per unit)
    /arm_profile_velocity std_msgs/Int32          (speed of all joints)
    /arm_stop             std_msgs/Empty

Keys (run in a real terminal; hold a key to repeat):
    a / d     joint 1  + / -      (A / D = 5x step)
    w / s     gripper  close / open   (W / S = 5x step)
    g         flip which direction counts as "close"  (press if w opens the gripper)
    ]  /  [   gripper current cap  + / -  (10 units)
    =  /  -   speed (profile velocity) + / -  (5 units)
    c         toggle stop-on-contact: when both pads first touch, the gripper is told to hold
              where it is right then (stops queued closing), and further closing is blocked
    space     STOP: /arm_stop, targets re-sync to where the arm is
    h         help          x or Ctrl-C   quit (torque stays on, arm holds its pose)

Run:
    ros2 run open_manipulator_x6 keyboard_teleop
    ros2 run open_manipulator_x6 keyboard_teleop --ros-args -p joint1_step:=0.01 -p close_sign:=-1.0
"""
import os
import select
import sys
import termios
import time
import tty

HELP = __doc__.split("Keys", 1)[1].split("Run:", 1)[0]


class TeleopCore:
    """ROS-free logic: targets, clamps and key handling."""

    def __init__(self, joint1="joint1", gripper="gripper", joint1_step=0.02, gripper_step=0.02,
                 joint1_range=0.8, gripper_range=0.8, close_sign=-1.0, cap=100, cap_step=10,
                 cap_min=0, cap_max=200, vel=40, vel_step=5, vel_min=5, vel_max=100,
                 contact_stop=True):
        self.joint1, self.gripper = joint1, gripper
        self.j1_step, self.g_step = joint1_step, gripper_step
        self.j1_range, self.g_range = joint1_range, gripper_range
        self.close_sign = 1.0 if close_sign >= 0 else -1.0
        self.cap, self.cap_step, self.cap_min, self.cap_max = cap, cap_step, cap_min, cap_max
        self.vel, self.vel_step, self.vel_min, self.vel_max = vel, vel_step, vel_min, vel_max
        self.contact_stop = contact_stop
        self.names, self.targets, self.origin = [], {}, {}
        self.message = ""

    @property
    def ready(self):
        return bool(self.names)

    def sync(self, names, positions):
        """(Re)start from the arm's measured pose. Limits are centred on this pose."""
        self.names = list(names)
        self.targets = dict(zip(names, positions))
        self.origin = dict(self.targets)

    def _move(self, name, delta, rng):
        lo, hi = self.origin[name] - rng, self.origin[name] + rng
        new = min(hi, max(lo, self.targets[name] + delta))
        hit = new != self.targets[name] + delta
        self.targets[name] = new
        return hit

    def resync_targets(self, names, positions):
        """Take the arm's measured pose as the new targets, and re-centre the +/- range on it.
        Used after a pause, so a key press never sends targets that another program has since
        moved (and never snaps a joint back into a range centred on an old pose)."""
        for n, p in zip(names, positions):
            if n in self.targets:
                self.targets[n] = float(p)
                self.origin[n] = float(p)

    def hold_gripper_at(self, present):
        """Contact just made: drop any queued closing by targeting the measured position.
        Returns True if a command should be sent."""
        if not self.ready or not self.contact_stop:
            return False
        lo, hi = self.origin[self.gripper] - self.g_range, self.origin[self.gripper] + self.g_range
        self.targets[self.gripper] = min(hi, max(lo, present))
        self.message = f"contact: gripper held at {self.targets[self.gripper]:+.3f}"
        return True

    def handle(self, key, contact_blocks_close=False):
        """Returns a set of actions: 'command', 'cap', 'vel', 'stop', 'quit'."""
        acts = set()
        if not self.ready and key not in ("x", "h", "\x03"):
            self.message = "waiting for /joint_states ..."
            return acts
        self.message = ""
        mult = 5.0 if key.isupper() else 1.0
        k = key.lower()
        if k in "ad" and k:
            sign = 1.0 if k == "a" else -1.0
            hit = self._move(self.joint1, sign * self.j1_step * mult, self.j1_range)
            acts.add("command")
            self.message = "joint 1 at its limit" if hit else ""
        elif k in "ws" and k:
            closing = (k == "w")
            if closing and self.contact_stop and contact_blocks_close:
                self.message = "both pads in contact: not closing further (c toggles this)"
                return acts
            sign = self.close_sign if closing else -self.close_sign
            hit = self._move(self.gripper, sign * self.g_step * mult, self.g_range)
            acts.add("command")
            self.message = "gripper at its limit" if hit else ""
        elif k == "g":
            self.close_sign = -self.close_sign
            self.message = f"w now moves the gripper position {'+' if self.close_sign > 0 else '-'}"
        elif key in "]":
            self.cap = min(self.cap_max, self.cap + self.cap_step)
            acts.add("cap")
        elif key in "[":
            self.cap = max(self.cap_min, self.cap - self.cap_step)
            acts.add("cap")
        elif key == "=":
            self.vel = min(self.vel_max, self.vel + self.vel_step)
            acts.add("vel")
        elif key == "-":
            self.vel = max(self.vel_min, self.vel - self.vel_step)
            acts.add("vel")
        elif k == "c":
            self.contact_stop = not self.contact_stop
            self.message = f"stop-on-contact {'ON' if self.contact_stop else 'OFF'}"
        elif key == " ":
            acts.add("stop")
        elif k == "x" or key == "\x03":
            acts.add("quit")
        elif k == "h":
            self.message = "help printed"
            acts.add("help")
        return acts


def main(args=None):  # pragma: no cover - needs ROS 2 and a terminal
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty, Int16, Int32

    try:
        from flexigrasp_msgs.msg import TactileFrame
    except ImportError:
        TactileFrame = None

    class Teleop(Node):
        def __init__(self):
            super().__init__("keyboard_teleop")
            P = self.declare_parameter
            P("joint1_name", "joint1"); P("gripper_name", "gripper")
            P("joint1_step", 0.02); P("gripper_step", 0.02)
            P("joint1_range", 0.8); P("gripper_range", 0.8)
            P("close_sign", -1.0)
            P("gripper_cap", 100); P("cap_max", 200)
            P("profile_velocity", 40); P("fingers", ["left", "right"])
            P("topic_prefix", "flexigrasp"); P("contact_stop", True)
            g = lambda n: self.get_parameter(n).value
            self.core = TeleopCore(g("joint1_name"), g("gripper_name"), g("joint1_step"),
                                   g("gripper_step"), g("joint1_range"), g("gripper_range"),
                                   g("close_sign"), g("gripper_cap"), cap_max=g("cap_max"),
                                   vel=g("profile_velocity"), contact_stop=g("contact_stop"))
            self.pub_cmd = self.create_publisher(JointState, "/arm_command", 10)
            self.pub_cap = self.create_publisher(Int16, "/arm_gripper_current", 10)
            self.pub_vel = self.create_publisher(Int32, "/arm_profile_velocity", 10)
            self.pub_stop = self.create_publisher(Empty, "/arm_stop", 10)
            self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
            self.effort = {}
            self.pads = {}
            self.fingers = list(g("fingers"))
            if TactileFrame is not None:
                for f in self.fingers:
                    self.create_subscription(
                        TactileFrame, f"/{g('topic_prefix')}/{f}/frame",
                        lambda m, f=f: self.on_pad(f, m), qos_profile_sensor_data)
            self.quit = False
            self.was_in_contact = False
            self.last_cmd_time = -1e9
            self.idle_resync_s = 1.0
            self.last_draw = 0.0
            self.create_timer(0.03, self.tick)

        def on_joints(self, m):
            self.effort = dict(zip(m.name, m.effort))
            self.latest = (list(m.name), list(m.position))
            if not self.core.ready:
                self.core.sync(*self.latest)

        def on_pad(self, f, m):
            self.pads[f] = (time.monotonic(), bool(m.in_contact), float(m.total))

        def both_in_contact(self):
            now = time.monotonic()
            if not self.fingers or len(self.pads) < len(self.fingers):
                return False
            return all(now - self.pads[f][0] < 0.5 and self.pads[f][1] for f in self.fingers)

        def send_command(self):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(self.core.names)
            msg.position = [self.core.targets[n] for n in msg.name]
            self.pub_cmd.publish(msg)
            self.last_cmd_time = time.monotonic()

        def apply(self, acts):
            if "command" in acts:
                self.send_command()
            if "cap" in acts:
                self.pub_cap.publish(Int16(data=int(self.core.cap)))
            if "vel" in acts:
                self.pub_vel.publish(Int32(data=int(self.core.vel)))
            if "stop" in acts:
                self.pub_stop.publish(Empty())
                self.core.message = "STOP sent; targets re-synced"
                self.core.names = []          # re-sync from the next /joint_states
            if "help" in acts:
                sys.stdout.write("\r\n" + HELP.replace("\n", "\r\n") + "\r\n")
            if "quit" in acts:
                self.quit = True

        def tick(self):
            contact = self.both_in_contact()
            active = time.monotonic() - self.last_cmd_time <= self.idle_resync_s
            # Only react to contact while this teleop is actually being used, so a sequence run
            # (or another teleop) is never overridden by stale targets from here.
            if contact and not self.was_in_contact and self.core.ready and active:
                names, positions = self.latest
                if self.core.gripper in names and self.core.hold_gripper_at(
                        positions[names.index(self.core.gripper)]):
                    self.send_command()
            self.was_in_contact = contact
            while select.select([sys.stdin], [], [], 0)[0]:
                data = os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
                if not data:
                    break
                for ch in data:
                    if (ch.lower() in "adws" and self.core.ready
                            and time.monotonic() - self.last_cmd_time > self.idle_resync_s):
                        self.core.resync_targets(*self.latest)
                    self.apply(self.core.handle(ch, self.both_in_contact()))
            now = time.monotonic()
            if now - self.last_draw > 0.2:
                self.last_draw = now
                self.draw()

        def draw(self):
            c = self.core
            if not c.ready:
                line = "waiting for /joint_states ..."
            else:
                pads = "  ".join(
                    f"{f[0].upper()}:{self.pads[f][2]:6.0f}{'*' if self.pads[f][1] else ' '}"
                    if f in self.pads else f"{f[0].upper()}:  --  " for f in self.fingers)
                ge = self.effort.get(c.gripper)
                line = (f"j1 {c.targets[c.joint1]:+.2f} | grip {c.targets[c.gripper]:+.2f} "
                        f"(w={'+' if c.close_sign > 0 else '-'}) | cap {c.cap} (~{c.cap * 2.69:.0f}mA) "
                        f"| vel {c.vel} | gripA {ge if ge is None else format(ge, '+.2f')} "
                        f"| pads {pads} | stop-on-contact {'ON' if c.contact_stop else 'off'}"
                        f"{' | ' + c.message if c.message else ''}")
            sys.stdout.write("\r\x1b[K" + line[:200])
            sys.stdout.flush()

    if not sys.stdin.isatty():
        raise SystemExit("keyboard_teleop needs an interactive terminal.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    rclpy.init(args=args)
    node = Teleop()
    try:
        tty.setcbreak(fd)
        sys.stdout.write(HELP.replace("\n", "\r\n") + "\r\n")
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\nbye (torque left on)\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()