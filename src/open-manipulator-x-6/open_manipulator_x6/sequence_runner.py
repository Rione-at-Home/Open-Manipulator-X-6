#!/usr/bin/env python3
"""
Plays a taught sequence on the X6 and marks each step for the logs.

    ros2 launch open_manipulator_x6 bottle_test.launch.py sequence:=<file> label:=bottle_1
or alone:
    ros2 run open_manipulator_x6 sequence_runner --ros-args -p sequence_file:=<file>

Publishes /sequence_step (std_msgs/String, JSON) at the start and end of every step, so the
session logger CSV and the bag can be split into phases. Writes a per-step summary CSV
(<out_dir>/<stamp>_<label>_steps.csv). Any /arm_stop (gamepad options, keyboard space, the
pendant STOP button) aborts the run.
"""
import csv
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Int16, Int32, String
from std_srvs.srv import Trigger

from . import sequence as sq
from .pose_store import PoseStore
from .sequence_executor import SequenceExecutor

try:
    from flexigrasp_msgs.msg import TactileFrame
except ImportError:
    TactileFrame = None


class SequenceRunner(Node):
    def __init__(self):
        super().__init__("sequence_runner")
        P = self.declare_parameter
        P("sequence_file", "")
        P("out_dir", "~/flexigrasp_logs")
        P("label", "")
        P("gripper_name", "gripper")
        P("fingers", ["left", "right"])
        P("topic_prefix", "flexigrasp")
        P("rate_hz", 50.0)
        P("pad_fresh_s", 0.5)
        P("start_delay_s", 3.0)
        g = lambda n: self.get_parameter(n).value

        path = os.path.expanduser(g("sequence_file"))
        if not path:
            raise RuntimeError("set -p sequence_file:=<path to the sequence YAML>")
        self.seq = sq.load(path)
        store = PoseStore(self.seq.poses_file)
        sq.check_against_poses(self.seq, store.names(), store.gripper_open)
        self.ex = SequenceExecutor(self.seq, store.poses, store.gripper_open, g("gripper_name"))

        stamp = time.strftime("%Y%m%d_%H%M%S")
        label = g("label") or self.seq.name
        out = os.path.expanduser(g("out_dir"))
        os.makedirs(out, exist_ok=True)
        self.summary_path = os.path.join(out, f"{stamp}_{label}_steps.csv")

        self.fingers = list(g("fingers"))
        self.pad_fresh_s = g("pad_fresh_s")
        self.pos, self.vel, self.pads = {}, {}, {}
        self.names = []
        self.start_at = time.monotonic() + g("start_delay_s")
        self.announced = False
        self.done = None

        self.pub_cmd = self.create_publisher(JointState, "/arm_command", 10)
        self.pub_vel = self.create_publisher(Int32, "/arm_profile_velocity", 10)
        self.pub_cap = self.create_publisher(Int16, "/arm_gripper_current", 10)
        self.pub_step = self.create_publisher(String, "/sequence_step", 10)
        self.pub_stop = self.create_publisher(Empty, "/arm_stop", 10)
        self.rezero = self.create_client(Trigger, f"/{g('topic_prefix')}/rezero_all")
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(Empty, "/arm_stop", self.on_stop, 10)
        if TactileFrame is not None:
            for f in self.fingers:
                self.create_subscription(TactileFrame, f"/{g('topic_prefix')}/{f}/frame",
                                         lambda m, f=f: self.on_pad(f, m), qos_profile_sensor_data)
        else:
            self.get_logger().warn("flexigrasp_msgs not found: gripper close will never see contact")
        self.create_timer(1.0 / g("rate_hz"), self.tick)
        n = len(self.ex.steps)
        self.get_logger().info(
            f"Loaded '{self.seq.name}': {n} steps ({self.seq.options['repeats']} repeats). "
            f"Starting in {g('start_delay_s'):.0f} s. Any /arm_stop aborts.")

    def on_js(self, m):
        self.names = list(m.name)
        self.pos = dict(zip(m.name, m.position))
        self.vel = dict(zip(m.name, m.velocity))

    def on_pad(self, f, m):
        self.pads[f] = (time.monotonic(), bool(m.in_contact))

    def on_stop(self, _m):
        self.act(self.ex.abort(time.monotonic(), "arm_stop received"))

    def contact_both(self):
        now = time.monotonic()
        return bool(self.fingers) and all(
            f in self.pads and now - self.pads[f][0] < self.pad_fresh_s and self.pads[f][1]
            for f in self.fingers)

    def tick(self):
        if self.done:
            return
        now = time.monotonic()
        if now < self.start_at:
            return
        if not self.pos:
            if not self.announced:
                self.get_logger().warn("Waiting for /joint_states (is arm_driver running?)")
                self.announced = True
            return
        self.act(self.ex.tick(now, self.pos, self.vel, self.contact_both()))

    def act(self, actions):
        for a in actions:
            kind = a[0]
            if kind == "command":
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = [n for n in self.names if n in a[1]] or list(a[1])
                msg.position = [float(a[1][n]) for n in msg.name]
                self.pub_cmd.publish(msg)
            elif kind == "velocity":
                self.pub_vel.publish(Int32(data=a[1]))
            elif kind == "cap":
                self.pub_cap.publish(Int16(data=a[1]))
            elif kind == "rezero":
                if self.rezero.service_is_ready():
                    self.rezero.call_async(Trigger.Request())
                else:
                    self.get_logger().warn("rezero_all service not available; pads not re-zeroed")
            elif kind == "event":
                self.pub_step.publish(String(data=json.dumps(a[1])))
                ev = a[1]
                if ev["status"] != "start":
                    self.get_logger().info(
                        f"[{ev['phase']} r{ev['repeat']} #{ev['index']}] {ev['step']}: {ev['status']}")
            elif kind == "done":
                self.done = a[1]
                self.write_summary()
                self.get_logger().info(f"Sequence finished: {self.done}. Summary: {self.summary_path}")

    def write_summary(self):
        rows = self.ex.results
        keys = ["phase", "repeat", "index", "step", "result", "duration_s"]
        extra = sorted({k for r in rows for k in r} - set(keys))
        with open(self.summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys + extra)
            w.writeheader()
            w.writerows(rows)


def main(args=None):  # pragma: no cover
    rclpy.init(args=args)
    try:
        node = SequenceRunner()
    except (RuntimeError, sq.SequenceError, OSError, ValueError) as e:
        print(f"sequence_runner: {e}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.pub_stop.publish(Empty())     # the arm stops where it is instead of finishing the move
        node.act(node.ex.abort(time.monotonic(), "Ctrl-C"))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if node.done == "ok" else 2)


if __name__ == "__main__":  # pragma: no cover
    main()