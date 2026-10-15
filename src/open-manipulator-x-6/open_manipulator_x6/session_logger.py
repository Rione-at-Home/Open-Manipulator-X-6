"""
Session logger: pad features + arm joint states on one clock, written to one CSV.

One row is written for every /joint_states message (~50 Hz). Each row carries the latest frame
from each finger (and its age in ms) plus every joint's position / velocity / effort, so tactile
events can be lined up against arm motion without any post-hoc syncing. All times are the ROS
system clock (the pad node stamps receive time with the system clock too).

For raw pad images use `ros2 bag record` as well; this CSV is the quick-look companion.

    ros2 run open_manipulator_x6 session_logger --ros-args -p out_dir:=~/flexigrasp_logs -p label:=tilt_sweep_1
"""
import csv
import os
import time
from typing import Dict, List, Optional

PAD_FIELDS = ["seq", "age_ms", "in_contact", "peak", "total", "contact_taxels",
              "centroid_row", "centroid_col", "bad_frames"]


class SessionWriter:
    """ROS-free CSV writer; the node feeds it plain numbers."""

    def __init__(self, path: str, fingers: List[str], joint_names: Optional[List[str]] = None):
        self.path, self.fingers = path, list(fingers)
        self.joint_names: Optional[List[str]] = list(joint_names) if joint_names else None
        self._f = None
        self._w = None
        self.rows = 0

    def _open(self, joint_names: List[str]):
        self.joint_names = list(joint_names)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        head = ["t"]
        for f in self.fingers:
            head += [f"{f}_{k}" for k in PAD_FIELDS]
        for j in self.joint_names:
            head += [f"{j}_pos", f"{j}_vel", f"{j}_effort"]
        self._w.writerow(head)

    def write(self, t: float, pads: Dict[str, Optional[dict]], names, pos, vel, eff):
        if self._w is None:
            self._open(names)
        idx = {n: i for i, n in enumerate(names)}
        row = [f"{t:.4f}"]
        for f in self.fingers:
            p = pads.get(f)
            if p is None:
                row += [""] * len(PAD_FIELDS)
            else:
                row += [p.get(k, "") if not isinstance(p.get(k), float) else f"{p[k]:.4f}"
                        for k in PAD_FIELDS]
        for j in self.joint_names:
            i = idx.get(j)
            def pick(a):
                return f"{a[i]:.5f}" if i is not None and i < len(a) else ""
            row += [pick(pos), pick(vel), pick(eff)]
        self._w.writerow(row)
        self.rows += 1
        if self.rows % 50 == 0:
            self._f.flush()

    def close(self):
        if self._f:
            self._f.flush()
            self._f.close()
            self._f = None


def make_path(out_dir: str, label: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{label}.csv" if label else f"{stamp}.csv"
    return os.path.join(os.path.expanduser(out_dir), name)


def main(args=None):  # pragma: no cover - needs ROS 2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from flexigrasp_msgs.msg import TactileFrame

    class SessionLogger(Node):
        def __init__(self):
            super().__init__("session_logger")
            self.declare_parameter("out_dir", "~/flexigrasp_logs")
            self.declare_parameter("label", "")
            self.declare_parameter("fingers", ["left", "right"])
            self.declare_parameter("topic_prefix", "flexigrasp")
            fingers = list(self.get_parameter("fingers").value)
            prefix = self.get_parameter("topic_prefix").value
            path = make_path(self.get_parameter("out_dir").value, self.get_parameter("label").value)
            self.writer = SessionWriter(path, fingers)
            self.latest: Dict[str, Optional[dict]] = {f: None for f in fingers}
            for f in fingers:
                self.create_subscription(TactileFrame, f"/{prefix}/{f}/frame",
                                         lambda m, f=f: self.on_pad(f, m), qos_profile_sensor_data)
            self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
            self.get_logger().info(f"Logging to {path}")

        def on_pad(self, f, m):
            self.latest[f] = dict(seq=m.seq, in_contact=int(m.in_contact), peak=float(m.peak),
                                  total=float(m.total), contact_taxels=m.contact_taxels,
                                  centroid_row=float(m.centroid_row), centroid_col=float(m.centroid_col),
                                  bad_frames=m.bad_frames,
                                  _t=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

        def on_joints(self, m):
            now = self.get_clock().now().nanoseconds * 1e-9
            pads = {}
            for f, p in self.latest.items():
                pads[f] = None if p is None else {**p, "age_ms": (now - p["_t"]) * 1000.0}
            self.writer.write(now, pads, list(m.name), list(m.position), list(m.velocity), list(m.effort))

        def destroy_node(self):
            self.writer.close()
            super().destroy_node()

    rclpy.init(args=args)
    node = SessionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()