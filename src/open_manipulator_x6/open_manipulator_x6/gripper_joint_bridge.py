#!/usr/bin/env python3
"""
Republishes arm_driver's /joint_states for the URDF: the driver's single 'gripper' angle (rad)
becomes 'left_finger_joint' (m) (right_finger_joint follows by mimic in the URDF). Everything else
is passed through unchanged.

    /joint_states  ->  gripper_joint_bridge  ->  /joint_states_urdf  ->  robot_state_publisher

Started by: ros2 launch open_manipulator_x6_description display.launch.py mode:=live

Calibration (config/gripper_bridge.yaml in the description package): the gripper angle and the
fingertip gap between the pad faces, fully closed and fully open. The gap is treated as linear
in the angle; cad_gap_m is the gap in the URDF's zero pose. Each finger moves half the change.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

PLACEHOLDER = {"gripper_rad_closed": 0.0, "gripper_rad_open": 1.0, "gap_closed_m": 0.0,
               "gap_open_m": 0.08}


def finger_from_gripper(g, g_closed, g_open, gap_closed, gap_open, cad_gap, lower, upper):
    """(left_finger_joint position in m, metres per radian). Clamped to the URDF limits."""
    if g_open == g_closed:
        raise ValueError("gripper_rad_open and gripper_rad_closed must differ")
    slope_gap = (gap_open - gap_closed) / (g_open - g_closed)      # m of gap per rad
    gap = gap_closed + (g - g_closed) * slope_gap
    q = (gap - cad_gap) / 2.0
    return min(upper, max(lower, q)), slope_gap / 2.0


def convert(names, positions, velocities, gripper_name, params):
    """New (names, positions, velocities) with the gripper replaced by left_finger_joint."""
    names, positions = list(names), list(positions)
    velocities = list(velocities) if len(velocities) == len(names) else []
    if gripper_name not in names:
        return names, positions, velocities
    i = names.index(gripper_name)
    q, rate = finger_from_gripper(positions[i], **params)
    names[i], positions[i] = "left_finger_joint", q
    if velocities:
        velocities[i] = velocities[i] * rate
    return names, positions, velocities


class GripperJointBridge(Node):
    def __init__(self):
        super().__init__("gripper_joint_bridge")
        P = self.declare_parameter
        P("gripper_name", "gripper")
        P("input_topic", "/joint_states")
        P("output_topic", "/joint_states_urdf")
        for k, v in PLACEHOLDER.items():
            P(k, v)
        P("cad_gap_m", 0.0617)
        P("finger_lower", -0.024)
        P("finger_upper", 0.02)
        g = lambda n: self.get_parameter(n).value
        self.gripper_name = g("gripper_name")
        self.params = dict(g_closed=float(g("gripper_rad_closed")), g_open=float(g("gripper_rad_open")),
                           gap_closed=float(g("gap_closed_m")), gap_open=float(g("gap_open_m")),
                           cad_gap=float(g("cad_gap_m")), lower=float(g("finger_lower")),
                           upper=float(g("finger_upper")))
        finger_from_gripper(0.0, **self.params)        # fail early on a bad calibration
        if all(float(g(k)) == v for k, v in PLACEHOLDER.items()):
            self.get_logger().warn("Gripper calibration is still the placeholder: the fingers in RViz "
                                   "will not match the real gap. See config/gripper_bridge.yaml.")
        self.pub = self.create_publisher(JointState, g("output_topic"), 10)
        self.create_subscription(JointState, g("input_topic"), self.on_states, 10)
        self.get_logger().info(f"{g('input_topic')} -> {g('output_topic')} ('{self.gripper_name}' -> "
                               f"left_finger_joint)")

    def on_states(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        out.name, out.position, out.velocity = convert(msg.name, msg.position, msg.velocity,
                                                       self.gripper_name, self.params)
        if len(msg.effort) == len(msg.name):
            out.effort = list(msg.effort)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GripperJointBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()