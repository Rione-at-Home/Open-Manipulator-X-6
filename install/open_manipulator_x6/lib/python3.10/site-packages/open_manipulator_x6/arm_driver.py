import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Int16, Int32, String
from std_srvs.srv import SetBool, Trigger
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from .dynamixel_driver import (
    CURRENT_BASED_POSITION_MODE,
    POSITION_CONTROL_MODE,
    DynamixelHardwareDriver,
)
from .utils import (
    rad_s_to_raw_vel,
    rad_to_ticks,
    raw_effort,
    raw_temperature_to_celsius,
    raw_vel_to_rad_s,
    raw_voltage_to_volts,
    ticks_to_rad,
)


class ArmDriver(Node):

    def __init__(self):
        super().__init__("arm_driver")

        # Parameters
        self.declare_parameter("port", "/dev/ttyUSB2") # Check the actual port name on your system
        self.declare_parameter("baudrate", 1000000)  # 1 Mbps . Check the actual baudrate for your Dynamixel motors

        # NOTE: gripper ID (6) is a placeholder — swap in the real ID once you
        # finish renumbering (you mentioned landing on 1 and 7 eventually).
        self.declare_parameter("joint_ids", [1, 2, 3, 4, 5, 6, 7])

        self.declare_parameter(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "wrist_rotate", "gripper"],

        )
        self.declare_parameter("home_positions", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Diagnostic thresholds (deg C) — tune to your servo's Temperature Limit register
        self.declare_parameter("temperature_warn_c", 70.0)
        self.declare_parameter("temperature_error_c", 80.0)

        # --- Safety / contact-test settings ---
        # Profile Velocity unit = 0.229 rpm (40 -> ~9 rpm, ~0.96 rad/s). 0 would mean "unlimited".
        self.declare_parameter("profile_velocity", 40)
        self.declare_parameter("profile_acceleration", 20)
        self.declare_parameter("max_profile_velocity", 200)  # ceiling for /arm_profile_velocity
        # Gripper runs in current-based position mode (5): Goal Current is a torque cap.
        # 1 unit = 2.69 mA, so 100 -> ~0.27 A. Start low and raise in steps via /arm_gripper_current.
        self.declare_parameter("gripper_name", "gripper")
        self.declare_parameter("gripper_current_mode", True)
        self.declare_parameter("gripper_goal_current", 100)
        self.declare_parameter("gripper_current_max", 300)  # hard ceiling for /arm_gripper_current
        # True: torque stays on when this node exits (the arm holds its pose instead of dropping).
        self.declare_parameter("leave_torque_on_shutdown", True)

        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.joint_ids = list(self.get_parameter("joint_ids").get_parameter_value().integer_array_value)
        self.joint_names = list(self.get_parameter("joint_names").get_parameter_value().string_array_value)
        self.home_positions = list(self.get_parameter("home_positions").get_parameter_value().double_array_value)
        self.temp_warn_c = self.get_parameter("temperature_warn_c").get_parameter_value().double_value
        self.temp_error_c = self.get_parameter("temperature_error_c").get_parameter_value().double_value

        gp = self.get_parameter
        self.profile_velocity = gp("profile_velocity").get_parameter_value().integer_value
        self.profile_acceleration = gp("profile_acceleration").get_parameter_value().integer_value
        self.max_profile_velocity = gp("max_profile_velocity").get_parameter_value().integer_value
        self.gripper_name = gp("gripper_name").get_parameter_value().string_value
        self.gripper_current_mode = gp("gripper_current_mode").get_parameter_value().bool_value
        self.gripper_goal_current = gp("gripper_goal_current").get_parameter_value().integer_value
        self.gripper_current_max = gp("gripper_current_max").get_parameter_value().integer_value
        self.leave_torque_on = gp("leave_torque_on_shutdown").get_parameter_value().bool_value

        self.joint_map = dict(zip(self.joint_names, self.joint_ids))
        self.gripper_id = self.joint_map.get(self.gripper_name)
        self.has_current = {}   # id -> True (XM430, effort in A) / False (XL430, effort = load fraction)
        self.gripper_in_current_mode = False
        self.id_to_index = {m_id: idx for idx, m_id in enumerate(self.joint_ids)}

        # Internal State
        self.is_enabled = False
        self.status = "INITIALIZING"
        self.current_positions = [0.0] * len(self.joint_ids)

        # Low-Level Hardware Interface 
        self.driver = DynamixelHardwareDriver(port=self.port, baudrate=self.baudrate)

        if not self.connect_and_initialize():
            self.get_logger().error("Hardware initialization failed!")
            self.status = "ERROR"

        # Publishers & Subscribers 
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.status_pub = self.create_publisher(String, "/arm_status", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/arm_diagnostics", 10)

        self.create_subscription(JointState, "/arm_command", self.arm_command_cb, 10)
        self.create_subscription(Bool, "/arm_enable", self.arm_enable_cb, 10)
        self.create_subscription(Empty, "/arm_home", self.arm_home_cb, 10)
        self.create_subscription(Empty, "/arm_stop", self.arm_stop_cb, 10)
        self.create_subscription(Int16, "/arm_gripper_current", self.gripper_current_cb, 10)
        self.create_subscription(Int32, "/arm_profile_velocity", self.profile_velocity_cb, 10)

        # Services
        self.create_service(SetBool, "~/enable_torque", self.enable_torque_srv)
        self.create_service(Trigger, "~/reboot_motors", self.reboot_motors_srv)

        # --- 50 Hz Update Loop ---
        self.timer = self.create_timer(0.02, self.update_loop)
        self.get_logger().info("OpenManipulator-X Arm Driver Ready.")

    def connect_and_initialize(self) -> bool:
        """
        Method to connect to the Dynamixel hardware and initialize the motors.
        Returns True if successful, False otherwise.
        """

        if not self.driver.connect():
            return False

        # Verify communication with all motors

        for m_id in self.joint_ids:

            if not self.driver.ping(m_id):
                
                self.get_logger().error(f"Failed to ping motor ID: {m_id}")
                return False

        # Torque off first so modes/limits can be written, then identify each servo family.
        self.driver.enable_torque(self.joint_ids, False)
        for m_id in self.joint_ids:
            sensor = self.driver.has_current_sensor(m_id)
            if sensor is None:
                self.get_logger().error(f"Could not read model number of ID {m_id}.")
                return False
            self.has_current[m_id] = sensor

        # Everything in plain position control except (optionally) the gripper.
        gripper_current = (
            self.gripper_current_mode
            and self.gripper_id is not None
            and self.has_current.get(self.gripper_id, False)
        )
        if self.gripper_current_mode and self.gripper_id is not None and not gripper_current:
            self.get_logger().warn(
                "gripper_current_mode requested but the gripper servo has no current sensor; "
                "falling back to plain position control (NO force limit)."
            )
        position_ids = [j for j in self.joint_ids if not (gripper_current and j == self.gripper_id)]
        if position_ids and not self.driver.set_operating_mode(position_ids, POSITION_CONTROL_MODE):
            return False
        if gripper_current:
            if not self.driver.set_operating_mode([self.gripper_id], CURRENT_BASED_POSITION_MODE):
                return False
            if not self.driver.write_goal_current(self.gripper_id, self.gripper_goal_current):
                self.get_logger().error("Failed to write gripper Goal Current; aborting init.")
                return False
        self.gripper_in_current_mode = gripper_current

        # Slow, smooth motion before any torque is applied.
        if not self.driver.write_profile(
            self.joint_ids, self.profile_velocity, self.profile_acceleration
        ):
            self.get_logger().error("Failed to write profile velocity/acceleration; aborting init.")
            return False

        self.enable_torque(True)
        if not self.is_enabled:
            return False
        self.get_logger().info(
            f"Init done: profile vel={self.profile_velocity} acc={self.profile_acceleration}, "
            f"gripper {'current-based (goal ' + str(self.gripper_goal_current) + ')' if gripper_current else 'position-only'}."
        )
        self.status = "READY"
        return True

    def enable_torque(self, enable: bool):
        # Enabling goes through the safe path: Goal Position := Present Position, then torque on.
        ok = (
            self.driver.enable_torque_holding_position(self.joint_ids)
            if enable
            else self.driver.enable_torque(self.joint_ids, False)
        )
        if ok:
            self.is_enabled = enable
            self.status = "READY" if enable else "DISABLED"
            self.get_logger().info(f"Torque {'enabled' if enable else 'disabled'}.")
        else:
            self.status = "ERROR"
            self.get_logger().error("Failed to toggle torque.")

    # --- Callbacks ---
    def arm_command_cb(self, msg: JointState):
        if not self.is_enabled:
            return

        target_ticks = list(self.current_positions)  # Fallback to current position if unassigned
        
        # Name-based mapping fallback to ordered index array
        if msg.name:
            target_dict = dict(zip(msg.name, msg.position))
            for idx, name in enumerate(self.joint_names):
                if name in target_dict:
                    target_ticks[idx] = rad_to_ticks(target_dict[name])
                else:
                    target_ticks[idx] = rad_to_ticks(self.current_positions[idx])
        elif len(msg.position) == len(self.joint_ids):
            target_ticks = [rad_to_ticks(p) for p in msg.position]
        else:
            return

        self.driver.write_positions(self.joint_ids, target_ticks)
        self.status = "MOVING"

    def arm_enable_cb(self, msg: Bool):
        self.enable_torque(msg.data)

    def arm_home_cb(self, _msg: Empty):
        if not self.is_enabled:
            return
        home_ticks = [rad_to_ticks(p) for p in self.home_positions]
        self.driver.write_positions(self.joint_ids, home_ticks)
        self.status = "MOVING"

    def arm_stop_cb(self, _msg: Empty):
        # Stop motion by commanding current position
        if self.is_enabled:
            stop_ticks = [rad_to_ticks(p) for p in self.current_positions]
            self.driver.write_positions(self.joint_ids, stop_ticks)
            self.status = "READY"

    def gripper_current_cb(self, msg: Int16):
        """Sets the gripper torque cap (raw units, 2.69 mA each), clamped to [0, gripper_current_max]."""
        if not self.gripper_in_current_mode:
            self.get_logger().warn("Gripper is not in current-based mode; ignoring /arm_gripper_current.")
            return
        value = max(0, min(int(msg.data), self.gripper_current_max))
        if value != msg.data:
            self.get_logger().warn(f"Gripper current {msg.data} clamped to {value}.")
        if self.driver.write_goal_current(self.gripper_id, value):
            self.gripper_goal_current = value
            self.get_logger().info(f"Gripper goal current = {value} (~{value * 2.69:.0f} mA).")
        else:
            self.get_logger().error("Failed to write gripper goal current.")

    def profile_velocity_cb(self, msg: Int32):
        """Sets Profile Velocity on all joints live, clamped to [1, max_profile_velocity]."""
        value = max(1, min(int(msg.data), self.max_profile_velocity))
        if self.driver.write_profile(self.joint_ids, value, self.profile_acceleration):
            self.profile_velocity = value
            self.get_logger().info(f"Profile velocity = {value} (~{value * 0.229:.1f} rpm).")

    def enable_torque_srv(self, request, response):
        self.enable_torque(request.data)
        response.success = self.is_enabled == request.data
        response.message = f"Torque {'enabled' if self.is_enabled else 'disabled'}"
        return response

    def reboot_motors_srv(self, _request, response):
        self.enable_torque(False)
        all_ok = True
        for m_id in self.joint_ids:
            if not self.driver.reboot(m_id):
                all_ok = False
        
        self.connect_and_initialize()
        response.success = all_ok
        response.message = "Motors rebooted successfully" if all_ok else "One or more motors failed reboot"
        return response

    # Diagonistcs to check the following from the motor:
    # - current
    # - voltage
    # - temperature
    def build_diagnostics(self, states: dict) -> DiagnosticArray:
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        for idx, m_id in enumerate(self.joint_ids):
            status = DiagnosticStatus()
            status.name = f"arm_driver: {self.joint_names[idx]} (ID {m_id})"
            status.hardware_id = str(m_id)

            state = states.get(m_id)
            if state is None:
                status.level = DiagnosticStatus.STALE
                status.message = "No data received from motor"
                diag_array.status.append(status)
                continue

            has_cur = self.has_current.get(m_id, True)
            current = raw_effort(state["current"], has_cur)
            voltage = raw_voltage_to_volts(state["voltage"]) if "voltage" in state else None
            temperature = (
                raw_temperature_to_celsius(state["temperature"]) if "temperature" in state else None
            )

            level = DiagnosticStatus.OK
            messages = []

            if temperature is not None:
                if temperature >= self.temp_error_c:
                    level = DiagnosticStatus.ERROR
                    messages.append(f"Overtemperature ({temperature:.1f} C)")
                elif temperature >= self.temp_warn_c:
                    level = max(level, DiagnosticStatus.WARN)
                    messages.append(f"High temperature ({temperature:.1f} C)")

            status.level = level
            status.message = "; ".join(messages) if messages else "OK"

            status.values.append(KeyValue(
                key="Current (A)" if has_cur else "Load (fraction of max output)",
                value=f"{current:.3f}"))
            status.values.append(KeyValue(key="Effort unit", value="A" if has_cur else "load_fraction"))
            if voltage is not None:
                status.values.append(KeyValue(key="Voltage (V)", value=f"{voltage:.2f}"))
            if temperature is not None:
                status.values.append(KeyValue(key="Temperature (C)", value=f"{temperature:.1f}"))

            diag_array.status.append(status)

        return diag_array

    def update_loop(self):
        states = self.driver.read_states(self.joint_ids)
        if not states:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        positions, velocities, efforts = [], [], []

        for idx, m_id in enumerate(self.joint_ids):
            if m_id in states:
                pos_rad = ticks_to_rad(states[m_id]["position"])
                vel_rad_s = raw_vel_to_rad_s(states[m_id]["velocity"])
                effort = raw_effort(states[m_id]["current"], self.has_current.get(m_id, True))

                positions.append(pos_rad)
                velocities.append(vel_rad_s)
                efforts.append(effort)

                self.current_positions[idx] = pos_rad
            else:
                positions.append(self.current_positions[idx])
                velocities.append(0.0)
                efforts.append(0.0)

        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts

        self.joint_pub.publish(msg)

        status_msg = String()
        status_msg.data = self.status
        self.status_pub.publish(status_msg)

        self.diag_pub.publish(self.build_diagnostics(states))

    def destroy_node(self):
        if not self.leave_torque_on:
            self.enable_torque(False)
        else:
            self.get_logger().warn("Exiting with torque ON (leave_torque_on_shutdown): the arm holds its pose.")
        self.driver.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmDriver()
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