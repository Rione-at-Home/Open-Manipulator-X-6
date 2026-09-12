import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import SetBool, Trigger
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from .dynamixel_driver import DynamixelHardwareDriver
from .utils import (
    rad_s_to_raw_vel,
    rad_to_ticks,
    raw_current_to_effort,
    raw_temperature_to_celsius,
    raw_vel_to_rad_s,
    raw_voltage_to_volts,
    ticks_to_rad,
)


class ArmDriver(Node):

    def __init__(self):
        super().__init__("arm_driver")

        # Parameters
        self.declare_parameter("port", "/dev/ttyACM0") # Check the actual port name on your system
        self.declare_parameter("baudrate", 1000000)  # 1 Mbps . Check the actual baudrate for your Dynamixel motors

        # NOTE: gripper ID (6) is a placeholder — swap in the real ID once you
        # finish renumbering (you mentioned landing on 1 and 6 eventually).
        self.declare_parameter("joint_ids", [11, 12, 13, 14, 15, 2, 6])

        self.declare_parameter(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "wrist_rotate", "gripper"],

        )
        self.declare_parameter("home_positions", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Diagnostic thresholds (deg C) — tune to your servo's Temperature Limit register
        self.declare_parameter("temperature_warn_c", 70.0)
        self.declare_parameter("temperature_error_c", 80.0)

        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.joint_ids = list(self.get_parameter("joint_ids").get_parameter_value().integer_array_value)
        self.joint_names = list(self.get_parameter("joint_names").get_parameter_value().string_array_value)
        self.home_positions = list(self.get_parameter("home_positions").get_parameter_value().double_array_value)
        self.temp_warn_c = self.get_parameter("temperature_warn_c").get_parameter_value().double_value
        self.temp_error_c = self.get_parameter("temperature_error_c").get_parameter_value().double_value

        self.joint_map = dict(zip(self.joint_names, self.joint_ids))
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

        self.driver.set_operating_mode(self.joint_ids)
        self.enable_torque(True)
        self.status = "READY"
        return True

    def enable_torque(self, enable: bool):
        if self.driver.enable_torque(self.joint_ids, enable):
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

    def enable_torque_srv(self, request, response):
        self.enable_torque(request.data)
        response.success = True
        response.message = f"Torque set to {request.data}"
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

            current = raw_current_to_effort(state["current"])
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

            status.values.append(KeyValue(key="Current (A)", value=f"{current:.3f}"))
            if voltage is not None:
                status.values.append(KeyValue(key="Voltage (V)", value=f"{voltage:.2f}"))
            if temperature is not None:
                status.values.append(KeyValue(key="Temperature (C)", value=f"{temperature:.1f}"))

            diag_array.status.append(status)

        return diag_array

    # --- Main Loop (50 Hz) ---
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
                effort = raw_current_to_effort(states[m_id]["current"])

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
        self.enable_torque(False)
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