import logging

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# Control Table Addresses (XM430 Series / Protocol 2.0)
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_STATE = 10  # Current (2B) + Velocity (4B) + Position (4B)

POSITION_CONTROL_MODE = 3
PROTOCOL_VERSION = 2.0


class DynamixelHardwareDriver:

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 57600):
        self.port_name = port
        self.baudrate = baudrate

        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        self.sync_write_pos = GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self.sync_read_state = GroupSyncRead(
            self.port_handler, self.packet_handler, ADDR_PRESENT_CURRENT, LEN_PRESENT_STATE
        )

        self.is_connected = False
        self.logger = logging.getLogger("DynamixelHardwareDriver")

    def connect(self) -> bool:
        """
        Method to connect to the Dynamixel hardware. 
        Returns True if successful, False otherwise.
        """

        if not self.port_handler.openPort():
            self.logger.error(f"Failed to open port: {self.port_name}")
            return False

        if not self.port_handler.setBaudRate(self.baudrate):
            self.logger.error(f"Failed to set baudrate: {self.baudrate}")
            return False

        self.is_connected = True
        return True

    def disconnect(self):
        """
        Method to disconnect from the Dynamixel hardware.
        """

        if self.is_connected:
            self.port_handler.closePort()
            self.is_connected = False

    def ping(self, motor_id: int) -> bool:
        """
        Method to ping a Dynamixel motor.
        Returns True if successful, False otherwise.
        """

        model_num, comm_result, error = self.packet_handler.ping(self.port_handler, motor_id)
        return comm_result == COMM_SUCCESS and error == 0

    def enable_torque(self, joint_ids: list, enable: bool) -> bool:
        """
        Method to enable or disable torque for a list of Dynamixel joints.
        Returns True if successful, False otherwise.
        """

        value = 1 if enable else 0
        success = True
        for m_id in joint_ids:
            comm_result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, m_id, ADDR_TORQUE_ENABLE, value
            )

            if comm_result != COMM_SUCCESS or error != 0:
                success = False

        return success

    def set_operating_mode(self, joint_ids: list, mode: int = POSITION_CONTROL_MODE):
        """
        Sets operating mode (Torque must be disabled first).
        """

        self.enable_torque(joint_ids, False)

        for m_id in joint_ids:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, m_id, ADDR_OPERATING_MODE, mode
            )

    def write_positions(self, joint_ids: list, target_ticks: list) -> bool:
        """
        Writes target positions to all joint IDs in a single packet.
        Returns True if successful, False otherwise.
        """

        self.sync_write_pos.clearParam()
        for m_id, ticks in zip(joint_ids, target_ticks):
            param = [
                (ticks & 0xFF),
                (ticks >> 8) & 0xFF,
                (ticks >> 16) & 0xFF,
                (ticks >> 24) & 0xFF,
            ]
            if not self.sync_write_pos.addParam(m_id, bytes(param)):
                return False

        comm_result = self.sync_write_pos.txPacket()
        return comm_result == COMM_SUCCESS

    def read_states(self, joint_ids: list) -> dict:
        """
        Reads current, velocity, and position for all joint IDs in a single packet.
        """
        self.sync_read_state.clearParam()
        for m_id in joint_ids:
            self.sync_read_state.addParam(m_id)

        comm_result = self.sync_read_state.rxPacket()
        states = {}

        if comm_result != COMM_SUCCESS:
            return states

        for m_id in joint_ids:
            
            if self.sync_read_state.isAvailable(m_id, ADDR_PRESENT_CURRENT, 2):
                curr = self.sync_read_state.getData(m_id, ADDR_PRESENT_CURRENT, 2)
                vel = self.sync_read_state.getData(m_id, ADDR_PRESENT_VELOCITY, 4)
                pos = self.sync_read_state.getData(m_id, ADDR_PRESENT_POSITION, 4)
                states[m_id] = {"current": curr, "velocity": vel, "position": pos}

        return states

    def reboot(self, motor_id: int) -> bool:
        comm_result, error = self.packet_handler.reboot(self.port_handler, motor_id)
        return comm_result == COMM_SUCCESS and error == 0