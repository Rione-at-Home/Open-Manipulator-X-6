import logging

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# Control Table Addresses (XM430 Series / Protocol 2.0)
ADDR_MODEL_NUMBER = 0
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

LEN_MODEL_NUMBER = 2
LEN_GOAL_POSITION = 4
LEN_GOAL_CURRENT = 2
LEN_GAIN = 2
# Current(2B) + Velocity(4B) + Position(4B) + Vel Trajectory(4B) + Pos Trajectory(4B)
# + Input Voltage(2B) + Temperature(1B) = 21B, read in one contiguous block (126-146)
LEN_PRESENT_STATE = 21

POSITION_CONTROL_MODE = 3
CURRENT_CONTROL_MODE = 0
PROTOCOL_VERSION = 2.0

# NOT pre-filled: I could not independently verify the raw Model Number
# register values for XM430-W210 vs W350 against an authoritative source.
# Read them once with read_model_number() on a known-W210 and a known-W350
# joint (cross-check against Dynamixel Wizard's Model Information panel),
# then fill this dict in yourself so later joins are labeled automatically.
KNOWN_MODEL_NUMBERS = {
    # 1234: "XM430-W350",  # fill in after confirming via Wizard
    # 5678: "XM430-W210",
}


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

    def read_model_number(self, motor_id: int) -> int | None:
        """
        Reads the raw Model Number register. Use this to confirm which
        servos are XM430-W210 vs XM430-W350 (or any other variant) rather
        than assuming from wiring/origin alone - see KNOWN_MODEL_NUMBERS
        for a best-effort (unverified) label lookup.
        """
        value, comm_result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler, motor_id, ADDR_MODEL_NUMBER
        )
        if comm_result != COMM_SUCCESS or error != 0:
            return None
        return value

    def read_position_gains(self, motor_id: int) -> dict | None:
        """
        Reads Position P/I/D Gain for one servo. Returns None on comm failure.
        """
        gains = {}
        for name, addr in (("p", ADDR_POSITION_P_GAIN),
                            ("i", ADDR_POSITION_I_GAIN),
                            ("d", ADDR_POSITION_D_GAIN)):
            value, comm_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, addr
            )
            if comm_result != COMM_SUCCESS or error != 0:
                return None
            gains[name] = value
        return gains

    def write_position_gain(self, motor_id: int, addr: int, value: int) -> bool:
        """
        Writes a single Position PID gain register. These are RAM-area
        registers on the X-series (unlike Operating Mode), so torque does
        NOT need to be disabled first.
        """
        comm_result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, addr, value
        )
        return comm_result == COMM_SUCCESS and error == 0

    def set_position_gains(
        self, motor_id: int, p: int | None = None, i: int | None = None, d: int | None = None
    ) -> bool:
        """
        Writes any subset of Position P/I/D gains for one servo.
        Pass only the terms you want to change; others are left untouched.
        """
        success = True
        if p is not None:
            success &= self.write_position_gain(motor_id, ADDR_POSITION_P_GAIN, p)
        if i is not None:
            success &= self.write_position_gain(motor_id, ADDR_POSITION_I_GAIN, i)
        if d is not None:
            success &= self.write_position_gain(motor_id, ADDR_POSITION_D_GAIN, d)
        return success

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

    def read_operating_mode(self, motor_id: int) -> int | None:
        """
        Reads back the Operating Mode register. Use this to CONFIRM a mode
        switch actually took effect - set_operating_mode() below does not
        check its own write results, so a silently rejected write (e.g.
        because torque wasn't fully disabled first) would otherwise go
        unnoticed and the servo would stay in its previous mode.
        """
        value, comm_result, error = self.packet_handler.read1ByteTxRx(
            self.port_handler, motor_id, ADDR_OPERATING_MODE
        )
        if comm_result != COMM_SUCCESS or error != 0:
            return None
        return value

    def set_operating_mode(self, joint_ids: list, mode: int = POSITION_CONTROL_MODE):
        """
        Sets operating mode (Torque must be disabled first).
        """

        self.enable_torque(joint_ids, False)

        for m_id in joint_ids:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, m_id, ADDR_OPERATING_MODE, mode
            )

    def write_goal_current(self, motor_id: int, raw_current: int) -> bool:
        """
        Writes a signed Goal Current (raw units, ~2.69 mA/unit) to one servo.
        Requires Current Control Mode (0). Caller is responsible for clamping
        raw_current to a safe magnitude BEFORE calling this - this method
        does not enforce any limit itself.
        """
        value = int(raw_current) & 0xFFFF  # two's-complement wrap for negatives
        comm_result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_CURRENT, value
        )
        return comm_result == COMM_SUCCESS and error == 0

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
        Reads current, velocity, position, voltage, and temperature for all
        joint IDs in a single packet.
        """
        self.sync_read_state.clearParam()
        for m_id in joint_ids:
            self.sync_read_state.addParam(m_id)

        comm_result = self.sync_read_state.txRxPacket()
        states = {}

        if comm_result != COMM_SUCCESS:
            return states

        for m_id in joint_ids:

            if self.sync_read_state.isAvailable(m_id, ADDR_PRESENT_CURRENT, 2):
                curr = self.sync_read_state.getData(m_id, ADDR_PRESENT_CURRENT, 2)
                vel = self.sync_read_state.getData(m_id, ADDR_PRESENT_VELOCITY, 4)
                pos = self.sync_read_state.getData(m_id, ADDR_PRESENT_POSITION, 4)
                states[m_id] = {"current": curr, "velocity": vel, "position": pos}

                if self.sync_read_state.isAvailable(m_id, ADDR_PRESENT_VOLTAGE, 2):
                    states[m_id]["voltage"] = self.sync_read_state.getData(
                        m_id, ADDR_PRESENT_VOLTAGE, 2
                    )

                if self.sync_read_state.isAvailable(m_id, ADDR_PRESENT_TEMPERATURE, 1):
                    states[m_id]["temperature"] = self.sync_read_state.getData(
                        m_id, ADDR_PRESENT_TEMPERATURE, 1
                    )

        return states

    def reboot(self, motor_id: int) -> bool:
        comm_result, error = self.packet_handler.reboot(self.port_handler, motor_id)
        return comm_result == COMM_SUCCESS and error == 0