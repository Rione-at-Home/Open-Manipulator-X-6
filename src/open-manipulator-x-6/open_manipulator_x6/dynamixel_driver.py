import logging
import time

from dynamixel_sdk import (
    COMM_SUCCESS,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

# Control Table Addresses (XM430 / XL430 Series / Protocol 2.0)
ADDR_MODEL_NUMBER = 0
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_PWM = 100
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
LEN_GOAL_PWM = 2
LEN_GAIN = 2
LEN_PRESENT_STATE = 21

POSITION_CONTROL_MODE = 3
CURRENT_CONTROL_MODE = 0
PWM_CONTROL_MODE = 16
PROTOCOL_VERSION = 2.0

KNOWN_MODEL_NUMBERS = {
    1060: "XL430-W250",
    1020: "XM430-W210",
    1030: "XM430-W350",
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
        if not self.port_handler.openPort():
            self.logger.error(f"Failed to open port: {self.port_name}")
            return False

        if not self.port_handler.setBaudRate(self.baudrate):
            self.logger.error(f"Failed to set baudrate: {self.baudrate}")
            return False

        self.is_connected = True
        return True

    def disconnect(self):
        if self.is_connected:
            self.port_handler.closePort()
            self.is_connected = False

    def ping(self, motor_id: int) -> bool:
        model_num, comm_result, error = self.packet_handler.ping(self.port_handler, motor_id)
        return comm_result == COMM_SUCCESS and error == 0

    def read_model_number(self, motor_id: int) -> int | None:
        value, comm_result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler, motor_id, ADDR_MODEL_NUMBER
        )
        if comm_result != COMM_SUCCESS or error != 0:
            return None
        return value

    def read_position_gains(self, motor_id: int) -> dict | None:
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
        comm_result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, addr, value
        )
        return comm_result == COMM_SUCCESS and error == 0

    def set_position_gains(
        self, motor_id: int, p: int | None = None, i: int | None = None, d: int | None = None
    ) -> bool:
        success = True
        if p is not None:
            success &= self.write_position_gain(motor_id, ADDR_POSITION_P_GAIN, p)
        if i is not None:
            success &= self.write_position_gain(motor_id, ADDR_POSITION_I_GAIN, i)
        if d is not None:
            success &= self.write_position_gain(motor_id, ADDR_POSITION_D_GAIN, d)
        return success

    def enable_torque(self, joint_ids: list, enable: bool) -> bool:
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
        value, comm_result, error = self.packet_handler.read1ByteTxRx(
            self.port_handler, motor_id, ADDR_OPERATING_MODE
        )
        if comm_result != COMM_SUCCESS or error != 0:
            return None
        return value

    def set_operating_mode(self, joint_ids: list, mode: int = POSITION_CONTROL_MODE, retries: int = 4) -> bool:
        """
        Sets operating mode with self-healing retries and diagnostic checks.
        """
        all_success = True

        for m_id in joint_ids:
            mode_set = False
            hw_err = 0
            torque_val = -1

            for attempt in range(retries):
                # 1. Force torque disable and wait for controller disengagement
                self.enable_torque([m_id], False)
                time.sleep(0.05)
                
                # Verify torque status
                torque_val, _, _ = self.packet_handler.read1ByteTxRx(self.port_handler, m_id, ADDR_TORQUE_ENABLE)
                if torque_val != 0:
                    self.logger.warning(
                        f"ID {m_id} refused to disable torque (readback {torque_val}). "
                        f"Retrying ({attempt + 1}/{retries})..."
                    )
                    continue

                # 2. Write operating mode to EEPROM
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, m_id, ADDR_OPERATING_MODE, mode
                )
                time.sleep(0.05)
                
                # 3. Verify write
                actual_mode = self.read_operating_mode(m_id)
                if actual_mode == mode:
                    mode_set = True
                    break

                # 4. Diagnostic read on failure
                hw_err, _, _ = self.packet_handler.read1ByteTxRx(self.port_handler, m_id, ADDR_HARDWARE_ERROR)
                
                self.logger.warning(
                    f"Mode write rejected for ID {m_id} (readback {actual_mode}, expected {mode}). "
                    f"Torque: {torque_val}, HW Error: {hw_err}. Retrying ({attempt + 1}/{retries})..."
                )

            if not mode_set:
                model_num = self.read_model_number(m_id)
                err_msg = f"Failed to set mode {mode} on ID {m_id} after {retries} attempts."
                
                if hw_err != 0:
                    err_msg += f" DIAGNOSIS: Joint is in a Hardware Error state (Code {hw_err}). Power-cycle required."
                elif model_num == 1060 and mode == CURRENT_CONTROL_MODE:
                    err_msg += f" DIAGNOSIS: ID {m_id} is an XL430-W250 (Model 1060). Lacks current sensor; use PWM Control Mode (16)."
                elif torque_val != 0:
                    err_msg += f" DIAGNOSIS: Torque stuck active. External process may be resetting torque."
                else:
                    err_msg += f" DIAGNOSIS: Rejected by firmware. Model Number: {model_num}."
                    
                self.logger.error(err_msg)
                all_success = False

        return all_success

    def write_goal_current(self, motor_id: int, raw_current: int) -> bool:
        value = int(raw_current) & 0xFFFF
        comm_result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_CURRENT, value
        )
        return comm_result == COMM_SUCCESS and error == 0

    def write_goal_pwm(self, motor_id: int, raw_pwm: int) -> bool:
        """
        Writes a signed Goal PWM (duty cycle, max ±885) to one servo.
        Requires PWM Control Mode (16).
        """
        value = int(raw_pwm) & 0xFFFF
        comm_result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_PWM, value
        )
        return comm_result == COMM_SUCCESS and error == 0

    def write_positions(self, joint_ids: list, target_ticks: list) -> bool:
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