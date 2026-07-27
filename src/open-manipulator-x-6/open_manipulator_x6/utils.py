import math

# XM430 Specifications
TICKS_PER_REV = 4096 # ticks per revolution
CENTER_TICK = 2048
RAD_PER_TICK = (2.0 * math.pi) / TICKS_PER_REV
TICKS_PER_RAD = TICKS_PER_REV / (2.0 * math.pi)

# Velocity & Current Units
RAW_VEL_TO_RAD_S = 0.229 * (2.0 * math.pi / 60.0)  # 1 unit = 0.229 RPM
RAD_S_TO_RAW_VEL = 1.0 / RAW_VEL_TO_RAD_S
RAW_CURRENT_TO_EFFORT = 2.69 / 1000.0              # 1 unit = 2.69 mA (~Torque representation)


def rad_to_ticks(rad: float) -> int:
    """
    Converts radians (0 at center) to raw encoder ticks (0-4095).
    """
    tick = int(CENTER_TICK + (rad * TICKS_PER_RAD))
    return max(0, min(TICKS_PER_REV - 1, tick))


def ticks_to_rad(ticks: int) -> float:
    """
    Converts raw encoder ticks to radians (0 at center).
    """
    return (ticks - CENTER_TICK) * RAD_PER_TICK


def raw_vel_to_rad_s(raw_vel: int) -> float:
    """
    Converts raw Dynamixel velocity units to rad/s.
    """
    # Convert signed 32-bit int
    if raw_vel > 2147483647:
        raw_vel -= 4294967296
    return float(raw_vel) * RAW_VEL_TO_RAD_S


def rad_s_to_raw_vel(rad_s: float) -> int:
    """
    Converts rad/s to raw Dynamixel velocity units.
    """
    return int(rad_s * RAD_S_TO_RAW_VEL)


def raw_current_to_effort(raw_current: int) -> float:
    """
    Converts raw current register reading to approximate effort (Amps).
    """
    if raw_current > 32767:
        raw_current -= 65536
    return float(raw_current) * RAW_CURRENT_TO_EFFORT