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

# Voltage & Temperature Units
RAW_VOLTAGE_TO_VOLTS = 0.1  # 1 unit = 0.1 V
# Present Temperature register already reports whole degrees Celsius (1 unit = 1 degC)


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


def raw_voltage_to_volts(raw_voltage: int) -> float:
    """
    Converts raw present-voltage register reading to volts.
    """
    return float(raw_voltage) * RAW_VOLTAGE_TO_VOLTS


def raw_temperature_to_celsius(raw_temp: int) -> float:
    """
    Converts raw present-temperature register reading to degrees Celsius.
    """
    return float(raw_temp)

# --- Model-aware effort -------------------------------------------------------------------
# Register 126 means different things on the two servo families:
#   XM430: Present Current, signed, 1 unit = 2.69 mA   -> reported in Amps
#   XL430: Present Load,    signed, 1 unit = 0.1 % PWM -> reported as a fraction (-1.0 .. 1.0)
# Both go in JointState.effort, so a consumer must know which joints are which. The arm driver
# publishes the family of every joint in /arm_diagnostics ("Effort unit").
RAW_LOAD_TO_FRACTION = 0.001  # 1 unit = 0.1 % of max output


def raw_load_to_fraction(raw_load: int) -> float:
    """XL430 Present Load register -> fraction of maximum output, signed (-1.0 .. 1.0)."""
    if raw_load > 32767:
        raw_load -= 65536
    return float(raw_load) * RAW_LOAD_TO_FRACTION


def raw_effort(raw: int, has_current_sensor: bool) -> float:
    """Amps for XM430 joints, load fraction for XL430 joints."""
    return raw_current_to_effort(raw) if has_current_sensor else raw_load_to_fraction(raw)


def amps_to_raw_current(amps: float) -> int:
    """Amps -> XM430 Goal Current register units (rounded, sign kept)."""
    return int(round(amps / RAW_CURRENT_TO_EFFORT))