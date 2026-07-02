"""Time-of-Flight (ToF) distance sensor management using VL53L0X."""

import sys
import threading
import time

try:
    import board
    import busio
    import adafruit_vl53l0x
except ImportError as e:
    print("Fehler: Adafruit-Bibliotheken fehlen.")
    print("  pip install adafruit-blinka adafruit-circuitpython-vl53l0x --break-system-packages")
    if __name__ == "__main__":
        sys.exit(1)
    else:
        raise e

try:
    from gpiozero import DigitalOutputDevice
except ImportError as e:
    print("Fehler: gpiozero fehlt (nur auf dem Raspberry Pi verfuegbar).")
    if __name__ == "__main__":
        sys.exit(1)
    else:
        raise e


# ==========================================
# 1. CONFIGURATION
# ==========================================
XSHUT_LEFT = 24
XSHUT_RIGHT = 25
ADDR_LEFT = 0x2A
ADDR_RIGHT = 0x2B

THRESHOLD_MM = 200
VERBOSE = False  # Mute by default in thread mode

# ==========================================
# 2. PUBLIC STATE
# ==========================================
state = {"L": 0, "R": 0}  # 1 = object close, 0 = free
raw = {"L": 0, "R": 0}    # Last measurement in mm


def set_threshold(mm):
    """Change the threshold value at runtime. Only applies to the current process.

    Args:
        mm (int): Threshold distance in millimeters.
    """
    global THRESHOLD_MM
    THRESHOLD_MM = mm


# VL53L0X internal registers
_REG_INT_STATUS = 0x13
_REG_INT_CLEAR = 0x0B
_REG_RANGE = 0x1E


# ==========================================
# 3. RAW I2C HELPERS
# ==========================================
def _wr(i2c, addr, data):
    """Write data to the specified I2C address.

    Args:
        i2c (busio.I2C): Active I2C bus instance.
        addr (int): I2C device address.
        data (list or bytes): Data bytes to write.
    """
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto(addr, bytes(data))
    finally:
        i2c.unlock()


def _rd1(i2c, addr, reg):
    """Read a single byte from the specified register of an I2C device.

    Args:
        i2c (busio.I2C): Active I2C bus instance.
        addr (int): I2C device address.
        reg (int): Register address to read from.

    Returns:
        int: Value of the read byte.
    """
    buf = bytearray(1)
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto_then_readfrom(addr, bytes([reg]), buf)
    finally:
        i2c.unlock()
    return buf[0]


def _rd2(i2c, addr, reg):
    """Read two bytes from the specified register of an I2C device.

    Args:
        i2c (busio.I2C): Active I2C bus instance.
        addr (int): I2C device address.
        reg (int): Register address to read from.

    Returns:
        int: Combined 16-bit integer value of the read bytes.
    """
    buf = bytearray(2)
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto_then_readfrom(addr, bytes([reg]), buf)
    finally:
        i2c.unlock()
    return (buf[0] << 8) | buf[1]


def _read_nb(i2c, addr):
    """Perform a non-blocking read of the distance measurement from the sensor.

    Args:
        i2c (busio.I2C): Active I2C bus instance.
        addr (int): I2C device address.

    Returns:
        int or None: Distance in millimeters, or None if no new measurement is ready.
    """
    if (_rd1(i2c, addr, _REG_INT_STATUS) & 0x07) != 0x04:
        return None
    dist = _rd2(i2c, addr, _REG_RANGE)
    _wr(i2c, addr, [_REG_INT_CLEAR, 0x01])
    return dist


# ==========================================
# 4. SENSOR SETUP
# ==========================================
def _try_connect(i2c, addr):
    """Attempt to connect to a VL53L0X sensor at the given address.

    Args:
        i2c (busio.I2C): Active I2C bus instance.
        addr (int): I2C address to connect to.

    Returns:
        adafruit_vl53l0x.VL53L0X or None: Sensor instance on success, None on failure.
    """
    try:
        return adafruit_vl53l0x.VL53L0X(i2c, address=addr)
    except Exception:
        return None


def _setup_sensors(i2c):
    """Initialize xshut pins, allocate addresses, and start continuous measurement.

    Args:
        i2c (busio.I2C): Active I2C bus instance.

    Returns:
        tuple: (left sensor object, right sensor object, (left xshut device, right xshut device)).
    """
    xshut_left = DigitalOutputDevice(XSHUT_LEFT, initial_value=False)
    xshut_right = DigitalOutputDevice(XSHUT_RIGHT, initial_value=False)
    time.sleep(0.2)

    xshut_left.on()
    time.sleep(0.1)
    left = _try_connect(i2c, ADDR_LEFT)
    if left:
        print(f"ToF Links: bereits bei 0x{ADDR_LEFT:02X}")
    else:
        left = _try_connect(i2c, 0x29)
        if left is None:
            print("Fehler: Linker ToF-Sensor nicht gefunden.")
            sys.exit(1)
        left.set_address(ADDR_LEFT)
        print(f"ToF Links: 0x29 -> 0x{ADDR_LEFT:02X}")

    xshut_right.on()
    time.sleep(0.1)
    right = _try_connect(i2c, ADDR_RIGHT)
    if right:
        print(f"ToF Rechts: bereits bei 0x{ADDR_RIGHT:02X}")
    else:
        right = _try_connect(i2c, 0x29)
        if right is None:
            print("Fehler: Rechter ToF-Sensor nicht gefunden.")
            sys.exit(1)
        right.set_address(ADDR_RIGHT)
        print(f"ToF Rechts: 0x29 -> 0x{ADDR_RIGHT:02X}")

    left.start_continuous()
    right.start_continuous()
    return left, right, (xshut_left, xshut_right)


# ==========================================
# 5. THREADING & ACQUISITION LOOP
# ==========================================
_thread = None


def _loop(i2c, left, right, xshut):
    """Continually polls distance measurements from sensors and updates global state."""
    while True:
        # Perform non-blocking reads on both sensors
        updated = False
        for label, addr in (("L", ADDR_LEFT), ("R", ADDR_RIGHT)):
            dist = _read_nb(i2c, addr)
            if dist is not None:
                raw[label] = dist
                state[label] = 1 if dist < THRESHOLD_MM else 0
                updated = True
        if VERBOSE and updated:
            print(f"L {state['L']}  R {state['R']}")
        # Wait briefly if no new sample is available, otherwise continue immediately
        time.sleep(0.005 if not updated else 0)


def start():
    """Initialize sensors and start the polling loop in a background daemon thread."""
    global _thread
    if _thread and _thread.is_alive():
        return  # Already started
    i2c = busio.I2C(board.SCL, board.SDA)
    left, right, xshut = _setup_sensors(i2c)
    _thread = threading.Thread(target=_loop, args=(i2c, left, right, xshut), daemon=True)
    _thread.start()
    print("ToF-Thread gestartet.")


# ==========================================
# 6. STANDALONE MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    VERBOSE = True
    start()
    print(f"Schwelle: < {THRESHOLD_MM} mm")
    print("Ctrl+C zum Beenden.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBeendet.")
