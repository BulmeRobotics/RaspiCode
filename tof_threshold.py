import time
import sys

# ==========================================
# 1. IMPORTS
# ==========================================
try:
    import board
    import busio
    import adafruit_vl53l0x
except ImportError:
    print("Fehler: Adafruit-Bibliotheken fehlen.")
    print("  pip install adafruit-blinka adafruit-circuitpython-vl53l0x --break-system-packages")
    sys.exit(1)

try:
    from gpiozero import DigitalOutputDevice
except ImportError:
    print("Fehler: gpiozero fehlt (nur auf dem Raspberry Pi verfuegbar).")
    sys.exit(1)

# ==========================================
# 2. KONFIGURATION
# ==========================================
XSHUT_LEFT    = 24
XSHUT_RIGHT   = 25
ADDR_LEFT     = 0x2A
ADDR_RIGHT    = 0x2B

THRESHOLD_MM  = 200
POLL_INTERVAL = 0.03   # s
VERBOSE       = True

# VL53L0X-Register fuer nicht-blockierendes Lesen
_REG_INT_STATUS = 0x13  # bits[2:0]: 0x04 = neue Messung bereit
_REG_INT_CLEAR  = 0x0B
_REG_RANGE      = 0x1E  # uint16 big-endian, mm

# ==========================================
# 3. RAW I2C HELPERS
# ==========================================
def _wr(i2c, addr, data):
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto(addr, bytes(data))
    finally:
        i2c.unlock()

def _rd1(i2c, addr, reg):
    buf = bytearray(1)
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto_then_readfrom(addr, bytes([reg]), buf)
    finally:
        i2c.unlock()
    return buf[0]

def _rd2(i2c, addr, reg):
    buf = bytearray(2)
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto_then_readfrom(addr, bytes([reg]), buf)
    finally:
        i2c.unlock()
    return (buf[0] << 8) | buf[1]

def _read_nb(i2c, addr):
    """Nicht-blockierend: gibt Distanz in mm zurueck wenn Messung fertig, sonst None."""
    if (_rd1(i2c, addr, _REG_INT_STATUS) & 0x07) != 0x04:
        return None
    dist = _rd2(i2c, addr, _REG_RANGE)
    _wr(i2c, addr, [_REG_INT_CLEAR, 0x01])
    return dist

# ==========================================
# 4. SENSOR-SETUP
# ==========================================
def _try_connect(i2c, addr):
    try:
        return adafruit_vl53l0x.VL53L0X(i2c, address=addr)
    except Exception:
        return None

def setup_sensors(i2c):
    xshut_left  = DigitalOutputDevice(XSHUT_LEFT,  initial_value=False)
    xshut_right = DigitalOutputDevice(XSHUT_RIGHT, initial_value=False)
    time.sleep(0.2)
    print("Beide XSHUT LOW")

    xshut_left.on()
    time.sleep(0.1)
    left = _try_connect(i2c, ADDR_LEFT)
    if left:
        print(f"Links: bereits bei 0x{ADDR_LEFT:02X}")
    else:
        left = _try_connect(i2c, 0x29)
        if left is None:
            print("Fehler: Linker Sensor nicht gefunden.")
            sys.exit(1)
        left.set_address(ADDR_LEFT)
        print(f"Links: 0x29 -> 0x{ADDR_LEFT:02X}")

    xshut_right.on()
    time.sleep(0.1)
    right = _try_connect(i2c, ADDR_RIGHT)
    if right:
        print(f"Rechts: bereits bei 0x{ADDR_RIGHT:02X}")
    else:
        right = _try_connect(i2c, 0x29)
        if right is None:
            print("Fehler: Rechter Sensor nicht gefunden.")
            sys.exit(1)
        right.set_address(ADDR_RIGHT)
        print(f"Rechts: 0x29 -> 0x{ADDR_RIGHT:02X}")

    # Kontinuierliches Ranging starten - beide Sensoren messen ab jetzt autonom
    left.start_continuous()
    right.start_continuous()

    return left, right, (xshut_left, xshut_right)

# ==========================================
# 5. MAIN
# ==========================================
def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    left, right, _xshut = setup_sensors(i2c)

    print()
    print("=" * 44)
    print(f"  VL53L0X | Schwelle: < {THRESHOLD_MM} mm | {int(POLL_INTERVAL * 1000)} ms")
    print("=" * 44)

    results = {"L": 0, "R": 0}

    try:
        while True:
            # Beide Sensoren nicht-blockierend lesen — kein sensor wartet auf den anderen
            for label, addr in (("L", ADDR_LEFT), ("R", ADDR_RIGHT)):
                dist = _read_nb(i2c, addr)
                if dist is not None:
                    results[label] = 1 if dist < THRESHOLD_MM else 0

            if VERBOSE:
                print(f"L {results['L']}  R {results['R']}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nBeendet.")
        left.stop_continuous()
        right.stop_continuous()


if __name__ == "__main__":
    main()
