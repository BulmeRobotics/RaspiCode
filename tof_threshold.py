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
XSHUT_LEFT   = 24
XSHUT_RIGHT  = 25
ADDR_LEFT    = 0x2A
ADDR_RIGHT   = 0x2B

THRESHOLD_MM   = 200
POLL_INTERVAL = 0.05   # s
VERBOSE        = True   # Messwerte ausgeben

# ==========================================
# 3. SENSOR-SETUP
# ==========================================
def _try_connect(i2c, addr):
    try:
        return adafruit_vl53l0x.VL53L0X(i2c, address=addr)
    except Exception:
        return None

def setup_sensors(i2c):
    """
    Robust gegen Neustart: Wenn Sensoren bereits bei 0x2A/0x2B sind,
    werden sie direkt verwendet ohne erneute Adressvergabe.
    """
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

    return left, right, (xshut_left, xshut_right)

# ==========================================
# 4. MAIN
# ==========================================
def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    left, right, _xshut = setup_sensors(i2c)

    print()
    print("=" * 44)
    print(f"  VL53L0X Polling | Schwelle: < {THRESHOLD_MM} mm")
    print("=" * 44)

    try:
        while True:
            results = {}
            for label, sensor in (("L", left), ("R", right)):
                try:
                    dist = sensor.range
                except Exception as e:
                    print(f"[{label}] Lesefehler: {e}")
                    results[label] = 0
                    continue
                results[label] = 1 if dist < THRESHOLD_MM else 0

            if VERBOSE:
                print(f"L {results.get('L', 0)}  R {results.get('R', 0)}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nBeendet.")


if __name__ == "__main__":
    main()
