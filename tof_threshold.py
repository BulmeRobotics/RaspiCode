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
ADDR_DEFAULT = 0x29
ADDR_LEFT    = 0x2A
ADDR_RIGHT   = 0x2B

THRESHOLD_MM  = 60
POLL_INTERVAL = 0.05   # s
MAX_VALID_MM  = 1200   # oberhalb = ausser Reichweite / Messfehler

# ==========================================
# 3. SENSOR-SETUP
# ==========================================
def _try_connect(i2c, addr):
    """Versucht, einen VL53L0X an der gegebenen Adresse zu oeffnen. Gibt None zurueck bei Fehler."""
    try:
        return adafruit_vl53l0x.VL53L0X(i2c, address=addr)
    except Exception:
        return None

def setup_sensors(i2c):
    """
    Beide Sensoren hochfahren und umadressieren.
    Robust gegen Neustart: Wenn Sensoren bereits bei 0x2A/0x2B sind (z.B. weil
    die XSHUT-Pins beim letzten Programmende HIGH blieben), werden sie direkt
    verwendet ohne erneute Adressvergabe.
    """
    xshut_left  = DigitalOutputDevice(XSHUT_LEFT,  initial_value=False)
    xshut_right = DigitalOutputDevice(XSHUT_RIGHT, initial_value=False)
    time.sleep(0.2)    # Sensoren Zeit zum Resetten geben (falls XSHUT verdrahtet)
    print("Beide XSHUT LOW")

    # --- Linken Sensor ---
    xshut_left.on()
    time.sleep(0.1)
    # Erst pruefen ob Sensor schon auf Zieladresse sitzt (Neustart-Fall)
    left = _try_connect(i2c, ADDR_LEFT)
    if left:
        print(f"Links: bereits bei 0x{ADDR_LEFT:02X}")
    else:
        left = _try_connect(i2c, ADDR_DEFAULT)
        if left is None:
            print(f"Fehler: Linker Sensor antwortet weder auf 0x{ADDR_LEFT:02X} noch auf 0x{ADDR_DEFAULT:02X}.")
            sys.exit(1)
        left.set_address(ADDR_LEFT)
        print(f"Links: 0x{ADDR_DEFAULT:02X} -> 0x{ADDR_LEFT:02X}")

    # --- Rechten Sensor ---
    xshut_right.on()
    time.sleep(0.1)
    right = _try_connect(i2c, ADDR_RIGHT)
    if right:
        print(f"Rechts: bereits bei 0x{ADDR_RIGHT:02X}")
    else:
        right = _try_connect(i2c, ADDR_DEFAULT)
        if right is None:
            print(f"Fehler: Rechter Sensor antwortet weder auf 0x{ADDR_RIGHT:02X} noch auf 0x{ADDR_DEFAULT:02X}.")
            sys.exit(1)
        right.set_address(ADDR_RIGHT)
        print(f"Rechts: 0x{ADDR_DEFAULT:02X} -> 0x{ADDR_RIGHT:02X}")

    return left, right, (xshut_left, xshut_right)


# ==========================================
# 4. MAIN
# ==========================================
def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    left, right, _xshut = setup_sensors(i2c)

    print()
    print("=" * 44)
    print(f"  VL53L0X Schwellwert-Modus aktiv")
    print(f"  Schwelle: < {THRESHOLD_MM} mm  |  Intervall: {int(POLL_INTERVAL * 1000)} ms")
    print("  (Ausgabe nur bei Zustandsaenderung)")
    print("=" * 44)

    state    = {"L": None, "R": None}
    sensors  = (("L", left), ("R", right))

    try:
        while True:
            for label, sensor in sensors:
                try:
                    dist = sensor.range
                except Exception as e:
                    print(f"[{label}] Lesefehler: {e}")
                    continue

                if dist <= 0 or dist >= MAX_VALID_MM:
                    continue

                near = dist < THRESHOLD_MM
                if near != state[label]:
                    tag = "NAH " if near else "frei"
                    print(f"[{label}] {tag}  {dist:4d} mm")
                    state[label] = near

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nBeendet.")


if __name__ == "__main__":
    main()
