import time
import sys

# ==========================================
# 1. IMPORTS
# ==========================================
try:
    import VL53L0X
except ImportError:
    print("Fehler: VL53L0X fehlt.")
    print("  pip install VL53L0X --break-system-packages")
    sys.exit(1)

try:
    from gpiozero import DigitalOutputDevice
except ImportError:
    print("Fehler: gpiozero fehlt (nur auf dem Raspberry Pi verfuegbar).")
    sys.exit(1)

# ==========================================
# 2. KONFIGURATION
# ==========================================
I2C_BUS      = 1
XSHUT_LEFT   = 24     # GPIO-Pin fuer XSHUT des linken Sensors
XSHUT_RIGHT  = 25     # GPIO-Pin fuer XSHUT des rechten Sensors
ADDR_DEFAULT = 0x29   # Werksadresse beider Sensoren
ADDR_LEFT    = 0x2A   # Neue Adresse linker Sensor
ADDR_RIGHT   = 0x2B   # Neue Adresse rechter Sensor

THRESHOLD_MM  = 60    # Schwellwert in mm
POLL_INTERVAL = 0.05  # Abfrageintervall in Sekunden
MAX_VALID_MM  = 1200  # Oberhalb = ausser Reichweite / Messfehler

# ==========================================
# 3. SENSOR-SETUP
# ==========================================
def setup_sensors():
    """Beide Sensoren sequenziell hochfahren und umadressieren."""
    xshut_left  = DigitalOutputDevice(XSHUT_LEFT,  initial_value=False)
    xshut_right = DigitalOutputDevice(XSHUT_RIGHT, initial_value=False)
    time.sleep(0.05)
    print("Beide XSHUT LOW (Reset)")

    # --- Linken Sensor hochfahren ---
    xshut_left.on()
    time.sleep(0.1)
    print(f"XSHUT Links  (GPIO{XSHUT_LEFT})  HIGH -> init @ 0x{ADDR_DEFAULT:02X} ...")
    left = VL53L0X.VL53L0X(i2c_bus=I2C_BUS, i2c_address=ADDR_DEFAULT)
    left.open()
    left.change_address(ADDR_LEFT)
    print(f"  -> umadressiert auf 0x{ADDR_LEFT:02X}")

    # --- Rechten Sensor hochfahren (links ist jetzt auf 0x2A, kein Konflikt) ---
    xshut_right.on()
    time.sleep(0.1)
    print(f"XSHUT Rechts (GPIO{XSHUT_RIGHT}) HIGH -> init @ 0x{ADDR_DEFAULT:02X} ...")
    right = VL53L0X.VL53L0X(i2c_bus=I2C_BUS, i2c_address=ADDR_DEFAULT)
    right.open()
    right.change_address(ADDR_RIGHT)
    print(f"  -> umadressiert auf 0x{ADDR_RIGHT:02X}")

    # Continuous Ranging auf beiden Sensoren starten
    left.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)
    right.start_ranging(VL53L0X.Vl53l0xAccuracyMode.GOOD)

    # XSHUT-Objekte als Tupel zurueckgeben, damit sie nicht vom GC eingesammelt
    # werden (Destruktor wuerde den Pin LOW ziehen = Sensor-Reset).
    return left, right, (xshut_left, xshut_right)


# ==========================================
# 4. MAIN
# ==========================================
def main():
    left, right, _xshut = setup_sensors()

    print()
    print("=" * 44)
    print(f"  VL53L0X Schwellwert-Modus aktiv")
    print(f"  Schwelle: < {THRESHOLD_MM} mm  |  Intervall: {int(POLL_INTERVAL * 1000)} ms")
    print("=" * 44)

    state = {"L": None, "R": None}

    try:
        while True:
            for label, sensor in (("L", left), ("R", right)):
                dist = sensor.get_distance()
                if dist <= 0 or dist >= MAX_VALID_MM:
                    continue                          # ungueltige Messung ueberspringen

                near = dist < THRESHOLD_MM
                if near != state[label]:
                    if near:
                        print(f"[{label}] Objekt NAH   {dist:4d} mm  (< {THRESHOLD_MM} mm)")
                    else:
                        print(f"[{label}] Objekt frei  {dist:4d} mm")
                    state[label] = near

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nSystem herunterfahren...")
        left.stop_ranging()
        right.stop_ranging()
        left.close()
        right.close()


if __name__ == "__main__":
    main()
