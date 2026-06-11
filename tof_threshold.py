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
    from gpiozero import DigitalOutputDevice, DigitalInputDevice
except ImportError:
    print("Fehler: gpiozero fehlt (nur auf dem Raspberry Pi verfuegbar).")
    sys.exit(1)

# ==========================================
# 2. KONFIGURATION
# ==========================================
XSHUT_LEFT   = 24
XSHUT_RIGHT  = 25
IRQ_LEFT     = 5    # Sensor GPIO1 -> Pi-Eingang
IRQ_RIGHT    = 6
ADDR_LEFT    = 0x2A
ADDR_RIGHT   = 0x2B

THRESHOLD_MM  = 200
POLL_INTERVAL = 0.05  # s

# VL53L0X-Register (8-Bit-Adressierung)
_REG_INTERRUPT_CONFIG = 0x0A  # Bits [2:0]: 0=off 1=below-low 2=above-high 4=new-sample(default)
_REG_INTERRUPT_CLEAR  = 0x0B  # Bit 0: 1 = Latch loeschen
_REG_THRESH_LOW       = 0x0E  # uint16 big-endian, mm
_REG_GPIO_MUX         = 0x84  # Bit 4: 0=active-low, 1=active-high

# ==========================================
# 3. HILFSFUNKTIONEN (Roh-I2C)
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

def _configure_threshold(i2c, addr):
    """GPIO1 als active-low Schwellwert-Interrupt konfigurieren."""
    # Polaritaet: active-low (Bit 4 loeschen)
    _wr(i2c, addr, [_REG_GPIO_MUX, _rd1(i2c, addr, _REG_GPIO_MUX) & ~0x10])
    # Interrupt-Bedingung: range < THRESH_LOW
    _wr(i2c, addr, [_REG_INTERRUPT_CONFIG, 0x01])
    # Schwellwert setzen (uint16 big-endian)
    _wr(i2c, addr, [_REG_THRESH_LOW,
                    (THRESHOLD_MM >> 8) & 0xFF,
                    THRESHOLD_MM & 0xFF])
    # Latch loeschen
    _wr(i2c, addr, [_REG_INTERRUPT_CLEAR, 0x01])

def _clear_interrupt(i2c, addr):
    _wr(i2c, addr, [_REG_INTERRUPT_CLEAR, 0x01])

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

    # Hardware-Schwellwert-Interrupt auf GPIO1 konfigurieren
    _configure_threshold(i2c, ADDR_LEFT)
    _configure_threshold(i2c, ADDR_RIGHT)

    # Kontinuierliches Ranging starten
    left.start_continuous()
    right.start_continuous()

    return left, right, (xshut_left, xshut_right)

# ==========================================
# 5. MAIN
# ==========================================
def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    left, right, _xshut = setup_sensors(i2c)

    irq_left  = DigitalInputDevice(IRQ_LEFT,  pull_up=True)
    irq_right = DigitalInputDevice(IRQ_RIGHT, pull_up=True)

    print()
    print("=" * 44)
    print(f"  VL53L0X GPIO-Modus | Schwelle: < {THRESHOLD_MM} mm")
    print("=" * 44)

    try:
        while True:
            l_state = "LOW" if not irq_left.value  else "HIGH"
            r_state = "LOW" if not irq_right.value else "HIGH"
            print(f"L {l_state}  R {r_state}")
            _clear_interrupt(i2c, ADDR_LEFT)
            _clear_interrupt(i2c, ADDR_RIGHT)
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nBeendet.")
        left.stop_continuous()
        right.stop_continuous()


if __name__ == "__main__":
    main()