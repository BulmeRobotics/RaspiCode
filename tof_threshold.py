import time
import sys

# ==========================================
# 1. IMPORT-BLOCK
# ==========================================
try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    print("Fehler: smbus2 fehlt. Bitte mit 'pip install smbus2' installieren.")
    sys.exit()

try:
    from gpiozero import DigitalOutputDevice, DigitalInputDevice
except ImportError:
    print("Fehler: gpiozero fehlt (nur auf dem Raspberry Pi verfuegbar).")
    sys.exit()

# ==========================================
# 2. KONFIGURATION
# ==========================================
I2C_BUS = 1                  # Standard-I2C-Bus am Raspberry Pi (Pins SDA1/SCL1)

# XSHUT (= GPIO0/CE am Sensor): zum sequenziellen Hochfahren fuer die Adressvergabe.
# LOW = Sensor im Reset/Standby, HIGH = Sensor aktiv.
XSHUT_LEFT = 24
XSHUT_RIGHT = 25

# GPIO1-Interrupt-Ausgaenge der Sensoren -> Pi-Eingaenge.
# GPIO1 ist Open-Drain: braucht Pull-up (hier intern via gpiozero aktiviert).
IRQ_LEFT = 5
IRQ_RIGHT = 6

# I2C-Adressen. Beide Sensoren starten auf 0x29 und werden umadressiert.
ADDR_DEFAULT = 0x29
ADDR_LEFT = 0x2A
ADDR_RIGHT = 0x2B

# Schwellwert: GPIO1 wird AKTIV, wenn die Distanz KLEINER als dieser Wert ist.
# Gueltig 0-255 mm (bei Scaling 1, dem Default).
THRESHOLD_MM = 60

# Continuous-Mode-Timing.
# Regel aus dem Datenblatt: max_convergence_time + 5 <= intermeasurement * 0.9
MAX_CONVERGENCE_TIME = 30    # ms (1-63)
INTERMEASUREMENT_MS = 50     # ms zwischen Messungen (Vielfaches von 10)

# Polaritaet des GPIO1-Interrupt-Ausgangs (Open-Drain).
#   False = Active-Low  -> Objekt nah = Leitung LOW,  frei = HIGH (Pull-up)
#   True  = Active-High
GPIO1_ACTIVE_HIGH = False

# ==========================================
# 3. VL6180X REGISTER (16-Bit-Index!)
# ==========================================
# Wichtig: Der VL6180X adressiert seine Register mit einem 16-Bit-Index.
# Deshalb funktionieren die ueblichen SMBus-Funktionen (8-Bit-Register) NICHT
# und es wird hier mit rohen I2C-Transaktionen (i2c_msg) gearbeitet.
SYSTEM__FRESH_OUT_OF_RESET        = 0x016
SYSTEM__MODE_GPIO1                = 0x011
SYSTEM__INTERRUPT_CONFIG_GPIO     = 0x014
SYSTEM__INTERRUPT_CLEAR           = 0x015
SYSRANGE__START                   = 0x018
SYSRANGE__THRESH_HIGH             = 0x019
SYSRANGE__THRESH_LOW              = 0x01A
SYSRANGE__INTERMEASUREMENT_PERIOD = 0x01B
SYSRANGE__MAX_CONVERGENCE_TIME    = 0x01C
RESULT__RANGE_STATUS              = 0x04D
RESULT__INTERRUPT_STATUS_GPIO     = 0x04F
RESULT__RANGE_VAL                 = 0x062
READOUT__AVERAGING_SAMPLE_PERIOD  = 0x10A
I2C_SLAVE__DEVICE_ADDRESS         = 0x212
IDENTIFICATION__MODEL_ID          = 0x000

# Pflicht-Tuning-Sequenz nach jedem Reset (ST AN4545 / Pololu configureDefault).
# Ohne diese Werte liefert das Ranging unbrauchbare Ergebnisse.
SR03_INIT = [
    (0x0207, 0x01), (0x0208, 0x01), (0x0096, 0x00), (0x0097, 0xFD),
    (0x00E3, 0x00), (0x00E4, 0x04), (0x00E5, 0x02), (0x00E6, 0x01),
    (0x00E7, 0x03), (0x00F5, 0x02), (0x00D9, 0x05), (0x00DB, 0xCE),
    (0x00DC, 0x03), (0x00DD, 0xF8), (0x009F, 0x00), (0x00A3, 0x3C),
    (0x00B7, 0x00), (0x00BB, 0x3C), (0x00B2, 0x09), (0x00CA, 0x09),
    (0x0198, 0x01), (0x01B0, 0x17), (0x01AD, 0x00), (0x00FF, 0x05),
    (0x0100, 0x05), (0x0199, 0x05), (0x01A6, 0x1B), (0x01AC, 0x3E),
    (0x01A7, 0x1F), (0x0030, 0x00),
]

# ==========================================
# 4. VL6180X TREIBER-KLASSE
# ==========================================
class VL6180X:
    def __init__(self, bus, address):
        self.bus = bus
        self.address = address

    # --- Roh-I2C mit 16-Bit-Index ---
    def write8(self, reg, data):
        msg = i2c_msg.write(self.address, [(reg >> 8) & 0xFF, reg & 0xFF, data & 0xFF])
        self.bus.i2c_rdwr(msg)

    def read8(self, reg):
        w = i2c_msg.write(self.address, [(reg >> 8) & 0xFF, reg & 0xFF])
        r = i2c_msg.read(self.address, 1)
        self.bus.i2c_rdwr(w, r)
        return list(r)[0]

    # Sensor-Identitaet pruefen (Model-ID muss 0xB4 sein)
    def check_present(self):
        return self.read8(IDENTIFICATION__MODEL_ID) == 0xB4

    # I2C-Adresse umschreiben (7-Bit). Danach self.address aktualisieren.
    def set_address(self, new_address):
        self.write8(I2C_SLAVE__DEVICE_ADDRESS, new_address & 0x7F)
        self.address = new_address

    # Pflicht-Init + Grundkonfiguration laden
    def init(self):
        # Nur laden, wenn der Sensor frisch aus dem Reset kommt
        if self.read8(SYSTEM__FRESH_OUT_OF_RESET) == 1:
            for reg, val in SR03_INIT:
                self.write8(reg, val)
            self.write8(SYSTEM__FRESH_OUT_OF_RESET, 0x00)

        # Readout-Averaging (Rauschunterdrueckung, Default 48)
        self.write8(READOUT__AVERAGING_SAMPLE_PERIOD, 0x30)
        # Max-Konvergenzzeit
        self.write8(SYSRANGE__MAX_CONVERGENCE_TIME, MAX_CONVERGENCE_TIME & 0x3F)

    # Hardware-Schwellwert + GPIO1-Interrupt konfigurieren und Continuous starten.
    # range_int_mode = 1 (Level Low): Interrupt feuert, wenn Distanz < THRESH_LOW.
    def start_threshold_mode(self):
        # Schwellwerte (THRESH_HIGH auf Maximum, da nur die untere Grenze zaehlt)
        self.write8(SYSRANGE__THRESH_LOW, THRESHOLD_MM & 0xFF)
        self.write8(SYSRANGE__THRESH_HIGH, 0xFF)

        # Mess-Intervall im Continuous-Mode (0 = 10ms, Schritt 10ms)
        self.write8(SYSRANGE__INTERMEASUREMENT_PERIOD, (INTERMEASUREMENT_MS // 10) - 1)

        # Interrupt-Quelle: Range "Level Low" (Bits [2:0] = 1)
        self.write8(SYSTEM__INTERRUPT_CONFIG_GPIO, 0x01)

        # GPIO1 als Interrupt-Ausgang: select = 1000 (Bits [4:1]) -> 0x10
        gpio1_cfg = 0x10
        if GPIO1_ACTIVE_HIGH:
            gpio1_cfg |= 0x20    # Polaritaets-Bit [5]
        self.write8(SYSTEM__MODE_GPIO1, gpio1_cfg)

        # Alle Interrupts loeschen und Continuous-Ranging starten
        self.clear_interrupt()
        self.write8(SYSRANGE__START, 0x03)   # mode=continuous, start

    # Letzten Distanzwert lesen (mm). Nur fuer Logging - die Schwellwert-
    # Entscheidung trifft der Sensor selbst in Hardware.
    def read_range(self):
        return self.read8(RESULT__RANGE_VAL)

    # Interrupt-Status lesen. Range-Bits [2:0]: 1 = Level-Low-Event (Objekt nah).
    def range_event(self):
        return (self.read8(RESULT__INTERRUPT_STATUS_GPIO) & 0x07) == 1

    # Interrupt-Latch loeschen, damit der naechste Messzyklus neu auswertet.
    def clear_interrupt(self):
        self.write8(SYSTEM__INTERRUPT_CLEAR, 0x07)


# ==========================================
# 5. INITIALISIERUNG & SENSOR-BRINGUP
# ==========================================
def setup_sensors(bus):
    """Beide Sensoren sequenziell hochfahren und umadressieren."""
    xshut_left = DigitalOutputDevice(XSHUT_LEFT)
    xshut_right = DigitalOutputDevice(XSHUT_RIGHT)

    # Beide Sensoren in den Reset zwingen
    xshut_left.off()
    xshut_right.off()
    time.sleep(0.02)

    # --- Linken Sensor hochfahren und umadressieren ---
    xshut_left.on()
    time.sleep(0.01)             # Boot-Zeit (min. 1ms laut Datenblatt)
    left = VL6180X(bus, ADDR_DEFAULT)
    if not left.check_present():
        print("Fehler: Linker Sensor antwortet nicht auf 0x29.")
        sys.exit()
    left.set_address(ADDR_LEFT)
    left.init()

    # --- Rechten Sensor hochfahren (jetzt eindeutig, da links umadressiert) ---
    xshut_right.on()
    time.sleep(0.01)
    right = VL6180X(bus, ADDR_DEFAULT)
    if not right.check_present():
        print("Fehler: Rechter Sensor antwortet nicht auf 0x29.")
        sys.exit()
    right.set_address(ADDR_RIGHT)
    right.init()

    # XSHUT-Objekte zurueckgeben, damit sie nicht vom Garbage Collector
    # eingesammelt werden (sonst fallen die Pins auf LOW = Reset).
    return left, right, (xshut_left, xshut_right)


# ==========================================
# 6. MAIN
# ==========================================
def main():
    # Timing-Regel aus dem Datenblatt hart pruefen (sonst undefiniertes Verhalten)
    assert MAX_CONVERGENCE_TIME + 5 <= INTERMEASUREMENT_MS * 0.9, (
        "Ungueltige Timing-Konfiguration: "
        "MAX_CONVERGENCE_TIME + 5 muss <= INTERMEASUREMENT_MS * 0.9 sein."
    )

    bus = SMBus(I2C_BUS)
    left, right, _xshut = setup_sensors(bus)

    # GPIO1-Leitungen der Sensoren als Pi-Eingaenge (Open-Drain -> Pull-up).
    irq_left = DigitalInputDevice(IRQ_LEFT, pull_up=True)
    irq_right = DigitalInputDevice(IRQ_RIGHT, pull_up=True)

    # Hardware-Schwellwertmodus auf beiden Sensoren starten
    left.start_threshold_mode()
    right.start_threshold_mode()

    print("=======================================")
    print(f" VL6180X Schwellwert-Modus aktiv (< {THRESHOLD_MM} mm)")
    print(f" Mess-Intervall: {INTERMEASUREMENT_MS} ms")
    print("=======================================")

    # Letzten Zustand merken, um nur bei Aenderungen zu loggen
    state = {"L": None, "R": None}

    try:
        while True:
            for name, sensor in (("L", left), ("R", right)):
                near = sensor.range_event()           # Sensor-Hardware-Vergleich
                if near != state[name]:
                    dist = sensor.read_range()
                    if near:
                        print(f"[{name}] Objekt NAH  ({dist} mm < {THRESHOLD_MM} mm)")
                    else:
                        print(f"[{name}] Objekt frei ({dist} mm)")
                    state[name] = near
                # Latch loeschen, damit der naechste Messzyklus neu auswertet
                sensor.clear_interrupt()

            time.sleep(INTERMEASUREMENT_MS / 1000.0)

    except KeyboardInterrupt:
        print("\nSystem herunterfahren...")
        # Continuous-Ranging stoppen
        left.write8(SYSRANGE__START, 0x00)
        right.write8(SYSRANGE__START, 0x00)
        bus.close()


if __name__ == "__main__":
    main()
