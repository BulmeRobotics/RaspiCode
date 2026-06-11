"""
Scannt mehrere GPIO-Pins gleichzeitig.
Starte das Script, halte die Hand vor den LINKEN Sensor,
und schau welcher Pin auf LOW wechselt.
"""
import time
from gpiozero import DigitalInputDevice

# Kandidaten: alle freien GPIO-Pins rund um GPIO5 (Pin 29)
# GPIO6 (Pin 31) ist der rechte Sensor - zum Vergleich mit dabei
CANDIDATES = [4, 7, 12, 13, 16, 19, 20, 21, 22, 23, 26, 27]

pins = {}
for g in CANDIDATES:
    try:
        pins[g] = DigitalInputDevice(g, pull_up=True)
    except Exception as e:
        print(f"GPIO{g}: nicht verfuegbar ({e})")

print(f"Scanne {len(pins)} Pins. Hand vor linken Sensor halten...")
print("Ctrl+C zum Beenden.\n")

prev = {g: p.value for g, p in pins.items()}

try:
    while True:
        for g, p in pins.items():
            val = p.value
            if val != prev[g]:
                state = "HIGH" if val else "LOW "
                print(f"GPIO{g:2d} -> {state}  <-- Aenderung!")
                prev[g] = val
        time.sleep(0.02)
except KeyboardInterrupt:
    print("\nFertig.")
