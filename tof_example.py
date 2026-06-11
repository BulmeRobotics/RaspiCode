import time
import tof

# Sensoren initialisieren und Thread starten
tof.start()

# Ab hier laeuft der Sensor-Thread im Hintergrund.
# tof.state["L"] und ["R"] werden automatisch aktualisiert.

while True:
    l = tof.state["L"]
    r = tof.state["R"]
    print(f"Links: {l}  Rechts: {r}")
    time.sleep(0.03)
