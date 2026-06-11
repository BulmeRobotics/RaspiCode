import time
import tof_threshold

# Sensoren initialisieren und Thread starten
tof_threshold.start()

# Ab hier laeuft der Sensor-Thread im Hintergrund.
# tof_threshold.state["L"] und ["R"] werden automatisch aktualisiert.

while True:
    l = tof_threshold.state["L"]
    r = tof_threshold.state["R"]
    print(f"Links: {l}  Rechts: {r}")
    time.sleep(0.03)
