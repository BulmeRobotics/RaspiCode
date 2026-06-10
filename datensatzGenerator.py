import cv2
import os
import time
import sys

# Picamera2 importieren
try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt.")
    sys.exit()

# --- EINSTELLUNGEN ---
FPS = 5                    # Wie viele Bilder pro Sekunde gespeichert werden sollen
SLEEP_TIME = 1.0 / FPS     # Pause zwischen den Bildern
SAVE_FOLDER = "dataset_neu" # Ordner, in dem die Bilder landen
CAMERA_ID = 0              # 0 für die linke Cam, 1 für die rechte (ggf. anpassen)

# Ordner automatisch erstellen, falls er nicht existiert
os.makedirs(SAVE_FOLDER, exist_ok=True)

def main():
    print(f"Initialisiere Kamera {CAMERA_ID}...")
    picam2 = Picamera2(CAMERA_ID)
    
    # EXAKT die gleichen Einstellungen wie im Hauptcode!
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    print("\n=======================================")
    print(f" Kamerasystem BEREIT. Speichere in: ./{SAVE_FOLDER}/")
    print(f" Nehme {FPS} Bilder pro Sekunde auf.")
    print("=======================================\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # 1. Frame holen (Identisch zum Live-Code)
            frame_rgb = picam2.capture_array()
            
            # 2. Konvertierung für OpenCV/Modell (Identisch zum Live-Code)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # 3. Bildname generieren (z.B. img_0001.jpg, img_0002.jpg)
            filename = os.path.join(SAVE_FOLDER, f"img_{frame_count:04d}.jpg")
            
            # 4. Bild in bester Qualität abspeichern
            cv2.imwrite(filename, frame_bgr)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[{frame_count}] Bilder gespeichert...")

            # Warten, um die gewünschte Framerate (FPS) zu halten
            processing_time = time.time() - loop_start
            time_to_wait = SLEEP_TIME - processing_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("\nAufnahme durch Benutzer abgebrochen.")
    finally:
        picam2.stop()
        dauer = time.time() - start_time
        print(f"Kamera gestoppt. {frame_count} Bilder in {dauer:.1f} Sekunden gespeichert.")

if __name__ == "__main__":
    main()
