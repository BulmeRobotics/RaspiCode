import cv2
import sys

# Kamera ID (0 oder 1)
CAM_ID = 1

def mouse_callback(event, x, y, flags, param):
    """
    Diese Funktion wird aufgerufen, wenn die Maus bewegt oder geklickt wird.
    Sie gibt die Koordinaten direkt in der Konsole aus.
    """
    if event == cv2.EVENT_MOUSEMOVE:
        # Hier könnten wir auch bei Bewegung spammen, 
        # aber ein Klick ist oft präziser zum Ablesen:
        pass
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"📍 Klick-Koordinate: X = {x}, Y = {y}")

try:
    from picamera2 import Picamera2
    picam2 = Picamera2(CAM_ID)
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
except Exception as e:
    print(f"Fehler: {e}")
    sys.exit()

cv2.namedWindow("Koordinaten-Tool")
cv2.setMouseCallback("Koordinaten-Tool", mouse_callback)

print("--- Koordinaten-Tool gestartet ---")
print("Klicke in das Bild, um die X/Y-Koordinaten zu sehen.")
print("Drücke 'q' zum Beenden.")

while True:
    frame_rgb = picam2.capture_array()
    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Koordinaten-Tool", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()