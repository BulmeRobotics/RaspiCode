import cv2
import numpy as np
import time
import sys

try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt. Bitte installieren.")
    sys.exit()

# ==========================================
# KONFIGURATION
# ==========================================
CAM_ID = 0  # 0 für Links, 1 für Rechts

def nothing(x):
    pass

# ==========================================
# GUI & TRACKBARS SETUP
# ==========================================
cv2.namedWindow("Size Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Size Calibration", 640, 650)

# Trackbars für die min/max Breite und Höhe
cv2.createTrackbar('Min Width', 'Size Calibration', 60, 300, nothing)
cv2.createTrackbar('Min Height', 'Size Calibration', 60, 300, nothing)

cv2.createTrackbar('Max Width', 'Size Calibration', 360, 600, nothing)
cv2.createTrackbar('Max Height', 'Size Calibration', 360, 600, nothing)

# Aspect Ratio (Seitenverhältnis) -> Werte werden durch 10 geteilt (5 = 0.5, 20 = 2.0)
cv2.createTrackbar('Min AR (*10)', 'Size Calibration', 5, 20, nothing)
cv2.createTrackbar('Max AR (*10)', 'Size Calibration', 20, 40, nothing)

# ==========================================
# KAMERA STARTEN
# ==========================================
print("Starte Kamera...")
try:
    picam2 = Picamera2(CAM_ID)
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
except Exception as e:
    print(f"Kamera-Fehler: {e}")
    sys.exit()

print("\n--- GRÖSSEN-KALIBRIERUNG GESTARTET ---")
print("Passe die Schieberegler im Fenster an.")
print("Grüne Box = Gültig (Größe stimmt & MITTELPUNKT im Feld)")
print("Rote Box = Ungültig (zu groß, zu klein, falsches Format)")
print("Drücke 'q' zum Beenden.\n")

frame_counter = 0

while True:
    frame_rgb = picam2.capture_array()
    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    display_frame = frame_bgr.copy()
    
    # 1. Aktuelle Trackbar-Werte auslesen
    min_w = cv2.getTrackbarPos('Min Width', 'Size Calibration')
    min_h = cv2.getTrackbarPos('Min Height', 'Size Calibration')
    max_w = cv2.getTrackbarPos('Max Width', 'Size Calibration')
    max_h = cv2.getTrackbarPos('Max Height', 'Size Calibration')
    
    # Durch 10 teilen, um Fließkommazahlen zu bekommen
    min_ar = cv2.getTrackbarPos('Min AR (*10)', 'Size Calibration') / 10.0
    max_ar = cv2.getTrackbarPos('Max AR (*10)', 'Size Calibration') / 10.0
    
    # 2. Sperrzonen definieren und einzeichnen
    cutoff_top_y = int(frame_rgb.shape[0] * 0.25)
    cutoff_bottom_y = int(frame_rgb.shape[0] * 0.875)
    cutoff_left_x = int(frame_rgb.shape[1] * (1/7))
    cutoff_right_x = int(frame_rgb.shape[1] * (6/7))
    
    cv2.line(display_frame, (0, cutoff_top_y), (display_frame.shape[1], cutoff_top_y), (0, 255, 255), 2)
    cv2.line(display_frame, (0, cutoff_bottom_y), (display_frame.shape[1], cutoff_bottom_y), (0, 255, 255), 2)
    cv2.line(display_frame, (cutoff_left_x, 0), (cutoff_left_x, display_frame.shape[0]), (0, 255, 255), 2)
    cv2.line(display_frame, (cutoff_right_x, 0), (cutoff_right_x, display_frame.shape[0]), (0, 255, 255), 2)

    # 3. Bildverarbeitung wie im Hauptcode
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    total_area = gray_frame.shape[0] * gray_frame.shape[1]
    
    blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < total_area * 0.99]

    square_contours = []
    
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w == 0: continue 
        
        # --- MITTELPUNKT BERECHNEN UND PRÜFEN ---
        cx = x + (w // 2)
        cy = y + (h // 2)
        
        # Ignoriere alles, dessen Mittelpunkt in den Sperrzonen liegt
        if cy < cutoff_top_y or cy > cutoff_bottom_y: continue
        if cx < cutoff_left_x or cx > cutoff_right_x: continue
        
        aspect_ratio = w / float(h)
        
        # GRÖSSEN- UND FORMAT-CHECK (gegen die Trackbars)
        is_valid_size = (w >= min_w and h >= min_h and 
                         w <= max_w and h <= max_h and 
                         min_ar <= aspect_ratio <= max_ar)
        
        if is_valid_size:
            square_contours.append(c)
            # Grüne Box für GÜLTIGE Objekte - cY wird angezeigt
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"W:{w} H:{h} cY:{cy}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Rote Box für UNGÜLTIGE Objekte - cY wird angezeigt
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(display_frame, f"W:{w} H:{h} cY:{cy}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 4. Den größten gültigen Kandidaten croppen und anzeigen
    if square_contours:
        largest_contour = max(square_contours, key=cv2.contourArea)
        x_b, y_b, w_b, h_b = cv2.boundingRect(largest_contour)
        cy_b = y_b + (h_b // 2) # Y-Zentrum des Zielobjekts
        
        letter_crop = gray_frame[y_b:y_b+h_b, x_b:x_b+w_b]
        
        # Exakte Padding-Logik aus deinem Hauptcode
        FILL_RATIO = 0.7
        max_dim = max(w_b, h_b)
        target_dim = int(max_dim / FILL_RATIO)
        pad_top = (target_dim - h_b) // 2
        pad_bottom = target_dim - h_b - pad_top
        pad_left = (target_dim - w_b) // 2
        pad_right = target_dim - w_b - pad_left
        bg_color = int(np.median(gray_frame[0:10, 0:10]))
        padded_img = cv2.copyMakeBorder(letter_crop, pad_top, pad_bottom, pad_left, pad_right, 
                                        cv2.BORDER_CONSTANT, value=bg_color)
        
        display_crop = cv2.resize(padded_img, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Crop Ansicht (Das sieht die KI)", display_crop)
        
        frame_counter += 1
        if frame_counter % 15 == 0:
            print(f"✅ ZIEL GEFUNDEN | B: {w_b}px | H: {h_b}px | cY (Höhe): {cy_b}px | Aspect Ratio: {(w_b/h_b):.2f}")
    else:
        blank = np.zeros((200, 200), dtype=np.uint8)
        cv2.putText(blank, "Kein Ziel", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)
        cv2.imshow("Crop Ansicht (Das sieht die KI)", blank)

    cv2.imshow("Size Calibration", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("KALIBRIERUNG BEENDET. HIER IST DEIN NEUER CODE:")
print("Tausche die Größenüberprüfung in deinem Hauptcode hiermit aus:\n")

print(f"if w_tmp < {min_w} or h_tmp < {min_h} or w_tmp > {max_w} or h_tmp > {max_h}: continue")
print(f"aspect_ratio = w_tmp / float(h_tmp)")
print(f"if {min_ar} <= aspect_ratio <= {max_ar}:")
print("    square_contours.append(c)")
print("="*50 + "\n")