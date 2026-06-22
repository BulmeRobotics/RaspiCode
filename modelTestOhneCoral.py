import cv2
import numpy as np
import time
import os
import sys
import math
from collections import Counter

# ==========================================
# 1. INITIALISIERUNG & IMPORT-BLOCK
# ==========================================
try:
    import ai_edge_litert.interpreter as tflite
    print("LiteRT erfolgreich geladen.")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Klassisches TFLite geladen.")
    except ImportError:
        print("Fehler: Weder LiteRT noch TFLite gefunden!")
        sys.exit()

try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt. Bitte installieren.")
    sys.exit()

# ==========================================
# 2. KONFIGURATION
# ==========================================
# WICHTIG: Nutze hier dein normales Modell OHNE "_edgetpu" im Namen!
MODEL_PATH = "trainedEdgeClean.tflite" 
LABEL_PATH = "labels.txt"
MIN_CONFIDENCE = 0.6

def load_labels(path):
    if os.path.exists(path):
        with open(path, 'r') as f: 
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "background", 1: "H", 2: "S", 3: "U"}

LABELS = load_labels(LABEL_PATH)

# ==========================================
# 3. GEOMETRIE- & FARBFUNKTIONEN (CIRCLE FALLBACK)
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_target_corners(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 
            
            width = rect[1][0]
            height = rect[1][1]
            if height == 0: continue
            aspect_ratio = width / height
            
            if 0.7 <= aspect_ratio <= 1.3:
                return order_points(box)
    return None

def warp_target(image_bgr, corners, output_size=200):
    dst_points = np.array([
        [0, 0], [output_size - 1, 0],
        [output_size - 1, output_size - 1], [0, output_size - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    return cv2.warpPerspective(image_bgr, matrix, (output_size, output_size))

def classify_color(hsv_pixel):
    h, s, v = hsv_pixel
    if v < 60: return "Black"
    if s < 50 and v > 200: return "White" 
    if h < 10 or h > 160: return "Red"
    elif 20 < h < 35: return "Yellow"
    elif 40 < h < 80: return "Green"
    elif 90 < h < 130: return "Blue"
    return "Unknown"

def scan_target_colors(warped_image_bgr):
    hsv_image = cv2.cvtColor(warped_image_bgr, cv2.COLOR_BGR2HSV)
    center = (100, 100)
    radii = [10, 30, 50, 70, 90]
    final_colors = []

    for r in radii:
        ring_colors = []
        for angle_deg in range(0, 360, 30):
            angle_rad = math.radians(angle_deg)
            x = max(0, min(199, int(center[0] + r * math.cos(angle_rad))))
            y = max(0, min(199, int(center[1] + r * math.sin(angle_rad))))
            
            color_name = classify_color(hsv_image[y, x])
            if color_name not in ["White", "Unknown"]:
                ring_colors.append(color_name)
        
        if ring_colors:
            final_colors.append(Counter(ring_colors).most_common(1)[0][0])
        else:
            final_colors.append("Unknown")
    return final_colors

def calculate_victim_health(colors):
    color_values = {"Yellow": 0, "Blue": 2, "Red": -1, "Black": -2, "Green": 1}
    total_sum = sum(color_values.get(c, 0) for c in colors)
    if total_sum == 0: return "U"
    elif total_sum == 1: return "S"
    elif total_sum == 2: return "H"
    return "Fake"

# ==========================================
# 4. TENSORFLOW LITE INTERPRETER STARTEN (NUR CPU)
# ==========================================
print("Lade TFLite Modell (Reiner CPU-Modus)...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ki_h, ki_w = input_details[0]['shape'][1:3]
    is_int8 = (input_details[0]['dtype'] in [np.int8, np.uint8])
    print(f"Modell geladen. Erwartet {ki_w}x{ki_h} Pixel. INT8: {is_int8}")
    
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    sys.exit()

# ==========================================
# 5. KAMERA STARTEN
# ==========================================
print("Starte Picamera2...")
try:
    picam2 = Picamera2(0) # Kamera 0
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
except Exception as e:
    print(f"Kamerafehler: {e}")
    sys.exit()

print("\nKamera läuft! Beenden mit Taste 'q' im Videofenster.")

# ==========================================
# 6. HAUPTSCHLEIFE (LIVE-STREAM)
# ==========================================
while True:
    # 1. Frame holen
    frame_rgb = picam2.capture_array()
    
    # --- NEU: HARDWARE-AUSRICHTUNG KORRIGIEREN ---
    # Das Bild wird physisch um 180 Grad gedreht, um die Kameramontage auszugleichen.
    # Dies muss vor allen anderen Konvertierungen passieren!
    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
    
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    display_frame = frame_bgr.copy()
    
    detected_label = "Nichts gefunden"
    confidence = 0.0
    color = (0, 0, 255) # Rot als Standard
    box_coords = None

    # --- PIPELINE 1: BUCHSTABEN-ERKENNUNG (Zweistufig) ---
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    total_area = gray_frame.shape[0] * gray_frame.shape[1]
    
    blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    cv2.imshow("Debug: Adaptive Maske", cv2.resize(thresh, (320, 240)))
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < total_area * 0.99]

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x_b, y_b, w_b, h_b = cv2.boundingRect(largest_contour)
        box_coords = (x_b, y_b, w_b, h_b)
        
        letter_crop = gray_frame[y_b:y_b+h_b, x_b:x_b+w_b]
        
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
        prep_img = cv2.resize(padded_img, (ki_w, ki_h), interpolation=cv2.INTER_AREA)
    else:
        prep_img = cv2.resize(gray_frame, (ki_w, ki_h))

    cv2.imshow("Das sieht die KI (96x96)", cv2.resize(prep_img, (200, 200), interpolation=cv2.INTER_NEAREST))

    # TFLite Inferenz (Auf der CPU)
    prep_img_expanded = np.expand_dims(prep_img, axis=-1)
    input_data = np.expand_dims(prep_img_expanded, axis=0)

    if is_int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data.astype(np.float32) / scale + zero_point).astype(np.int8)
    else:
        input_data = input_data.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    if is_int8:
        out_scale, out_zero_point = output_details[0]['quantization']
        scores = (output_data.astype(np.float32) - out_zero_point) * out_scale
    else:
        scores = output_data

    best_class_id = np.argmax(scores)
    confidence = scores[best_class_id]

    if confidence > MIN_CONFIDENCE:
        label_str = LABELS.get(best_class_id)
        if label_str and label_str.lower() != "background":
            detected_label = f"Buchstabe: {label_str} ({confidence*100:.0f}%)"
            color = (0, 255, 0) # Grün
            
            if box_coords:
                bx, by, bw, bh = box_coords
                cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), color, 2)

    # --- PIPELINE 2: FARBRINGE (Fallback) ---
    if detected_label == "Nichts gefunden":
        corners = find_target_corners(frame_bgr)
        if corners is not None:
            pts = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(display_frame, [pts], True, (255, 165, 0), 3)
            
            warped = warp_target(frame_bgr, corners)
            colors = scan_target_colors(warped)
            circle_status = calculate_victim_health(colors)
            
            if circle_status in ["H", "S", "U"]:
                detected_label = f"Ring: {circle_status}"
                color = (255, 165, 0)
                cv2.imshow("Debug: Warped Ring", warped)

    # --- ERGEBNIS ANZEIGEN ---
    cv2.putText(display_frame, detected_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Live-Kamera Pi 5", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aufräumen
picam2.stop()
cv2.destroyAllWindows()
print("Live-Test beendet.")