"""
Isoliertes Debug-Skript für Cognitive Targets ('C') und die Farbauswertung.
Führt KI-Inferenz und Ecken-Filterung auf einem Live-Stream aus und visualisiert die Zwischenschritte.
"""

import cv2
import numpy as np
import math
import collections
import time
import os
import sys

# ==========================================
# 0. HARDWARE & KI IMPORTS
# ==========================================
try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt.")
    sys.exit()

try:
    import ai_edge_litert.interpreter as tflite
    print("LiteRT erfolgreich geladen.")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Klassisches TFLite geladen.")
    except ImportError:
        print("Fehler: TFLite nicht gefunden!")
        sys.exit()

# ==========================================
# 1. KONFIGURATION
# ==========================================
CAMERA_ID = 0           # 0 für Right, 1 für Left
SIDE_CODE = "R"         # "R" oder "L" (Wichtig für die Auswurf-Grenzen)
ROTATE_180 = True       # Bild um 180 Grad drehen?

MODEL_PATH = "trainedWMWhiteBackground.tflite"
LABEL_PATH = "labelsCog.txt"
MIN_CONFIDENCE_C = 0.8

# ==========================================
# 2. HILFSFUNKTIONEN (Farbe & Geometrie)
# ==========================================
def load_labels(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "C", 1: "H", 2: "S", 3: "U"}

LABELS = load_labels(LABEL_PATH)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_target(image_bgr, corners, output_size=200):
    dst_points = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    return cv2.warpPerspective(image_bgr, matrix, (output_size, output_size))

def classify_color(hsv_pixel):
    h, s, v = hsv_pixel
    if v < 80:
        return "Black"
    if s < 50 and v > 180:
        return "White"
    if h < 12 or h > 100:
        return "Red"
    elif 69 < h < 103:
        return "Yellow"
    elif 40 < h < 70:
        return "Green"
    elif 10 < h < 30:
        return "Blue"
    return "Unknown"

def is_valid_background_crop(hsv_image):
    """Debug-fähiger Eckenfilter mit Konsolenausgabe."""
    background_samples = []
    corner_clusters = [
        (5, 5), (10, 5), (5, 10),
        (194, 5), (189, 5), (194, 10),
        (5, 194), (10, 194), (5, 189),
        (194, 194), (189, 194), (194, 189)
    ]
    
    for cx, cy in corner_clusters:
        background_samples.append(classify_color(hsv_image[cy, cx]))

    white_count = background_samples.count("White")
    white_ratio = white_count / len(background_samples)
    
    print(f"    [ECKEN-CHECK] Weißanteil: {white_ratio*100:.0f}% ({white_count}/12 Pixel)")
    
    if white_ratio > 0.65:
        return True
    else:
        print(f"    [ECKEN-CHECK] FEHLGESCHLAGEN! Gefundene Farben: {collections.Counter(background_samples)}")
        return False

def scan_target_colors(warped_image_bgr):
    hsv_image = cv2.cvtColor(warped_image_bgr, cv2.COLOR_BGR2HSV)

    if not is_valid_background_crop(hsv_image): 
        return None
    
    center = (100, 100)
    radii = [10, 30, 50, 70, 90]
    final_colors = []

    for r in radii:
        ring_colors = []
        for angle_deg in range(0, 360, 30):
            angle_rad = math.radians(angle_deg)
            x = int(center[0] + r * math.cos(angle_rad))
            y = int(center[1] + r * math.sin(angle_rad))
            x, y = max(0, min(199, x)), max(0, min(199, y))

            color_name = classify_color(hsv_image[y, x])
            if color_name not in ["White", "Unknown"]:
                ring_colors.append(color_name)

        if ring_colors:
            most_common = collections.Counter(ring_colors).most_common(1)[0][0]
            final_colors.append(most_common)
        else:
            final_colors.append("Unknown")
            
    print(f"    [FARB-SCAN] Extrahierte Ringe (innen nach außen): {final_colors}")
    return final_colors

def calculate_victim_health(colors):
    color_values = {"Yellow": 0, "Blue": 2, "Red": -1, "Black": -2, "Green": 1}
    total_sum = sum(color_values.get(c, 0) for c in colors)
    yellow_count = colors.count("Yellow")

    status = "F"
    if total_sum == 0: status = "U"
    elif total_sum == 1: status = "S"
    elif total_sum == 2: status = "H"
    elif total_sum > 2: status = "F"
    
    if yellow_count >= 5: status = "F"
    return status, total_sum

# ==========================================
# 3. GRENZ- UND KONTUREN-BERECHNUNG
# ==========================================
def get_boundary_coords(img_shape, side):
    height, width = img_shape[:2]
    return {
        "top_y": int(height * 0.25),
        "bottom_y": int(height * 0.85),
        "left_x": int(width * (1 / 7)),
        "right_x": int(width * (6 / 7)),
        "left_auswurf_bottom": [260, 478] if side == "R" else [300, 478],
        "left_auswurf_top": [0, 202] if side == "R" else [0, 193],
        "right_auswurf_bottom": [320, 478] if side == "R" else [377, 478],
        "right_auswurf_top": [636, 185] if side == "R" else [636, 210],
    }

def is_in_boundary(cx, cy, img_shape, side):
    coords = get_boundary_coords(img_shape, side)
    if cy < coords["top_y"] or cy > coords["bottom_y"]: return False
    if cx < coords["left_x"] or cx > coords["right_x"]: return False

    lt, lb = coords["left_auswurf_top"], coords["left_auswurf_bottom"]
    if cy > lt[1] and cx < lb[0]:
        line_y = lt[1] + (cx - lt[0]) * ((lb[1] - lt[1]) / (lb[0] - lt[0]))
        if line_y < cy: return False

    rt, rb = coords["right_auswurf_top"], coords["right_auswurf_bottom"]
    if cx > rb[0] and cy > rt[1]:
        line_y = rt[1] + (cx - rt[0]) * ((rb[1] - rt[1]) / (rb[0] - rt[0]))
        if line_y < cy: return False

    return True

def find_target_corners(image_bgr, side):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    coords = get_boundary_coords(image_bgr.shape, side)
    pts_left = np.array([
        coords["left_auswurf_top"], coords["left_auswurf_bottom"],
        [coords["left_auswurf_bottom"][0], image_bgr.shape[0]], [0, image_bgr.shape[0]]
    ], dtype=np.int32)
    pts_right = np.array([
        coords["right_auswurf_bottom"], coords["right_auswurf_top"],
        [image_bgr.shape[1], coords["right_auswurf_top"][1]],
        [image_bgr.shape[1], image_bgr.shape[0]],
        [coords["right_auswurf_bottom"][0], image_bgr.shape[0]]
    ], dtype=np.int32)
    
    cv2.fillPoly(edges, [pts_left], 0)
    cv2.fillPoly(edges, [pts_right], 0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + (w // 2), y + (h // 2)

        if not is_in_boundary(cx, cy, image_bgr.shape, side):
            continue

        if cv2.contourArea(cnt) > 1000:
            rect = cv2.minAreaRect(cnt)
            box = np.int32(cv2.boxPoints(rect))
            width, height = rect[1][0], rect[1][1]
            if height == 0: continue
            aspect_ratio = width / height
            if 0.85 <= aspect_ratio <= 1.15:
                return order_points(box)
    return None


# ==========================================
# 4. MAIN DEBUG LOOP
# ==========================================
def main_debug_loop():
    print(f"\n--- STARTE COGNITIVE TARGET DEBUGGER (Kamera {SIDE_CODE}) ---")
    
    # 1. Modell laden
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        ki_h, ki_w = input_details[0]["shape"][1:3]
        is_int8 = input_details[0]["dtype"] in [np.int8, np.uint8]
        print("[OK] TFLite Modell erfolgreich geladen.")
    except Exception as e:
        print(f"[FEHLER] Modell konnte nicht geladen werden: {e}")
        return

    # 2. Kamera initialisieren
    try:
        picam2 = Picamera2(CAMERA_ID)
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("[OK] Picamera2 erfolgreich gestartet.")
    except Exception as e:
        print(f"[FEHLER] Kamera Zugriff verweigert: {e}")
        return

    cv2.namedWindow("Main Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Threshold Debug", cv2.WINDOW_AUTOSIZE)

    print("\nDrücke 'q' im Bildfenster um zu beenden.")
    print("Suche nach Zielen...\n")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            if ROTATE_180:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
            
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            display_frame = frame_bgr.copy()
            
            # --- BILDVERARBEITUNG & THRESHOLD ---
            gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            
            # Auswurf-Grenzen ausmaskieren (für Threshold-Ansicht)
            coords = get_boundary_coords(frame_rgb.shape, SIDE_CODE)
            pts_left = np.array([
                coords["left_auswurf_top"], coords["left_auswurf_bottom"],
                [coords["left_auswurf_bottom"][0], 480], [0, 480]
            ], dtype=np.int32)
            pts_right = np.array([
                coords["right_auswurf_bottom"], coords["right_auswurf_top"],
                [640, coords["right_auswurf_top"][1]], [640, 480],
                [coords["right_auswurf_bottom"][0], 480]
            ], dtype=np.int32)
            
            cv2.fillPoly(thresh, [pts_left], 0)
            cv2.fillPoly(thresh, [pts_right], 0)
            
            cv2.imshow("Threshold Debug", thresh)

            # --- KONTUREN & KI INFERENZ ---
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < (640*480) * 0.99]
            
            square_contours = []
            for c in valid_contours:
                x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(c)
                if h_tmp == 0: continue
                cx_tmp, cy_tmp = x_tmp + (w_tmp // 2), y_tmp + (h_tmp // 2)
                
                if not is_in_boundary(cx_tmp, cy_tmp, gray_frame.shape, SIDE_CODE): continue
                if w_tmp < 100 or h_tmp < 100 or w_tmp > 360 or h_tmp > 360: continue
                
                if 0.85 <= (w_tmp / float(h_tmp)) <= 1.15:
                    square_contours.append(c)

            if square_contours:
                largest_contour = max(square_contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(largest_contour)
                
                cv2.rectangle(display_frame, (x_b, y_b), (x_b + w_b, y_b + h_b), (255, 0, 0), 2)
                
                letter_crop = gray_frame[y_b:y_b + h_b, x_b:x_b + w_b]
                fill_ratio = 0.7
                max_dim = max(w_b, h_b)
                target_dim = int(max_dim / fill_ratio)
                pad_t = (target_dim - h_b) // 2
                pad_b = target_dim - h_b - pad_t
                pad_l = (target_dim - w_b) // 2
                pad_r = target_dim - w_b - pad_l
                
                bg_color = int(np.median(gray_frame[0:10, 0:10]))
                padded_img = cv2.copyMakeBorder(letter_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=bg_color)
                prep_img = cv2.resize(padded_img, (ki_w, ki_h), interpolation=cv2.INTER_AREA)

                input_data = np.expand_dims(np.expand_dims(prep_img, axis=-1), axis=0)
                
                if is_int8:
                    scale, zero_point = input_details[0]["quantization"]
                    input_data = (input_data.astype(np.float32) / scale + zero_point).astype(np.int8)
                else:
                    input_data = input_data.astype(np.float32)

                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])[0]

                if is_int8:
                    out_scale, out_zero_point = output_details[0]["quantization"]
                    scores = (output_data.astype(np.float32) - out_zero_point) * out_scale
                else:
                    scores = output_data

                best_class_id = np.argmax(scores)
                confidence = scores[best_class_id]
                label_str = LABELS.get(best_class_id, "")

                if confidence > MIN_CONFIDENCE_C and label_str.lower() != "background":
                    cv2.putText(display_frame, f"{label_str} ({confidence:.2f})", (x_b, y_b - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # --- FARBAUSWERTUNG AUSLÖSEN WENN 'C' ERKANNT ---
                    if label_str.upper() == "C":
                        print(f"\n[EVENT] Kognitives Ziel 'C' erkannt! (Confidence: {confidence:.2f})")
                        
                        detected_corners = find_target_corners(frame_bgr, SIDE_CODE)
                        if detected_corners is not None:
                            # Polygon auf dem Stream einzeichnen
                            pts = detected_corners.reshape((-1, 1, 2)).astype(np.int32)
                            cv2.polylines(display_frame, [pts], True, (0, 165, 255), 3)
                            
                            # Bild warpen und extrahiertes Bild anzeigen
                            warped = warp_target(frame_bgr, detected_corners)
                            cv2.imshow("Warped Target (200x200)", warped)
                            
                            # Den Evaluierungsprozess durchlaufen
                            colors = scan_target_colors(warped)
                            
                            if colors is not None:
                                status, points = calculate_victim_health(colors)
                                print(f"    [ERGEBNIS] Target valide! Status: {status} (Punkte: {points})")
                            else:
                                print("    [ERGEBNIS] Target vom Ecken-Filter verworfen. Warte auf besseren Crop...")
                        else:
                            print("    [WARNUNG] KI sah ein 'C', aber Canny-Kantenerkennung fand kein sauberes Viereck.")
                            
                        # Kurze Pause einlegen, damit die Konsolenausgabe lesbar bleibt,
                        # aber kurz genug, dass der Stream nicht zu sehr stottert.
                        cv2.waitKey(200) 

            # Auswurfgrenzen in den Main Stream zeichnen (als Overlay)
            cv2.line(display_frame, (coords["left_x"], 0), (coords["left_x"], 480), (0, 0, 255), 1)
            cv2.line(display_frame, (coords["right_x"], 0), (coords["right_x"], 480), (0, 0, 255), 1)
            cv2.imshow("Main Stream", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\nBeende Debug-Sitzung...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_debug_loop()