import cv2
import numpy as np
import threading
import time
import os
import sys
import serial
import math
from collections import Counter
from gpiozero import DigitalOutputDevice

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
    print("Fehler: Picamera2 fehlt. Bitte mit 'sudo apt install python3-picamera2' installieren.")
    sys.exit()

# ==========================================
# 2. KONFIGURATION
# ==========================================
MODEL_PATH = "trainedEdgeClean.tflite" 
LABEL_PATH = "labels.txt"
MIN_CONFIDENCE = 0.6
SERIAL_PORT = '/dev/ttyAMA0'  # Ggf. anpassen auf /dev/ttyUSB0
BAUD_RATE = 115200
TRIGGER_PIN = 17              # Gemeinsamer Pin für beide Kameras

# Globaler Pin (Shared Resource)
output_pin = DigitalOutputDevice(TRIGGER_PIN)

# Serial Setup
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Erfolgreich mit Serial {SERIAL_PORT} verbunden.")
except Exception as e:
    print(f"Serial nicht gefunden ({e}) -> Simulationsmodus aktiv.")
    ser = None

def load_labels(path):
    if os.path.exists(path):
        with open(path, 'r') as f: 
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "background", 1: "H", 2: "S", 3: "U"}

LABELS = load_labels(LABEL_PATH)

# ==========================================
# 3. SERIAL HELFER
# ==========================================
def SerialWrite(obj, camside=None):
    """Sendet Nachrichten im Format <OK> oder <LH>, <RS> etc. an den Arduino."""
    if ser:
        msg = f"<{camside}{obj}>\n" if camside else f"<{obj}>\n"
        ser.write(msg.encode('utf-8'))
        print(f"[SERIAL] Gesendet: {msg.strip()}")

# ==========================================
# 4. GEOMETRIE- & FARBFUNKTIONEN (CIRCLE DETECTION)
# ==========================================
def order_points(pts):
    """Sortiert 4 Koordinaten für das korrekte Entzerren (Warping)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_target_corners(image_bgr):
    cutoff_top_y = int(image_bgr.shape[0] * 0.25)
    cutoff_bottom_y = int(image_bgr.shape[0] * 0.875)
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Ring in toter Zone ignorieren
        if y < cutoff_top_y or (y + h) > cutoff_bottom_y: 
            continue
            
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
    """Generiert eine perfekt zentrierte, flache Aufsicht des Targets."""
    dst_points = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    return cv2.warpPerspective(image_bgr, matrix, (output_size, output_size))

def classify_color(hsv_pixel):
    """Ordnet HSV-Pixel vordefinierten Farbräumen zu (inkl. Glare-Filter)."""
    h, s, v = hsv_pixel
    if v < 60: return "Black"
    if s < 50 and v > 200: return "White" 

    if h < 10 or h > 160: return "Red"
    elif 20 < h < 35: return "Yellow"
    elif 40 < h < 80: return "Green"
    elif 90 < h < 130: return "Blue"
    return "Unknown"

def scan_target_colors(warped_image_bgr):
    """Führt den 12-Speichen Stern-Scan auf den 5 Ring-Radien durch."""
    hsv_image = cv2.cvtColor(warped_image_bgr, cv2.COLOR_BGR2HSV)
    center = (100, 100)
    radii = [10, 30, 50, 70, 90]
    final_colors = []

    for r in radii:
        ring_colors = []
        for angle_deg in range(0, 360, 30):
            angle_rad = math.radians(angle_deg)
            x = int(center[0] + r * math.cos(angle_rad))
            y = int(center[1] + r * math.sin(angle_rad))
            
            x = max(0, min(199, x))
            y = max(0, min(199, y))
            
            color_name = classify_color(hsv_image[y, x])
            if color_name != "White" and color_name != "Unknown":
                ring_colors.append(color_name)
        
        if ring_colors:
            most_common_color = Counter(ring_colors).most_common(1)[0][0]
            final_colors.append(most_common_color)
        else:
            final_colors.append("Unknown")
    return final_colors

def calculate_victim_health(colors):
    """Berechnet den Zustand der Ring-Opfer laut Regelwerk."""
    color_values = {"Yellow": 0, "Blue": 2, "Red": -1, "Black": -2, "Green": 1}
    total_sum = 0
    for color in colors:
        total_sum += color_values.get(color, 0)
        
    status = "Fake"
    if total_sum == 0: status = "U"   # Unharmed
    elif total_sum == 1: status = "S" # Stable
    elif total_sum == 2: status = "H" # Harmed
    return status, total_sum

# ==========================================
# 5. MULTITHREADED KAMERA AI + CIRCLE KLASSE
# ==========================================
class CameraAIThread(threading.Thread):
    def __init__(self, cam_id, side_code):
        super().__init__()
        self.cam_id = cam_id
        self.side_code = side_code
        self.enabled = False
        self.running = True
        
        # Sicherheits-Zähler
        self.Counter_Harmed = 0
        self.Counter_Safe = 0
        self.Counter_Unharmed = 0
        self.frame_counter = 0
        
        # Watchdog
        self.last_detection_time = 0.0
        self.TIMEOUT_DURATION = 3.0
        
        # ==========================================
        # TFLite Modell laden (REINER CPU MODUS)
        # ==========================================
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.ki_h, self.ki_w = self.input_details[0]['shape'][1:3]
            self.is_int8 = (self.input_details[0]['dtype'] in [np.int8, np.uint8])
            self.ready = True
            print(f"Cam {self.side_code} CPU Modell erfolgreich geladen.")
            
        except Exception as e:
            print(f"Cam {self.side_code} TFLite-Fehler: {e}")
            self.ready = False

    def reset_logic(self):
        self.Counter_Harmed = self.Counter_Safe = self.Counter_Unharmed = 0
        self.frame_counter = 0
        output_pin.off()

    def run(self):
        if not self.ready: return
        
        try:
            picam2 = Picamera2(self.cam_id)
            config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
        except Exception as e:
            print(f"Hardware-Fehler Cam {self.side_code}: {e}")
            return

        print(f"Thread {self.side_code} (Cam {self.cam_id}) aktiv und bereit.")

        while self.running:
            if not self.enabled:
                time.sleep(0.1)
                continue

            frame_rgb = picam2.capture_array()
            # 180 Grad Drehung für Hardwareausgleich
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # --- Visualisierungsvorbereitung (Nur für Rechte Kamera) ---
            if self.side_code == "R":
                display_frame = frame_bgr.copy()
            
            # Sperrzonen berechnen
            cutoff_top_y = int(frame_rgb.shape[0] * 0.25)
            cutoff_bottom_y = int(frame_rgb.shape[0] * 0.875)
            
            if self.side_code == "R":
                cv2.line(display_frame, (0, cutoff_top_y), (frame_bgr.shape[1], cutoff_top_y), (0, 255, 255), 2)
                cv2.line(display_frame, (0, cutoff_bottom_y), (frame_bgr.shape[1], cutoff_bottom_y), (0, 255, 255), 2)
            
            detected_frame_label = None
            box_coords = None
            detected_corners = None

            # --- ERKENNUNG 1: TFLite Buchstaben (Gefiltert) ---
            gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            total_area = gray_frame.shape[0] * gray_frame.shape[1]
            
            blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 21, 5)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < total_area * 0.99]

            square_contours = []
            for c in valid_contours:
                x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(c)
                if h_tmp == 0: continue 
                
                # 1. Tote Zonen (Sperrzonen oben und unten)
                if y_tmp < cutoff_top_y or (y_tmp + h_tmp) > cutoff_bottom_y:
                    continue
                    
                # 2. Absolute Pixel-Grenzen (mindestens 60px, maximal 360px)
                if w_tmp < 60 or h_tmp < 60:
                    continue
                if w_tmp > 360 or h_tmp > 360:
                    continue
                    
                # 3. Aspekt-Ratio (Rechteckigkeit)
                aspect_ratio = w_tmp / float(h_tmp)
                if 0.5 <= aspect_ratio <= 2.0:
                    square_contours.append(c)

            # --- KI-INFERENZ (Wird NUR gestartet, wenn alles gepasst hat) ---
            if square_contours:
                largest_contour = max(square_contours, key=cv2.contourArea)
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
                prep_img = cv2.resize(padded_img, (self.ki_w, self.ki_h), interpolation=cv2.INTER_AREA)

                # Vorbereitung für den Interpreter
                prep_img_expanded = np.expand_dims(prep_img, axis=-1)
                input_data = np.expand_dims(prep_img_expanded, axis=0)

                if self.is_int8:
                    scale, zero_point = self.input_details[0]['quantization']
                    input_data = (input_data.astype(np.float32) / scale + zero_point).astype(np.int8)
                else:
                    input_data = input_data.astype(np.float32)

                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                if self.is_int8:
                    out_scale, out_zero_point = self.output_details[0]['quantization']
                    scores = (output_data.astype(np.float32) - out_zero_point) * out_scale
                else:
                    scores = output_data

                best_class_id = np.argmax(scores)
                confidence = scores[best_class_id]

                if confidence > MIN_CONFIDENCE:
                    label_str = LABELS.get(best_class_id)
                    if label_str and label_str.lower() != "background":
                        detected_frame_label = label_str

            # --- ERKENNUNG 2: Farbringe (falls kein Buchstabe Vorrang hatte) ---
            if not detected_frame_label:
                detected_corners = find_target_corners(frame_bgr)
                if detected_corners is not None:
                    warped = warp_target(frame_bgr, detected_corners)
                    colors = scan_target_colors(warped)
                    circle_status, _ = calculate_victim_health(colors)
                    
                    if circle_status in ["H", "S", "U"]:
                        detected_frame_label = circle_status

            # --- STREAM FÜR RECHTE KAMERA AUSGEBEN ---
            if self.side_code == "R":
                if detected_frame_label:
                    color = (0, 255, 0)
                    cv2.putText(display_frame, f"Erkannt: {detected_frame_label}", (10, cutoff_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Rahmen zeichnen, je nachdem was gefunden wurde
                    if box_coords:
                        bx, by, bw, bh = box_coords
                        cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), color, 2)
                    elif detected_corners is not None:
                        pts = detected_corners.reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(display_frame, [pts], True, (255, 165, 0), 3)
                else:
                    cv2.putText(display_frame, "Suche...", (10, cutoff_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                cv2.imshow("Kamera Rechts", display_frame)
                cv2.waitKey(1)

            # --- FILTERUNG, ABSICHERUNG & WATCHDOG ---
            if detected_frame_label:
                # Erfolgreicher Fund
                self.last_detection_time = time.time()
                self.frame_counter += 1
                
                # Hardware-Pin HIGH
                if self.frame_counter == 1:
                    output_pin.on()
                    print(f"[{self.side_code}] Target gesichtet! Pin HIGH.")

                # Zähler erhöhen
                if detected_frame_label == "H": self.Counter_Harmed += 1
                elif detected_frame_label == "S": self.Counter_Safe += 1
                elif detected_frame_label == "U": self.Counter_Unharmed += 1

            else:
                # Kein Fund in diesem Frame: Watchdog prüfen
                if self.frame_counter > 0:
                    verstrichene_zeit = time.time() - self.last_detection_time
                    if verstrichene_zeit > self.TIMEOUT_DURATION:
                        print(f"[{self.side_code}] Watchdog: 3s ohne Kontakt. Daten verworfen, Pin LOW.")
                        self.reset_logic()

            # --- ERGEBNIS ÜBERTRAGEN ---
            if self.frame_counter >= 5:
                counts = {'H': self.Counter_Harmed, 'S': self.Counter_Safe, 'U': self.Counter_Unharmed}
                cam_transmit = max(counts, key=counts.get)
                
                SerialWrite(cam_transmit, self.side_code)
                
                output_pin.off()
                print(f"[{self.side_code}] Transfer abgeschlossen. Pin LOW.")
                self.reset_logic()
                self.enabled = False 

        picam2.stop()
        if self.side_code == "R":
            cv2.destroyWindow("Kamera Rechts")

# ==========================================
# 6. MAIN-STEUERUNG: PROTOKOLL LISTENER
# ==========================================
cam_left = CameraAIThread(0, "L")
cam_right = CameraAIThread(1, "R")
cam_left.start()
cam_right.start()

print("Warte auf Befehle vom Arduino...")

try:
    buffer = ""
    while True:
        if ser and ser.in_waiting > 0:
            char = ser.read().decode('utf-8', errors='ignore')
            buffer += char
            
            if ">" in buffer:
                start = buffer.find("<")
                end = buffer.find(">")
                if start != -1 and end > start:
                    cmd = buffer[start+1:end]
                    
                    if cmd == "I":
                        SerialWrite("OK")
                    elif cmd == "E":
                        cam_left.enabled = True
                        cam_right.enabled = True
                        SerialWrite("OK")
                    elif cmd == "RE":
                        cam_right.enabled = True
                        SerialWrite("OK")
                    elif cmd == "D":
                        cam_left.enabled = False
                        cam_left.reset_logic()
                        cam_right.enabled = False
                        cam_right.reset_logic()
                        SerialWrite("OK")
                    elif cmd == "RD":
                        cam_right.enabled = False
                        cam_right.reset_logic()
                        SerialWrite("OK")
                
                buffer = "" 
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("System herunterfahren...")
    cam_left.running = cam_right.running = False
    cam_left.join()
    cam_right.join()
    output_pin.off()