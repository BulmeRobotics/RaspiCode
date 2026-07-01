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

# --- ToF Modul importieren ---
try:
    import tof
except ImportError:
    print("Warnung: tof.py nicht gefunden. ToF-Logik wird übersprungen.")
    tof = None

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
MODEL_PATH = "trainedCog.tflite" 
LABEL_PATH = "labelsCog.txt"
MIN_CONFIDENCE = 0.7
MIN_CONFIDENCE_C = 0.4  # Lower threshold for cognitive target 'C'
SERIAL_PORT = '/dev/ttyAMA0'    #Serialler Port zum Arduino
BAUD_RATE = 115200

# --- PINS DEFINIEREN ---
TRIGGER_PIN = 17       # ALERT Pin              CM5 GPIO17 -> Giga 31
ACTIVE_PIN_L = 27      # Status Pin Kamera L    CM5 GPIO27 -> Giga 29
ACTIVE_PIN_R = 22      # Status Pin Kamera R    CM5 GPIO22 -> Giga 28

# --- STREAM CONFIGURATION (Defines) ---
SHOW_STREAM_L = False
SHOW_STREAM_R = False

# GPIO Objekte erstellen
output_pin = DigitalOutputDevice(TRIGGER_PIN)

#labels laden
def load_labels(path):
    if os.path.exists(path):
        with open(path, 'r') as f: 
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "C", 1: "H", 2: "S", 3: "U"}

LABELS = load_labels(LABEL_PATH)

# ==========================================
# 3. SERIAL MANAGER KLASSE
# ==========================================
class SerialManager:
    """Kapselt die serielle Kommunikation und das Buffer-Management."""
    def __init__(self, port, baud_rate):
        self.buffer = ""
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=0.1)
            print(f"Erfolgreich mit Serial {port} verbunden.")
        except Exception as e:
            print(f"Serial nicht gefunden ({e}) -> Simulationsmodus aktiv.")
            self.ser = None

    def write(self, obj, camside=None):
        if self.ser:
            msg = f"<{camside}{obj}>\n" if camside else f"<{obj}>\n"
            self.ser.write(msg.encode('utf-8'))
            print(f"[SERIAL] Gesendet: {msg.strip()}")

    def read_commands(self):
        """Liest den Puffer aus und extrahiert alle vollständigen <Befehle>."""
        commands = []
        if self.ser and self.ser.in_waiting > 0:
            chars = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
            self.buffer += chars
            
            while "<" in self.buffer and ">" in self.buffer:
                start = self.buffer.find("<")
                end = self.buffer.find(">", start)
                
                if start != -1 and end != -1:
                    cmd = self.buffer[start+1:end]
                    commands.append(cmd)
                    self.buffer = self.buffer[end+1:]
                else:
                    break
        return commands


# ==========================================
# 4. GEOMETRIE- & FARBFUNKTIONEN
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
    cutoff_top_y = int(image_bgr.shape[0] * 0.125)
    cutoff_bottom_y = int(image_bgr.shape[0] * 0.9375)
    cutoff_left_x = int(image_bgr.shape[1] * (1/7))
    cutoff_right_x = int(image_bgr.shape[1] * (6/7))
    cutoff_left_auswurf_bottom_x = (232)        #Boarder Links Unten X
    cutoff_left_auswurf_bottom_y = (478)        #Boarder Links Unten Y
    cutoff_left_auswurf_top_x = (0)             #Boarder Links Oben X
    cutoff_left_auswurf_top_y = (228)           #Boarder Links Oben Y
    cutoff_right_auswurf_bottom_x = (346)       #Boarder Rechts Unten X
    cutoff_right_auswurf_bottom_y = (476)       #Boarder Rechts Unten Y
    cutoff_right_auswurf_top_x = (636)          #Boarder Rechts Oben X
    cutoff_right_auswurf_top_y = (185)          #Boarder Rechts Oben Y

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # --- NEU: MITTELPUNKT BERECHNEN ---
        cx = x + (w // 2)
        cy = y + (h // 2)
        
        # Prüfen ob der MITTELPUNKT in der legalen Zone liegt
        if cy < cutoff_top_y or cy > cutoff_bottom_y: continue
        if cx < cutoff_left_x or cx > cutoff_right_x: continue

        if (cutoff_left_auswurf_top_x < cx < cutoff_left_auswurf_bottom_x and
            cutoff_left_auswurf_top_y < cy < cutoff_left_auswurf_bottom_y):continue
            
        if (cutoff_right_auswurf_top_x < cx < cutoff_right_auswurf_bottom_x and
            cutoff_right_auswurf_top_y < cy < cutoff_right_auswurf_bottom_y):continue
            

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
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    return cv2.warpPerspective(image_bgr, matrix, (output_size, output_size))

def classify_color(hsv_pixel):
    h, s, v = hsv_pixel
    if v < 80: return "Black"
    if h < 12 or h > 100: return "Red"
    elif 69 < h < 103: return "Yellow"
    elif 40 < h < 70: return "Green"
    elif 10 < h < 30: return "Blue"
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
    color_values = {"Yellow": 0, "Blue": 2, "Red": -1, "Black": -2, "Green": 1}
    total_sum = sum(color_values.get(color, 0) for color in colors)
        
    status = "F"
    if total_sum == 0: status = "U"   
    elif total_sum == 1: status = "S" 
    elif total_sum == 2: status = "H" 
    return status, total_sum


# ==========================================
# 5. MULTITHREADED KAMERA AI THREAD KLASSE
# ==========================================
class CameraAIThread(threading.Thread):
    def __init__(self, cam_id, side_code, status_gpio_pin, serial_mgr, show_stream=False):
        super().__init__()
        self.cam_id = cam_id
        self.side_code = side_code
        self.serial_mgr = serial_mgr
        self.show_stream = show_stream
        
        self.enabled = False
        self.running = True
        self.waiting_for_reset = False
        self._alert_lock = threading.Lock()
        self._alert_active = False
        
        self.status_pin = DigitalOutputDevice(status_gpio_pin)
        self.status_pin.off()
        
        self.Counter_Harmed = 0
        self.Counter_Safe = 0
        self.Counter_Unharmed = 0
        self.frame_counter = 0
        
        self.last_detection_time = 0.0
        self.TIMEOUT_DURATION = 2.0
        self.fps = 0.0
        # ToF debounce
        self.tof_filtered = True
        self._tof_last_raw = None
        self._tof_last_change = 0.0
        self.TOF_DEBOUNCE = 0.2
        
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.ki_h, self.ki_w = self.input_details[0]['shape'][1:3]
            self.is_int8 = (self.input_details[0]['dtype'] in [np.int8, np.uint8])
            self.ready = True
            print(f"Cam {self.side_code} CPU Modell geladen.")
        except Exception as e:
            print(f"Cam {self.side_code} Fehler: {e}")
            self.ready = False

    @property
    def alert_active(self):
        with self._alert_lock:
            return self._alert_active

    @alert_active.setter
    def alert_active(self, value):
        with self._alert_lock:
            self._alert_active = bool(value)

    def reset_logic(self):
        self.Counter_Harmed = self.Counter_Safe = self.Counter_Unharmed = 0
        self.frame_counter = 0

    def evaluate_color_target(self, picam2, samples=7, delay=0.03):
        votes = []
        for i in range(samples):
            try:
                sample_rgb = picam2.capture_array()
            except Exception:
                continue
            sample_rgb = cv2.rotate(sample_rgb, cv2.ROTATE_180)
            sample_bgr = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR)

            detected_corners = find_target_corners(sample_bgr)
            if detected_corners is not None:
                warped = warp_target(sample_bgr, detected_corners)
                colors = scan_target_colors(warped)
                circle_status, _ = calculate_victim_health(colors)
                if circle_status in ["H", "S", "U"]:
                    votes.append(circle_status)
            time.sleep(delay)

        if votes:
            most = Counter(votes).most_common(1)[0][0]
            print(f"[{self.side_code}] Color-eval votes: {Counter(votes)} -> {most}")
            return most
        return None

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

        print(f"Thread {self.side_code} aktiv. (Stream: {self.show_stream})")
        frame_count = 0
        last_fps_time = time.time()

        while self.running:
            current_time = time.time()

            if tof:
                try:
                    raw = (tof.state[self.side_code] != 0)
                except Exception:
                    raw = True
            else:
                raw = True

            if self._tof_last_raw is None:
                self._tof_last_raw = raw
                self._tof_last_change = current_time

            if raw != self._tof_last_raw:
                self._tof_last_raw = raw
                self._tof_last_change = current_time

            if not raw and self.tof_filtered:
                self.tof_filtered = False
            elif raw and not self.tof_filtered:
                if (current_time - self._tof_last_change) >= self.TOF_DEBOUNCE:
                    self.tof_filtered = True

            if self.waiting_for_reset:
                self.status_pin.on()
                time.sleep(0.1)
                continue

            if not self.enabled:
                self.status_pin.off()
                self.alert_active = False
                time.sleep(0.1)
                continue

            if tof and not self.tof_filtered:
                self.status_pin.off()
                self.alert_active = False
                if self.frame_counter > 0:
                    self.reset_logic()
                    print(f"[{self.side_code}] Wand verloren. Reset.")
                time.sleep(0.05)
                continue

            self.status_pin.on()
            frame_rgb = picam2.capture_array()
            current_time = time.time()
            frame_count += 1
            
            if current_time - last_fps_time > 1.0:
                self.fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time

            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            #
            cutoff_top_y = int(frame_rgb.shape[0] * 0.125)
            cutoff_bottom_y = int(frame_rgb.shape[0] * 0.9)
            cutoff_left_x = int(frame_rgb.shape[1] * (1/7))
            cutoff_right_x = int(frame_rgb.shape[1] * (6/7))
            
            display_frame = None
            if self.show_stream:
                display_frame = frame_bgr.copy()
                cv2.line(display_frame, (0, cutoff_top_y), (frame_bgr.shape[1], cutoff_top_y), (0, 255, 255), 2)
                cv2.line(display_frame, (0, cutoff_bottom_y), (frame_bgr.shape[1], cutoff_bottom_y), (0, 255, 255), 2)
                cv2.line(display_frame, (cutoff_left_x, 0), (cutoff_left_x, frame_bgr.shape[0]), (0, 255, 255), 2)
                cv2.line(display_frame, (cutoff_right_x, 0), (cutoff_right_x, frame_bgr.shape[0]), (0, 255, 255), 2)

            detected_frame_label = None
            box_coords = None
            detected_corners = None

            # --- ERKENNUNG 1: Buchstaben ---
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
                
                # --- NEU: MITTELPUNKT BERECHNEN ---
                cx_tmp = x_tmp + (w_tmp // 2)
                cy_tmp = y_tmp + (h_tmp // 2)
                
                # Wenn der MITTELPUNKT draußen liegt, ignorieren!
                if cy_tmp < cutoff_top_y or cy_tmp > cutoff_bottom_y: continue
                if cx_tmp < cutoff_left_x or cx_tmp > cutoff_right_x: continue
                
                if w_tmp < 60 or h_tmp < 60 or w_tmp > 360 or h_tmp > 360: continue
                
                aspect_ratio = w_tmp / float(h_tmp)
                if 0.5 <= aspect_ratio <= 2.0:
                    square_contours.append(c)

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

                if confidence > MIN_CONFIDENCE_C:
                    label_str = LABELS.get(best_class_id)
                    if label_str and label_str.lower() != "background":
                        if label_str.upper() == 'C':
                            detected_frame_label = 'C'
                            print(f"[{self.side_code}] Kognitives Ziel erkannt (C). Confidence: {confidence:.3f}. Starte Farbauswertung...")
                            color_result = self.evaluate_color_target(picam2, samples=3, delay=0.03)
                            if color_result:
                                self.serial_mgr.write(color_result, self.side_code)
                                print(f"[{self.side_code}] Farbauswertung Ergebnis: {color_result} - gesendet. Warte auf <D>.")
                                self.reset_logic()
                                self.enabled = False
                                self.waiting_for_reset = True
                                self.alert_active = True
                                continue
                            else:
                                print(f"[{self.side_code}] Farbauswertung lieferte kein Ergebnis.")
                        else:
                            detected_frame_label = label_str

            # --- ERKENNUNG 2: Farbringe ---
            if not detected_frame_label:
                detected_corners = find_target_corners(frame_bgr)
                if detected_corners is not None:
                    warped = warp_target(frame_bgr, detected_corners)
                    colors = scan_target_colors(warped)
                    circle_status, _ = calculate_victim_health(colors)
                    if circle_status in ["H", "S", "U"]:
                        detected_frame_label = circle_status

            # --- GUI DARSTELLUNG ---
            if self.show_stream:
                if detected_frame_label:
                    color = (0, 255, 0)
                    cv2.putText(display_frame, f"Erkannt: {detected_frame_label}", (10, cutoff_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    if box_coords:
                        bx, by, bw, bh = box_coords
                        cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), color, 2)
                    elif detected_corners is not None:
                        pts = detected_corners.reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(display_frame, [pts], True, (255, 165, 0), 3)
                else:
                    cv2.putText(display_frame, "Suche...", (10, cutoff_top_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                cv2.imshow(f"Kamera {self.side_code}", display_frame)
                cv2.waitKey(1)

            # --- FILTERUNG & WATCHDOG ---
            if detected_frame_label:
                self.last_detection_time = current_time
                self.frame_counter += 1
                self.alert_active = True
                print(f"[{self.side_code}] Detektion: {detected_frame_label}")
                if detected_frame_label == "H": self.Counter_Harmed += 1
                elif detected_frame_label == "S": self.Counter_Safe += 1
                elif detected_frame_label == "U": self.Counter_Unharmed += 1
            else:
                if self.last_detection_time > 0:
                    verstrichene_zeit = current_time - self.last_detection_time
                    if verstrichene_zeit > self.TIMEOUT_DURATION:
                        self.reset_logic()
                        self.alert_active = False

            # --- ERGEBNIS ÜBERTRAGEN ---
            if self.frame_counter >= 10:
                counts = {'H': self.Counter_Harmed, 'S': self.Counter_Safe, 'U': self.Counter_Unharmed}
                cam_transmit = max(counts, key=counts.get)
                self.serial_mgr.write(cam_transmit, self.side_code)
                print(f"[{self.side_code}] Transfer abgeschlossen. Warte auf <R>.")
                self.reset_logic()
                self.enabled = False
                self.waiting_for_reset = True

        picam2.stop()
        self.status_pin.off()
        if self.show_stream:
            cv2.destroyWindow(f"Kamera {self.side_code}")


# ==========================================
# 6. MAIN-STEUERUNG
# ==========================================

if tof:
    print("Starte TOF Sensoren...")
    tof.start()

serial_mgr = SerialManager(SERIAL_PORT, BAUD_RATE)

cam_right = CameraAIThread(0, "R", ACTIVE_PIN_R, serial_mgr, show_stream=SHOW_STREAM_R)
cam_left = CameraAIThread(1, "L", ACTIVE_PIN_L, serial_mgr, show_stream=SHOW_STREAM_L)

cam_left.start()
cam_right.start()

print("Warte auf Befehle vom Arduino...")

try:
    while True:
        commands = serial_mgr.read_commands()
        for cmd in commands:
            print("Arduino Command: ", cmd)
            
            if cmd == "I":
                serial_mgr.write("OK")
            elif cmd == "E":
                cam_left.enabled = True
                cam_right.enabled = True
                serial_mgr.write("OK")
                print("Enabled")
            elif cmd == "D":
                cam_left.enabled = False
                cam_left.waiting_for_reset = False
                cam_left.reset_logic()
                cam_left.alert_active = False
                
                cam_right.enabled = False
                cam_right.waiting_for_reset = False
                cam_right.reset_logic()
                cam_right.alert_active = False
                serial_mgr.write("OK")
                print("Disabled")

        def cam_has_wall(cam):
            try:
                return getattr(cam, 'tof_filtered', True)
            except Exception:
                return True

        left_valid = cam_left.alert_active and cam_has_wall(cam_left)
        right_valid = cam_right.alert_active and cam_has_wall(cam_right)

        if left_valid or right_valid:
            output_pin.on()
        else:
            output_pin.off()

        last_det = max(cam_left.last_detection_time, cam_right.last_detection_time)
        if last_det > 0 and (time.time() - last_det) > 3.0:
            if cam_left.alert_active or cam_right.alert_active:
                print("Auto-reset: keine neue Erkennung innerhalb von 3s. Reset.")
                cam_left.alert_active = False
                cam_right.alert_active = False
                cam_left.waiting_for_reset = False
                cam_right.waiting_for_reset = False
                cam_left.reset_logic()
                cam_right.reset_logic()
                output_pin.off()

        time.sleep(0.01)

except KeyboardInterrupt:
    print("System herunterfahren...")
    cam_left.running = cam_right.running = False
    cam_left.join()
    cam_right.join()
    output_pin.off()