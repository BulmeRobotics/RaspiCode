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
    # Versuche das neue LiteRT (Standard ab 2025/2026)
    import ai_edge_litert.interpreter as tflite
    print("LiteRT erfolgreich geladen.")
except ImportError:
    try:
        # Fallback auf das klassische TFLite
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
MODEL_PATH = "trained.tflite"
LABEL_PATH = "labels.txt"
MIN_CONFIDENCE = 0.4
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
    """Sucht die Bounding Box des Ring-Targets basierend auf Kontrastkanten."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box) # Verwende stabiles int32 für aktuelle NumPy-Versionen
            
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
        self.side_code = side_code # "L" oder "R"
        self.enabled = False
        self.running = True
        
        # Gemeinsame Sicherheits-Zähler für Buchstaben- und Ring-Erkennungen
        self.Counter_Harmed = 0
        self.Counter_Safe = 0
        self.Counter_Unharmed = 0
        self.frame_counter = 0
        
        # Watchdog-Variablen
        self.last_detection_time = 0.0
        self.TIMEOUT_DURATION = 3.0  # 3 Sekunden Timeout-Grenze
        
        # TFLite Modell laden
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.ki_h, self.ki_w = self.input_details[0]['shape'][1:3]
            self.is_int8 = (self.input_details[0]['dtype'] in [np.int8, np.uint8])
            self.ready = True
        except Exception as e:
            print(f"Cam {self.side_code} TFLite-Fehler: {e}")
            self.ready = False

    def reset_logic(self):
        """Setzt die Sicherheits-Zähler zurück."""
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

            # Frame-Aufnahme (Liefert RGB888)
            frame_rgb = picam2.capture_array()
            # Für die Geometrie und Farbauswertung der Ringe in OpenCV-BGR konvertieren
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            detected_frame_label = None

            # --- ERKENNUNG 1: TFLite Buchstaben ---
            
            # 1. Bild für das KI-Modell in Graustufen umwandeln (1 Farbkanal)
            gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            
            # 2. Skalieren auf die erwartete KI-Eingangsgröße
            prep_img = cv2.resize(gray_frame, (self.ki_w, self.ki_h))
            
            # 3. Dimensionen anpassen: Von (H, W) zu (H, W, 1) und dann zu (Batch=1, H, W, 1)
            prep_img = np.expand_dims(prep_img, axis=-1)  # Fügt den fehlenden Farbkanal (1) hinzu
            input_data = np.expand_dims(prep_img, axis=0) # Fügt die Batch-Dimension (1) hinzu
            
            # 4. Typisierung für int8 / float32
            if self.is_int8:
                input_data = (input_data.astype(np.float32) - 128).astype(np.int8)
            else:
                input_data = (input_data / 255.0).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            grid_h, grid_w, num_classes = output_data.shape

            max_score = 0
            for y in range(grid_h):
                for x in range(grid_w):
                    for class_id in range(1, num_classes):
                        raw_score = output_data[y][x][class_id]
                        score = (float(raw_score) + 128.0) / 255.0 if self.is_int8 else float(raw_score)
                        
                        if score > MIN_CONFIDENCE and score > max_score:
                            max_score = score
                            detected_frame_label = LABELS.get(class_id, "??")

            # --- ERKENNUNG 2: Farbringe (falls kein Buchstabe Vorrang hatte) ---
            if not detected_frame_label:
                corners = find_target_corners(frame_bgr)
                if corners is not None:
                    warped = warp_target(frame_bgr, corners)
                    colors = scan_target_colors(warped)
                    circle_status, _ = calculate_victim_health(colors)
                    
                    if circle_status in ["H", "S", "U"]:
                        detected_frame_label = circle_status

            # --- FILTERUNG, ABSICHERUNG & WATCHDOG ---
            if detected_frame_label:
                # 1. Erfolgreicher Fund: Zeitstempel auf JETZT setzen
                self.last_detection_time = time.time()
                self.frame_counter += 1
                
                # Erster valider Kontakt -> Hardware-Pin HIGH
                if self.frame_counter == 1:
                    output_pin.on()
                    print(f"[{self.side_code}] Target gesichtet! Pin HIGH.")

                # Zähler erhöhen
                if detected_frame_label == "H": self.Counter_Harmed += 1
                elif detected_frame_label == "S": self.Counter_Safe += 1
                elif detected_frame_label == "U": self.Counter_Unharmed += 1

            else:
                # 2. Kein Fund in diesem Frame: Watchdog prüfen!
                if self.frame_counter > 0:
                    verstrichene_zeit = time.time() - self.last_detection_time
                    
                    if verstrichene_zeit > self.TIMEOUT_DURATION:
                        print(f"[{self.side_code}] Watchdog: 3s ohne Kontakt. Daten verworfen, Pin LOW.")
                        self.reset_logic()
                        # self.enabled bleibt True, Kamera sucht sofort weiter!

            # --- ERGEBNIS ÜBERTRAGEN ---
            # Wenn 5 Übereinstimmungen gesammelt wurden -> Auswertung absenden
            if self.frame_counter >= 5:
                counts = {'H': self.Counter_Harmed, 'S': self.Counter_Safe, 'U': self.Counter_Unharmed}
                cam_transmit = max(counts, key=counts.get)
                
                # Übertragung via Serial (bspw. <LH>)
                SerialWrite(cam_transmit, self.side_code)
                
                # Hardware-Pin wieder LOW, Zähler zurücksetzen und in den Ruhezustand wechseln
                output_pin.off()
                print(f"[{self.side_code}] Transfer abgeschlossen. Pin LOW.")
                self.reset_logic()
                self.enabled = False 

        picam2.stop()

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
                    
                    # --- BEFEHLSAUSWERTUNG ---
                    
                    # INITIALISIERUNG
                    if cmd == "I":
                        SerialWrite("OK")
                    
                    # AKTIVIERUNG BOTH CAMERAS / REAKTIVIEREN
                    elif cmd == "E":
                        cam_left.enabled = True
                        cam_right.enabled = True
                        SerialWrite("OK")
                        
                    elif cmd == "RE":
                        cam_right.enabled = True
                        SerialWrite("OK")
                    
                    # DEAKTIVIERUNG / IDLE
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
                
                buffer = "" # Puffer leeren
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("System herunterfahren...")
    cam_left.running = cam_right.running = False
    cam_left.join()
    cam_right.join()
    output_pin.off()
