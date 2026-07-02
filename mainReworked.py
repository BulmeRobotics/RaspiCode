"""Main script for Raspberry Pi target detection, classification, and communication."""

import collections
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import serial
from gpiozero import DigitalOutputDevice

# Import Time-of-Flight (ToF) module if available
try:
    import tof
except ImportError:
    print("Warnung: tof.py nicht gefunden. ToF-Logik wird übersprungen.")
    tof = None

# Load LiteRT/TFLite interpreter
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

# Load Picamera2 library
try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt. Bitte mit 'sudo apt install python3-picamera2' installieren.")
    sys.exit()


# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "trainedCog.tflite"
LABEL_PATH = "labelsCog.txt"
MIN_CONFIDENCE = 0.9
MIN_CONFIDENCE_C = 0.8  # Lower threshold for cognitive target 'C'
SERIAL_PORT = "/dev/ttyAMA0"  # Serial port connected to the Arduino
BAUD_RATE = 115200

# GPIO Pin Configuration
TRIGGER_PIN = 17   # ALERT Pin (CM5 GPIO17 -> Giga 31)
ACTIVE_PIN_L = 27  # Status Pin Camera L (CM5 GPIO27 -> Giga 29)
ACTIVE_PIN_R = 22  # Status Pin Camera R (CM5 GPIO22 -> Giga 28)

# Camera Stream Visualization Settings
SHOW_STREAM_L = False
SHOW_STREAM_R = False

# Initialize GPIO output devices
output_pin = DigitalOutputDevice(TRIGGER_PIN)


def load_labels(path):
    """Load class labels from a text file, fallback to defaults if not found.

    Args:
        path (str): Path to the labels file.

    Returns:
        dict: Mapping of index (int) to label string (str).
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    return {0: "C", 1: "H", 2: "S", 3: "U"}


LABELS = load_labels(LABEL_PATH)


# ==========================================
# 2. SERIAL MANAGER CLASS
# ==========================================
class SerialManager:
    """Encapsulates serial communication and buffer management."""

    def __init__(self, port, baud_rate):
        """Initialize serial connection or enter simulation mode if connection fails.

        Args:
            port (str): Serial port path.
            baud_rate (int): Baud rate for the serial connection.
        """
        self.buffer = ""
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=0.1)
            print(f"Erfolgreich mit Serial {port} verbunden.")
        except Exception as e:
            print(f"Serial nicht gefunden ({e}) -> Simulationsmodus aktiv.")
            self.ser = None

    def write(self, obj, camside=None):
        """Write a formatted message to the serial port.

        Args:
            obj (str): The object identifier to send.
            camside (str, optional): The camera identifier side ('L' or 'R'). Defaults to None.
        """
        if self.ser:
            msg = f"<{camside}{obj}>\n" if camside else f"<{obj}>\n"
            self.ser.write(msg.encode("utf-8"))
            print(f"[SERIAL] Gesendet: {msg.strip()}")

    def read_commands(self):
        """Read the serial buffer and extract all complete commands wrapped in '<' and '>'.

        Returns:
            list: A list of extracted command strings.
        """
        commands = []
        if self.ser and self.ser.in_waiting > 0:
            chars = self.ser.read(self.ser.in_waiting).decode("utf-8", errors="ignore")
            self.buffer += chars

            while "<" in self.buffer and ">" in self.buffer:
                start = self.buffer.find("<")
                end = self.buffer.find(">", start)

                if start != -1 and end != -1:
                    cmd = self.buffer[start + 1:end]
                    commands.append(cmd)
                    self.buffer = self.buffer[end + 1:]
                else:
                    break
        return commands


# ==========================================
# 3. GEOMETRY & COLOR FUNCTIONS
# ==========================================
def order_points(pts):
    """Order coordinates in clockwise order starting from top-left.

    Args:
        pts (numpy.ndarray): 4x2 array of coordinate points.

    Returns:
        numpy.ndarray: Ordered 4x2 array of points (top-left, top-right, bottom-right, bottom-left).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_target(image_bgr, corners, output_size=200):
    """Warp the target area defined by corners to a standardized square size.

    Args:
        image_bgr (numpy.ndarray): Input source image in BGR format.
        corners (numpy.ndarray): 4 coordinate points defining the target area.
        output_size (int, optional): Standardized size of the output image. Defaults to 200.

    Returns:
        numpy.ndarray: Warped destination image.
    """
    dst_points = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    return cv2.warpPerspective(image_bgr, matrix, (output_size, output_size))


def classify_color(hsv_pixel):
    """Classify a single HSV pixel into discrete color names.

    Args:
        hsv_pixel (numpy.ndarray or list): HSV values (hue, saturation, value).

    Returns:
        str: Name of the classified color.
    """
    h, s, v = hsv_pixel
    if v < 80:
        return "Black"
    if h < 12 or h > 100:
        return "Red"
    elif 69 < h < 103:
        return "Yellow"
    elif 40 < h < 70:
        return "Green"
    elif 10 < h < 30:
        return "Blue"
    return "Unknown"


def scan_target_colors(warped_image_bgr):
    """Scan the warped target at concentric ring radii to identify colors.

    Args:
        warped_image_bgr (numpy.ndarray): Warped square BGR target image.

    Returns:
        list: List of identified dominant color strings for each ring.
    """
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
            most_common_color = collections.Counter(ring_colors).most_common(1)[0][0]
            final_colors.append(most_common_color)
        else:
            final_colors.append("Unknown")
    return final_colors


def calculate_victim_health(colors):
    """Calculate the victim status/health based on concentric ring colors.

    Args:
        colors (list): List of color names detected from rings.

    Returns:
        tuple: (status string 'H'/'S'/'U'/'F', total point sum).
    """
    color_values = {"Yellow": 0, "Blue": 2, "Red": -1, "Black": -2, "Green": 1}
    total_sum = sum(color_values.get(color, 0) for color in colors)

    yellow_count = 0
    for color in colors:
        if color_values.get(color, 0) == 0:
            yellow_count += 1

    status = "F"
    if total_sum == 0:
        status = "U"
    elif total_sum == 1:
        status = "S"
    elif total_sum == 2:
        status = "H"

    if yellow_count >= 5:
        status = "F"

    return status, total_sum


# ==========================================
# 4. CAMERA AI THREAD CLASS
# ==========================================
class CameraAIThread(threading.Thread):
    """Thread for handling image acquisition, AI classification, color analysis, and GPIO signal toggling."""

    def __init__(self, cam_id, side_code, status_gpio_pin, serial_mgr, show_stream=False):
        """Initialize the Camera AI Thread.

        Args:
            cam_id (int): ID of the camera.
            side_code (str): 'L' (Left) or 'R' (Right) camera side.
            status_gpio_pin (int): GPIO pin number for status feedback.
            serial_mgr (SerialManager): Manager instance for sending/receiving serial messages.
            show_stream (bool, optional): Whether to display visual preview window. Defaults to False.
        """
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

        # Detection counters
        self.Counter_Harmed = 0
        self.Counter_Safe = 0
        self.Counter_Unharmed = 0
        self.frame_counter = 0

        self.last_detection_time = 0.0
        self.TIMEOUT_DURATION = 2.0
        self.fps = 0.0

        # ToF Debouncing State
        self.tof_filtered = True
        self._tof_last_raw = None
        self._tof_last_change = 0.0
        self.TOF_DEBOUNCE = 0.2

        # Initialize TensorFlow Lite model
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.ki_h, self.ki_w = self.input_details[0]["shape"][1:3]
            self.is_int8 = self.input_details[0]["dtype"] in [np.int8, np.uint8]
            self.ready = True
            print(f"Cam {self.side_code} CPU Modell geladen.")
        except Exception as e:
            print(f"Cam {self.side_code} Fehler: {e}")
            self.ready = False

    @property
    def alert_active(self):
        """Thread-safe getter for alert_active flag."""
        with self._alert_lock:
            return self._alert_active

    @alert_active.setter
    def alert_active(self, value):
        """Thread-safe setter for alert_active flag."""
        with self._alert_lock:
            self._alert_active = bool(value)

    def reset_logic(self):
        """Reset detection counters and timers."""
        self.Counter_Harmed = self.Counter_Safe = self.Counter_Unharmed = 0
        self.frame_counter = 0
        self.last_detection_time = 0.0

    def get_boundary_coords(self, img_shape, side=None):
        """Get the boundary coordinate configurations for detection exclusion.

        Args:
            img_shape (tuple): Shape of the image (height, width, ...).
            side (str, optional): The camera side ('L' or 'R'). Defaults to self.side_code.

        Returns:
            dict: Coordinate mappings for rectangular boundaries and angled cutoffs.
        """
        if side is None:
            side = self.side_code

        height, width = img_shape[:2]

        cutoff_top_y = int(height * 0.25)
        cutoff_bottom_y = int(height * 0.8)

        cutoff_left_x = int(width * (1 / 7))
        cutoff_right_x = int(width * (6 / 7))

        # Default angled cutoff parameters for right side ("R")
        cutoff_left_auswurf_bottom = [260, 478]   # Left-bottom border
        cutoff_left_auswurf_top = [0, 202]       # Left-top border
        cutoff_right_auswurf_bottom = [320, 478]  # Right-bottom border
        cutoff_right_auswurf_top = [636, 185]     # Right-top border

        # Angled cutoff parameters for left side ("L")
        if side == "L":
            cutoff_left_auswurf_bottom = [300, 478]   # Left-bottom border
            cutoff_left_auswurf_top = [0, 193]       # Left-top border
            cutoff_right_auswurf_bottom = [377, 478]  # Right-bottom border
            cutoff_right_auswurf_top = [636, 210]     # Right-top border

        return {
            "top_y": cutoff_top_y,
            "bottom_y": cutoff_bottom_y,
            "left_x": cutoff_left_x,
            "right_x": cutoff_right_x,
            "left_auswurf_bottom": cutoff_left_auswurf_bottom,
            "left_auswurf_top": cutoff_left_auswurf_top,
            "right_auswurf_bottom": cutoff_right_auswurf_bottom,
            "right_auswurf_top": cutoff_right_auswurf_top,
        }

    def is_in_boundary(self, cx, cy, img_shape, side=None):
        """Check if a coordinate point (cx, cy) is within the legal detection boundaries.

        Args:
            cx (int): X coordinate of the point.
            cy (int): Y coordinate of the point.
            img_shape (tuple): Shape of the image.
            side (str, optional): The camera side. Defaults to self.side_code.

        Returns:
            bool: True if the coordinate is within boundaries, False otherwise.
        """
        coords = self.get_boundary_coords(img_shape, side)

        # Check simple rectangular bounds
        if cy < coords["top_y"] or cy > coords["bottom_y"]:
            return False
        if cx < coords["left_x"] or cx > coords["right_x"]:
            return False

        # Check left side angled cutoff zone
        left_top = coords["left_auswurf_top"]
        left_bottom = coords["left_auswurf_bottom"]
        if cy > left_top[1] and cx < left_bottom[0]:
            line_y = left_top[1] + (cx - left_top[0]) * (
                (left_bottom[1] - left_top[1]) / (left_bottom[0] - left_top[0])
            )
            if line_y < cy:
                return False

        # Check right side angled cutoff zone
        right_top = coords["right_auswurf_top"]
        right_bottom = coords["right_auswurf_bottom"]
        if cx > right_bottom[0] and cy > right_top[1]:
            line_y = right_top[1] + (cx - right_top[0]) * (
                (right_bottom[1] - right_top[1]) / (right_bottom[0] - right_top[0])
            )
            if line_y < cy:
                return False

        return True

    def find_target_corners(self, image_bgr, side=None):
        """Find the 4 corners of the target square in the BGR image.

        Args:
            image_bgr (numpy.ndarray): Input BGR image.
            side (str, optional): The camera side. Defaults to self.side_code.

        Returns:
            numpy.ndarray or None: Ordered 4 corner points if a valid target is found, else None.
        """
        if side is None:
            side = self.side_code

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Clear edges in the angled cutoff zones
        coords = self.get_boundary_coords(image_bgr.shape, side)
        pts_left = np.array([
            coords["left_auswurf_top"],
            coords["left_auswurf_bottom"],
            [coords["left_auswurf_bottom"][0], image_bgr.shape[0]],
            [0, image_bgr.shape[0]]
        ], dtype=np.int32)
        cv2.fillPoly(edges, [pts_left], 0)

        pts_right = np.array([
            coords["right_auswurf_bottom"],
            coords["right_auswurf_top"],
            [image_bgr.shape[1], coords["right_auswurf_top"][1]],
            [image_bgr.shape[1], image_bgr.shape[0]],
            [coords["right_auswurf_bottom"][0], image_bgr.shape[0]]
        ], dtype=np.int32)
        cv2.fillPoly(edges, [pts_right], 0)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            cx = x + (w // 2)
            cy = y + (h // 2)

            # Check if the center point lies inside the boundary zone
            if not self.is_in_boundary(cx, cy, image_bgr.shape, side=side):
                continue

            area = cv2.contourArea(cnt)
            if area > 7000:                         #Von 1000 auf 7000
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                width = rect[1][0]
                height = rect[1][1]
                if height == 0:
                    continue
                aspect_ratio = width / height

                if 0.85 <= aspect_ratio <= 1.15:
                    return order_points(box)
        return None

    def evaluate_color_target(self, picam2, samples=25, delay=0.03):
        """Capture multiple frame samples to evaluate and vote on the target color status.

        Args:
            picam2 (Picamera2): The Picamera2 camera instance.
            samples (int, optional): Number of frame samples to capture. Defaults to 25.
            delay (float, optional): Inter-frame delay in seconds. Defaults to 0.03.

        Returns:
            str or None: The voted status 'H'/'S'/'U'/'F' if votes exist, else None.
        """
        votes = []
        for i in range(samples):
            try:
                sample_rgb = picam2.capture_array()
            except Exception:
                continue
            sample_rgb = cv2.rotate(sample_rgb, cv2.ROTATE_180)
            sample_bgr = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR)

            detected_corners = self.find_target_corners(sample_bgr, self.side_code)
            if detected_corners is not None:
                warped = warp_target(sample_bgr, detected_corners)
                colors = scan_target_colors(warped)
                circle_status, _ = calculate_victim_health(colors)
                if circle_status in ["H", "S", "U", "F"]:
                    votes.append(circle_status)

        if votes:
            most = collections.Counter(votes).most_common(1)[0][0]
            print(f"[{self.side_code}] Color-eval votes: {collections.Counter(votes)} -> {most}")
            return most
        return None

    def run(self):
        """Run the main loop of the camera thread. Handles frame capture, detection, and logic updates."""
        if not self.ready:
            return

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

            coords = self.get_boundary_coords(frame_rgb.shape, self.side_code)

            # Prepare corner polygons (do not draw on image)
            pts_left = np.array([
                coords["left_auswurf_top"],
                coords["left_auswurf_bottom"],
                [coords["left_auswurf_bottom"][0], frame_rgb.shape[0]],
                [0, frame_rgb.shape[0]]
            ], dtype=np.int32)

            pts_right = np.array([
                coords["right_auswurf_bottom"],
                coords["right_auswurf_top"],
                [frame_rgb.shape[1], coords["right_auswurf_top"][1]],
                [frame_rgb.shape[1], frame_rgb.shape[0]],
                [coords["right_auswurf_bottom"][0], frame_rgb.shape[0]]
            ], dtype=np.int32)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            coords = self.get_boundary_coords(frame_bgr.shape, self.side_code)
            cutoff_top_y = coords["top_y"]

            display_frame = None
            if self.show_stream:
                display_frame = frame_bgr.copy()

                # Draw rectangular boundary from get_boundary_coords in red (0, 0, 255)
                cv2.line(display_frame, (0, coords["top_y"]), (frame_bgr.shape[1], coords["top_y"]), (0, 0, 255), 2)
                cv2.line(display_frame, (0, coords["bottom_y"]), (frame_bgr.shape[1], coords["bottom_y"]), (0, 0, 255), 2)
                cv2.line(display_frame, (coords["left_x"], 0), (coords["left_x"], frame_bgr.shape[0]), (0, 0, 255), 2)
                cv2.line(display_frame, (coords["right_x"], 0), (coords["right_x"], frame_bgr.shape[0]), (0, 0, 255), 2)

                # Draw angled cutoffs
                ang_left_bottom = tuple(coords["left_auswurf_bottom"])
                ang_left_top = tuple(coords["left_auswurf_top"])
                ang_right_bottom = tuple(coords["right_auswurf_bottom"])
                ang_right_top = tuple(coords["right_auswurf_top"])

                cv2.line(display_frame, ang_left_top, ang_left_bottom, (0, 0, 255), 2)
                cv2.line(display_frame, ang_right_top, ang_right_bottom, (0, 0, 255), 2)

            detected_frame_label = None
            box_coords = None
            detected_corners = None

            # --- DETECTION 1: Letters ---
            gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            total_area = gray_frame.shape[0] * gray_frame.shape[1]

            blurred = cv2.GaussianBlur(gray_frame, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
            )

            # Remove angled cutoffs from thresholded image
            cv2.fillPoly(thresh, [pts_left], 0)
            cv2.fillPoly(thresh, [pts_right], 0)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if 10 < cv2.contourArea(c) < total_area * 0.99]

            square_contours = []
            for c in valid_contours:
                x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(c)
                if h_tmp == 0:
                    continue

                # Calculate center point
                cx_tmp = x_tmp + (w_tmp // 2)
                cy_tmp = y_tmp + (h_tmp // 2)

                # Ignore if center point is out of boundary
                if not self.is_in_boundary(cx_tmp, cy_tmp, gray_frame.shape, self.side_code):
                    continue

                if w_tmp < 100 or h_tmp < 100 or w_tmp > 300 or h_tmp > 300:        #von 60 auf 100, weil Größe sonst zu klein möglich wäre
                    continue

                aspect_ratio = w_tmp / float(h_tmp)
                if 0.85 <= aspect_ratio <= 1.15:
                    square_contours.append(c)

            if square_contours:
                largest_contour = max(square_contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(largest_contour)
                box_coords = (x_b, y_b, w_b, h_b)

                letter_crop = gray_frame[y_b:y_b + h_b, x_b:x_b + w_b]

                fill_ratio = 0.55       #von 0.7 auf 0.55, testweise, für mehr rand wege Cogn Targets
                max_dim = max(w_b, h_b)
                target_dim = int(max_dim / fill_ratio)
                pad_top = (target_dim - h_b) // 2
                pad_bottom = target_dim - h_b - pad_top
                pad_left = (target_dim - w_b) // 2
                pad_right = target_dim - w_b - pad_left
                bg_color = int(np.median(gray_frame[0:10, 0:10]))
                padded_img = cv2.copyMakeBorder(
                    letter_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bg_color
                )
                prep_img = cv2.resize(padded_img, (self.ki_w, self.ki_h), interpolation=cv2.INTER_AREA)

                prep_img_expanded = np.expand_dims(prep_img, axis=-1)
                input_data = np.expand_dims(prep_img_expanded, axis=0)

                if self.is_int8:
                    scale, zero_point = self.input_details[0]["quantization"]
                    input_data = (input_data.astype(np.float32) / scale + zero_point).astype(np.int8)
                else:
                    input_data = input_data.astype(np.float32)

                self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

                if self.is_int8:
                    out_scale, out_zero_point = self.output_details[0]["quantization"]
                    scores = (output_data.astype(np.float32) - out_zero_point) * out_scale
                else:
                    scores = output_data

                best_class_id = np.argmax(scores)
                confidence = scores[best_class_id]

                if confidence > MIN_CONFIDENCE_C:
                    label_str = LABELS.get(best_class_id)
                    if label_str and label_str.lower() != "background":
                        if label_str.upper() == "C":
                            detected_frame_label = "C"
                            print(f"[{self.side_code}] Kognitives Ziel erkannt (C). Confidence: {confidence:.3f}. Starte Farbauswertung...")
                            color_result = self.evaluate_color_target(picam2, samples=25, delay=0.03)
                            if color_result:
                                if not self.enabled:
                                    print(f"[{self.side_code}] Wurde während Farbauswertung deaktiviert.")
                                    continue
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

            # --- DETECTION 2: Color Rings ---
            if not detected_frame_label:
                detected_corners = self.find_target_corners(frame_bgr, self.side_code)
                if detected_corners is not None:
                    warped = warp_target(frame_bgr, detected_corners)
                    colors = scan_target_colors(warped)
                    circle_status, _ = calculate_victim_health(colors)
                    if circle_status in ["H", "S", "U"]:
                        detected_frame_label = circle_status

            # --- GUI VISUALIZATION ---
            if self.show_stream:
                if detected_frame_label:
                    color = (0, 255, 0)
                    cv2.putText(
                        display_frame,
                        f"Erkannt: {detected_frame_label}",
                        (10, cutoff_top_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

                    if box_coords:
                        bx, by, bw, bh = box_coords
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
                    elif detected_corners is not None:
                        pts = detected_corners.reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(display_frame, [pts], True, (255, 165, 0), 3)
                else:
                    cv2.putText(
                        display_frame,
                        "Suche...",
                        (10, cutoff_top_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(f"Kamera {self.side_code}", display_frame)
                cv2.waitKey(1)

            # --- FILTERING & WATCHDOG ---
            if not self.enabled:
                continue

            if detected_frame_label:
                self.last_detection_time = current_time
                self.frame_counter += 1
                self.alert_active = True
                print(f"[{self.side_code}] Detektion: {detected_frame_label}")
                if detected_frame_label == "H":
                    self.Counter_Harmed += 1
                elif detected_frame_label == "S":
                    self.Counter_Safe += 1
                elif detected_frame_label == "U":
                    self.Counter_Unharmed += 1
            else:
                if self.last_detection_time > 0:
                    elapsed_time = current_time - self.last_detection_time
                    if elapsed_time > self.TIMEOUT_DURATION:
                        self.reset_logic()
                        self.alert_active = False

            # --- TRANSMIT RESULT ---
            if self.frame_counter >= 10:
                counts = {"H": self.Counter_Harmed, "S": self.Counter_Safe, "U": self.Counter_Unharmed}
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
# 5. MAIN CONTROL LOOP
# ==========================================
def cam_has_wall(cam):
    """Check if the given camera's associated ToF sensor detects a wall.

    Args:
        cam (CameraAIThread): The camera thread instance.

    Returns:
        bool: True if ToF sensor filtered state indicates wall presence or on error/fallback.
    """
    try:
        return getattr(cam, "tof_filtered", True)
    except Exception:
        return True


if __name__ == "__main__":
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

            left_valid = (
                cam_left.alert_active
                and cam_has_wall(cam_left)
                and (cam_left.enabled or cam_left.waiting_for_reset)
            )
            right_valid = (
                cam_right.alert_active
                and cam_has_wall(cam_right)
                and (cam_right.enabled or cam_right.waiting_for_reset)
            )

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