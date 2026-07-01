import cv2
import numpy as np
import time
import math
import sys
from collections import Counter

try:
    from picamera2 import Picamera2
except ImportError:
    print("Fehler: Picamera2 fehlt. Bitte installieren.")
    sys.exit()

# ==========================================
# KONFIGURATION
# ==========================================
CAM_ID = 1          # 0 für Left, 1 für Right
SAMPLES = 20         # Anzahl der Bilder, die für die Bewertung gemacht werden
DELAY_SEC = 0.03    # Pause zwischen den Aufnahmen

# ==========================================
# GEOMETRIE- & FARBFUNKTIONEN (Aus Hauptcode)
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
    # Exakte Sperrzonen aus deinem Hauptcode
    cutoff_top_y = int(image_bgr.shape[0] * 0.25)
    cutoff_bottom_y = int(image_bgr.shape[0] * 0.875)
    cutoff_left_x = int(image_bgr.shape[1] * (1/7))
    cutoff_right_x = int(image_bgr.shape[1] * (6/7))
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < cutoff_top_y or (y + h) > cutoff_bottom_y: continue
        if x < cutoff_left_x or (x + w) > cutoff_right_x: continue
            
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
    if v < 80: return "Black"
    #if s < 50 and v > 200: return "White" 

    if h < 12 or h > 100: return "Red"
    elif 69 < h < 103: return "Yellow"
    elif 40 < h < 70: return "Green"
    elif 10 < h < 30: return "Blue"
    return "Unknown"

def scan_target_colors(warped_image_bgr, debug=False):
    hsv_image = cv2.cvtColor(warped_image_bgr, cv2.COLOR_BGR2HSV)
    center = (100, 100)
    radii = [10, 30, 50, 70, 90]
    final_colors = []

    samples_info = []  # list of lists: for each ring a list of (x,y,color)

    for r in radii:
        ring_colors = []
        ring_info = []
        for angle_deg in range(0, 360, 30):
            angle_rad = math.radians(angle_deg)
            x = int(center[0] + r * math.cos(angle_rad))
            y = int(center[1] + r * math.sin(angle_rad))

            x = max(0, min(199, x))
            y = max(0, min(199, y))

            color_name = classify_color(hsv_image[y, x])
            ring_info.append((x, y, color_name))
            if color_name != "White" and color_name != "Unknown":
                ring_colors.append(color_name)

        if ring_colors:
            most_common_color = Counter(ring_colors).most_common(1)[0][0]
            final_colors.append(most_common_color)
        else:
            final_colors.append("Unknown")

        samples_info.append(ring_info)

    if debug:
        return final_colors, samples_info
    return final_colors


def visualize_warped_sampling(warped_image_bgr, samples_info, final_colors=None):
    vis = warped_image_bgr.copy()
    # color mapping for visualization (BGR)
    color_map = {
        'Red': (0, 0, 255),
        'Yellow': (0, 255, 255),
        'Green': (0, 255, 0),
        'Blue': (255, 0, 0),
        'Black': (0, 0, 0),
        'White': (255, 255, 255),
        'Unknown': (180, 180, 180)
    }

    center = (100, 100)
    # draw concentric reference circles
    for r in [10, 30, 50, 70, 90]:
        cv2.circle(vis, center, r, (200, 200, 200), 1)

    # draw sample points and small labels
    for ring_idx, ring in enumerate(samples_info):
        for (x, y, color_name) in ring:
            col = color_map.get(color_name, (0, 0, 0))
            # filled circle with contrasting border
            cv2.circle(vis, (x, y), 4, col, -1)
            cv2.circle(vis, (x, y), 6, (50, 50, 50), 1)

    # optionally write final detected ring colors
    if final_colors is not None:
        text = 'Rings: ' + ','.join(final_colors)
        cv2.putText(vis, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return vis

def calculate_victim_health(colors):
    """Return status (H/S/U/Fake) and an integer score based on detected ring colors."""
    color_values = {"Yellow": 0, "Blue": 2, "Red": -1, "Black": -2, "Green": 1}
    total_sum = sum(color_values.get(color, 0) for color in colors)
    status = "Fake"
    if total_sum == 0:
        status = "U"
    elif total_sum == 1:
        status = "S"
    elif total_sum == 2:
        status = "H"
    return status, total_sum

# ==========================================
# AUSWERTE-PROZESS SIMULATOR
# ==========================================
def run_cognitive_test(picam2):
    print(f"\n--- Starte Cognitive Target Analyse ({SAMPLES} Samples) ---")
    votes = []

    for i in range(SAMPLES):
        try:
            frame_rgb = picam2.capture_array()
        except Exception as e:
            print(f" Fehler bei Frame {i+1}: {e}")
            continue

        # Rotation aus deinem Hauptcode anwenden
        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        corners = find_target_corners(frame_bgr)
        if corners is not None:
            warped = warp_target(frame_bgr, corners)
            # get debug info (points per sample)
            colors, samples_info = scan_target_colors(warped, debug=True)
            status, score = calculate_victim_health(colors)

            # Visual debug: show detected box on original frame
            debug_frame = frame_bgr.copy()
            try:
                pts = np.int32(corners).reshape((-1, 1, 2))
                cv2.polylines(debug_frame, [pts], True, (0, 0, 255), 2)
            except Exception:
                pass

            cv2.putText(debug_frame, f"Sample {i+1}: {status} ({score})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Detected Box", cv2.resize(debug_frame, (640, 480)))

            if status in ["H", "S", "U"]:
                votes.append(status)
                print(f" Sample {i+1}: Ring gefunden -> Status: {status} (Punkte: {score}) | Farben: {colors}")

                # Warped Image zur visuellen Kontrolle anzeigen mit allen Sample-Punkten
                warped_vis = visualize_warped_sampling(warped, samples_info, colors)
                warped_large = cv2.resize(warped_vis, (400, 400), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Cognitive Scan", warped_large)
                # kurze Anzeige, damit Fenster aktualisiert bleibt
                cv2.waitKey(100)
            else:
                print(f" Sample {i+1}: Ring gefunden, aber ungueltiger Status ({status}) | Farben: {colors}")
                warped_vis = visualize_warped_sampling(warped, samples_info, colors)
                cv2.imshow("Cognitive Scan", cv2.resize(warped_vis, (400, 400)))
                cv2.waitKey(100)

        else:
            print(f" Sample {i+1}: Kein Ring in den aktiven Zonen gefunden.")

        time.sleep(DELAY_SEC)

    print("\n---------------- AUSWERTUNG ----------------")
    if votes:
        most_common = Counter(votes).most_common(1)[0][0]
        print(f" Gesammelte Stimmen: {Counter(votes)}")
        print(f" Das Skript wuerde uebertragen: >>> {most_common} <<<")
    else:
        print(" Fehler: Keine gueltigen Ringe in den Samples gefunden.")
    print("--------------------------------------------\n")
    cv2.destroyWindow("Cognitive Scan")
    cv2.destroyWindow("Detected Box")


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print("Starte Kamera...")
    try:
        picam2 = Picamera2(CAM_ID)
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
    except Exception as e:
        print(f"Kamera-Fehler: {e}")
        sys.exit()

    print("Kamera laeuft.")
    print("-> Druecke 'c' im Videofenster, um die Cognitive Target Auswertung zu starten.")
    print("-> Druecke 'q' zum Beenden.")

    while True:
        frame_rgb = picam2.capture_array()
        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        display_frame = frame_bgr.copy()
        
        # Sperrzonen visuell einzeichnen, damit man weiss, wo man den Ring hinlegen darf
        cutoff_top_y = int(display_frame.shape[0] * 0.25)
        cutoff_bottom_y = int(display_frame.shape[0] * 0.875)
        cutoff_left_x = int(display_frame.shape[1] * (1/7))
        cutoff_right_x = int(display_frame.shape[1] * (6/7))
        
        cv2.line(display_frame, (0, cutoff_top_y), (display_frame.shape[1], cutoff_top_y), (0, 255, 255), 2)
        cv2.line(display_frame, (0, cutoff_bottom_y), (display_frame.shape[1], cutoff_bottom_y), (0, 255, 255), 2)
        cv2.line(display_frame, (cutoff_left_x, 0), (cutoff_left_x, display_frame.shape[0]), (0, 255, 255), 2)
        cv2.line(display_frame, (cutoff_right_x, 0), (cutoff_right_x, display_frame.shape[0]), (0, 255, 255), 2)
        
        cv2.putText(display_frame, "Druecke 'C' fuer Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Stream", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            run_cognitive_test(picam2)

    picam2.stop()
    cv2.destroyAllWindows()