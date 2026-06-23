import cv2
import numpy as np
import math
import sys
from picamera2 import Picamera2

# ==========================================
# KONFIGURATION
# ==========================================
CAM_ID = 0  # 0 für links, 1 für rechts (je nachdem welche du kalibrieren willst)

# Globale Variable für den Klick-Sensor
hsv_frame_global = None

def mouse_click(event, x, y, flags, param):
    """Gibt den exakten HSV-Wert des angeklickten Pixels in der Konsole aus."""
    global hsv_frame_global
    if event == cv2.EVENT_LBUTTONDOWN and hsv_frame_global is not None:
        h, s, v = hsv_frame_global[y, x]
        print(f"👉 Klick auf X:{x} Y:{y} | HSV-Werte: H={h:3d}, S={s:3d}, V={v:3d}")

def nothing(x):
    pass

# ==========================================
# GUI UND TRACKBARS SETUP
# ==========================================
cv2.namedWindow("Einstellungen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Einstellungen", 400, 600)

# Startwerte (deine aktuellen Werte aus dem Hauptcode)
cv2.createTrackbar('Black V Max', 'Einstellungen', 60, 255, nothing)
cv2.createTrackbar('White S Max', 'Einstellungen', 50, 255, nothing)
cv2.createTrackbar('White V Min', 'Einstellungen', 200, 255, nothing)

cv2.createTrackbar('Red H Low', 'Einstellungen', 10, 179, nothing)
cv2.createTrackbar('Red H High', 'Einstellungen', 160, 179, nothing)

cv2.createTrackbar('Yellow H Min', 'Einstellungen', 20, 179, nothing)
cv2.createTrackbar('Yellow H Max', 'Einstellungen', 35, 179, nothing)

cv2.createTrackbar('Green H Min', 'Einstellungen', 40, 179, nothing)
cv2.createTrackbar('Green H Max', 'Einstellungen', 80, 179, nothing)

cv2.createTrackbar('Blue H Min', 'Einstellungen', 90, 179, nothing)
cv2.createTrackbar('Blue H Max', 'Einstellungen', 130, 179, nothing)

# ==========================================
# DEINE GEOMETRIE FUNKTIONEN (Exakte Kopie)
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
    cutoff_top_y = int(image_bgr.shape[0] * 0.25)
    cutoff_bottom_y = int(image_bgr.shape[0] * 0.875)
    
    cutoff_left_x = int(image_bgr.shape[1] * (1/7))
    cutoff_right_x = int(image_bgr.shape[1] * (6/7))
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Leicht stärkerer Blur gegen Bildrauschen
    
    # 1. Sensiblerer Canny-Filter
    edges = cv2.Canny(blurred, 30, 100)
    
    # 2. NEU: Dilatation (Linien dicker machen, um Lücken im Kanten-Ring zu schließen)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 3. NEU: RETR_LIST statt RETR_EXTERNAL, um wirklich alle Ebenen zu betrachten
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Absteigend nach Größe sortieren (der größte gültige Ring gewinnt immer)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Horizontale Sperre (Oben/Unten)
        if y < cutoff_top_y or (y + h) > cutoff_bottom_y: 
            continue
            
        # Vertikale Sperre (Links/Rechts)
        if x < cutoff_left_x or (x + w) > cutoff_right_x:
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
            
            # Toleranz für das Rechteck/Quadrat (Ring von der Seite gesehen ist gestaucht)
            if 0.7 <= aspect_ratio <= 1.3:
                return order_points(box)
    return None
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

cv2.namedWindow("Live Video")
cv2.setMouseCallback("Live Video", mouse_click)

print("\n--- KALIBRIERUNG GESTARTET ---")
print("1. Klicke in das 'Live Video' Fenster auf eine Farbe, um den HSV Wert zu sehen.")
#print("2. Passe die Schieberegler im Fenster 'Einstellungen' an.")
print("2. Drücke 'q' um zu beenden und den Code zu generieren.\n")

# ==========================================
# HAUPTSCHLEIFE
# ==========================================
while True:
    frame_rgb = picam2.capture_array()
    frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Globales Update für den Maus-Klick Sensor
    hsv_frame_global = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. Aktuelle Werte von den Schiebereglern holen
    blk_v = cv2.getTrackbarPos('Black V Max', 'Einstellungen')
    wht_s = cv2.getTrackbarPos('White S Max', 'Einstellungen')
    wht_v = cv2.getTrackbarPos('White V Min', 'Einstellungen')
    red_l = cv2.getTrackbarPos('Red H Low', 'Einstellungen')
    red_h = cv2.getTrackbarPos('Red H High', 'Einstellungen')
    yel_min = cv2.getTrackbarPos('Yellow H Min', 'Einstellungen')
    yel_max = cv2.getTrackbarPos('Yellow H Max', 'Einstellungen')
    grn_min = cv2.getTrackbarPos('Green H Min', 'Einstellungen')
    grn_max = cv2.getTrackbarPos('Green H Max', 'Einstellungen')
    blu_min = cv2.getTrackbarPos('Blue H Min', 'Einstellungen')
    blu_max = cv2.getTrackbarPos('Blue H Max', 'Einstellungen')

    # 2. Ring-Suche durchführen
    corners = find_target_corners(frame_bgr)
    
    if corners is not None:
        # Bounding Box ins Live-Video zeichnen
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame_bgr, [pts], True, (255, 0, 255), 2)
        
        # Ring ausschneiden
        warped = warp_target(frame_bgr, corners)
        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # Test-Punkte analysieren und zeichnen
        center = (100, 100)
        radii = [10, 30, 50, 70, 90]
        
        for r in radii:
            for angle_deg in range(0, 360, 30):
                angle_rad = math.radians(angle_deg)
                x = int(center[0] + r * math.cos(angle_rad))
                y = int(center[1] + r * math.sin(angle_rad))
                
                x = max(0, min(199, x))
                y = max(0, min(199, y))
                
                h, s, v = warped_hsv[y, x]
                
                # Live-Klassifizierung mit den Trackbar-Werten
                color_name = "Unknown"
                draw_color = (128, 128, 128) # Grau für Unknown
                
                if v < blk_v:
                    color_name = "Black"
                    draw_color = (0, 0, 0)
                elif s < wht_s and v > wht_v:
                    color_name = "White"
                    draw_color = (255, 255, 255)
                elif h < red_l or h > red_h:
                    color_name = "Red"
                    draw_color = (0, 0, 255)
                elif yel_min < h < yel_max:
                    color_name = "Yellow"
                    draw_color = (0, 255, 255)
                elif grn_min < h < grn_max:
                    color_name = "Green"
                    draw_color = (0, 255, 0)
                elif blu_min < h < blu_max:
                    color_name = "Blue"
                    draw_color = (255, 0, 0)
                
                # Punkt einzeichnen
                cv2.circle(warped, (x, y), 3, draw_color, -1)
                # Dünner Rand, damit man weiße Punkte sieht
                cv2.circle(warped, (x, y), 3, (0,0,0), 1)

        # Warped Ansicht vergrößert anzeigen für bessere Sichtbarkeit
        warped_display = cv2.resize(warped, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Kalibrierung - Ring", warped_display)
    else:
        # Falls kein Ring da ist, Fenster leeren oder Text anzeigen
        blank = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(blank, "Kein Ring gefunden", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Kalibrierung - Ring", blank)

    # Live-Video anzeigen
    cv2.imshow("Live Video", frame_bgr)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aufräumen
picam2.stop()
cv2.destroyAllWindows()

# ==========================================
# CODE GENERATOR
# ==========================================
print("\n" + "="*50)
print("KALIBRIERUNG BEENDET. HIER IST DEIN NEUER CODE:")
print("Ersetze die Funktion 'classify_color' in deinem Hauptcode hiermit:\n")

print("def classify_color(hsv_pixel):")
print("    h, s, v = hsv_pixel")
print(f"    if v < {blk_v}: return \"Black\"")
print(f"    if s < {wht_s} and v > {wht_v}: return \"White\"")
print("    ")
print(f"    if h < {red_l} or h > {red_h}: return \"Red\"")
print(f"    elif {yel_min} < h < {yel_max}: return \"Yellow\"")
print(f"    elif {grn_min} < h < {grn_max}: return \"Green\"")
print(f"    elif {blu_min} < h < {blu_max}: return \"Blue\"")
print("    return \"Unknown\"")
print("="*50 + "\n")