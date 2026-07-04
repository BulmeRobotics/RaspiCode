import cv2
import numpy as np

def get_boundary_coords(img_shape, side="R"):
    height, width = img_shape[:2]
    cutoff_top_y    = int(height * 0.25)
    cutoff_bottom_y = int(height * 0.9)
    cutoff_left_x   = int(width * (1/7))
    cutoff_right_x  = int(width * (6/7))

    cutoff_left_auswurf_bottom  = [260, 478]
    cutoff_left_auswurf_top     = [0, 202]
    cutoff_right_auswurf_bottom = [320, 478]
    cutoff_right_auswurf_top    = [636, 185]

    if side == "L":
        cutoff_left_auswurf_bottom  = [300, 478]
        cutoff_left_auswurf_top     = [0, 193]
        cutoff_right_auswurf_bottom = [377, 478]
        cutoff_right_auswurf_top    = [636, 210]

    return {
        'top_y': cutoff_top_y,
        'bottom_y': cutoff_bottom_y,
        'left_x': cutoff_left_x,
        'right_x': cutoff_right_x,
        'left_auswurf_bottom': cutoff_left_auswurf_bottom,
        'left_auswurf_top': cutoff_left_auswurf_top,
        'right_auswurf_bottom': cutoff_right_auswurf_bottom,
        'right_auswurf_top': cutoff_right_auswurf_top,
    }

def is_in_boundary(cx, cy, img_shape, side="R"):
    coords = get_boundary_coords(img_shape, side)
    
    if cy < coords['top_y'] or cy > coords['bottom_y']: return False
    if cx < coords['left_x'] or cx > coords['right_x']: return False

    left_top = coords['left_auswurf_top']
    left_bottom = coords['left_auswurf_bottom']
    if (cy > left_top[1] and cx < left_bottom[0]):
        lineY = left_top[1] + (cx - left_top[0]) * ((left_bottom[1]-left_top[1])/((left_bottom[0]-left_top[0])))
        if (lineY < cy): return False

    right_top = coords['right_auswurf_top']
    right_bottom = coords['right_auswurf_bottom']
    if (cx > right_bottom[0] and cy > right_top[1]):
        lineY = right_top[1] + (cx - right_top[0]) * ((right_bottom[1]-right_top[1])/((right_bottom[0]-right_top[0])))
        if (lineY < cy): return False
                    
    return True

def get_boundary_mask(img_shape, side="R"):
    coords = get_boundary_coords(img_shape, side)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    cv2.rectangle(mask, (coords['left_x'], coords['top_y']), (coords['right_x'], coords['bottom_y']), 255, -1)
    
    left_top = coords['left_auswurf_top']
    left_bottom = coords['left_auswurf_bottom']
    pts_left = np.array([
        left_top,
        left_bottom,
        [left_bottom[0], img_shape[0]],
        [0, img_shape[0]]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts_left], 0)
    
    right_top = coords['right_auswurf_top']
    right_bottom = coords['right_auswurf_bottom']
    pts_right = np.array([
        right_bottom,
        right_top,
        [img_shape[1], right_top[1]],
        [img_shape[1], img_shape[0]],
        [right_bottom[0], img_shape[0]]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts_right], 0)
    
    return mask

def test_mask():
    shape = (480, 640, 3)
    for side in ["R", "L"]:
        mask = get_boundary_mask(shape, side)
        errors = 0
        for y in range(0, shape[0], 5):
            for x in range(0, shape[1], 5):
                allowed_func = is_in_boundary(x, y, shape, side)
                allowed_mask = mask[y, x] > 0
                if allowed_func != allowed_mask:
                    errors += 1
                    print(f"Mismatch at {x}, {y} for {side}: func={allowed_func}, mask={allowed_mask}")
                    if errors > 10: 
                        print("Too many errors, stopping.")
                        return
        print(f"Side {side} tested, errors: {errors}")

test_mask()
