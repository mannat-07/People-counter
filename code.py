import cv2
import datetime
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import time

# Initialize detector with better parameters
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Video input from webcam (real-time)
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    print("Please check if your camera is connected and not being used by another application")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera initialized successfully!")
print("Press 'q' to quit, 'r' to reset counters, 'd' to toggle debug mode")

# Variables
textIn, textOut = 0, 0
people_inside = 0  # Net count of people currently inside
# Vertical lines (for a 600px-wide frame). Adjust as needed.
line_blue_x = 150  # Blue line (left side)
line_red_x = 450   # Red line (right side)
next_id = 0
object_last_position = {}
tracked_objects = {}  # id -> list of recent centers (trajectory)
# Per-object state for robust crossing detection
# object_state[id] = {
#   'last_side': 'left' | 'right' | 'mid',
#   'visited_blue': bool,
#   'visited_red': bool,
#   'last_count_time': float
# }
object_state = {}
frame_skip = 2  # Process every 2nd frame for performance
frame_count = 0
debug_mode = False
last_detection_time = {}
trajectory_length = 20  # Number of points to track for each object (longer history)
min_trajectory_for_counting = 3  # Minimum trajectory points before counting
# Crossing stability controls
margin_px = 8    # Pixel margin around lines to reduce jitter-triggered side flips
debounce_sec = 1.0  # Minimum time between counts for same ID
# Detection confidence threshold (adjustable at runtime with [ and ])
weight_thresh = 0.4
ui_message = ""
ui_message_until = 0.0
# Midpoint-crossing and ROI
roi_pad = 20        # extra width around the two lines to accept detections
mid_margin = 10     # tolerance around the midpoint for side change
# Detection filtering and counting stability
min_box_area = 1800            # filter out tiny detections
min_aspect, max_aspect = 0.25, 0.8  # acceptable w/h aspect range for a person
min_dx_for_count = 10          # require at least this much net x movement across recent points
trend_len = 3                  # number of recent points to check for directional trend

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Always resize frame for consistent display
    frame = imutils.resize(frame, width=600)
    
    frame_count += 1
    
    # Skip detection on some frames but always display
    if frame_count % frame_skip != 0:
        # Just display the frame without detection
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            textIn, textOut = 0, 0
            people_inside = 0
            object_last_position.clear()
            tracked_objects.clear()
            object_state.clear()
            last_detection_time.clear()
            print("Counters reset!")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key in (ord('a'), ord('d'), ord('j'), ord('l')):
            # Move lines for calibration: a/d for blue, j/l for red
            width = frame.shape[1]
            if key == ord('a'):
                line_blue_x = max(10, line_blue_x - 5)
            elif key == ord('d'):
                line_blue_x = min(width - 30, line_blue_x + 5)
            elif key == ord('j'):
                line_red_x = max(30, line_red_x - 5)
            elif key == ord('l'):
                line_red_x = min(width - 10, line_red_x + 5)
            # Enforce minimum separation
            if line_red_x - line_blue_x < 20:
                line_red_x = line_blue_x + 20
            ui_message = f"Lines: BLUE={line_blue_x}, RED={line_red_x}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        elif key == ord('['):
            weight_thresh = max(0.0, round(weight_thresh - 0.05, 2))
            ui_message = f"Sensitivity: {weight_thresh:.2f}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        elif key == ord(']'):
            weight_thresh = min(1.0, round(weight_thresh + 0.05, 2))
            ui_message = f"Sensitivity: {weight_thresh:.2f}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        elif key == ord('z'):
            roi_pad = max(0, roi_pad - 2)
            ui_message = f"ROI pad: {roi_pad}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        elif key == ord('x'):
            roi_pad = min(100, roi_pad + 2)
            ui_message = f"ROI pad: {roi_pad}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        elif key == ord('c'):
            mid_margin = max(2, mid_margin - 1)
            ui_message = f"Mid margin: {mid_margin}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        elif key == ord('v'):
            mid_margin = min(40, mid_margin + 1)
            ui_message = f"Mid margin: {mid_margin}"
            ui_message_until = time.time() + 1.5
            print(ui_message)
        
        # Draw lines and a clean header overlay with counters on skipped frames
        cv2.line(frame, (line_blue_x, 0), (line_blue_x, frame.shape[0]), (255, 0, 0), 3)
        cv2.line(frame, (line_red_x, 0), (line_red_x, frame.shape[0]), (0, 0, 255), 3)

        overlay = frame.copy()
        header_h = 80
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        cv2.putText(frame, f"IN: {textIn}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 0), 2)
        cv2.putText(frame, f"OUT: {textOut}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2)
        cv2.putText(frame, f"INSIDE: {people_inside}", (200, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        # Always show current sensitivity
        cv2.putText(frame, f"SENS: {weight_thresh:.2f}", (frame.shape[1] - 170, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
        # Transient UI message
        if time.time() < ui_message_until and ui_message:
            cv2.putText(frame, ui_message, (frame.shape[1] - 340, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("People Counter", frame)
        continue

    # Preprocess for more robust detection (equalize illumination on Y channel)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = clahe.apply(y)
    ycrcb = cv2.merge((y, cr, cb))
    detect_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Detect people with optimized parameters, restricted to ROI (doorway band) for fewer false positives
    # Compute band first (uses current line positions)
    x_mid = (line_blue_x + line_red_x) // 2
    band_left = max(0, min(line_blue_x, line_red_x) - roi_pad)
    band_right = min(frame.shape[1] - 1, max(line_blue_x, line_red_x) + roi_pad)

    roi_img = detect_img[:, band_left:band_right] if band_right > band_left else detect_img
    (rects, weights) = hog.detectMultiScale(roi_img, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Map ROI rectangles back to full-frame coordinates
    rects = [(x + band_left, y, w, h) for (x, y, w, h) in rects]

    # Confidence filter
    if len(rects) > 0 and len(weights) == len(rects):
        rects_weights = [((x, y, w, h), float(wt)) for (x, y, w, h), wt in zip(rects, weights)]
        rects = [(x, y, w, h) for (x, y, w, h), wt in rects_weights if wt >= weight_thresh]
        weights = [wt for (x, y, w, h), wt in rects_weights if wt >= weight_thresh]

    # Size/aspect filters to remove spurious boxes
    filtered = []
    for i, (x, y, w, h) in enumerate(rects):
        area = w * h
        aspect = (w / float(h)) if h > 0 else 0
        if area < min_box_area:
            continue
        if not (min_aspect <= aspect <= max_aspect):
            continue
        filtered.append((x, y, w, h))
    rects = filtered

    # Convert to (x1, y1, x2, y2) for NMS
    rects_np = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    probs = np.array(weights).reshape(-1) if len(rects) > 0 and len(weights) == len(rects) else None
    picks = non_max_suppression(rects_np, probs=probs, overlapThresh=0.6) if len(rects_np) > 0 else []

    current_positions = []
    for (x1, y1, x2, y2) in picks:
        w, h = (x2 - x1), (y2 - y1)
        cx, cy = x1 + w // 2, y1 + h // 2
        # Already restricted to ROI; accept directly
        current_positions.append(((x1, y1, w, h), (cx, cy)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Track & Count with robust vertical line crossing logic (Blue->Red = IN, Red->Blue = OUT)
    current_time = time.time()
    
    # Clean up old objects (not seen for more than 5 seconds)
    for obj_id in list(object_last_position.keys()):
        if obj_id in last_detection_time:
            if current_time - last_detection_time[obj_id] > 5.0:
                del object_last_position[obj_id]
                if obj_id in tracked_objects:
                    del tracked_objects[obj_id]
                if obj_id in last_detection_time:
                    del last_detection_time[obj_id]
    
    for (box, center) in current_positions:
        matched_id = None
        min_distance = float('inf')
        (bx, by, bw, bh) = box
        dyn_thresh = max(60, int(0.6 * max(bw, bh)))  # Dynamic match threshold by size

        # Find closest existing object
        for obj_id, last_center in object_last_position.items():
            dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
            if dist < dyn_thresh and dist < min_distance:
                matched_id = obj_id
                min_distance = dist

        if matched_id is None:
            matched_id = next_id
            next_id += 1
            tracked_objects[matched_id] = []
            # Initialize last_side based on position vs midpoint
            init_side = 'left' if center[0] < (x_mid - mid_margin) else ('right' if center[0] > (x_mid + mid_margin) else 'mid')
            # Initialize visited flags based on initial placement relative to Blue/Red
            blue_on_left = (line_blue_x < line_red_x)
            on_blue_side = (init_side == 'left') if blue_on_left else (init_side == 'right')
            on_red_side = (init_side == 'right') if blue_on_left else (init_side == 'left')
            object_state[matched_id] = {
                'last_side': init_side,
                'visited_blue': True if on_blue_side else False,
                'visited_red': True if on_red_side else False,
                'last_count_time': 0.0,
            }

        object_last_position[matched_id] = center
        last_detection_time[matched_id] = current_time

        # Store trajectory (limit to recent points)
        tracked_objects[matched_id].append(center)
        if len(tracked_objects[matched_id]) > trajectory_length:
            tracked_objects[matched_id] = tracked_objects[matched_id][-trajectory_length:]

        # Count when crossing the MIDPOINT (robust and simple)
        if len(tracked_objects[matched_id]) >= min_trajectory_for_counting:
            x_curr = tracked_objects[matched_id][-1][0]
            # Determine side relative to midpoint
            if x_curr < (x_mid - mid_margin):
                side = 'left'
            elif x_curr > (x_mid + mid_margin):
                side = 'right'
            else:
                side = 'mid'

            st = object_state.get(matched_id)
            if st is None:
                st = {'last_side': side, 'visited_blue': False, 'visited_red': False, 'last_count_time': 0.0}
                object_state[matched_id] = st

            prev_side = st['last_side']
            # Update visited flags based on which color side this position represents
            blue_on_left = (line_blue_x < line_red_x)
            if side != 'mid':
                on_blue_side = (side == 'left') if blue_on_left else (side == 'right')
                on_red_side = (side == 'right') if blue_on_left else (side == 'left')
                if on_blue_side:
                    st['visited_blue'] = True
                if on_red_side:
                    st['visited_red'] = True

            # Helper: directional trend check to avoid counting on jitter
            trend_ok = True
            if len(tracked_objects[matched_id]) >= trend_len:
                recent = tracked_objects[matched_id][-trend_len:]
                dx = recent[-1][0] - recent[0][0]
                # For L->R expect positive dx, for R->L negative dx
                # We'll validate against the intended side change below with min_dx_for_count
            
            # Color-aware counting based on Blue→Red = IN, Red→Blue = OUT
            if blue_on_left:
                # Blue is on the LEFT, Red on the RIGHT
                # IN: left -> right, but only if previously visited BLUE side
                if (prev_side in ('left', 'mid')) and side == 'right' and st.get('visited_blue', False):
                    # check directional movement magnitude
                    if len(tracked_objects[matched_id]) >= trend_len:
                        recent = tracked_objects[matched_id][-trend_len:]
                        dx = recent[-1][0] - recent[0][0]
                        trend_ok = dx >= min_dx_for_count
                    else:
                        trend_ok = True
                    
                    if (current_time - st['last_count_time']) > debounce_sec:
                        if trend_ok:
                            textIn += 1
                            people_inside += 1
                            st['last_count_time'] = current_time
                            # reset visited flags after a successful count to avoid double-counting
                            st['visited_blue'] = False
                            st['visited_red'] = False
                            if debug_mode:
                                print(f"✅ ID {matched_id}: IN (Blue→Red via L→R). In={textIn}, Inside={people_inside}")

                # OUT: right -> left, but only if previously visited RED side
                if (prev_side in ('right', 'mid')) and side == 'left' and st.get('visited_red', False):
                    if len(tracked_objects[matched_id]) >= trend_len:
                        recent = tracked_objects[matched_id][-trend_len:]
                        dx = recent[-1][0] - recent[0][0]
                        trend_ok = dx <= -min_dx_for_count
                    else:
                        trend_ok = True
                    if (current_time - st['last_count_time']) > debounce_sec:
                        if trend_ok:
                            textOut += 1
                            people_inside = max(0, people_inside - 1)
                            st['last_count_time'] = current_time
                            st['visited_blue'] = False
                            st['visited_red'] = False
                            if debug_mode:
                                print(f"❌ ID {matched_id}: OUT (Red→Blue via R→L). Out={textOut}, Inside={people_inside}")
            else:
                # Blue is on the RIGHT, Red on the LEFT
                # IN: right -> left (Blue→Red), only if visited BLUE side
                if (prev_side in ('right', 'mid')) and side == 'left' and st.get('visited_blue', False):
                    if len(tracked_objects[matched_id]) >= trend_len:
                        recent = tracked_objects[matched_id][-trend_len:]
                        dx = recent[-1][0] - recent[0][0]
                        trend_ok = dx <= -min_dx_for_count
                    else:
                        trend_ok = True
                    if (current_time - st['last_count_time']) > debounce_sec:
                        if trend_ok:
                            textIn += 1
                            people_inside += 1
                            st['last_count_time'] = current_time
                            st['visited_blue'] = False
                            st['visited_red'] = False
                            if debug_mode:
                                print(f"✅ ID {matched_id}: IN (Blue→Red via R→L). In={textIn}, Inside={people_inside}")
                # OUT: left -> right (Red→Blue), only if visited RED side
                if (prev_side in ('left', 'mid')) and side == 'right' and st.get('visited_red', False):
                    if len(tracked_objects[matched_id]) >= trend_len:
                        recent = tracked_objects[matched_id][-trend_len:]
                        dx = recent[-1][0] - recent[0][0]
                        trend_ok = dx >= min_dx_for_count
                    else:
                        trend_ok = True
                    if (current_time - st['last_count_time']) > debounce_sec:
                        if trend_ok:
                            textOut += 1
                            people_inside = max(0, people_inside - 1)
                            st['last_count_time'] = current_time
                            st['visited_blue'] = False
                            st['visited_red'] = False
                            if debug_mode:
                                print(f"❌ ID {matched_id}: OUT (Red→Blue via L→R). Out={textOut}, Inside={people_inside}")

            # Update last side
            st['last_side'] = side

    # Draw VERTICAL lines
    cv2.line(frame, (line_blue_x, 0), (line_blue_x, frame.shape[0]), (255, 0, 0), 3)
    cv2.line(frame, (line_red_x, 0), (line_red_x, frame.shape[0]), (0, 0, 255), 3)

    # Clean header overlay with counters
    overlay = frame.copy()
    header_h = 80
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(frame, f"IN: {textIn}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 0), 2)
    cv2.putText(frame, f"OUT: {textOut}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2)
    cv2.putText(frame, f"INSIDE: {people_inside}", (200, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    # Always show current sensitivity
    cv2.putText(frame, f"SENS: {weight_thresh:.2f}", (frame.shape[1] - 170, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
    # Draw midpoint line for visual aid
    x_mid = (line_blue_x + line_red_x) // 2
    cv2.line(frame, (x_mid, 0), (x_mid, frame.shape[0]), (180, 180, 180), 1)

    # Transient UI message
    if time.time() < ui_message_until and ui_message:
        cv2.putText(frame, ui_message, (frame.shape[1] - 340, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("People Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset counters
        textIn, textOut = 0, 0
        people_inside = 0
        object_last_position.clear()
        tracked_objects.clear()
        object_state.clear()
        last_detection_time.clear()
        print("Counters reset!")
    elif key == ord('['):
        weight_thresh = max(0.0, round(weight_thresh - 0.05, 2))
        ui_message = f"Sensitivity: {weight_thresh:.2f}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key == ord(']'):
        weight_thresh = min(1.0, round(weight_thresh + 0.05, 2))
        ui_message = f"Sensitivity: {weight_thresh:.2f}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key == ord('z'):
        roi_pad = max(0, roi_pad - 2)
        ui_message = f"ROI pad: {roi_pad}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key == ord('x'):
        roi_pad = min(100, roi_pad + 2)
        ui_message = f"ROI pad: {roi_pad}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key == ord('c'):
        mid_margin = max(2, mid_margin - 1)
        ui_message = f"Mid margin: {mid_margin}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key == ord('v'):
        mid_margin = min(40, mid_margin + 1)
        ui_message = f"Mid margin: {mid_margin}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key in (ord('a'), ord('d'), ord('j'), ord('l')):
        # Move lines for calibration: a/d for blue, j/l for red
        width = frame.shape[1]
        if key == ord('a'):
            line_blue_x = max(10, line_blue_x - 5)
        elif key == ord('d'):
            line_blue_x = min(width - 30, line_blue_x + 5)
        elif key == ord('j'):
            line_red_x = max(30, line_red_x - 5)
        elif key == ord('l'):
            line_red_x = min(width - 10, line_red_x + 5)
        # Enforce minimum separation
        if line_red_x - line_blue_x < 20:
            line_red_x = line_blue_x + 20
        ui_message = f"Lines: BLUE={line_blue_x}, RED={line_red_x}"
        ui_message_until = time.time() + 1.5
        print(ui_message)
    elif key == ord('d'):
        # Toggle debug mode
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()