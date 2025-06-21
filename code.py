import cv2
import datetime
import imutils

# Initialize detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Video input
cap = cv2.VideoCapture("testing.mp4")

# Variables
textIn, textOut = 0, 0
line_in_y = 220
line_out_y = 280

next_id = 0
object_last_position = {}
tracked_objects = {}
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)

    # Detect people
    (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    current_positions = []

    for (x, y, w, h) in rects:
        cx, cy = x + w // 2, y + h // 2
        current_positions.append(((x, y, w, h), (cx, cy)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Track & Count
    for (box, center) in current_positions:
        matched_id = None

        for obj_id, last_center in object_last_position.items():
            dist = abs(center[0] - last_center[0]) + abs(center[1] - last_center[1])
            if dist < 60:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        object_last_position[matched_id] = center

        if matched_id not in tracked_objects:
            tracked_objects[matched_id] = []

        tracked_objects[matched_id].append(center)

        if matched_id not in counted_ids and len(tracked_objects[matched_id]) >= 2:
            y_prev = tracked_objects[matched_id][-2][1]
            y_curr = tracked_objects[matched_id][-1][1]

            if y_prev < line_in_y and y_curr >= line_in_y:
                textIn += 1
                counted_ids.add(matched_id)

            elif y_prev > line_out_y and y_curr <= line_out_y:
                textOut += 1
                counted_ids.add(matched_id)

    # Draw lines
    cv2.line(frame, (0, line_in_y), (frame.shape[1], line_in_y), (255, 0, 0), 2) #red
    cv2.line(frame, (0, line_out_y), (frame.shape[1], line_out_y), (0, 0, 255), 2) #blue

    # Display count & timestamp
    cv2.putText(frame, f"In: {textIn}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {textOut}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imshow("Security Feed", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
