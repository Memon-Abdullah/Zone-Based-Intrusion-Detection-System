# Import YOLO from ultralytics
from ultralytics import YOLO

# Import opencv for camera and drawing
import cv2

# Import numpy for array handling
import numpy as np

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# ─────────────────────────────────────────
# ZONE DRAWING SETUP
# We will store points clicked by mouse here
zone_points = []  # list of (x, y) points
zone_complete = False  # becomes True when zone is drawn

# ─────────────────────────────────────────
# MOUSE CALLBACK FUNCTION
# This function runs automatically whenever you click the mouse
def draw_zone(event, x, y, flags, param):
    global zone_points, zone_complete

    # LEFT CLICK = add a new point to zone
    if event == cv2.EVENT_LBUTTONDOWN:
        if not zone_complete:
            zone_points.append((x, y))  # save clicked point
            print(f"Point added: ({x}, {y})")

    # RIGHT CLICK = complete/finish the zone
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(zone_points) >= 3:  # need at least 3 points for a polygon
            zone_complete = True
            print("Zone complete!")

# ─────────────────────────────────────────
# FUNCTION TO CHECK IF POINT IS IN ZONE
# Uses OpenCV pointPolygonTest
# Technically: uses ray casting + cross product internally
def is_in_zone(point, zone):
    # Convert zone points to numpy array (required by opencv)
    # Shape must be (N, 1, 2) for pointPolygonTest
    zone_array = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))

    # pointPolygonTest returns:
    # +ve number → point is INSIDE
    # -ve number → point is OUTSIDE
    #  0         → point is ON the edge
    result = cv2.pointPolygonTest(zone_array, point, False)

    return result >= 0  # True if inside or on edge

# ─────────────────────────────────────────
# FUNCTION TO GET BOTTOM CENTER OF BOUNDING BOX
# Why bottom center? Because feet touch the ground
# More accurate to check if person is in zone by feet position
def get_foot_point(x1, y1, x2, y2):
    foot_x = int((x1 + x2) / 2)  # center horizontally
    foot_y = int(y2)              # bottom of bounding box
    return (foot_x, foot_y)

# ─────────────────────────────────────────
# OPEN CAMERA
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # width in pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # height in pixels

# Create window and attach mouse callback to it
cv2.namedWindow("YOLO Zone Detection")
cv2.setMouseCallback("YOLO Zone Detection", draw_zone)

print("LEFT CLICK to add zone points")
print("RIGHT CLICK to complete the zone")
print("Press Q to quit")

# ─────────────────────────────────────────
# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── DRAW ZONE ON FRAME ──
    if len(zone_points) > 0:

        # Draw each point as a small circle
        for point in zone_points:
            cv2.circle(frame, point, 5, (0, 255, 255), -1)  # yellow dot

        # Draw lines between points
        for i in range(1, len(zone_points)):
            cv2.line(frame, zone_points[i-1], zone_points[i], (0, 255, 255), 2)

        # If zone is complete, close the polygon (connect last point to first)
        if zone_complete:
            cv2.line(frame, zone_points[-1], zone_points[0], (0, 255, 255), 2)

            # Fill the zone with transparent color
            zone_array = np.array(zone_points, dtype=np.int32)
            overlay = frame.copy()  # copy frame
            cv2.fillPoly(overlay, [zone_array], (0, 255, 255))  # fill yellow
            # Blend original frame with overlay (30% transparent)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # ── RUN YOLO DETECTION ──
    results = model(frame, stream=True, conf=0.5)

    for result in results:
        boxes = result.boxes  # get all detected boxes

        for box in boxes:
            # Get class id and class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Only process PERSON detections (class_id 0 = person in COCO)
            if class_name == "person":

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get confidence score
                conf = float(box.conf[0])

                # Get foot point (bottom center of box)
                foot = get_foot_point(x1, y1, x2, y2)

                # ── CHECK IF PERSON IS IN ZONE ──
                if zone_complete:
                    in_zone = is_in_zone(foot, zone_points)
                else:
                    in_zone = False

                # Set color based on zone status
                # RED = in zone, GREEN = outside zone
                color = (0, 0, 255) if in_zone else (0, 255, 0)

                # Draw bounding box around person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw foot point
                cv2.circle(frame, foot, 6, color, -1)

                # Show label with confidence
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show ALERT if person is in zone
                if in_zone:
                    cv2.putText(frame, "!! PERSON IN ZONE !!",
                               (30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               1.2, (0, 0, 255), 3)

    # Show instructions on screen
    if not zone_complete:
        cv2.putText(frame, "Left Click: Add Point | Right Click: Complete Zone",
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the final frame
    cv2.imshow("YOLO Zone Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()