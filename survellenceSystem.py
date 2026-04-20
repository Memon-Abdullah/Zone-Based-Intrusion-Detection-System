from ultralytics import YOLO
import cv2
import numpy as np
import json
import os

# ─────────────────────────────────────────
model = YOLO("yolov8n.pt")

zone_points = []
zone_complete = False
zone_file = "zone.json"

# ─────────────────────────────────────────
def draw_zone(event, x, y, flags, param):
    global zone_points, zone_complete

    if event == cv2.EVENT_LBUTTONDOWN:
        if not zone_complete:
            zone_points.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        if len(zone_points) >= 3:
            zone_complete = True

# ─────────────────────────────────────────
def save_zone():
    with open(zone_file, "w") as f:
        json.dump(zone_points, f)

def load_zone():
    global zone_points, zone_complete
    if os.path.exists(zone_file):
        with open(zone_file, "r") as f:
            zone_points = json.load(f)
            zone_points = [tuple(p) for p in zone_points]
            zone_complete = True
            return True
    return False

# ─────────────────────────────────────────
def is_in_zone(point):
    zone_array = np.array(zone_points, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(zone_array, point, False) >= 0

# ─────────────────────────────────────────
# 🔥 3 POINT VOTING SYSTEM
def get_points(x1, y1, x2, y2):
    bottom = (int((x1 + x2) / 2), int(y2))
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    top = (int((x1 + x2) / 2), int(y1))
    return [bottom, center, top]

def voting(points):
    votes = sum([is_in_zone(p) for p in points])
    return votes >= 2

# ─────────────────────────────────────────
# 🎥 VIDEO
cap = cv2.VideoCapture("v2.mp4")
# 🔥 IMPORTANT: FPS control
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

ret, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (1280, 720))

cv2.namedWindow("Zone Setup")
cv2.setMouseCallback("Zone Setup", draw_zone)

zone_loaded = load_zone()

# ─────────────────────────────────────────
# 🟡 ZONE SETUP
while True:
    frame_copy = first_frame.copy()

    if zone_loaded:
        cv2.putText(frame_copy, "ENTER = use saved | R = redraw",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    for p in zone_points:
        cv2.circle(frame_copy, p, 5, (0,255,255), -1)

    for i in range(1, len(zone_points)):
        cv2.line(frame_copy, zone_points[i-1], zone_points[i], (0,255,255), 2)

    if zone_complete:
        cv2.line(frame_copy, zone_points[-1], zone_points[0], (0,255,255), 2)

    cv2.imshow("Zone Setup", frame_copy)

    key = cv2.waitKey(1)

    if key == ord('r'):
        zone_points = []
        zone_complete = False

    if key == 13 and zone_complete:
        save_zone()
        break

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# close setup window
cv2.destroyWindow("Zone Setup")

# ─────────────────────────────────────────
# 🟢 DETECTION LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # draw zone
    overlay = frame.copy()
    zone_array = np.array(zone_points, dtype=np.int32)
    cv2.fillPoly(overlay, [zone_array], (0,255,255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # YOLO
    results = model(frame, conf=0.5)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if model.names[cls] != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pts = get_points(x1, y1, x2, y2)
            in_zone = voting(pts)

            color = (0,0,255) if in_zone else (0,255,0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            for p in pts:
                cv2.circle(frame, p, 4, color, -1)

            if in_zone:
                cv2.putText(frame, "ALERT",
                           (30,50),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, (0,0,255), 3)

    cv2.imshow("Detection", frame)

    # 🔥 REAL SPEED CONTROL
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()