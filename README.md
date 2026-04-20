# 🔐 Zone-Based Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10.11-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-1.x-orange?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

A real-time computer vision surveillance system that detects human intrusions into user-defined restricted zones using **YOLOv8** object detection, **polygon-based zone geometry**, and a **3-point voting system** for robust, noise-resistant alerting.

---

## 📸 Demo

> Draw your zone → Watch detections → Get alerted when a person enters

```
[Camera/Video Feed]
       ↓
[YOLOv8 Person Detection]
       ↓
[3-Point Bounding Box Voting]
       ↓
[Zone Intersection Test (pointPolygonTest)]
       ↓
[🔴 ALERT if person in zone | 🟢 Safe if outside]
```

---

## 🧠 How It Works — Technical Deep Dive

### 1. Object Detection — YOLOv8 (CNN-Based)

YOLOv8 is built on a **Convolutional Neural Network (CNN)** backbone. Unlike traditional sliding-window detectors, YOLO runs a **single forward pass** over the entire image, predicting bounding boxes and class probabilities simultaneously.

```
Input Image (640x640)
       ↓
[ Backbone — CSPDarknet ]   → Extracts features: edges, shapes, textures
       ↓
[ Neck — PANet/FPN ]        → Multi-scale feature fusion
       ↓
[ Head — Detection ]        → Outputs: (x, y, w, h, confidence, class)
```

**Model used:** `yolov8n.pt` (Nano) — fastest variant, optimized for real-time inference on CPU.

| Model    | Parameters | Speed  | mAP   |
|----------|-----------|--------|-------|
| yolov8n  | 3.2M      | ⚡⚡⚡  | 37.3  |
| yolov8s  | 11.2M     | ⚡⚡    | 44.9  |
| yolov8m  | 25.9M     | ⚡     | 50.2  |
| yolov8l  | 43.7M     | 🐢    | 52.9  |
| yolov8x  | 68.2M     | 🐢🐢   | 53.9  |

---

### 2. Zone Definition — Polygon Drawing

The user draws a **custom polygon zone** by clicking points on the first frame. This zone is stored as a list of `(x, y)` coordinates and **persisted to disk** as `zone.json` for reuse across sessions.

```python
zone_points = [(x1,y1), (x2,y2), (x3,y3), ...]  # polygon vertices
```

Zone is saved/loaded automatically:
```
zone.json  →  persists between runs
             no need to redraw every time
```

---

### 3. Point-in-Polygon Test — Ray Casting Algorithm

To check if a person is inside the zone, the system uses **OpenCV's `pointPolygonTest`**, which internally implements the **Ray Casting algorithm** based on **cross products**.

**How Ray Casting works:**

```
Cast a ray from the point →→→→→→→→→→→→→→→→→→
Count how many times it crosses polygon edges:

Even crossings  →  OUTSIDE ❌
Odd  crossings  →  INSIDE  ✅
```

**Cross product basis:**
```
For each edge (A→B) of polygon:
  vector = B - A
  cross  = vector × (point - A)

  cross > 0  →  point is LEFT of edge
  cross < 0  →  point is RIGHT of edge
```

`pointPolygonTest` returns:
```
+ve  →  INSIDE the polygon
 0   →  ON the edge
-ve  →  OUTSIDE the polygon
```

---

### 4. 3-Point Voting System — Robust Detection

Instead of checking a single point (e.g. foot only), the system samples **3 key points** from the bounding box and uses a **majority voting** strategy:

```
Bounding Box Points:
  ● Bottom Center  (foot point)
  ● Body Center    (torso)
  ● Top Center     (head)

Voting Rule:
  2 or more points inside zone  →  Person IS in zone ✅
  0 or 1 point inside zone      →  Person NOT in zone ❌
```

**Why voting?**

Single-point checks suffer from:
- Edge cases where foot is slightly outside
- Partial occlusion of person
- Slight bounding box jitter between frames

Voting makes the system **noise-resistant and stable**.

---

### 5. FPS-Aware Playback

The system reads the video's native FPS and computes the correct frame delay:

```python
fps   = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)  # milliseconds per frame
```

This ensures video plays at **real speed** instead of running as fast as the CPU allows.

---

## 🗂️ Project Structure

```
yolo_project/
│
├── venv/                  # Python virtual environment
├── yolov8n.pt             # YOLOv8 Nano model weights
├── surveillance.py        # Main application code
├── zone.json              # Saved zone coordinates (auto-generated)
├── test_video.mp4         # Input video file
└── README.md              # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10.11
- pip
- Windows / Linux / macOS

### Step 1 — Create Virtual Environment
```bash
mkdir yolo_project
cd yolo_project
python -m venv venv
```

### Step 2 — Activate Environment
```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### Step 4 — Add Your Video
Place your video file in the project folder and update this line in the code:
```python
cap = cv2.VideoCapture("your_video.mp4")
```

### Step 5 — Run
```bash
python surveillance.py
```

---

## 🎮 Controls

| Key / Action        | Description                        |
|---------------------|------------------------------------|
| `Left Click`        | Add a zone point                   |
| `Right Click`       | Complete and close the zone        |
| `Enter`             | Confirm zone and start detection   |
| `R`                 | Reset and redraw zone              |
| `Q`                 | Quit application                   |

---

## 🔄 System Flow

```
START
  │
  ├─► Load zone.json (if exists)
  │       └─► Show "use saved / redraw" option
  │
  ├─► Zone Setup Screen (first frame)
  │       └─► User draws polygon by clicking
  │       └─► Right click to close polygon
  │       └─► Enter to confirm → zone saved to zone.json
  │
  ├─► Detection Loop
  │       ├─► Read frame from video
  │       ├─► Resize to 1280×720
  │       ├─► Draw transparent zone overlay
  │       ├─► Run YOLOv8 (conf ≥ 0.5, class = person only)
  │       ├─► For each detected person:
  │       │       ├─► Extract 3 bounding box points
  │       │       ├─► Run pointPolygonTest on each point
  │       │       ├─► Voting: 2/3 inside → ALERT
  │       │       └─► Draw RED box + ALERT text if in zone
  │       └─► Display frame with delay = 1000/fps ms
  │
  └─► END → release camera → destroy windows
```

---

## 🧩 Key Concepts Summary

| Concept               | Technology Used              | Purpose                             |
|-----------------------|------------------------------|-------------------------------------|
| Object Detection      | YOLOv8 (CNN)                 | Detect persons in each frame        |
| Zone Geometry         | Polygon (pointPolygonTest)   | Define restricted area              |
| Point-in-Polygon      | Ray Casting + Cross Product  | Check if person is inside zone      |
| Robust Detection      | 3-Point Majority Voting      | Reduce false positives              |
| Zone Persistence      | JSON file (zone.json)        | Save/load zone across sessions      |
| Real-time Playback    | FPS-aware frame delay        | Correct video speed                 |

---

## 🚀 Future Improvements

- [ ] Multi-zone support (multiple restricted areas)
- [ ] Person tracking with unique IDs (DeepSORT)
- [ ] Email / SMS alert on intrusion
- [ ] Save alert clips automatically
- [ ] Web dashboard for live monitoring
- [ ] Fine-tune YOLOv8 on custom surveillance dataset
- [ ] Export to edge devices (Raspberry Pi / Jetson Nano)

---

## 👤 Author

**Abdullah Memon;**     

---

