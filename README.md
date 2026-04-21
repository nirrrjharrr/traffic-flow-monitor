# Traffic Flow Monitor

A computer vision pipeline for vehicle detection, tracking, and counting from fixed-camera road footage. Built with YOLOv8 and OpenCV, it uses persistent multi-object tracking and a line-crossing logic to produce per-class vehicle counts and flow rate estimates.

---

## What Problem It Solves

Manual traffic surveys are labor-intensive and inconsistent. This system automates vehicle counting from pre-recorded or real-time video, providing structured output (total count, per-class breakdown, flow rate) that can inform traffic signal timing, congestion studies, or transport planning — without requiring roadside hardware beyond a camera.

---

## Key Features

- **YOLOv8s detection** with class-specific confidence thresholds (lower threshold for motorcycles to compensate for their smaller visual footprint)
- **ByteTrack multi-object tracking** with persistent IDs across frames, preventing double-counts due to temporary occlusion
- **Binary line-crossing counter** with a configurable dead-band zone to suppress jitter near the count line
- **ROI masking** via a binary PNG mask to restrict detection to a specific road region, reducing false positives from background clutter
- **Per-class counting** for cars, trucks, buses, and motorcycles, with a flow rate (vehicles/minute) calculated from elapsed time since first detection
- **Minimum bounding-box area filter** to discard spurious small detections
- **Video output** written to disk alongside real-time display

---

## How It Works

```
Input Video
    │
    ▼
Mask Application (bitwise AND with ROI mask)
    │
    ▼
YOLOv8s Detection + ByteTrack Tracking
    │   (runs on masked frame; returns boxes, class labels, track IDs, confidence)
    │
    ▼
Detection Filtering
    │   - class must be in {car, truck, bus, motorcycle}
    │   - confidence ≥ threshold (class-specific)
    │   - bounding box area ≥ MIN_BOX_AREA
    │
    ▼
LineCrossCounter.update()
    │   - centroid y-coordinate classified as above / below / inside dead-band
    │   - a count is registered when a tracked ID transitions from one side to the other
    │   - each ID is counted at most once
    │
    ▼
Annotated Frame (bounding boxes, IDs, centroids, HUD overlay)
    │
    ▼
Display + Video Write (output.mp4)
```

The dead-band (`LINE_TOLERANCE = 8 px`) prevents a centroid hovering at the line from triggering multiple counts in consecutive frames.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8s.pt`) |
| Multi-object tracking | ByteTrack (via `ultralytics` tracker config) |
| Video I/O & rendering | OpenCV (`cv2`) |
| Numerical operations | NumPy (via OpenCV internals) |
| Python | 3.10+ |

---

## Setup and Usage

### Prerequisites

```bash
pip install ultralytics opencv-python
```

The YOLOv8s weights (`yolov8s.pt`) are downloaded automatically by Ultralytics on first run.

### File structure expected

```
project/
├── main.py
├── mask.png          # binary mask (white = ROI, black = ignored)
└── videos/
    └── traffic_main.mp4
```

### Running

```bash
python main.py
```

**Controls during playback:**

| Key | Action |
|---|---|
| `S` | Save current frame as `sample_output.png` |
| `ESC` | Exit and print final counts |

Output video is saved to `output.mp4`.

### Configuration

All tunable parameters are grouped at the top of `main.py`:

```python
VIDEO_PATH             = "videos/traffic_main.mp4"
MODEL_PATH             = "yolov8s.pt"
MASK_PATH              = "mask.png"
CONF_THRESHOLD         = 0.45
MOTORCYCLE_CONF_THRESHOLD = 0.30
MIN_BOX_AREA           = 400
LINE_Y_RATIO           = 0.55   # count line as fraction of frame height
LINE_TOLERANCE         = 8      # dead-band half-width in pixels
```

---

## Example Output

At exit, the terminal prints a structured summary:

```
── FINAL COUNT ─────────────────────────
  Total vehicles : 148
  Flow rate      : 12.40 vehicles/min
  car            : 97
  motorcycle     : 31
  truck          : 14
  bus            : 6
  Sum check      : 148 == 148
─────────────────────────────────────────
```

The HUD overlay on each frame shows running totals and flow rate in real time. A `sample_output.png` can be saved mid-run by pressing `S`.

---

## Limitations

These are genuine constraints of the current design, not implementation bugs. Any deployment should account for them.

**1. The mask is manually drawn and not transferable.**
The ROI mask (`mask.png`) is created for one specific camera angle and scene. Using a different camera position, zoom level, or mounting height requires drawing a new mask from scratch. There is no automatic scene adaptation.

**2. Performance is sensitive to camera angle and motion direction.**
The line-crossing logic assumes vehicles move predominantly in one direction (top-to-bottom or bottom-to-top across the count line). Intersections with crossing flows, U-turns, or vehicles that stop on the line will produce incorrect counts. The counter also has no directionality — it counts any transition, regardless of travel direction.

**3. YOLO misclassification is a real source of error.**
The pretrained COCO weights were not optimized for traffic surveillance. Common failure modes observed: trucks classified as cars at distance, motorcycles missed when occluded by another vehicle or when viewed at an angle, and three-wheelers (common in local traffic) having no corresponding class at all. The lower confidence threshold for motorcycles partially compensates but increases false positive risk.

**4. Lighting and weather degrade performance.**
Detection quality degrades under low light, glare, rain, or fog. The model has no domain adaptation for these conditions and was not fine-tuned on local footage. Night-time operation is unreliable without supplemental preprocessing (e.g., histogram equalization, IR camera feed).

**5. Occlusion causes track fragmentation.**
When a vehicle is fully occluded behind another (common in dense traffic), ByteTrack may assign it a new ID when it re-emerges. If the centroid crosses the line after the ID changes, this can result in double-counting.

**6. Resolution and inference speed trade-off is fixed.**
Inference runs at `imgsz=960`, which provides reasonable accuracy for small objects (motorcycles) but increases per-frame latency. On CPU, real-time processing is unlikely. No frame-skipping or adaptive resolution logic is currently implemented.

**7. Not validated against ground truth.**
There is no evaluation against a manually annotated count. Reported counts should be treated as estimates. Accuracy on a given video depends heavily on scene complexity and camera quality.

---

## Possible Improvements

**Adaptive or learned ROI selection**
Replace the static binary mask with a learned or semi-automatic region proposal — for example, using perspective transform to define a road polygon from lane markings, or training a segmentation model to identify drivable area.

**Domain-specific fine-tuning**
Fine-tune YOLOv8 on a labeled dataset from local traffic conditions. This would improve detection of motorcycles, auto-rickshaws, and other vehicle types underrepresented in COCO. Datasets such as MIO-TCD or custom-annotated local footage would be appropriate starting points.

**Bidirectional and multi-lane counting**
Extend the counter to distinguish direction of travel (using centroid trajectory slope) and to handle multiple lanes independently via separate virtual lines or lane-aware tracking zones.

**Small object detection improvement**
Replace or supplement YOLOv8s with a model variant tuned for small object detection, or apply super-resolution preprocessing on high-density regions of the frame.

**Trajectory-based analysis**
Store per-track centroid histories to enable speed estimation (via homography mapping from image to world coordinates), lane-change detection, and vehicle queue analysis.

**Edge deployment and live stream support**
Integrate with RTSP streams from CCTV cameras. Profile and optimize for deployment on edge hardware (e.g., NVIDIA Jetson, Raspberry Pi with Coral accelerator), including model quantization (INT8) and TensorRT export.

**Evaluation framework**
Add a ground-truth annotation tool and evaluation script to measure precision, recall, and counting accuracy (MOTA/MOTP or similar) on labeled video segments.

---

## License

MIT
