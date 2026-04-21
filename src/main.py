import time
import cv2
from collections import defaultdict
from ultralytics import YOLO

#CONFIG

VIDEO_PATH   = "videos/traffic_main.mp4"
MODEL_PATH   = "yolov8s.pt"
WINDOW_NAME  = "Traffic Counter"
MASK_PATH    = "assets/mask.png"

CONF_THRESHOLD = 0.45
MOTORCYCLE_CONF_THRESHOLD = 0.30
MIN_BOX_AREA   = 400

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

LINE_Y_RATIO   = 0.55
LINE_TOLERANCE = 8

CLASS_COLORS = {
    "car":        (0,  200, 255),
    "truck":      (255, 80,  80),
    "bus":        (80, 255, 100),
    "motorcycle": (255,  80, 200),
}
DEFAULT_COLOR   = (200, 200, 200)
COUNTED_COLOR   = (0,   255,   0)
UNCOUNTED_COLOR = (0,   0,   255)
LINE_COLOR      = (0,   255, 255)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

#COUNTER

class LineCrossCounter:
    def __init__(self, line_y: int):
        self.line_y     = line_y
        self.prev_side: dict[int, int] = {}
        self.counted:   set[int]       = set()
        self.total      = 0
        self.per_class: dict[str, int] = defaultdict(int)
        self._start_time: float | None = None

    def _side(self, cy: int) -> int:
        if cy < self.line_y - LINE_TOLERANCE:
            return -1   # above line
        if cy > self.line_y + LINE_TOLERANCE:
            return 1    # below line
        return 0        # inside dead-band — don't commit

    def flow_rate(self) -> float:
        if self._start_time is None or self.total == 0:
            return 0.0
        elapsed = (time.time() - self._start_time) / 60.0
        return self.total / elapsed if elapsed > 0 else 0.0

    def update(self, detections: list[dict]) -> None:
        for det in detections:
            tid          = det["id"]
            cy           = det["centroid"][1]
            current_side = self._side(cy)

            if current_side == 0:
                continue  # centroid inside dead-band, skip

            if tid in self.prev_side:
                prev = self.prev_side[tid]
                if prev != 0 and prev != current_side and tid not in self.counted:
                    self.counted.add(tid)
                    self.total += 1
                    self.per_class[det["label"]] += 1
                    if self._start_time is None:
                        self._start_time = time.time()
                    print(f"  ✔ COUNTED  ID={tid}  cls={det['label']}  TOTAL={self.total}")

            self.prev_side[tid] = current_side

#HELPERS

def build_vehicle_class_ids(model) -> set[int]:
    return {idx for idx, name in model.names.items() if name in VEHICLE_CLASSES}


def parse_detections(result, model, vehicle_ids: set[int]) -> list[dict]:
    if result.boxes.id is None:
        return []

    detections = []
    for box, tid, cls, conf in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.id.cpu().numpy().astype(int),
        result.boxes.cls.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
    ):
        if cls not in vehicle_ids:
            continue

        label = model.names[cls]

        # Allow weaker motorcycle detections while keeping other classes strict.
        if label == "motorcycle":
            if conf < MOTORCYCLE_CONF_THRESHOLD:
                continue
        else:
            if conf < CONF_THRESHOLD:
                continue

        x1, y1, x2, y2 = map(int, box)
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append({
            "box":      (x1, y1, x2, y2),
            "id":       int(tid),
            "label":    label,
            "conf":     float(conf),
            "centroid": (cx, cy),
        })

    return detections

#DRAWING

def draw_detections(frame, detections, counter):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx, cy          = det["centroid"]
        label           = det["label"]
        tid             = det["id"]
        color           = CLASS_COLORS.get(label, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

        text = f"{label} #{tid}  {det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(frame, text, (x1 + 3, ty - 2), FONT, FONT_SCALE, (20, 20, 20), THICKNESS - 1)

        dot_color = COUNTED_COLOR if tid in counter.counted else UNCOUNTED_COLOR
        cv2.circle(frame, (cx, cy), 5, dot_color, -1)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

def draw_counting_line(frame, line_y, w):
    """Visible line for calibration. Once counting works, you can remove this call."""
    cv2.line(frame, (0, line_y), (w, line_y), LINE_COLOR, 2)
    cv2.putText(frame, f"COUNT LINE  y={line_y}", (8, line_y - 8),
                FONT, 0.45, LINE_COLOR, 1)


def draw_hud(frame, counter):
    fr = counter.flow_rate()
    flow_str = f"{fr:.1f} v/min" if fr > 0 else "waiting..."

    lines = [
        ("TOTAL", str(counter.total)),
        ("flow",  flow_str),
    ]
    for cls in ("car", "truck", "bus", "motorcycle"):
        n = counter.per_class.get(cls, 0)
        if n > 0:
            lines.append((cls, str(n)))

    # Layout constants
    TITLE      = "Traffic Flow Monitor"
    PAD        = 8
    LINE_H     = 18
    FONT_SMALL = 0.50
    FONT_TITLE = 0.55
    W_BOX      = 200
    TITLE_H    = 20

    box_h = PAD + TITLE_H + PAD // 2 + len(lines) * LINE_H + PAD

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (6 + W_BOX, 6 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Title
    cv2.putText(frame, TITLE,
                (PAD + 2, 6 + PAD + TITLE_H - 4),
                FONT, FONT_TITLE, (0, 255, 0), 1)

    # Divider line
    div_y = 6 + PAD + TITLE_H + 2
    cv2.line(frame, (10, div_y), (6 + W_BOX - 4, div_y), (60, 60, 60), 1)

    # Rows
    for i, (key, val) in enumerate(lines):
        y = div_y + PAD + i * LINE_H + LINE_H - 4

        if i == 0:
            color  = (0, 255, 180)
            weight = 1
            scale  = FONT_SMALL
        elif i == 1:
            color  = (180, 220, 255)
            weight = 1
            scale  = FONT_SMALL
        else:
            color  = (210, 210, 210)
            weight = 1
            scale  = FONT_SMALL

        cv2.putText(frame, f"{key:<12}", (PAD + 2, y),
                    FONT, scale, color, weight)

        (vw, _), _ = cv2.getTextSize(val, FONT, scale, weight)
        cv2.putText(frame, val, (6 + W_BOX - vw - PAD, y),
                    FONT, scale, color, weight)

#MAIN LOOP

def run(model, cap, mask, out):
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_y = int(h * LINE_Y_RATIO)

    vehicle_ids = build_vehicle_class_ids(model)
    counter     = LineCrossCounter(line_y)

    print(f"Frame size : {w}×{h}")
    print(f"Count line : y={line_y}  ({LINE_Y_RATIO:.0%} down the frame)")
    print(f"Dead-band  : y={line_y - LINE_TOLERANCE}–{line_y + LINE_TOLERANCE}")
    print("Press ESC or close window to stop.\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("End of video.")
            break

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        results = model.track(
            masked_frame,
            conf=min(CONF_THRESHOLD, MOTORCYCLE_CONF_THRESHOLD),
            classes=list(vehicle_ids),
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=960,
            verbose=False,
        )

        detections = []
        for result in results:
            detections.extend(parse_detections(result, model, vehicle_ids))

        counter.update(detections)

        #draw_counting_line(frame, line_y, w)
        draw_detections(frame, detections, counter)
        draw_hud(frame, counter)
        
        out.write(frame)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        
        # press 's' to save image
        if key == ord('s'):
            cv2.imwrite("assets/sample_output.png", frame)
            print("Image saved!")

        if key == 27:
            print("ESC — exiting.")
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed — exiting.")
            break

    print("\n── FINAL COUNT ─────────────────────────")
    print(f"  Total vehicles : {counter.total}")
    print(f"  Flow rate      : {counter.flow_rate():.2f} vehicles/min")
    for cls, n in sorted(counter.per_class.items()):
        print(f"  {cls:<14}: {n}")
    print(f"  Sum check      : {sum(counter.per_class.values())} == {counter.total}")
    print("─────────────────────────────────────────")

#ENTRY

def main():
    print("Loading model …")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {VIDEO_PATH}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))
    
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")
    mask = cv2.resize(mask, (int(cap.get(3)), int(cap.get(4))))

    try:
        run(model, cap, mask, out)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()