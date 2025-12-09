import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time

# -------------------------------
# Configurations
# -------------------------------
input_path = 'notstampede.mp4'         # Input video path
model_path = 'yolov8s.pt'           # 'yolov8n.pt'|'yolov8s.pt'|'yolov8m.pt'
conf_thres = 0.25                   # YOLO confidence
fps_smooth = 0.9                    # FPS EMA smoothing

# Define bottleneck ROIs as polygons (clockwise). Edit these to match your scene.
# Example: two exits and one stair. Points are (x, y) in image pixels.
ROIS = [
    {
        "name": "Exit A",
        "type": "exit",
        "poly": np.array([(50, 350), (250, 350), (250, 470), (50, 470)], dtype=np.int32)
    },
    {
        "name": "Exit B",
        "type": "exit",
        "poly": np.array([(900, 330), (1230, 330), (1230, 470), (900, 470)], dtype=np.int32)
    },
    {
        "name": "Stairs",
        "type": "stairs",
        "poly": np.array([(520, 140), (760, 140), (760, 320), (520, 320)], dtype=np.int32)
    }
]

# Thresholds (pixel-based to start; tune for your video)
# These are normalized per-ROI area for density, and per-frame counts for inflow.
DENSITY_HI = 2.0e-4      # people per pixel inside ROI (e.g., 0.0002)
INFLOW_HI = 4            # people newly entering ROI per second (approx)
SPEED_LOW = 1.5          # pixels/frame (below this = slow/stop)
PRESSURE_HI = 8.0        # arbitrary; density * speed_std
MAX_MATCH_DIST = 35      # px for nearest-neighbor centroid match between frames

# Visualization
ALPHA_FILL = 0.25
COLOR_SAFE = (60, 200, 60)
COLOR_WARN = (0, 215, 255)
COLOR_ALERT = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

# -------------------------------
# Helpers
# -------------------------------
def draw_translucent_poly(frame, poly, color, alpha=0.25, border=2):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=border)

def points_in_poly(points, poly):
    # Returns boolean mask for points inside polygon
    mask = []
    for (x, y) in points:
        res = cv2.pointPolygonTest(poly, (float(x), float(y)), False)  # >=0 inside/on-edge
        mask.append(res >= 0)
    return np.array(mask, dtype=bool)

def match_points(prev_pts, curr_pts, max_dist=30):
    # Simple nearest-neighbor matching for displacement estimation (no IDs)
    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return []
    prev = np.array(prev_pts, dtype=np.float32)
    curr = np.array(curr_pts, dtype=np.float32)
    matches = []
    used_prev = set()
    used_curr = set()
    for i, cp in enumerate(curr):
        dists = np.linalg.norm(prev - cp[None, :], axis=1)
        j = int(np.argmin(dists))
        if j not in used_prev and dists[j] <= max_dist:
            matches.append((prev[j], cp))
            used_prev.add(j)
            used_curr.add(i)
    return matches

def roi_area_pixels(poly):
    return cv2.contourArea(poly.astype(np.float32))

# -------------------------------
# State
# -------------------------------
model = YOLO(model_path)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ema_fps = fps
prev_time = time.time()

# Per-ROI history buffers
roi_hist = {
    roi["name"]: {
        "prev_centroids": [],          # list of (x, y) from previous frame
        "enter_count_buffer": deque(maxlen=30),  # per-frame entering counts
        "speed_buffer": deque(maxlen=30),        # per-frame mean speed
        "density_buffer": deque(maxlen=30),      # per-frame density
    }
    for roi in ROIS
}

# -------------------------------
# Main loop
# -------------------------------
win_name = "Stampede Risk (Bottleneck Monitoring)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference (persons only, class 0)
    results = model(frame, classes=[0], conf=conf_thres, verbose=False)
    boxes = results[0].boxes
    centroids = []

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centroids.append((cx, cy))
        # Draw boxes (optional; comment to reduce clutter)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    centroids_np = np.array(centroids, dtype=np.int32)

    # Per-ROI analytics
    for roi in ROIS:
        name = roi["name"]
        poly = roi["poly"]
        area_px = max(roi_area_pixels(poly), 1.0)

        hist = roi_hist[name]
        prev_pts = hist["prev_centroids"]

        # Inside/outside classification
        inside_mask = points_in_poly(centroids_np, poly) if len(centroids_np) else np.array([], dtype=bool)
        curr_inside = centroids_np[inside_mask].tolist()

        # Count entries: current inside that had no close predecessor inside last frame
        # Build "previous all" to detect entering events
        prev_all = np.array(prev_pts, dtype=np.float32) if len(prev_pts) else np.zeros((0, 2), dtype=np.float32)
        entering_count = 0
        for p in curr_inside:
            if len(prev_all) == 0:
                entering_count += 1
            else:
                dists = np.linalg.norm(prev_all - np.array(p, dtype=np.float32)[None, :], axis=1)
                if (dists.min() if len(dists) else 1e9) > MAX_MATCH_DIST:
                    entering_count += 1

        # Speed estimation for points currently inside (nearest-neighbor to previous frame)
        matches = match_points(prev_pts, curr_inside, max_dist=MAX_MATCH_DIST)
        speeds = [float(np.linalg.norm(c - p)) for (p, c) in matches]
        mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
        std_speed = float(np.std(speeds)) if len(speeds) else 0.0

        # Density = people per pixel in ROI
        density = (len(curr_inside) / area_px)

        # Pressure proxy (Helbing-inspired): density * speed variance
        pressure = density * (std_speed ** 2)

        # Convert entering_count per-frame to approx per-second using EMA FPS
        inflow_per_sec = entering_count * ema_fps

        # Save history (for smoothing/plots later if needed)
        hist["enter_count_buffer"].append(inflow_per_sec)
        hist["speed_buffer"].append(mean_speed)
        hist["density_buffer"].append(density)
        hist["prev_centroids"] = centroids_np.tolist()

        # Decision logic
        alert = False
        warn = False

        if (density > DENSITY_HI and inflow_per_sec > INFLOW_HI and mean_speed < SPEED_LOW) or (pressure > PRESSURE_HI):
            alert = True
        elif (density > 0.6 * DENSITY_HI and inflow_per_sec > 0.6 * INFLOW_HI):
            warn = True

        color = COLOR_ALERT if alert else (COLOR_WARN if warn else COLOR_SAFE)
        draw_translucent_poly(frame, poly, color, alpha=ALPHA_FILL, border=2)

        # Info text
        x, y = int(poly[:, 0].mean()), int(poly[:, 1].min()) - 10
        label = (f"{name} | dens:{density:.4f} | inflow/s:{inflow_per_sec:.1f} | "
                 f"spd(px/f):{mean_speed:.2f} | P:{pressure:.1f}")
        cv2.putText(frame, label, (max(10, x - 260), max(20, y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

        if alert:
            cv2.putText(frame, f"ALERT: {name}", (max(10, x - 90), max(40, y + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Global HUD
    cv2.putText(frame, f"People: {len(centroids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # FPS estimate
    now = time.time()
    inst_fps = 1.0 / max(now - prev_time, 1e-6)
    ema_fps = fps_smooth * ema_fps + (1 - fps_smooth) * inst_fps
    prev_time = now
    cv2.putText(frame, f"FPS: {ema_fps:.1f}", (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2, cv2.LINE_AA)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
