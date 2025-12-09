import cv2
import numpy as np
from ultralytics import YOLO
import math
import tkinter as tk
from tkinter import simpledialog
from collections import deque

# -------------------------------
# Globals
# -------------------------------
current_polygon = []
base_ground = []
polygons = []
temp_frame = None
drawing_base_done = False

# -------------------------------
# Video & YOLO settings
# -------------------------------
video_path = '1.0.mp4'
model_path = 'yolov8s.pt'
confidence_threshold = 0.25

# Realistic thresholds
density_threshold = 0.05      # people per m^2
speed_threshold = 0.05        # meters per frame
frames_to_average = 10
prediction_frames_ahead = 5
approach_count_threshold = 1  # 1 person approaching triggers predictive warning

# -------------------------------
# Mouse callback for drawing polygons
# -------------------------------
def draw_polygon(event, x, y, flags, param):
    global current_polygon, polygons, base_ground, temp_frame, drawing_base_done
    frame = temp_frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN and len(current_polygon) >= 3:
        if not drawing_base_done:
            base_ground.extend(current_polygon)
            drawing_base_done = True
        else:
            polygons.append(current_polygon.copy())
        current_polygon.clear()

    # Draw base ground (green)
    if base_ground:
        cv2.polylines(frame, [np.array(base_ground, np.int32)], True, (0,255,0), 2)
    # Draw all ROIs (red)
    for poly in polygons:
        cv2.polylines(frame, [np.array(poly, np.int32)], True, (0,0,255), 2)
    # Draw current polygon (blue)
    for i, pt in enumerate(current_polygon):
        cv2.circle(frame, pt, 4, (255,0,0), -1)
        if i > 0:
            cv2.line(frame, current_polygon[i-1], pt, (255,255,0), 2)

    temp_frame = frame
    cv2.imshow("Draw Base & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save", frame)

# -------------------------------
# Load video first frame
# -------------------------------
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video: check video_path")
frame = cv2.resize(frame, (1280, 720))
temp_frame = frame.copy()

# -------------------------------
# Interactive Drawing Loop
# -------------------------------
cv2.namedWindow("Draw Base & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save")
cv2.setMouseCallback("Draw Base & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save", draw_polygon)

print("Step 1: Draw base ground (green). Right click to finish base.")
print("Step 2: Draw ROIs (red). Right click to finish each ROI.")
print("Press 'u' to undo last point. Press 's' to finish drawing and start detection.")

while True:
    cv2.imshow("Draw Base & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u') and len(current_polygon) > 0:
        current_polygon.pop()
    if key == ord('s') and drawing_base_done:
        break

cv2.destroyAllWindows()
if not base_ground:
    raise RuntimeError("Base ground not drawn. Exiting.")

# -------------------------------
# Tkinter GUI to enter side lengths
# -------------------------------
def get_side_lengths(n_sides):
    root = tk.Tk()
    root.withdraw()
    lengths = []
    for i in range(n_sides):
        prompt = f"Length of side {i+1} (from point {i} to {(i+1)%n_sides}) in meters:"
        length = simpledialog.askfloat(title="Enter Side Length", prompt=prompt)
        if length is None:
            root.destroy()
            raise RuntimeError("User cancelled input")
        lengths.append(length)
    root.destroy()
    return lengths

# -------------------------------
# Remove duplicate closing point if any
# -------------------------------
if len(base_ground) >= 2 and base_ground[0] == base_ground[-1]:
    base_ground = base_ground[:-1]

n_sides = len(base_ground)
if n_sides < 3:
    raise RuntimeError("Base ground must have at least 3 points (polygon).")

real_lengths = get_side_lengths(n_sides)

# -------------------------------
# Real world coordinate reconstruction
# -------------------------------
real_pts = [[0.0, 0.0]]
for i in range(1, n_sides):
    x1, y1 = base_ground[i-1]
    x2, y2 = base_ground[i]
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    angle = math.atan2(dy, dx)
    prev = real_pts[-1]
    new_x = prev[0] + real_lengths[i-1] * math.cos(angle)
    new_y = prev[1] + real_lengths[i-1] * math.sin(angle)
    real_pts.append([new_x, new_y])

real_pts = np.array(real_pts, dtype=np.float32)
video_pts = np.array(base_ground, dtype=np.float32)
H, mask = cv2.findHomography(video_pts, real_pts, method=cv2.RANSAC)
if H is None:
    raise RuntimeError("Homography computation failed.")
H_inv = np.linalg.inv(H)
print("Homography computed.")

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO(model_path)

# -------------------------------
# Helper functions
# -------------------------------
def point_in_polygon(pt, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.float32), (float(pt[0]), float(pt[1])), False) >= 0

def image_to_real(pts_img):
    if len(pts_img) == 0:
        return np.zeros((0,2), dtype=np.float32)
    arr = np.array(pts_img, dtype=np.float32).reshape(-1,1,2)
    transformed = cv2.perspectiveTransform(arr, H).reshape(-1,2)
    return transformed

def real_to_topdown(pts_real, scale_x, scale_y):
    return [(int(p[0]*scale_x), int(p[1]*scale_y)) for p in pts_real]

# -------------------------------
# Top-down settings
# -------------------------------
topdown_size = (600, 600)
max_x = max(np.max(real_pts[:,0]), 1.0)
max_y = max(np.max(real_pts[:,1]), 1.0)
scale_x = topdown_size[0] / (max_x * 1.1)
scale_y = topdown_size[1] / (max_y * 1.1)

# -------------------------------
# Initialize ROI histories
# -------------------------------
roi_history = []
for _ in polygons:
    roi_history.append({
        "densities": deque(maxlen=frames_to_average),
        "prev_centroids": [],
        "prev_avg_density": 0.0,
        "warning_cooldown": 0
    })

prev_centroids_global = []

# -------------------------------
# Nearest neighbor matching for velocities
# -------------------------------
def match_prev_positions(current_pts, prev_pts):
    velocities = [np.array([0.0,0.0], dtype=np.float32) for _ in range(len(current_pts))]
    if len(prev_pts) == 0 or len(current_pts) == 0:
        return np.array(velocities, dtype=np.float32)
    prev_arr = np.array(prev_pts, dtype=np.float32)
    cur_arr = np.array(current_pts, dtype=np.float32)
    for i, c in enumerate(cur_arr):
        dists = np.linalg.norm(prev_arr - c, axis=1)
        min_idx = int(np.argmin(dists))
        velocities[i] = c - prev_arr[min_idx]
    return np.array(velocities, dtype=np.float32)

# -------------------------------
# Detection loop
# -------------------------------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 25.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    annotated = frame.copy()

    # YOLO detection (people class)
    results = model(frame, classes=[0], conf=confidence_threshold)
    boxes = results[0].boxes if len(results)>0 else []

    centroids_img = []
    for b in boxes:
        xy = b.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
        cx, cy = (x1+x2)/2, (y1+y2)/2
        centroids_img.append([cx, cy])
        cv2.circle(annotated, (int(cx),int(cy)), 3, (0,0,255), -1)

    # Transform centroids to real-world coordinates
    centroids_real = image_to_real(centroids_img)

    # Compute velocities w.r.t previous global centroids
    velocities = match_prev_positions(centroids_real.tolist(), prev_centroids_global)

    # Prepare top-down view
    topdown = np.ones((topdown_size[1], topdown_size[0], 3), dtype=np.uint8) * 255
    base_real = cv2.perspectiveTransform(np.array([base_ground], np.float32), H)[0]
    base_top = np.array(real_to_topdown(base_real, scale_x, scale_y), np.int32)
    cv2.polylines(topdown, [base_top], True, (0,255,0), 2)

    for pt in centroids_real:
        tx, ty = int(pt[0]*scale_x), int(pt[1]*scale_y)
        cv2.circle(topdown, (tx, ty), 4, (0,0,255), -1)

    # Process each ROI
    for idx, roi in enumerate(polygons):
        roi_real = cv2.perspectiveTransform(np.array([roi], np.float32), H)[0]
        roi_top_pts = np.array(real_to_topdown(roi_real, scale_x, scale_y), np.int32)
        roi_color = (0,255,0)  # default safe

        # People inside ROI
        people_in_roi = []
        people_outside_near = []

        for i, p in enumerate(centroids_real):
            if point_in_polygon(p, roi_real):
                people_in_roi.append(p)
            else:
                x_min, y_min = np.min(roi_real, axis=0)
                x_max, y_max = np.max(roi_real, axis=0)
                pad = max(1.0, np.mean([x_max-x_min, y_max-y_min])*0.2)
                if (p[0] >= x_min-pad and p[0] <= x_max+pad and p[1] >= y_min-pad and p[1] <= y_max+pad):
                    people_outside_near.append((i,p))

        count_inside = len(people_in_roi)
        roi_area = abs(cv2.contourArea(roi_real))
        density = (count_inside/roi_area) if roi_area>1e-6 else 0.0
        roi_history[idx]["densities"].append(density)
        avg_density = float(np.mean(roi_history[idx]["densities"]))

        # Average speed in ROI
        prev_in_roi = roi_history[idx]["prev_centroids"]
        speeds = []
        if len(prev_in_roi) > 0 and len(people_in_roi) > 0:
            prev_arr = np.array(prev_in_roi, dtype=np.float32)
            for p in people_in_roi:
                dists = np.linalg.norm(prev_arr - p, axis=1)
                min_idx = int(np.argmin(dists))
                speed = float(np.linalg.norm(p - prev_arr[min_idx]))
                speeds.append(speed)
        avg_speed_in_roi = float(np.mean(speeds)) if speeds else 0.0
        roi_history[idx]["prev_centroids"] = [np.array(x, dtype=np.float32) for x in people_in_roi]

        # Predict approaching people
        approaching_count = 0
        approx_times = []
        for i_out, p_out in people_outside_near:
            v = velocities[i_out] if i_out < len(velocities) else np.array([0.0,0.0], dtype=np.float32)
            speed_norm = float(np.linalg.norm(v))
            if speed_norm < 1e-6: 
                continue
            predicted = p_out + v*float(prediction_frames_ahead)
            if point_in_polygon(predicted, roi_real):
                approaching_count += 1
                roi_centroid = np.mean(roi_real, axis=0)
                dist = float(np.linalg.norm(roi_centroid - p_out))
                t_est = dist / (speed_norm + 1e-6)
                approx_times.append(t_est)
        avg_approach_time = float(np.mean(approx_times)) if approx_times else float('inf')

        # Density trend
        prev_avg_density = roi_history[idx].get("prev_avg_density",0.0)
        density_increase = avg_density - prev_avg_density

        # Reactive stampede
        reactive_stampede = (avg_density > density_threshold and avg_speed_in_roi > speed_threshold)

        # Predictive warning
        predictive_warning = False
        if approaching_count >= approach_count_threshold:
            if avg_approach_time < 10 or density_increase > 0.01 or avg_density > (0.6*density_threshold):
                predictive_warning = True

        # Cooldown
        if roi_history[idx]["warning_cooldown"] > 0:
            roi_history[idx]["warning_cooldown"] -= 1

        # Choose color and label
        label_text = f"Dens:{avg_density:.2f} In:{count_inside} App:{approaching_count} Sp:{avg_speed_in_roi:.2f}"
        if reactive_stampede:
            roi_color = (0,0,255)
            label_text = "STAMPede DETECTED!"
            roi_history[idx]["warning_cooldown"] = 30
        elif predictive_warning and roi_history[idx]["warning_cooldown"]==0:
            roi_color = (0,255,255)
            label_text = "PREDICTED STAMPEDE SOON"
            roi_history[idx]["warning_cooldown"] = 30
        else:
            roi_color = (0,255,0)

        # Draw ROI
        cv2.polylines(topdown, [roi_top_pts], True, roi_color, 2)
        M = cv2.moments(roi_top_pts)
        if M['m00'] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.putText(topdown, label_text, (cx-60, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_color, 2)
        cv2.polylines(annotated, [np.array(roi, np.int32)], True, roi_color, 2)
        rx_min, ry_min = np.min(np.array(roi, np.int32), axis=0)
        cv2.putText(annotated, label_text, (int(rx_min), int(ry_min)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        roi_history[idx]["prev_avg_density"] = avg_density

    # Show frames
    cv2.imshow("Original Frame", annotated)
    cv2.imshow("Top-Down View", topdown)

    # Update global centroids
    prev_centroids_global = [np.array(x,dtype=np.float32) for x in centroids_real]
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
