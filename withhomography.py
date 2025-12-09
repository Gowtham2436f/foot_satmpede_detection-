import cv2
import numpy as np
from ultralytics import YOLO
import math
import tkinter as tk
from tkinter import simpledialog

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
video_path = 'notstampede.mp4'
model_path = 'yolov8s.pt'
confidence_threshold = 0.25
density_threshold = 0.5  # people/m²

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
    cv2.imshow("Draw Base Ground & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save", frame)

# -------------------------------
# Load video first frame
# -------------------------------
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")
temp_frame = frame.copy()

# -------------------------------
# Interactive Drawing Loop
# -------------------------------
cv2.namedWindow("Draw Base Ground & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save")
cv2.setMouseCallback("Draw Base Ground & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save", draw_polygon)

print("Step 1: Draw base ground (green). Right click to finish base.")
print("Step 2: Draw ROIs (red). Right click to finish each ROI.")
print("Press 'u' to undo last point. Press 's' to finish drawing and start detection.")

while True:
    cv2.imshow("Draw Base Ground & ROIs - LClick:Add, RClick:Finish, U:Undo, S:Save", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u') and len(current_polygon) > 0:
        current_polygon.pop()
    if key == ord('s') and drawing_base_done:
        break

cv2.destroyAllWindows()
print("Base ground drawn.")

# -------------------------------
# Tkinter GUI to enter side lengths
# -------------------------------
def get_side_lengths(n_sides):
    root = tk.Tk()
    root.withdraw()  # hide main window
    lengths = []
    for i in range(n_sides):
        length = simpledialog.askfloat(
            title="Enter Side Length",
            prompt=f"Length of side {i+1} (from point {i} to {(i+1)%n_sides}) in meters:"
        )
        if length is None:
            raise RuntimeError("User cancelled input")
        lengths.append(length)
    root.destroy()
    return lengths

# -------------------------------
# Remove duplicate closing point if any
# -------------------------------
if base_ground[0] == base_ground[-1]:
    base_ground = base_ground[:-1]

n_sides = len(base_ground)
real_lengths = get_side_lengths(n_sides)

# -------------------------------
# Fully automatic real_pts reconstruction
# -------------------------------
real_pts = [[0, 0]]  # start at origin
for i in range(1, n_sides):
    x1, y1 = base_ground[i-1]
    x2, y2 = base_ground[i]
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)
    prev = real_pts[-1]
    new_x = prev[0] + real_lengths[i-1] * math.cos(angle)
    new_y = prev[1] + real_lengths[i-1] * math.sin(angle)
    real_pts.append([new_x, new_y])

real_pts = np.array(real_pts, dtype=np.float32)
video_pts = np.array(base_ground, dtype=np.float32)
H, _ = cv2.findHomography(video_pts, real_pts)
print("Homography computed for any polygon shape.")

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO(model_path)

# -------------------------------
# Helper function
# -------------------------------
def point_in_polygon(pt, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), pt, False) >= 0

# -------------------------------
# Top-down view mapping
# -------------------------------
topdown_size = (500, 500)
scale_x = topdown_size[0] / max(real_pts[:,0])
scale_y = topdown_size[1] / max(real_pts[:,1])

def map_to_topdown(pt):
    pt = np.array([[pt]], dtype=np.float32)
    top_pt = cv2.perspectiveTransform(pt, H)[0][0]
    x = int(top_pt[0] * scale_x)
    y = int(top_pt[1] * scale_y)
    return (x, y)

# -------------------------------
# YOLO detection loop
# -------------------------------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()
    results = model(frame, classes=[0], conf=confidence_threshold)
    boxes = results[0].boxes
    centroids = []

    # Compute centroids
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        centroids.append((cx, cy))
        cv2.circle(annotated_frame, (cx, cy), 5, (0,0,255), -1)

    # Draw ROIs on original frame
    cv2.polylines(annotated_frame, [np.array(base_ground, np.int32)], True, (0,255,0), 2)
    for roi in polygons:
        cv2.polylines(annotated_frame, [np.array(roi, np.int32)], True, (0,0,255), 2)

    # Top-down homography view
    topdown = np.ones((topdown_size[1], topdown_size[0], 3), dtype=np.uint8) * 255
    # base ground
    base_top = np.array([map_to_topdown(pt) for pt in base_ground], np.int32)
    cv2.polylines(topdown, [base_top], True, (0,255,0), 2)
    # ROIs
    for roi in polygons:
        roi_top = np.array([map_to_topdown(pt) for pt in roi], np.int32)
        cv2.polylines(topdown, [roi_top], True, (0,0,255), 2)
    # people
    for pt in centroids:
        tx, ty = map_to_topdown(pt)
        cv2.circle(topdown, (tx, ty), 5, (255,0,0), -1)

    # Compute density and label on original frame
    for roi in polygons:
        people_in_roi = [pt for pt in centroids if point_in_polygon(pt, roi)]
        if len(people_in_roi) == 0:
            continue
        roi_pts_m = cv2.perspectiveTransform(np.array([roi], dtype=np.float32), H)[0]
        roi_area_m2 = cv2.contourArea(roi_pts_m)
        density = len(people_in_roi)/roi_area_m2 if roi_area_m2 > 0 else 0

        M = cv2.moments(np.array(roi, np.int32))
        cx = int(M['m10']/M['m00']) if M['m00'] != 0 else roi[0][0]
        cy = int(M['m01']/M['m00']) if M['m00'] != 0 else roi[0][1]
        color = (0,255,255) if density < density_threshold else (0,0,255)
        cv2.putText(annotated_frame, f'Density: {density:.2f}', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show original + top-down separately
    cv2.imshow('YOLO Detection', annotated_frame)
    cv2.imshow('Top-Down Homography View', topdown)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
