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
video_path = '3.mp4'
model_path = 'yolov8s.pt'
confidence_threshold = 0.25
density_threshold = 0.5  # people/m²
speed_threshold = 0.5    # meters/frame
frames_to_average = 10     # moving average frames

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
    root.withdraw()
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
# Real world coordinate reconstruction
# -------------------------------
real_pts = [[0, 0]]
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
    return cv2.pointPolygonTest(np.array(polygon, np.float32), (pt[0], pt[1]), False) >= 0

def map_to_topdown(pt, scale_x, scale_y):
    return int(pt[0]*scale_x), int(pt[1]*scale_y)

def image_to_real(pts_img):
    if len(pts_img)==0: return np.zeros((0,2),dtype=np.float32)
    arr = np.array(pts_img, dtype=np.float32).reshape(-1,1,2)
    return cv2.perspectiveTransform(arr,H).reshape(-1,2)

# -------------------------------
# Top-down settings
# -------------------------------
topdown_size = (600,600)
max_x = max(real_pts[:,0]) if np.max(real_pts[:,0])>0 else 1
max_y = max(real_pts[:,1]) if np.max(real_pts[:,1])>0 else 1
scale_x = topdown_size[0]/(max_x*1.1)
scale_y = topdown_size[1]/(max_y*1.1)

# -------------------------------
# Initialize ROI histories
# -------------------------------
roi_history = [{"densities": deque(maxlen=frames_to_average),
                "prev_centroids": []} for _ in polygons]

# -------------------------------
# Function to match centroids to previous frame
# -------------------------------
def match_prev(current_pts, prev_pts):
    if len(prev_pts)==0 or len(current_pts)==0:
        return current_pts
    matched = []
    for c in current_pts:
        distances = [np.linalg.norm(c-p) for p in prev_pts]
        min_idx = np.argmin(distances)
        matched.append(current_pts[np.argmin(distances)])
    return matched

# -------------------------------
# Detection loop
# -------------------------------
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    annotated = frame.copy()

    # YOLO detection
    results = model(frame, classes=[0], conf=confidence_threshold)
    boxes = results[0].boxes
    centroids_img = []
    for b in boxes:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
        centroids_img.append([(x1+x2)/2, (y1+y2)/2])
        cv2.circle(annotated, (int((x1+x2)/2), int((y1+y2)/2)), 3, (0,0,255), -1)

    # Transform to real-world coords
    centroids_real = image_to_real(centroids_img)

    # Top-down view
    topdown = np.ones((topdown_size[1],topdown_size[0],3), dtype=np.uint8)*255
    base_real = cv2.perspectiveTransform(np.array([base_ground],np.float32),H)[0]
    cv2.polylines(topdown, [np.array([map_to_topdown(p,scale_x,scale_y) for p in base_real],np.int32)], True, (0,255,0),2)

    for pt in centroids_real:
        tx,ty = map_to_topdown(pt,scale_x,scale_y)
        cv2.circle(topdown,(tx,ty),4,(0,0,255),-1)

    # Process each ROI
    for idx, roi in enumerate(polygons):
        roi_real = cv2.perspectiveTransform(np.array([roi],np.float32),H)[0]
        roi_top = np.array([map_to_topdown(p,scale_x,scale_y) for p in roi_real],np.int32)
        cv2.polylines(topdown,[roi_top],True,(0,0,255),2)

        # people inside ROI
        people_in_roi = [pt for pt in centroids_real if point_in_polygon(pt,roi_real)]
        count = len(people_in_roi)
        roi_area = abs(cv2.contourArea(roi_real))
        density = count/roi_area if roi_area>1e-6 else 0.0
        roi_history[idx]["densities"].append(density)
        avg_density = np.mean(roi_history[idx]["densities"])

        # speed calculation (track nearest)
        prev = roi_history[idx]["prev_centroids"]
        speeds=[]
        for p in people_in_roi:
            if len(prev)>0:
                distances = [np.linalg.norm(p-v) for v in prev]
                min_d = min(distances)
                speeds.append(min_d)
        avg_speed = np.mean(speeds) if speeds else 0.0
        roi_history[idx]["prev_centroids"]=people_in_roi

        # stampede decision
        is_stampede = avg_density>density_threshold and avg_speed>speed_threshold
        color = (0,0,255) if is_stampede else (0,255,0)

        # label density
        M = cv2.moments(roi_top)
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(topdown,f"{avg_density:.2f}",(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    cv2.imshow("Original Frame", annotated)
    cv2.imshow("Top-Down Homography", topdown)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
