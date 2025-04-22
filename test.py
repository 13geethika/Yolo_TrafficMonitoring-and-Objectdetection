import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------------- Configuration ------------------- #
#model = YOLO('yolov8s.pt')  # or yolov8s.pt depending on your choice
# Train
#model.train(data='./data.yaml', epochs=5, project="vehicle_detection_project", name="yolov8n_custom", imgsz=640)
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720  
heavy_traffic_threshold = 14
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
background_color = (0, 0, 255)
MODEL_NAME = 'yolov8s.pt'
pixels_per_meter = 8  

# ------------------- Input ------------------- #
xyz = input("Enter your input file name (without .mp4): ")
input_video_path = xyz + '.mp4'

if not os.path.exists(input_video_path):
    print("File does not exist! Please enter a valid file name.")
    exit()

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# ------------------- Model and Tracker Initialization ------------------- #
model = YOLO(MODEL_NAME)
tracker = DeepSort(max_age=30)  

# ------------------- Output Setup ------------------- #
os.makedirs("./Frames", exist_ok=True)
count = 0
coordinates = defaultdict(lambda: deque(maxlen=int(fps)))  # For storing positions

# ------------------- Main Loop ------------------- #
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD)
    processed_frame = results[0].plot(line_width=1)
    detections = results[0].boxes

    # Preparing detections for DeepSORT
    det_list = []
    for i, (xyxy, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
        x1, y1, x2, y2 = map(int, xyxy)
        det_list.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), cls.item()))  # [x, y, w, h]

    # Update tracker
    tracks = tracker.update_tracks(det_list, frame=processed_frame)
    vehicles = 0

    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip unconfirmed tracks

        track_id = track.track_id
        ltrb = track.to_ltrb()  # l, t, r, b
        x1, y1, x2, y2 = map(int, ltrb)
        center_y = (y1 + y2) // 2

        coordinates[track_id].append(center_y)  # Track vertical position
        vehicles += 1

        # Speed estimation
        if len(coordinates[track_id]) >= int(fps / 2):  
            y_start = coordinates[track_id][0]
            y_end = coordinates[track_id][-1]
            distance_pixels = abs(y_end - y_start)
            distance_meters = distance_pixels / pixels_per_meter
            time_elapsed = len(coordinates[track_id]) / fps
            speed = (distance_meters / time_elapsed) * 3.6  # km/h

        else:
            speed = 0

        # Draw bounding box and label
        label = f"#{track_id} {int(speed)} km/h"
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(processed_frame, label, (x1, y1 - 10), font, 0.7, (0, 255, 0), 2)

    # Traffic intensity detection
    traffic_intensity = "Heavy" if vehicles > heavy_traffic_threshold else "Smooth"

    # Display vehicle count
    cv2.rectangle(processed_frame, (0, 0), (400, 40), background_color, -1)
    cv2.putText(processed_frame, f"Vehicles: {vehicles}", (10, 30), font, 1, font_color, 2)

    # Display traffic intensity
    cv2.rectangle(processed_frame, (0, 50), (400, 90), background_color, -1)
    cv2.putText(processed_frame, f"Traffic: {traffic_intensity}", (10, 80), font, 1, font_color, 2)

    # Save frame
    cv2.imwrite(f"./Frames/{count:04d}.jpg", processed_frame)
    count += 1
# Show live output
    cv2.imshow('Vehicle Detection & Speed Estimation', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- Video Writing ------------------- #
abc = input("Enter output file name (without .mp4): ")
output_video_path = abc + '.mp4'

if os.path.exists(output_video_path):
    print("File already exists. Please choose another name.")
    exit()

# Sort and create video
frames = sorted(os.listdir('./Frames'), key=lambda x: int(x.split('.')[0]))
frame_example = cv2.imread(f'./Frames/{frames[0]}')
height, width, _ = frame_example.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for frame_name in frames:
    frame = cv2.imread(f'./Frames/{frame_name}')
    video_writer.write(frame)
    print(f'Writing frame: {frame_name}')

video_writer.release()
print(f"Output video saved as {output_video_path}")