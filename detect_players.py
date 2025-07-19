# detect_players.py
import cv2
import torch
from ultralytics import YOLO
import os

def run_detection(video_path, model_path, output_dir, label='broadcast'):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []

        for i, box in enumerate(results.boxes.xyxy):
            cls = int(results.boxes.cls[i])
            name = model.names[cls]
            if name.lower() != "player":
                continue  # only keep player

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            crop_path = os.path.join(output_dir, f"{label}_f{frame_id}_p{i}.jpg")
            cv2.imwrite(crop_path, crop)

            detections.append({
                'frame': frame_id,
                'bbox': (x1, y1, x2, y2),
                'crop_path': crop_path,
            })

        all_detections.append(detections)
        frame_id += 1

    cap.release()
    return all_detections
