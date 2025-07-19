# annotate_video.py
import cv2
import os

def draw_annotations(video_path, detections, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id < len(detections):
            for det in detections[frame_id]:
                x1, y1, x2, y2 = det['bbox']
                pid = det.get('id', None)
                if pid:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 2)
                    cv2.putText(frame, f"Player {pid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
