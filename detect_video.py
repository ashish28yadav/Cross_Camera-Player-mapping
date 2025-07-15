import cv2
from ultralytics import YOLO
import os

# Load the fine-tuned YOLOv8 model
model = YOLO(r"C:\Users\ranja\DDA Python\best (1).pt")  # Update path if needed

# Choose the video (can use either)
video_path = r"C:\Users\ranja\DDA Python\broadcast.mp4"  # or tacticam.mp4
cap = cv2.VideoCapture(video_path)

# Output video writer
output_path = "output_detected.mp4"
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)[0]

    annotated_frame = results.plot()  # draw boxes on frame

    out.write(annotated_frame)
    frame_count += 1

    # Optional: Show preview
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Detection complete! Output saved as {output_path}")
