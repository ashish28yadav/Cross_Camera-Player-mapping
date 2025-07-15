import cv2
from ultralytics import YOLO
import numpy as np
import math

# Load trained model
model = YOLO(r"C:\Users\ranja\DDA Python\best (1).pt")

video_path = r"C:\Users\ranja\DDA Python\broadcast.mp4"
cap = cv2.VideoCapture(video_path)

output_path = "output_final_custom_styled.mp4"
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Font & visuals
font = cv2.FONT_HERSHEY_SIMPLEX
trail = np.zeros((height, width, 3), dtype=np.uint8)

# Player ID tracking logic
player_id_counter = 1
player_tracker = {}

def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def assign_id(cx, cy, threshold=50):
    global player_id_counter
    for pid, (px, py) in player_tracker.items():
        if math.hypot(cx - px, cy - py) < threshold:
            player_tracker[pid] = (cx, cy)
            return pid
    player_tracker[player_id_counter] = (cx, cy)
    player_id_counter += 1
    return player_id_counter - 1

def draw_styled_box(frame, box, obj_id, class_name):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = get_centroid(x1, y1, x2, y2)

    # Define style by object type
    if 'goalkeeper' in class_name.lower():
        color = (0, 255, 0)           # Green
        shadow_color = (0, 0, 0)      # Black shadow
        label = f"Goalkeeper {obj_id}"
    elif 'ball' in class_name.lower():
        color = (0, 0, 255)           # Red
        shadow_color = (200, 200, 200)  # Light grey shadow
        label = "Ball"
    else:
        color = (60, 60, 60)          # Light black
        shadow_color = (0, 0, 0)      # Black shadow
        label = f"Player {obj_id}"

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    # Shadowed label
    cv2.putText(frame, label, (x1, y1 - 10), font, 0.7, shadow_color, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, y1 - 10), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw trail point for animation
    cv2.circle(trail, (cx, cy), 4, color, -1)

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)[0]
    overlay = frame.copy()

    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(cls_id)]
        cx, cy = get_centroid(x1, y1, x2, y2)

        # Assign ID only to players and goalkeepers
        if 'ball' in class_name.lower():
            draw_styled_box(overlay, box, None, class_name)
        else:
            obj_id = assign_id(cx, cy)
            draw_styled_box(overlay, box, obj_id, class_name)

    # Combine trail and overlay
    final_frame = cv2.addWeighted(overlay, 0.85, trail, 0.15, 0)
    out.write(final_frame)
    cv2.imshow("Styled Player Mapping", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    trail[:] = (trail * 0.92).astype(np.uint8)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Styled detection complete! Output saved as {output_path}")
