# Cross_Camera-Player-mapping
🏃‍♂️ Cross-Camera Player and Ball Detection using YOLOv8
A powerful sports video analysis tool that detects players, goalkeepers, and the ball, assigns unique IDs to players, and renders aesthetically styled bounding boxes with animated trails. The system uses a fine-tuned YOLOv8 model from Ultralytics and supports player tracking using centroid-based logic.


📁 Project Structure
bash
Copy
Edit
📦 DDA Python
├── best (1).pt                # Fine-tuned YOLOv8 model
├── broadcast.mp4              # Main video input (match footage)
├── tacticam.mp4               # Optional secondary angle video
├── detect_and_track.py        # Core Python script
├── output_with_black_frame.mp4# Output video with styled detections
├── README.md                  # Project documentation
🧠 Features
✅ Detects players, goalkeepers, and the ball

✅ Assigns persistent Player IDs

✅ Adds animated motion trails

✅ Renders styled bounding boxes:

Light black for players

Green for goalkeepers

Red for ball (with grey label shadow)

✅ Clean GUI window (cv2.imshow) with real-time display

✅ Lightweight tracking without external libraries

🚀 How the Code Works
The core logic is in detect_and_track.py:

Load YOLOv8 model (best (1).pt) trained to detect player, ball, goalkeeper classes.

Read input video (broadcast.mp4) using OpenCV.

For every frame:

Run detection using YOLOv8.

For each object:

If ball: draw red box with grey shadow.

If goalkeeper: green box with label.

If player: assign ID using centroid matching, draw light black box.

Add animation trail for each object’s movement.

Save output to output_with_black_frame.mp4.

🛠️ How to Run on Another PC
1. ✅ Prerequisites
Make sure you have:

Python 3.8+

pip (Python package manager)

A CUDA-compatible GPU (recommended for speed, but CPU also works)

2. 📦 Install Dependencies
bash
Copy
Edit
pip install ultralytics opencv-python numpy
3. 📁 File Setup
Ensure the following files are in the same folder:

detect_and_track.py

best (1).pt (your trained YOLOv8 model)

broadcast.mp4 (your input video)

4. ▶️ Run the Script
bash
Copy
Edit
python detect_and_track.py
Press Q to stop the preview early.

Final video will be saved as output_with_black_frame.mp4.

🧪 Example Output
“Player 1” and “Goalkeeper 3” bounding boxes move with animated trails, while the ball is tracked with a red box — all labels styled for visual clarity.

🧩 Model Note
The project assumes the YOLOv8 model (best (1).pt) is trained with the following custom classes:

"player"

"goalkeeper"

"ball"

If your model uses different labels, update the class name checks in the script.

