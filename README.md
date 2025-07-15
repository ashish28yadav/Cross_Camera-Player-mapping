# ⚽ Cross-Camera Player Detection

Detect and map players, goalkeepers, and the ball in football match videos using a fine-tuned [YOLOv8](https://github.com/ultralytics/ultralytics) model. This project applies **real-time detection**, **aesthetic bounding boxes**, and **animated tracking trails**, making it ideal for sports analytics, highlight generation, or tactical breakdowns.
![Detection Demo 1](https://github.com/user-attachments/assets/b125d4d1-6dfc-4dc8-adad-41c45f5c25b7)
![Detection Demo 2](https://github.com/user-attachments/assets/f213ece2-bd23-4a10-9130-b0d50995ba21)
![Detection Demo 3](https://github.com/user-attachments/assets/a47a0305-1186-4804-a77e-b71103aa7052)






---

## 🧠 Features

- 🎯 **YOLO-Based Object Detection**
- 🧍 Identifies:
  - Players (with unique IDs)
  - Goalkeepers (green box)
  - Ball (red box with light gray shadow)
- 🪄 Aesthetic bounding boxes with shadows
- 🌀 Animated motion trails
- 🎥 Live preview with keyboard interrupt (`q`)
- 💾 Final video output with all overlays

---

## 📁 Project Structure

```bash
📦 DDA Python/
├── best (1).pt                  # Fine-tuned YOLOv8 model
├── broadcast.mp4                # Primary input video
├── tacticam.mp4                 # (Optional) Alternative angle
├── detect_and_track.py          # Main Python script
├── output_detected.mp4          #primary output video
├── output_with_black_frame.mp4  # Output video with styled detections
├── README.md                    # Project documentation

🚀 Getting Started
✅ Prerequisites
Ensure Python 3.8+ is installed.

Install required libraries:

bash
Copy
Edit
pip install ultralytics opencv-python numpy
💡 Tip: If using GPU, make sure PyTorch with CUDA is properly installed.

⚙️ How It Works
The detect_and_track.py script performs the following steps:

Load YOLOv8 model from best (1).pt (fine-tuned to detect "player", "goalkeeper", and "ball").

Open the input video (broadcast.mp4) using OpenCV.

For each frame:

Perform detection using YOLOv8.

Assign unique IDs to players using centroid tracking.

Style bounding boxes:

🟩 Goalkeepers → Green box

⚫ Players → Light black box with white label

🔴 Ball → Red box with light gray label shadow

Draw motion trails for smooth tracking effect.

Save the output as output_with_black_frame.mp4.

▶️ How to Run
Clone this repo or download the project folder.

Ensure these files exist in the same directory:

detect_and_track.py

best (1).pt (your trained YOLOv8 model)

broadcast.mp4 (input match video)

Then run:

bash
Copy
Edit
python detect_and_track.py
Press q at any time to exit the preview window.

The final annotated video will be saved as:
output_with_black_frame.mp4

<details> <summary>
🔧 Customization Tips</summary>
Change video source:
Modify the path in video_path = "..." inside detect_and_track.py.

Update class labels:
If your model uses different class names (e.g., "person" or "gk"), update the string conditions like:

if 'goalkeeper' in class_name.lower()
Export tracking data:
You can log each player's bounding box and ID to a CSV file for post-game analysis.

</details>
🧪 Sample Output
Object	Bounding Box	Shadow Color	Label Example
Player	Light Black	Black	Player 4
Goalkeeper	Green	Black	Goalkeeper 1
Ball	Red	Light Grey	Ball

📽️ Model Training
This project assumes you already trained a YOLOv8 model using Ultralytics with custom classes:


['player', 'goalkeeper', 'ball']
If you haven’t trained your own model yet, refer to:
👉 Ultralytics YOLO Docs


