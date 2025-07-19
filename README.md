# ⚽ Player Re-Identification In Sports Footage

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
├── broadcast.mp4 # Broadcast view input video
├── tacticam.mp4 # Tacticam view input video
├── best (1).pt # YOLOv8 custom trained model
├── broadcast_output.mp4 # Final annotated broadcast video
├── tacticam_output.mp4 # Final annotated tacticam video
├── main.py # Main pipeline (run this!)
├── detect_players.py # Detection and crop logic
├── extract_features.py # ResNet50-based embedding generator
├── match_players.py # Feature-based matching across views
├── annotate_video.py # Drawing IDs and saving output video


---

## 🚀 How It Works

### 🔄 Pipeline Overview

1. **Detect players** in each frame using YOLOv8
2. **Crop bounding boxes** for all detected players
3. **Extract ResNet50 embeddings** from each crop
4. **Match players** from tacticam to broadcast view using cosine similarity
5. **Assign a consistent `player_id`** to each player across both views
6. **Overlay bounding boxes and IDs**, and save annotated output videos

---

## ⚙️ Setup Instructions

### ✅ Requirements

Install all dependencies:

```bash
pip install ultralytics opencv-python numpy torch torchvision scikit-learn
⚠️ Python 3.8+ recommended
💡 GPU-accelerated PyTorch will significantly improve performance

▶️ How to Run
Make sure the following files are present:

broadcast.mp4, tacticam.mp4 (your videos)

best (1).pt (YOLOv8 model trained to detect player, ball, goalkeeper)

All .py scripts from this repo
python main.py
✅ That’s it! The pipeline will:

Run detection

Match identities across videos

Output two new annotated videos:

broadcast_output.mp4

tacticam_output.mp4

Press q in the preview window (optional) to stop early.

🧪 Example Output Format
Object	View	Bounding Box Color	Label
Player	Both	Light Black	Player 4
Goalkeeper	Both (future)	Green	Goalkeeper 1
Ball	Both (future)	Red + Grey Shadow	Ball
