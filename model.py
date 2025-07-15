from ultralytics import YOLO

# Correct: Load YOLOv8 model
model = YOLO(r"C:\Users\ranja\DDA Python\best (1).pt")

# Run prediction on an image or frame
results = model("test_image.jpg")  # Or use a video frame
results[0].show()
