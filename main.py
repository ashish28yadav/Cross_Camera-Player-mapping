# main.py
from detect_players import run_detection
from extract_features import extract_embeddings
from match_players import match_players
from annotate_video import draw_annotations

# Paths
model_path = r"C:\Users\ranja\DDA Python\best (1).pt"
broadcast_video = r"C:\Users\ranja\DDA Python\broadcast.mp4"
tacticam_video  = r"C:\Users\ranja\DDA Python\tacticam.mp4"
out_broadcast = "broadcast_output.mp4"
out_tacticam  = "tacticam_output.mp4"

# Step 1: Detect and crop
broadcast_dets = run_detection(broadcast_video, model_path, "broadcast_crops", label="b")
tacticam_dets  = run_detection(tacticam_video, model_path, "tacticam_crops", label="t")

# Step 2: Extract features
broadcast_embeds = extract_embeddings(broadcast_dets)
tacticam_embeds  = extract_embeddings(tacticam_dets)

# Step 3: Match players
broadcast_matched, tacticam_matched = match_players(broadcast_embeds, tacticam_embeds)

# Step 4: Draw and save output videos
draw_annotations(broadcast_video, broadcast_matched, out_broadcast)
draw_annotations(tacticam_video, tacticam_matched, out_tacticam)

print("✅ Cross-camera player mapping completed.")
