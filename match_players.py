# match_players.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def match_players(broadcast_embeds, tacticam_embeds, similarity_threshold=0.8):
    matched_ids = {}
    global_id = 1

    for frame_idx in range(min(len(broadcast_embeds), len(tacticam_embeds))):
        broadcast = broadcast_embeds[frame_idx]
        tacticam = tacticam_embeds[frame_idx]

        if not broadcast or not tacticam:
            continue

        B = np.array([x['embedding'] for x in broadcast])
        T = np.array([x['embedding'] for x in tacticam])

        sim = cosine_similarity(B, T)

        for b_idx, b_det in enumerate(broadcast):
            t_idx = np.argmax(sim[b_idx])
            if sim[b_idx, t_idx] >= similarity_threshold:
                b_det['id'] = global_id
                tacticam[t_idx]['id'] = global_id
                matched_ids[(frame_idx, b_idx)] = global_id
                global_id += 1

    return broadcast_embeds, tacticam_embeds
