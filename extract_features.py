# extract_features.py
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import cv2
import numpy as np

def extract_embeddings(detections):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    embeddings = []

    with torch.no_grad():
        for group in detections:
            group_embeds = []
            for det in group:
                image = cv2.imread(det['crop_path'])
                if image is None:
                    continue
                img_tensor = transform(image).unsqueeze(0)
                emb = model(img_tensor).squeeze().numpy()
                det['embedding'] = emb
                group_embeds.append(det)
            embeddings.append(group_embeds)

    return embeddings
