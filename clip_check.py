# paste into a new file clip_check.py or inline in your pipeline
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2

# load model once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def clip_text_image_similarity(cv2_bgr_image, text_prompts=["a painting by Margo Veillon"]):
    # cv2 image (BGR) -> PIL RGB
    img_rgb = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    inputs = processor(text=text_prompts, images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        image_emb = outputs.image_embeds   # (1, D)
        text_emb = outputs.text_emb       # (len(prompts), D)
        # normalize
        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        # cosine similarities
        sims = (image_emb @ text_emb.T).squeeze(0).cpu().numpy()
    return sims  # array of similarity scores for each prompt