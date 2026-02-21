import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from config import CLIP_MODEL, BATCH_SIZE

def compute_clip_embeddings(records: list[dict]) -> np.ndarray:
    """Returns (N, D) embedding matrix, also stores in rec['embedding']."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="CLIP embeddings"):
        batch = records[i:i + BATCH_SIZE]
        images = [Image.fromarray(r["array"]) for r in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)

            if hasattr(out, "pooler_output"):
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state[:, 0, :]
            else:
                feats = out  # already a tensor

            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs = feats.cpu().numpy()
        for j, rec in enumerate(batch):
            rec["embedding"] = embs[j]
        all_embeddings.append(embs)

    return np.vstack(all_embeddings)
