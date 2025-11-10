# siglip_leaf_gate.py
# Zero-shot Leaf Gate ด้วย SigLIP/CLIP (ใช้ตอน inference)

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# เลือกโมเดล SigLIP หรือ CLIP ที่รองรับภาพได้ดี
MODEL_NAME = "google/siglip-base-patch16-384"  # ถ้าโหลดไม่ได้จะใช้ CLIP แทน

def load_model():
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        print(f"[LeafGate] Loaded {MODEL_NAME}")
    except Exception:
        fallback = "openai/clip-vit-base-patch32"
        processor = AutoProcessor.from_pretrained(fallback)
        model = AutoModel.from_pretrained(fallback)
        print(f"[LeafGate] Fallback to {fallback}")
    model.to(DEVICE)
    model.eval()
    return processor, model


def compute_leaf_score(processor, model, image_path: str) -> float:
    image = Image.open(image_path).convert("RGB")

    # กลุ่ม prompt ฝั่งใบไม้
    leaf_texts = [
        "a close-up photo of a single leaf",
        "a photo of green leaf",
        "a photo of plant leaf on plain background",
        "macro photo of leaf with visible veins",
    ]

    # กลุ่ม prompt ฝั่งไม่ใช่ใบไม้
    non_leaf_texts = [
        "a photo without leaves",
        "a photo of a person or object",
        "a landscape without focus on a single leaf",
        "a random indoor object",
    ]

    all_texts = leaf_texts + non_leaf_texts

    inputs = processor(
        text=all_texts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # รองรับทั้ง SigLIP/CLIP (ดู field ที่มีอยู่)
    if hasattr(outputs, "image_embeds"):
        img_emb = outputs.image_embeds[0]
    else:
        img_emb = outputs.last_hidden_state[:, 0, :][0]

    if hasattr(outputs, "text_embeds"):
        txt_emb = outputs.text_embeds
    else:
        txt_emb = outputs.text_model_output.last_hidden_state[:, 0, :]

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    sims = (txt_emb @ img_emb)  # [num_text]
    sims = sims / 0.02  # temperature เล็กๆ ให้ softmax คมขึ้น

    sims_leaf = sims[:len(leaf_texts)]
    sims_non_leaf = sims[len(leaf_texts):]

    # log-sum-exp รวมเป็นคะแนนกลุ่ม
    leaf_score = torch.logsumexp(sims_leaf, dim=0)
    non_leaf_score = torch.logsumexp(sims_non_leaf, dim=0)
    scores = torch.stack([leaf_score, non_leaf_score], dim=0)
    probs = F.softmax(scores, dim=0)

    p_leaf = float(probs[0].item())
    return p_leaf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="path รูปภาพที่ต้องการเช็ค")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="ถ้า p_leaf >= threshold จะถือว่าเป็นใบไม้")
    args = parser.parse_args()

    processor, model = load_model()
    p_leaf = compute_leaf_score(processor, model, args.image)

    is_leaf = p_leaf >= args.threshold

    print(f"[LeafGate] p_leaf = {p_leaf:.4f}")
    print(f"[LeafGate] threshold = {args.threshold}")
    if is_leaf:
        print("[LeafGate] ✅ ดูเหมือนใบไม้ (ผ่าน gate)")
    else:
        print("[LeafGate] ❌ ไม่มั่นใจว่าเป็นใบไม้ (ไม่ผ่าน gate)")


if __name__ == "__main__":
    main()
