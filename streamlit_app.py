# streamlit_app_leaf_demo.py
# Streamlit demo: CLIPSeg Leaf Gate + ViT classifier

import os
import io
from typing import Tuple

import streamlit as st
import torch
from PIL import Image
import numpy as np

from leaf_gate_clipseg import LeafGateCLIPSeg
from infer_vit_with_clipseg_gate import load_classifier

st.set_page_config(page_title="Leaf Classifier Demo", page_icon="🌿", layout="wide")

@st.cache_resource(show_spinner=False)
def load_gate(prompts: Tuple[str, ...], thresh: float):
    clean = []
    for p in prompts:
        if isinstance(p, (list, tuple)):
            for q in p:
                s = str(q).strip()
                if s:
                    clean.append(s)
        else:
            s = str(p).strip()
            if s:
                clean.append(s)
    if not clean:
        clean = ["a close-up photo of a single leaf"]
    return LeafGateCLIPSeg(prompts=clean, thresh=thresh)



@st.cache_resource(show_spinner=False)
def load_clf(ckpt_path: str):
    return load_classifier(ckpt_path)

def pil_from_upload(uploaded_file) -> Image.Image:
    byts = uploaded_file.read()
    return Image.open(io.BytesIO(byts)).convert("RGB")


st.title("🌿 Leaf Gate (CLIPSeg) + ViT Classifier Demo")

with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Path to checkpoint (.pt)", value="./checkpoints_vit_lora/best_vit_lora_vit_small_patch14_dinov2.pt")
    default_prompts = (
        "a close-up photo of a single leaf",
        "green leaf with visible veins",
        "a plant leaf on plain background",
        "macro photo of leaf veins",
    )
    prompts_txt = st.text_area("Prompts (one per line)", value="\n".join(default_prompts), height=150)
    prompt_list = tuple([p.strip() for p in prompts_txt.splitlines() if p.strip()])
    thresh = st.slider("Mask Threshold", 0.0, 1.0, 0.5, 0.01)
    run_btn = st.button("Run Prediction", type="primary")

col1, col2, col3 = st.columns([1, 1, 1])

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded and run_btn:
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()
    with st.spinner("Loading models... (first time may download weights)"):
        gate = load_gate(prompt_list, thresh)
        model, class_names, img_size, tfm = load_clf(ckpt_path)

    pil = pil_from_upload(uploaded)

    crop, dbg, fb = gate.crop_leaf_from_pil(pil, return_debug=True)

    x = tfm(crop).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

    with col1:
        st.subheader("Original")
        st.image(pil, use_column_width=True)
    with col2:
        st.subheader("Mask Overlay")
        cap = "Fallback (full image)" if fb else "Leaf region detected"
        st.image(dbg, use_column_width=True, caption=cap)
    with col3:
        st.subheader("Leaf Crop → ViT Input")
        st.image(crop, use_column_width=True)

    st.markdown("---")
    st.subheader("Prediction")
    st.write(f"**Predicted:** {class_names[pred_idx]}  \n**Confidence:** {probs[pred_idx]:.3f}")
    st.bar_chart({cls: probs[i] for i, cls in enumerate(class_names)})
    st.caption("Note: Bars show softmax probabilities over your 3 classes.")

else:
    st.info("⬅️ Upload an image and press **Run Prediction**.")
    st.caption("Tip: the first run may take a while to download CLIPSeg weights.")
