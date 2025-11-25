import os
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torchvision import transforms as T
import joblib

from leaf_gate_clipseg import LeafGateCLIPSeg

# =========================================================
# 1. ConvNeXt-style 1D head  (ใช้เหมือนตอนเทรน)
# =========================================================

class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, layer_scale_init_value: float = 1e-6):
        super().__init__()
        # depthwise conv 1D
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  (เราใช้ L=1)
        """
        shortcut = x  # [B, L, D]

        # [B, L, D] -> [B, D, L] เพื่อใช้ conv1d
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # กลับเป็น [B, L, D]

        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x + shortcut
        return x


class ConvNeXt1DHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int = 2,
    ):
        """
        in_dim      = dim ของ feature จาก ViT (เช่น 384) หรือจาก PCA
        hidden_dim  = dim ภายใน ConvNeXt block
        num_classes = จำนวนคลาส (เช่น 3)
        """
        super().__init__()
        # โปรเจกต์จาก feature_dim -> hidden_dim
        self.proj = nn.Linear(in_dim, hidden_dim)

        # สร้างหลาย ๆ ConvNeXt block
        self.blocks = nn.Sequential(
            *[ConvNeXt1DBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # norm ด้านท้าย
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # classifier สุดท้าย
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F]  (F = feature_dim จาก ViT หรือ PCA)
        return: [B, num_classes]
        """
        # 1) โปรเจกต์ก่อน
        x = self.proj(x)          # [B, H]

        # 2) ใส่มิติ L=1 เพื่อผ่าน ConvNeXt1DBlock
        x = x.unsqueeze(1)        # [B, 1, H]
        x = self.blocks(x)        # [B, 1, H]

        # 3) ดึงออกมาเหลือ [B, H]
        x = x[:, 0, :]            # [B, H]

        # 4) norm + fc
        x = self.norm(x)          # [B, H]
        x = self.fc(x)            # [B, num_classes]
        return x


# =========================================================
# 2. ViT feature extractor
# =========================================================

IMG_SIZE_VIT = 518
VIT_MODEL_NAME = "vit_small_patch14_dinov2"
MODEL_DIR = "models"  # ที่เก็บ .pkl / .pth / .npy


def get_eval_transform(img_size: int = IMG_SIZE_VIT):
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
    ])


@st.cache_resource
def load_vit_backbone(
    model_name: str = VIT_MODEL_NAME,
    img_size: int = IMG_SIZE_VIT,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 -> return features
    vit.eval().to(device)
    tfm = get_eval_transform(img_size)
    return vit, tfm, device


@torch.no_grad()
def extract_vit_feature_from_pil(
    pil_img: Image.Image,
    vit: nn.Module,
    tfm,
    device: torch.device,
) -> np.ndarray:
    """
    รับ PIL.Image 1 รูป -> คืน feature vector np.array (D,)
    ให้ logic สอดคล้องกับ extract_vit_features.py
    """
    img = pil_img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)   # [1, 3, H, W]

    out = vit.forward_features(x)

    if isinstance(out, dict):
        # timm บางรุ่นคืน dict
        if "x_norm_clstoken" in out and isinstance(out["x_norm_clstoken"], torch.Tensor):
            feat = out["x_norm_clstoken"]     # [1, D]
        elif "cls_token" in out and isinstance(out["cls_token"], torch.Tensor):
            feat = out["cls_token"]           # [1, D]
        else:
            tensors_2d = [
                v for v in out.values()
                if isinstance(v, torch.Tensor) and v.ndim == 2
            ]
            if not tensors_2d:
                raise RuntimeError("forward_features(dict) ไม่มี tensor ขนาด [B, D]")
            feat = tensors_2d[0]
    elif isinstance(out, torch.Tensor):
        if out.ndim == 4:
            # [B, C, H, W] -> GAP -> [B, C]
            feat = out.mean(dim=[2, 3])
        elif out.ndim == 3:
            # [B, N, D] -> เอา CLS token
            feat = out[:, 0, :]
        elif out.ndim == 2:
            feat = out
        else:
            raise RuntimeError(f"ไม่รู้จัก shape ของ features: {out.shape}")
    else:
        raise RuntimeError("forward_features คืน type ที่ไม่รองรับ")

    feat_np = feat[0].detach().cpu().numpy().astype(np.float32)  # (D,)
    return feat_np


# =========================================================
# 3. โหลด ML models + PCA + ConvNeXt1D
# =========================================================

@st.cache_resource
def load_ml_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- class names -----
    raw = np.load(os.path.join(MODEL_DIR, "class_names.npy"), allow_pickle=True)
    if isinstance(raw, np.ndarray):
        class_names = [str(x) for x in raw.tolist()]
    else:
        class_names = [str(raw)]

    num_classes = len(class_names)

    # ----- SVM / RF (no PCA) -----
    svm = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "rf.pkl"))

    # ----- ConvNeXt1D (no PCA) -----
    feat_dim = 384  # ต้องตรงกับ feature dim จาก ViT
    conv_no_pca = ConvNeXt1DHead(
        in_dim=feat_dim,
        hidden_dim=384,
        num_classes=num_classes,
        num_blocks=2,
    )
    state_no_pca = torch.load(os.path.join(MODEL_DIR, "convnext1d_no_pca.pth"), map_location=device)
    conv_no_pca.load_state_dict(state_no_pca)
    conv_no_pca.to(device).eval()

    # ----- PCA + SVM / RF / ConvNeXt -----
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    svm_pca = joblib.load(os.path.join(MODEL_DIR, "svm_pca.pkl"))
    rf_pca = joblib.load(os.path.join(MODEL_DIR, "rf_pca.pkl"))

    pca_dim = pca.n_components_
    conv_pca = ConvNeXt1DHead(
        in_dim=pca_dim,
        hidden_dim=pca_dim,
        num_classes=num_classes,
        num_blocks=2,
    )
    state_pca = torch.load(os.path.join(MODEL_DIR, "convnext1d_pca.pth"), map_location=device)
    conv_pca.load_state_dict(state_pca)
    conv_pca.to(device).eval()

    return {
        "device": device,
        "class_names": class_names,
        "svm": svm,
        "rf": rf,
        "conv_no_pca": conv_no_pca,
        "pca": pca,
        "svm_pca": svm_pca,
        "rf_pca": rf_pca,
        "conv_pca": conv_pca,
    }

# =========================================================
# 4. Ensemble prediction
# =========================================================

def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max()
    exps = np.exp(logits)
    return exps / exps.sum()

def predict_all_models(feat_vec: np.ndarray, models: dict):
    """
    feat_vec: np.array shape (D,) จาก ViT
    models: dict ที่ได้จาก load_ml_models()
    """
    device = models["device"]
    class_names = models["class_names"]

    svm = models["svm"]
    rf = models["rf"]
    conv_no_pca = models["conv_no_pca"]

    pca = models["pca"]
    svm_pca = models["svm_pca"]
    rf_pca = models["rf_pca"]
    conv_pca = models["conv_pca"]

    # --------- เตรียม input ---------
    x = feat_vec.reshape(1, -1).astype(np.float32)   # (1, D)

    # ===== Non-PCA =====
    proba_svm = svm.predict_proba(x)[0]
    proba_rf = rf.predict_proba(x)[0]

    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        logits_conv = conv_no_pca(x_t)          # [1, C]
        proba_conv = softmax_np(logits_conv.cpu().numpy()[0])

    proba_ens_non_pca = (proba_svm + proba_rf + proba_conv) / 3.0

    # ===== PCA =====
    x_pca = pca.transform(x).astype(np.float32)

    proba_svm_pca = svm_pca.predict_proba(x_pca)[0]
    proba_rf_pca = rf_pca.predict_proba(x_pca)[0]

    with torch.no_grad():
        x_pca_t = torch.from_numpy(x_pca).to(device)
        logits_conv_pca = conv_pca(x_pca_t)
        proba_conv_pca = softmax_np(logits_conv_pca.cpu().numpy()[0])

    proba_ens_pca = (proba_svm_pca + proba_rf_pca + proba_conv_pca) / 3.0

    def idx2name(idx):
        return class_names[int(idx)]

    res = {
        "non_pca": {
            "svm": {
                "proba": proba_svm,
                "pred_idx": int(proba_svm.argmax()),
            },
            "rf": {
                "proba": proba_rf,
                "pred_idx": int(proba_rf.argmax()),
            },
            "convnext": {
                "proba": proba_conv,
                "pred_idx": int(proba_conv.argmax()),
            },
            "ensemble": {
                "proba": proba_ens_non_pca,
                "pred_idx": int(proba_ens_non_pca.argmax()),
            },
        },
        "pca": {
            "svm_pca": {
                "proba": proba_svm_pca,
                "pred_idx": int(proba_svm_pca.argmax()),
            },
            "rf_pca": {
                "proba": proba_rf_pca,
                "pred_idx": int(proba_rf_pca.argmax()),
            },
            "convnext_pca": {
                "proba": proba_conv_pca,
                "pred_idx": int(proba_conv_pca.argmax()),
            },
            "ensemble_pca": {
                "proba": proba_ens_pca,
                "pred_idx": int(proba_ens_pca.argmax()),
            },
        },
    }

    # ใส่ label text เพิ่ม
    for g in res.values():
        for k, v in g.items():
            v["pred_label"] = idx2name(v["pred_idx"])

    return res


# =========================================================
# 5. Leaf Gate (CLIPSeg)
# =========================================================

@st.cache_resource
def load_leaf_gate():
    """
    LeafGateCLIPSeg ภายในจะจัดการ device เอง
    """
    gate = LeafGateCLIPSeg()
    return gate


# =========================================================
# 6. Streamlit UI
# =========================================================
def main():

    st.title("Leaf Classification Web Service 🌿")

    # โหลดโมเดลหลักทั้งหมด (cache เพื่อลดเวลาโหลดซ้ำ)
    gate = load_leaf_gate()                      # Leaf Gate (CLIPSeg)
    vit, vit_tfm, vit_device = load_vit_backbone()  # ViT feature extractor
    models = load_ml_models()                    # รวม ConvNeXt และข้อมูลอื่น ๆ

    # อัปโหลดรูปจากผู้ใช้
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปใบไม้ (jpg, png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    # อ่านรูปเป็น PIL.Image
    pil = Image.open(uploaded_file).convert("RGB")

    # ----------------------------
    # 1) Leaf Gate ทำงานอัตโนมัติ (ไม่มี checkbox แล้ว)
    # ----------------------------
    with st.spinner("กำลังตรวจหาบริเวณใบไม้ ..."):
        try:
            # คืนเฉพาะภาพใบไม้ที่ถูกครอปแล้ว (ขนาดใกล้เคียง 518x518)
            leaf_img = gate.crop_leaf_from_pil(pil)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในขั้นตอน Leaf Gate: {e}")
            return

    if leaf_img is None:
        st.warning("ไม่พบใบไม้ชัดเจนในภาพนี้ กรุณาลองอัปโหลดรูปอื่น")
        return

    # แสดงเฉพาะรูปหลังผ่าน Leaf Gate ตามที่ต้องการ
    st.subheader("ภาพใบไม้หลังผ่าน Leaf Gate")
    st.image(leaf_img, caption="Leaf Gate Output", use_column_width=True)

    # ----------------------------
    # 2) ปุ่ม Predict ด้วย ConvNeXt เพียงอย่างเดียว
    # ----------------------------
    if st.button("🔍 Predict ด้วย ConvNeXt"):
        # 2.1 ดึง feature จาก ViT ใช้ภาพที่ผ่าน Leaf Gate แล้วเท่านั้น
        with st.spinner("กำลังดึงคุณลักษณะจาก ViT และทำนายผลด้วย ConvNeXt..."):
            feat = extract_vit_feature_from_pil(
                leaf_img, vit, vit_tfm, vit_device
            )  # shape = (D,)

            device = models["device"]
            conv_model = models["conv_no_pca"]      # ใช้ ConvNeXt (non-PCA) ตัวที่แม่นยำสุด
            class_names = models["class_names"]

            x = feat.reshape(1, -1).astype(np.float32)  # (1, D)

            with torch.no_grad():
                x_t = torch.from_numpy(x).to(device)
                logits = conv_model(x_t)               # [1, num_classes]
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # 2.2 แสดงผลลัพธ์จาก ConvNeXt
        pred_idx = int(probs.argmax())
        pred_label = class_names[pred_idx]

        st.subheader("ผลการจำแนกจาก ConvNeXt")
        st.markdown(f"**Predicted class:** `{pred_label}`")

        st.write("**ความน่าจะเป็นของแต่ละคลาส:**")
        for name, p in zip(class_names, probs):
            st.write(f"- {name}: {p:.3f}")


if __name__ == "__main__":
    main()

