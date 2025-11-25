@ -1,144 +1,221 @@
import os
from typing import Tuple, List

import numpy as np
import streamlit as st
from PIL import Image

import streamlit as st
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
# -------------------------------------------------
# 0. Streamlit basic config + simple pastel title color
# -------------------------------------------------

st.set_page_config(
    page_title="Leaf Classification (ViT + ConvNeXt + LeafGate)",
    page_icon="🌿",
    layout="centered",
)

# เปลี่ยน "สีตัวอักษร" ของหัวข้อให้เป็นเขียวพาสเทล #CCFFCC
# โดยไม่ไปยุ่งกับ padding / background ของ layout เดิม
st.markdown(
    """
    <style>
    h1, h2, h3 {
        color: #CCFFCC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 1. ConvNeXt 1D block + head (ต้องเหมือนตอนเทรนใน train_ml_with_ensemble.py)
# -------------------------------------------------


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, layer_scale_init_value: float = 1e-6):
    """
    1D ConvNeXt block:
    - depthwise conv (Conv1d with groups=C)
    - LayerNorm
    - pointwise MLP 2 ชั้น (Linear up -> GELU -> Linear down)
    - gamma (learnable scale) + residual
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        # depthwise conv 1D
        up_dim = in_channels * mlp_ratio

        # depthwise conv: ทำงานบนแกนเวลา/ลำดับ (L) โดยไม่เปลี่ยนจำนวน channel
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            groups=in_channels,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)

        # layer norm บน dim channel
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        # pointwise MLP: C -> 4C -> C (เหมือน ConvNeXt ปกติ)
        self.pw1 = nn.Linear(in_channels, up_dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.pw2 = nn.Linear(up_dim, in_channels)

        # gamma เป็น learnable scale สำหรับ output ของ block
        self.gamma = nn.Parameter(scale_init_value * torch.ones(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  (เราใช้ L=1)
        x: [B, C, L]
        """
        shortcut = x  # [B, L, D]

        # [B, L, D] -> [B, D, L] เพื่อใช้ conv1d
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # กลับเป็น [B, L, D]

        shortcut = x

        # Conv1d ทำงานกับ [B, C, L]
        # แต่ LayerNorm / Linear ของเราตั้งค่าให้ normalize ที่ dim สุดท้าย
        # จึงต้อง transpose ไป-กลับ
        x = x
        x = self.dwconv(x)         # [B, C, L]
        x = x.transpose(1, 2)      # [B, L, C]
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x * self.gamma
        x = x.transpose(1, 2)      # [B, C, L]

        x = x + shortcut
        return x


class ConvNeXt1DHead(nn.Module):
    """
    หัว ConvNeXt 1D แบบง่ายสำหรับเอา ViT feature vector มา classify 3 class
    ใช้ได้ทั้งกรณี input เป็น feature ตรง ๆ หรือ feature ที่ลดมิติด้วย PCA แล้ว
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        in_dim: int = 384,     # ขนาด feature จาก ViT
        num_classes: int = 3,  # dicot / monocot / other
        num_blocks: int = 2,
    ):
        """
        in_dim      = dim ของ feature จาก ViT (เช่น 384) หรือจาก PCA
        hidden_dim  = dim ภายใน ConvNeXt block
        num_classes = จำนวนคลาส (เช่น 3)
        """
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        # โปรเจกต์จาก feature_dim -> hidden_dim
        self.proj = nn.Linear(in_dim, hidden_dim)

        # สร้างหลาย ๆ ConvNeXt block
        self.blocks = nn.Sequential(
            *[ConvNeXt1DBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # norm ด้านท้าย
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # แปลง [B, D] -> [B, D, 1] แล้วรันผ่าน ConvNeXt block หลายชั้น
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ConvNeXt1DBlock(
                    in_channels=in_dim,
                    kernel_size=7,
                    mlp_ratio=mlp_ratio,
                    scale_init_value=1e-6,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # normalize feature ก่อนเข้า fc
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)

        # classifier สุดท้าย
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F]  (F = feature_dim จาก ViT หรือ PCA)
        return: [B, num_classes]
        x: [B, D]
        """
        # 1) โปรเจกต์ก่อน
        x = self.proj(x)          # [B, H]
        # เพิ่มแกน L=1 เพื่อใช้กับ Conv1d
        x = x.unsqueeze(-1)           # [B, D, 1]
        x = self.blocks(x)            # [B, D, 1]

        # 2) ใส่มิติ L=1 เพื่อผ่าน ConvNeXt1DBlock
        x = x.unsqueeze(1)        # [B, 1, H]
        x = self.blocks(x)        # [B, 1, H]
        # global pooling ตามแกน L (ซึ่งมีขนาด 1 อยู่แล้ว)
        x = x.mean(dim=-1)            # [B, D]

        # 3) ดึงออกมาเหลือ [B, H]
        x = x[:, 0, :]            # [B, H]

        # 4) norm + fc
        x = self.norm(x)          # [B, H]
        x = self.fc(x)            # [B, num_classes]
        x = self.norm(x)
        x = self.fc(x)                # [B, num_classes]
        return x


# =========================================================
# -------------------------------------------------
# 2. ViT feature extractor
# =========================================================
# -------------------------------------------------

IMG_SIZE_VIT = 518
VIT_MODEL_NAME = "vit_small_patch14_dinov2"
MODEL_DIR = "models"  # ที่เก็บ .pkl / .pth / .npy
MODEL_DIR = "models"


def get_eval_transform(img_size: int = IMG_SIZE_VIT):
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
    ])
    """
    Transform สำหรับเตรียมรูปภาพก่อนเข้า ViT:
    - Resize (ขยายเผื่อ) แล้ว CenterCrop
    - แปลงเป็น Tensor
    - Normalize ให้อยู่ในสเกลใกล้เคียงตอน pretrain
    """
    return T.Compose(
        [
            T.Resize(int(img_size * 1.15)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


@st.cache_resource
def load_vit_backbone(
    model_name: str = VIT_MODEL_NAME,
    img_size: int = IMG_SIZE_VIT,
):
def load_vit_and_convnext() -> Tuple[nn.Module, nn.Module, T.Compose, List[str], torch.device]:
    """
    โหลด ViT backbone + ConvNeXt1D head + class_names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 -> return features
    vit.eval().to(device)
    tfm = get_eval_transform(img_size)
    return vit, tfm, device

    # ViT feature extractor (ไม่ใช้ classifier head)
    vit = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0)
    vit.eval()
    vit.to(device)

    tfm = get_eval_transform(IMG_SIZE_VIT)

    # class names
    class_path = os.path.join(MODEL_DIR, "class_names.npy")
    raw = np.load(class_path, allow_pickle=True)
    if isinstance(raw, np.ndarray):
        class_names = [str(x) for x in raw.tolist()]
    else:
        class_names = [str(x) for x in raw]

    num_classes = len(class_names)

    # ConvNeXt head (no PCA) – in_dim ต้องตรงกับ dim ของ ViT feature (384)
    conv = ConvNeXt1DHead(
        in_dim=384,
        num_classes=num_classes,
        num_blocks=2,
        mlp_ratio=4,
    )
    ckpt_path = os.path.join(MODEL_DIR, "convnext1d_no_pca.pth")
    state = torch.load(ckpt_path, map_location=device)
    conv.load_state_dict(state)
    conv.to(device)
    conv.eval()

    return vit, conv, tfm, class_names, device


@torch.no_grad()
@ -189,320 +266,110 @@ def extract_vit_feature_from_pil(
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
def predict_convnext(
    feat_vec: np.ndarray,
    conv: nn.Module,
    class_names: List[str],
    device: torch.device,
):
    """
    feat_vec: np.array shape (D,) จาก ViT
    models: dict ที่ได้จาก load_ml_models()
    รับ feature vector 1 รูป -> คืน (pred_label, prob_per_class)
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

    x = torch.from_numpy(feat_vec.reshape(1, -1)).to(device)
    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        logits_conv = conv_no_pca(x_t)          # [1, C]
        proba_conv = softmax_np(logits_conv.cpu().numpy()[0])
        logits = conv(x)               # [1, C]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    proba_ens_non_pca = (proba_svm + proba_rf + proba_conv) / 3.0
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    return pred_label, probs

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
# -------------------------------------------------
# 3. LeafGate (CLIPSeg) – ตัดใบไม้ออกมา + overlay สีเขียว
# -------------------------------------------------

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
def load_leaf_gate() -> LeafGateCLIPSeg:
    """
    LeafGateCLIPSeg ภายในจะจัดการ device เอง
    โหลด LeafGateCLIPSeg เพียงครั้งเดียว แล้ว cache ไว้ใช้ซ้ำ
    """
    gate = LeafGateCLIPSeg()
    return gate


# =========================================================
# 6. Streamlit UI
# =========================================================
# -------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------


def main():
    st.title("Leaf Classification Web Service 🌿")
    st.title("LeafGate Leaf Classification 🌿")

    st.write(
        "ระบบจำแนกชนิดใบไม้ด้วย ViT + ConvNeXt "
        "พร้อม Leaf Gate (CLIPSeg) ที่ช่วยระบายสีเขียวเฉพาะบริเวณใบไม้ให้ดูชัดเจน"
    )

    # โหลดโมเดลหลักต่าง ๆ (มี cache แล้ว)
    gate = load_leaf_gate()                         # CLIPSeg LeafGate
    vit, vit_tfm, vit_device = load_vit_backbone()  # ViT feature extractor
    models = load_ml_models()                       # SVM, RF, ConvNeXt, PCA, ฯลฯ
    # โหลดโมเดล / ฟังก์ชันต่าง ๆ
    gate = load_leaf_gate()
    vit, conv, vit_tfm, class_names, device = load_vit_and_convnext()

    # ===== 1) อัปโหลดรูปภาพ =====
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปใบไม้ (jpg, png)",
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดรูปภาพก่อน")
        # ถ้าเปลี่ยนไฟล์ใหม่ ให้ล้างผลเก่าออก (กันสับสน)
        st.session_state.pop("feat", None)
        st.session_state.pop("results", None)
        st.info("กรุณาอัปโหลดรูปใบไม้ก่อนนะครับ")
        return

    # อ่านรูปต้นฉบับ
    pil = Image.open(uploaded_file).convert("RGB")
    st.image(pil, caption="ภาพต้นฉบับ", use_container_width=True)

    # ===== 2) เลือกว่าจะใช้ Leaf Gate (CLIPSeg) หรือไม่ =====
    use_gate = st.checkbox("ใช้ Leaf Gate (CLIPSeg) ตัดเฉพาะส่วนใบไม้", value=True)

    leaf_img = pil       # รูปที่จะส่งเข้า ViT (เริ่มต้น = รูปเต็ม)
    debug_imgs = None    # สำหรับเก็บรูป debug (overlay / mask ฯลฯ)

    if use_gate:
        with st.spinner("กำลังตรวจหาบริเวณใบไม้ด้วย CLIPSeg..."):
            crop, dbg, fb = gate.crop_leaf_from_pil(pil, return_debug=True)

        if crop is None:
            st.warning("ไม่พบใบไม้ชัดเจนในภาพนี้ (mask มีน้อยกว่า threshold)")
            return

        leaf_img = crop
        debug_imgs = dbg

        st.subheader("ผล Leaf Gate (CLIPSeg)")
        c1, c2 = st.columns(2)

        with c1:
            st.image(leaf_img, caption="Crop เฉพาะใบไม้", use_container_width=True)

        with c2:
            # ป้องกัน error: dbg อาจเป็น dict หรือเป็นรูปเดี่ยว
            if debug_imgs is not None:
                if isinstance(debug_imgs, dict):
                    if "overlay" in debug_imgs:
                        st.image(
                            debug_imgs["overlay"],
                            caption="Overlay mask",
                            use_container_width=True
                        )
                    elif "mask" in debug_imgs:
                        st.image(
                            debug_imgs["mask"],
                            caption="Mask",
                            use_container_width=True
                        )
                elif isinstance(debug_imgs, Image.Image):
                    st.image(
                        debug_imgs,
                        caption="Overlay / Mask",
                        use_container_width=True
                    )

    st.subheader("ขั้นตอนถัดไป: สร้าง Feature ด้วย ViT และทำนายด้วย ML")

    # ===== 3) เลือกโมเดลหลักที่จะโชว์ผล (เลือกได้ตั้งแต่ก่อน predict) =====
    model_choice = st.selectbox(
        "เลือกโมเดลสำหรับผลหลัก",
        [
            "SVM",
            "Random Forest",
            "ConvNeXt",
            "Ensemble (non-PCA)",
            "SVM + PCA",
            "Random Forest + PCA",
            "ConvNeXt + PCA",
            "Ensemble with PCA",
        ],
        key="model_choice"
    )

    choice_to_key = {
        "SVM": ("non_pca", "svm"),
        "Random Forest": ("non_pca", "rf"),
        "ConvNeXt": ("non_pca", "convnext"),
        "Ensemble (non-PCA)": ("non_pca", "ensemble"),
        "SVM + PCA": ("pca", "svm_pca"),
        "Random Forest + PCA": ("pca", "rf_pca"),
        "ConvNeXt + PCA": ("pca", "convnext_pca"),
        "Ensemble with PCA": ("pca", "ensemble_pca"),
    }

    # ===== 4) ปุ่ม Predict -> คำนวณและเก็บลง session_state =====
    if st.button("🔍 Predict ด้วย ML Models"):
        with st.spinner("กำลังดึง Feature จาก ViT..."):
            feat = extract_vit_feature_from_pil(leaf_img, vit, vit_tfm, vit_device)

        with st.spinner("กำลังทำนายด้วย SVM / RF / ConvNeXt / PCA / Ensemble..."):
            results = predict_all_models(feat, models)

        st.session_state["feat"] = feat
        st.session_state["results"] = results

    # ===== 5) ถ้ามีผลลัพธ์แล้ว (ใน session_state) ให้แสดงตาม model_choice =====
    if "results" in st.session_state:
        feat = st.session_state["feat"]
        results = st.session_state["results"]

        st.success(f"ได้ feature vector ขนาด {feat.shape[0]} มิติ จาก ViT")

        group_key, inner_key = choice_to_key[model_choice]
        main_res = results[group_key][inner_key]

        probs_main = main_res["proba"]
        label_main = main_res["pred_label"]
    # ใช้ Leaf Gate แบบอัตโนมัติทันทีที่มีรูป
    st.subheader("ผลจาก Leaf Gate (ระบายสีเขียวเฉพาะใบไม้)")
    with st.spinner("กำลังใช้ Leaf Gate ตรวจหาบริเวณใบไม้..."):
        crop_img, overlay_img, used_fallback = gate.crop_leaf_from_pil(
            pil,
            return_debug=True,
        )

        st.markdown(f"### ✅ ผลจากโมเดล: **{model_choice}**")
        st.write(f"**Predicted class**: `{label_main}`")
        st.write("**Probabilities:**")
        for cls_name, p in zip(models["class_names"], probs_main):
            st.write(f"- {cls_name}: {p:.3f}")
    if crop_img is None:
        # ถ้า LeafGate ล้มเหลว ให้ใช้ภาพเต็มแทน และแจ้งเตือน
        st.warning("Leaf Gate ไม่สามารถหาใบไม้ได้อย่างชัดเจน – จะใช้ทั้งภาพในการจำแนกแทน")
        leaf_for_vit = pil
        st.image(pil, caption="ใช้ภาพเต็มในการจำแนก", use_column_width=True)
    else:
        st.info("กดปุ่ม **Predict ด้วย ML Models** ก่อน เพื่อดูผลการทำนาย")
        # แสดงเฉพาะ overlay ที่ระบายสีเขียวตามที่ร้องขอ
        # (ไม่แสดงภาพ 518x518 crop)
        st.image(
            overlay_img,
            caption="Leaf Gate: ใบไม้ถูกระบายสีเขียว (ใช้ส่วนนี้เป็นบริเวณใบไม้)",
            use_column_width=True,
        )
        if used_fallback:
            st.info("Leaf Gate ใช้โหมด fallback (ใบไม้มีขนาดเล็ก หรือแยกขอบเขตยาก)")
        # สำหรับจำแนก ใช้ crop_img (ขนาด 518×518 ที่พร้อมเข้า ViT)
        leaf_for_vit = crop_img

    # ปุ่มทำนายด้วย ConvNeXt เพียงตัวเดียว
    if st.button("🔍 ทำนายชนิดใบไม้ด้วย ConvNeXt"):
        with st.spinner("กำลังดึงคุณลักษณะจาก ViT และทำนายด้วย ConvNeXt..."):
            feat_vec = extract_vit_feature_from_pil(
                leaf_for_vit, vit, vit_tfm, device
            )
            pred_label, probs = predict_convnext(
                feat_vec, conv, class_names, device
            )

        st.subheader("ผลการทำนายด้วย ConvNeXt")
        st.markdown(f"### ✅ คำตอบหลัก: **{pred_label}**")

        st.write("ความน่าจะเป็นของแต่ละคลาส:")
        for cls_name, p in zip(class_names, probs):
            st.write(f"- **{cls_name}**: {p:.3f}")


if __name__ == "__main__":
