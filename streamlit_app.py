import os
from typing import Tuple, List

import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms as T

from leaf_gate_clipseg import LeafGateCLIPSeg

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
        up_dim = in_channels * mlp_ratio

        # depthwise conv: ทำงานบนแกนเวลา/ลำดับ (L) โดยไม่เปลี่ยนจำนวน channel
        self.dwconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # layer norm บน dim channel
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        # pointwise MLP: C -> 4C -> C (เหมือน ConvNeXt ปกติ)
        self.pw1 = nn.Linear(in_channels, up_dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(up_dim, in_channels)

        # gamma เป็น learnable scale สำหรับ output ของ block
        self.gamma = nn.Parameter(scale_init_value * torch.ones(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        """
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
        in_dim: int = 384,     # ขนาด feature จาก ViT
        num_classes: int = 3,  # dicot / monocot / other
        num_blocks: int = 2,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()

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
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        """
        # เพิ่มแกน L=1 เพื่อใช้กับ Conv1d
        x = x.unsqueeze(-1)           # [B, D, 1]
        x = self.blocks(x)            # [B, D, 1]

        # global pooling ตามแกน L (ซึ่งมีขนาด 1 อยู่แล้ว)
        x = x.mean(dim=-1)            # [B, D]

        x = self.norm(x)
        x = self.fc(x)                # [B, num_classes]
        return x


# -------------------------------------------------
# 2. ViT feature extractor
# -------------------------------------------------

IMG_SIZE_VIT = 518
VIT_MODEL_NAME = "vit_small_patch14_dinov2"
MODEL_DIR = "models"


def get_eval_transform(img_size: int = IMG_SIZE_VIT):
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
def load_vit_and_convnext() -> Tuple[nn.Module, nn.Module, T.Compose, List[str], torch.device]:
    """
    โหลด ViT backbone + ConvNeXt1D head + class_names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def predict_convnext(
    feat_vec: np.ndarray,
    conv: nn.Module,
    class_names: List[str],
    device: torch.device,
):
    """
    รับ feature vector 1 รูป -> คืน (pred_label, prob_per_class)
    """
    x = torch.from_numpy(feat_vec.reshape(1, -1)).to(device)
    with torch.no_grad():
        logits = conv(x)               # [1, C]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    return pred_label, probs


# -------------------------------------------------
# 3. LeafGate (CLIPSeg) – ตัดใบไม้ออกมา + overlay สีเขียว
# -------------------------------------------------


@st.cache_resource
def load_leaf_gate() -> LeafGateCLIPSeg:
    """
    โหลด LeafGateCLIPSeg เพียงครั้งเดียว แล้ว cache ไว้ใช้ซ้ำ
    """
    gate = LeafGateCLIPSeg()
    return gate


# -------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------


def main():
    st.title("LeafGate Leaf Classification 🌿")

    st.write(
        "ระบบจำแนกชนิดใบไม้ด้วย ViT + ConvNeXt "
        "พร้อม Leaf Gate (CLIPSeg) ที่ช่วยระบายสีเขียวเฉพาะบริเวณใบไม้ให้ดูชัดเจน"
    )

    # โหลดโมเดล / ฟังก์ชันต่าง ๆ
    gate = load_leaf_gate()
    vit, conv, vit_tfm, class_names, device = load_vit_and_convnext()

    uploaded_file = st.file_uploader(
        "อัปโหลดรูปใบไม้ (jpg, png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดรูปใบไม้ก่อนนะครับ")
        return

    # อ่านรูปต้นฉบับ
    pil = Image.open(uploaded_file).convert("RGB")

    # ใช้ Leaf Gate แบบอัตโนมัติทันทีที่มีรูป
    st.subheader("ผลจาก Leaf Gate (ระบายสีเขียวเฉพาะใบไม้)")
    with st.spinner("กำลังใช้ Leaf Gate ตรวจหาบริเวณใบไม้..."):
        crop_img, overlay_img, used_fallback = gate.crop_leaf_from_pil(
            pil,
            return_debug=True,
        )

    if crop_img is None:
        # ถ้า LeafGate ล้มเหลว ให้ใช้ภาพเต็มแทน และแจ้งเตือน
        st.warning("Leaf Gate ไม่สามารถหาใบไม้ได้อย่างชัดเจน – จะใช้ทั้งภาพในการจำแนกแทน")
        leaf_for_vit = pil
        st.image(pil, caption="ใช้ภาพเต็มในการจำแนก", use_column_width=True)
    else:
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
    main()
