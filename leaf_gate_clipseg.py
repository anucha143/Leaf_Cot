# leaf_gate_clipseg.py
# Zero-shot Leaf ROI via CLIPSeg (prompted segmentation) — robust prompts + padding/truncation

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, Iterable
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "CIDAS/clipseg-rd64-refined"


def _sanitize_prompts(prompts: Optional[Iterable]) -> List[str]:
    """
    ทำ prompts ให้เป็นลิสต์สตริงแบบแบน (flat), ตัดช่องว่าง, ตัดค่าว่าง, ตัดซ้ำ
    รองรับกรณีผู้ใช้ป้อน tuple/list ซ้อน (list of lists)
    """
    default = [
        "a close-up photo of a single leaf",
        "green leaf with visible veins",
        "a plant leaf on plain background",
        "macro photo of leaf veins",
    ]
    if prompts is None:
        prompts = default

    out: List[str] = []

    def _push(x):
        s = str(x).strip()
        if s:
            out.append(s)

    if isinstance(prompts, (list, tuple)):
        for p in prompts:
            if isinstance(p, (list, tuple)):
                for q in p:
                    _push(q)
            else:
                _push(p)
    else:
        _push(prompts)

    # unique แบบคงลำดับ
    seen = set()
    flat = []
    for s in out:
        if s not in seen:
            seen.add(s)
            flat.append(s)

    if not flat:
        flat = default

    return flat


class LeafGateCLIPSeg:
    """
    Zero-shot prompted segmentation เพื่อหา mask ของ 'leaf'
    คืนภาพครอป (PIL) รอบบริเวณใบที่ใหญ่สุด ถ้าไม่เจอ mask ที่มั่นใจจะ fallback เป็นภาพเต็ม
    """
    def __init__(self, prompts: Optional[Iterable] = None, thresh: float = 0.5):
        self.processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        self.prompts = _sanitize_prompts(prompts)
        self.thresh = float(thresh)

    # ในไฟล์ leaf_gate_clipseg.py
    # แทนที่ทั้งเมธอด _predict_mask_logits ด้วยเวอร์ชันนี้

    @torch.no_grad()
    def _predict_mask_logits(self, image_pil: Image.Image) -> np.ndarray:
        # 1) ทำ prompts ให้เป็นลิสต์สตริงแบบแบนและสะอาด
        texts = _sanitize_prompts(self.prompts)

        # 2) ทำสำเนาภาพซ้ำให้ "ยาวเท่ากับจำนวน prompt"
        images = [image_pil] * len(texts)

        # 3) เรียก processor โดยเปิด padding / truncation (สำคัญ)
        inputs = self.processor(
            text=texts,
            images=images,  # << สำคัญ: ใช้ list ของภาพให้เท่ากับจำนวน texts
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        out = self.model(**inputs)  # logits shape ขึ้นกับโหมด batching
        logits = out.logits  # คาดว่าเป็น [N, 1, Hm, Wm] เมื่อ N = len(texts)

        # 4) ทำให้เป็นรูป [N_prompts, Hm, Wm] เสมอ แล้วรวมด้วย logsumexp
        if logits.ndim == 4:
            # หลาย ๆ กรณีจะได้ [N, 1, H, W] → บีบแกนช่องให้กลายเป็น [N, H, W]
            if logits.shape[1] == 1:
                maps = logits.squeeze(1)  # [N, H, W]
            else:
                # บางเวอร์ชัน (โหมดเดิม) อาจเป็น [1, N, H, W] → ดึงแกน batch ออกแทน
                # กรณีนี้เลือกแกนที่มีขนาดมากกว่า 1 เป็น N
                if logits.shape[0] == 1:
                    maps = logits[0]  # [N, H, W]
                else:
                    # fallback ปลอดภัย: สมมติแกน 0 คือ N
                    maps = logits[:, 0, :, :]
        elif logits.ndim == 3:
            maps = logits  # [N, H, W]
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        # รวมหลาย prompt ให้เป็น heatmap เดียวแบบคม ๆ
        maps = maps.to(torch.float32)
        logits_lse = torch.logsumexp(maps, dim=0)  # [H, W]

        return logits_lse.detach().cpu().numpy()

    def _resize_to(self, arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        H, W = shape_hw
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_CUBIC)

    def _postprocess_mask(self, prob_map: np.ndarray) -> np.ndarray:
        # prob_map ∈ [0,1] → binary mask → morphology ให้เนียนขึ้น
        mask = (prob_map > self.thresh).astype(np.uint8) * 255
        if mask.sum() == 0:
            return mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _tight_bbox(self, bin_mask: np.ndarray, min_area_ratio=0.01) -> Optional[Tuple[int, int, int, int]]:
        H, W = bin_mask.shape
        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area_ratio * (H * W):
            return None
        x, y, bw, bh = cv2.boundingRect(cnt)
        # padding เล็กน้อย
        px, py = int(0.03 * W), int(0.03 * H)
        x = max(0, x - px)
        y = max(0, y - py)
        bw = min(W - x, bw + 2 * px)
        bh = min(H - y, bh + 2 * py)
        return x, y, bw, bh

    def crop_leaf_from_pil(self, image_pil: Image.Image, out_size: int = 518, return_debug: bool = False):
        np_img = np.array(image_pil.convert("RGB"))
        H, W = np_img.shape[:2]

        logits = self._predict_mask_logits(image_pil)
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        prob = self._resize_to(prob, (H, W))
        bin_mask = self._postprocess_mask(prob)

        bbox = self._tight_bbox(bin_mask)
        if bbox is None:
            crop_rgb = np_img
            used_fallback = True
        else:
            x, y, bw, bh = bbox
            crop_rgb = np_img[y:y+bh, x:x+bw]
            used_fallback = False

        crop_pil = Image.fromarray(crop_rgb).resize((out_size, out_size), Image.BICUBIC)

        if return_debug:
            overlay = np_img.copy()
            if bin_mask.sum() > 0:
                overlay[bin_mask == 255] = (
                    overlay[bin_mask == 255] * 0.6 + np.array([0, 255, 0]) * 0.4
                ).astype(np.uint8)
            dbg = Image.fromarray(overlay)
            return crop_pil, dbg, used_fallback

        return crop_pil
