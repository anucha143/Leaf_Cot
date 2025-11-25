# leaf_gate_clipseg.py

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, Iterable
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "CIDAS/clipseg-rd64-refined"


def _sanitize_prompts(prompts: Optional[Iterable]) -> List[str]:
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
    def __init__(self, prompts: Optional[Iterable] = None, thresh: float = 0.5):
        self.processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        self.prompts = _sanitize_prompts(prompts)
        self.thresh = float(thresh)

    @torch.no_grad()
    def _predict_mask_logits(self, image_pil: Image.Image) -> np.ndarray:
        texts = _sanitize_prompts(self.prompts)
        images = [image_pil] * len(texts)

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        out = self.model(**inputs)
        logits = out.logits  # expected [N, 1, H, W] or [1, N, H, W]

        if logits.ndim == 4:
            if logits.shape[1] == 1:
                maps = logits.squeeze(1)      # [N, H, W]
            elif logits.shape[0] == 1:
                maps = logits[0]              # [N, H, W]
            else:
                maps = logits[:, 0, :, :]
        elif logits.ndim == 3:
            maps = logits
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        maps = maps.to(torch.float32)
        logits_lse = torch.logsumexp(maps, dim=0)  # [H, W]
        return logits_lse.detach().cpu().numpy()

    def _resize_to(self, arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        H, W = shape_hw
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_CUBIC)

    def _postprocess_mask(self, prob_map: np.ndarray) -> np.ndarray:
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
        prob = 1.0 / (1.0 + np.exp(-logits))
        prob = self._resize_to(prob, (H, W))
        bin_mask = self._postprocess_mask(prob)

        bbox = self._tight_bbox(bin_mask)
        if bbox is None:
            crop_rgb = np_img
            used_fallback = True
        else:
            x, y, bw, bh = bbox
            crop_rgb = np_img[y:y + bh, x:x + bw]
            used_fallback = False

        crop_pil = Image.fromarray(crop_rgb).resize((out_size, out_size), Image.BICUBIC)

        if return_debug:
            overlay = np_img.copy()

            # 🔽 ปรับความจางของสีเขียวตรงนี้
            alpha = 0.25  # ยิ่งตัวเลขน้อย สีเขียวจะยิ่งจาง (ลองปรับ 0.1–0.2 ได้)
            mask_idx = (bin_mask == 255)
            green = np.array([0, 255, 0], dtype=np.float32)

            overlay_f = overlay.astype(np.float32)
            overlay_f[mask_idx] = (
                    overlay_f[mask_idx] * (1.0 - alpha) + green * alpha
            )
            overlay = overlay_f.astype(np.uint8)

            dbg = Image.fromarray(overlay)
            return crop_pil, dbg, used_fallback

        return crop_pil



