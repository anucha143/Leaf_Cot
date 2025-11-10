# infer_vit_with_clipseg_gate.py
# Inference: CLIPSeg Leaf Gate (ROI crop) -> ViT+LoRA classifier

import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import timm

# --- import leaf gate ---
from leaf_gate_clipseg import LeafGateCLIPSeg

# --- LoRA definitions (must match training) ---
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * (self.alpha / self.r)
        return base_out + lora_out


def apply_lora_to_vit(model: nn.Module, r=8, alpha=16, lora_dropout=0.05, target_modules=("qkv", "proj")):
    replaced = 0
    for _, module in model.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear) and "qkv" in target_modules:
            module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, lora_dropout=lora_dropout)
            replaced += 1
        if hasattr(module, "proj") and isinstance(module.proj, nn.Linear) and "proj" in target_modules:
            module.proj = LoRALinear(module.proj, r=r, alpha=alpha, lora_dropout=lora_dropout)
            replaced += 1
    return replaced


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier(ckpt_path: str):
    state = torch.load(ckpt_path, map_location=DEVICE)
    args = state["args"]
    class_names = state["class_names"]

    model_name = args["model"]
    num_classes = len(class_names)
    lora_r = args.get("lora_r", 8)
    lora_alpha = args.get("lora_alpha", 16)
    lora_dropout = args.get("lora_dropout", 0.05)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    apply_lora_to_vit(model, r=lora_r, alpha=lora_alpha, lora_dropout=lora_dropout)
    model.load_state_dict(state["model_state"])
    model.to(DEVICE).eval()

    img_size = int(args.get("img_size", 518))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return model, class_names, img_size, tfm


@torch.no_grad()
def predict_with_gate(image_path: str, ckpt_path: str, thresh: float = 0.5, save_debug_dir: str = None) -> Dict:
    gate = LeafGateCLIPSeg(thresh=thresh)
    crop, dbg, fb = gate.crop_leaf_from_pil(Image.open(image_path).convert("RGB"), return_debug=True)

    model, class_names, img_size, tfm = load_classifier(ckpt_path)

    x = tfm(crop).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)
        crop.save(os.path.join(save_debug_dir, "crop.jpg"))
        dbg.save(os.path.join(save_debug_dir, "overlay.jpg"))

    return {
        "pred_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {cls: float(p) for cls, p in zip(class_names, probs)},
        "used_fallback": bool(fb),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--save_debug_dir", type=str, default=None)
    args = ap.parse_args()

    out = predict_with_gate(args.image, args.ckpt, args.thresh, args.save_debug_dir)
    print(out)


if __name__ == "__main__":
    main()
