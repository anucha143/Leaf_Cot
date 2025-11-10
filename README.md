# Leaf Venation ML — 2‑Stage Pipeline (Zero‑shot Leaf Gate + ViT‑LoRA)

This project contains **two steps**:
1) **Leaf Gate (zero‑shot)** using **SigLIP** to decide if an image is a leaf or not — **no training required**.
2) **Tri‑class classifier** (monocot/dicot/other) using **Vision Transformer (ViT) with LoRA fine‑tuning**.

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data structure
For step (2), organize your dataset as ImageFolder:

```
dataset/
  train/
    monocot/  *.jpg
    dicot/    *.jpg
    other/    *.jpg
  val/
    monocot/  *.jpg
    dicot/    *.jpg
    other/    *.jpg
  test/
    monocot/  *.jpg
    dicot/    *.jpg
    other/    *.jpg
```

For optional **threshold calibration** of the Gate, prepare:
```
leaf_gate_dataset/
  val/
    non_leaf/  *.jpg
    leaf/      *.jpg
```

> Ensure that `val` subfolders are **exactly** `['non_leaf', 'leaf']` for calibration.

## Step 1 — Zero‑shot Leaf Gate (SigLIP)
### Quick predict (no training)
```bash
python siglip_leaf_gate.py predict --image some_image.jpg --threshold 0.85
```

### Calibrate a better threshold (optional, recommended)
```bash
python siglip_leaf_gate.py calibrate --val_dir ./leaf_gate_dataset/val
# It prints best_threshold to use (e.g., 0.88)
```

## Step 2 — Train ViT‑LoRA (Tri‑class)
```bash
python train_vit_lora.py   --data_dir ./dataset   --model vit_small_patch14_dinov2   --epochs 40   --img_size 384   --batch_size 32   --lr 1e-4   --lora_r 8   --lora_alpha 16   --mixup 0.2
```
If your `timm` build does not have `vit_small_patch14_dinov2`, the script will automatically fallback to `vit_base_patch16_224.augreg_in21k`.

## Inference for the trained tri‑class model (optional)
```bash
python infer_vit_lora.py --ckpt ./checkpoints_vit_lora/best_vit_lora_vit_small_patch14_dinov2.pt --image path/to/leaf.jpg
```

## Full pipeline — Gate + Tri‑class
```bash
# Use your calibrated threshold here (default 0.85).
python pipeline_siglip_vit.py --image path/to/any_image.jpg --threshold 0.85   --tri_ckpt ./checkpoints_vit_lora/best_vit_lora_vit_small_patch14_dinov2.pt
```

The pipeline returns a JSON telling whether it passed the leaf gate and, if so, the predicted class (monocot/dicot/other) with probabilities.

## Why two steps?
- The **Gate** filters out non‑leaf images so the classifier is only applied to relevant inputs. This reduces false positives and makes the system more robust to “random background” images.
- **ViT‑LoRA** gives you a modern, efficient fine‑tuning method to classify venation into **monocot / dicot / other** using your custom dataset. LoRA trains only small adapter layers, keeping compute and overfitting under control.

## Files
- `siglip_leaf_gate.py`  — Zero‑shot leaf detector using SigLIP (predict + optional calibration)
- `train_vit_lora.py`    — ViT + LoRA training script (tri‑class)
- `infer_vit_lora.py`    — Simple inference script for the trained ViT‑LoRA
- `pipeline_siglip_vit.py` — End‑to‑end pipeline: Gate then Tri‑class
