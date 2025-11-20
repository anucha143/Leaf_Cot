# extract_vit_features.py
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
import timm


# ---------- Utils ----------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_rgb(img):
    """แปลงภาพให้เป็น RGB (กันปัญหา palette / transparency)"""
    return img.convert("RGB")


def get_eval_transform(img_size: int):
    return T.Compose([
        T.Lambda(to_rgb),
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


@torch.no_grad()
def extract_split_features(model, loader, device):
    """
    ดึง feature จาก ViT แล้วแปลงเป็น 2D array (N_samples, feat_dim)
    รองรับกรณีที่ forward_features คืน:
    - dict (บางโมเดล / DINOv2)
    - tensor 4D: (B, C, H, W)
    - tensor 3D: (B, N_tokens, C)
    - tensor 2D: (B, C)
    """
    model.eval()
    feats = []
    labels = []

    for images, y in tqdm(loader, desc="Extract features"):
        images = images.to(device)

        out = model.forward_features(images)

        # --- กรณี output เป็น dict ---
        if isinstance(out, dict):
            if "x_norm_clstoken" in out and isinstance(out["x_norm_clstoken"], torch.Tensor):
                f = out["x_norm_clstoken"]  # (B, C)
            elif "cls_token" in out and isinstance(out["cls_token"], torch.Tensor):
                f = out["cls_token"]
            else:
                # พยายามหา tensor 2D จาก dict
                tensors_2d = [
                    v for v in out.values()
                    if isinstance(v, torch.Tensor) and v.ndim == 2
                ]
                if not tensors_2d:
                    raise RuntimeError(
                        "forward_features(dict) ไม่มี tensor 2D ที่ใช้เป็น feature ได้"
                    )
                f = tensors_2d[0]

        # --- กรณี output เป็น tensor ---
        elif isinstance(out, torch.Tensor):
            if out.ndim == 4:
                # (B, C, H, W) → Global Average Pooling
                f = out.mean(dim=[2, 3])  # (B, C)
            elif out.ndim == 3:
                # (B, N_tokens, C)
                # ใช้ CLS token (token แรก) เป็น feature vector
                # เช่น shape [32, 1370, 384] → เอาออกมาเป็น [32, 384]
                f = out[:, 0, :]
            elif out.ndim == 2:
                # (B, C) อยู่แล้ว
                f = out
            else:
                raise RuntimeError(f"ไม่รู้จัก shape ของ features: {out.shape}")
        else:
            raise RuntimeError("forward_features คืน type ที่ไม่รองรับ")

        feats.append(f.cpu())
        labels.append(y)

    feats = torch.cat(feats).numpy()   # (N_samples, feat_dim)
    labels = torch.cat(labels).numpy() # (N_samples,)
    return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="โฟลเดอร์ที่มี train/val/test (ImageFolder)")
    parser.add_argument("--model", type=str, default="vit_small_patch14_dinov2")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="บน Windows แนะนำ 0 ถ้าไม่ชัวร์เรื่อง multiprocessing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str,
                        default="./features/vit_leaf_features.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) สร้าง ViT เป็น feature extractor (ไม่มี classifier head → num_classes=0)
    vit = timm.create_model(args.model, pretrained=True, num_classes=0)
    vit.to(device)

    # 2) Dataset + Loader
    eval_tf = get_eval_transform(args.img_size)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    train_set = datasets.ImageFolder(train_dir, transform=eval_tf)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_set.classes
    print("[Classes]", class_names)

    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem
    )

    # 3) Extract features ทีละ split
    print("\n[Train split]")
    X_train, y_train = extract_split_features(vit, train_loader, device)
    print("  train features:", X_train.shape)

    print("\n[Val split]")
    X_val, y_val = extract_split_features(vit, val_loader, device)
    print("  val features:", X_val.shape)

    print("\n[Test split]")
    X_test, y_test = extract_split_features(vit, test_loader, device)
    print("  test features:", X_test.shape)

    # 4) Save .npz
    np.savez(
        args.save_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        class_names=np.array(class_names, dtype=object),
    )
    print(f"\n[SAVE] Features saved to {args.save_path}")


if __name__ == "__main__":
    main()
