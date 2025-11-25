# =============================================================================
# extract_vit_features.py
#
# หน้าที่ของไฟล์นี้:
#   1) โหลดภาพใบไม้จากโฟลเดอร์ dataset (train / val / test) ด้วย ImageFolder
#   2) ใช้ Vision Transformer (ViT) จาก timm เป็น "Feature Extractor"
#      โดยไม่ใช้หัว classifier (num_classes=0)
#   3) ดึง feature vector ของแต่ละภาพ → ได้เป็นเมทริกซ์:
#          X_train: (N_train, feat_dim)
#          X_val:   (N_val, feat_dim)
#          X_test:  (N_test, feat_dim)
#   4) เซฟทุกอย่างเป็นไฟล์ .npz เพื่อไปใช้เทรน ML ต่อ (SVM, RF, ConvNeXt1D)
# =============================================================================

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
    """
    ตั้งค่า random seed ให้เหมือนกันทุกครั้งที่รัน
    เพื่อให้ผลลัพธ์การสุ่ม (เช่น การแบ่ง batch) มีความคงที่ (reproducible)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_rgb(img):
    """
    แปลงภาพให้เป็นโหมดสี RGB เสมอ
    ใช้แก้ปัญหาภาพที่มี palette/transparency (เช่น PNG, GIF)
    ซึ่งอาจทำให้ torchvision ทำงานผิดพลาดได้
    """
    return img.convert("RGB")


def get_eval_transform(img_size: int):
    """
    สร้างชุด Transform สำหรับเตรียมภาพให้เข้ากับ ViT
    ขั้นตอน:
      1) แปลงภาพเป็น RGB
      2) Resize ให้ใหญ่กว่าขนาดที่ต้องการเล็กน้อย (img_size * 1.15)
      3) CenterCrop กลางภาพให้พอดีกับ img_size x img_size
      4) แปลงเป็น Tensor (ค่าอยู่ในช่วง [0,1])
      5) Normalize ด้วย mean/std = (0.5, 0.5, 0.5)
         ทำให้ข้อมูลกระจายดีขึ้น และเข้ากับการเทรนของโมเดล
    """
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

    รองรับกรณีที่ model.forward_features() คืนค่า:
      - dict    (เช่น โมเดลสาย DINOv2 บางรุ่น)
      - tensor 4D: (B, C, H, W)          → ทำ Global Average Pooling
      - tensor 3D: (B, N_tokens, C)      → ใช้ token แรก (CLS token)
      - tensor 2D: (B, C)                → ใช้โดยตรง

    Parameters
    ----------
    model : torch.nn.Module
        โมเดล ViT ที่ใช้เป็น feature extractor
    loader : DataLoader
        Dataloader ที่วิ่งภาพ batch ละ B ภาพ
    device : torch.device
        อุปกรณ์ที่ใช้รัน (cuda หรือ cpu)

    Returns
    -------
    feats : np.ndarray
        เมทริกซ์ feature (N_samples, feat_dim)
    labels : np.ndarray
        label ของแต่ละภาพ (N_samples,)
    """
    model.eval()
    feats = []
    labels = []

    for images, y in tqdm(loader, desc="Extract features"):
        # images: [B, 3, H, W]
        images = images.to(device)

        # ใช้ forward_features เพื่อดึง representation จาก ViT
        out = model.forward_features(images)

        # --- กรณี output เป็น dict (เช่น DINOv2) ---
        if isinstance(out, dict):
            # ถ้ามี key "x_norm_clstoken" ให้ใช้เป็น feature vector
            if "x_norm_clstoken" in out and isinstance(out["x_norm_clstoken"], torch.Tensor):
                f = out["x_norm_clstoken"]  # (B, C)
            # หรือถ้ามี "cls_token" ให้ใช้ตัวนั้น
            elif "cls_token" in out and isinstance(out["cls_token"], torch.Tensor):
                f = out["cls_token"]
            else:
                # ถ้าไม่รู้ว่าจะเลือกอันไหน → พยายามหา tensor 2D ตัวแรกมาใช้
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
                # (B, C, H, W) → ใช้ Global Average Pooling ให้เหลือ (B, C)
                f = out.mean(dim=[2, 3])
            elif out.ndim == 3:
                # (B, N_tokens, C) → ใช้ CLS token (token แรก)
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

    # รวม batch ทั้งหมดเป็นเมทริกซ์เดียว
    feats = torch.cat(feats).numpy()   # (N_samples, feat_dim)
    labels = torch.cat(labels).numpy() # (N_samples,)
    return feats, labels


def main():
    """
    ฟังก์ชันหลักของสคริปต์:
      - รับ argument จาก command line
      - โหลด dataset (train/val/test)
      - โหลด ViT model
      - ดึง features จากทุกภาพ
      - บันทึกเป็นไฟล์ .npz
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="โฟลเดอร์ที่มี train/val/test (ImageFolder)")
    parser.add_argument("--model", type=str, default="vit_small_patch14_dinov2",
                        help="ชื่อโมเดล ViT จาก timm ที่ใช้เป็น feature extractor")
    parser.add_argument("--img_size", type=int, default=518,
                        help="ขนาดภาพ (height=width=img_size) ที่ป้อนให้ ViT")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="ขนาด batch ในการดึง features")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="บน Windows แนะนำ 0 เพื่อลดปัญหา multiprocessing/pickling")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed สำหรับควบคุมความสุ่ม")
    parser.add_argument("--save_path", type=str,
                        default="./features/vit_leaf_features.npz",
                        help="ไฟล์ .npz สำหรับเซฟ features ทั้งหมด")
    args = parser.parse_args()

    # สร้างโฟลเดอร์ปลายทาง ถ้ายังไม่มี
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # ตั้งค่า seed ให้ผลซ้ำได้
    set_seed(args.seed)

    # เลือก device (ใช้ GPU ถ้ามี)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) สร้าง ViT เป็น feature extractor (ไม่มี classifier head → num_classes=0)
    vit = timm.create_model(args.model, pretrained=True, num_classes=0)
    vit.to(device)

    # 2) Dataset + Loader
    eval_tf = get_eval_transform(args.img_size)

    # สมมติว่า data_dir มีโครงสร้าง:
    #   data_dir/
    #       train/<class_name>/*.jpg
    #       val/<class_name>/*.jpg
    #       test/<class_name>/*.jpg
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    # ImageFolder จะอ่านชื่อ sub-folder เป็น class label อัตโนมัติ
    train_set = datasets.ImageFolder(train_dir, transform=eval_tf)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    # class_names เช่น ['dicot', 'monocot', 'other']
    class_names = train_set.classes
    print("[Classes]", class_names)

    # pin_memory = True ถ้าใช้ CUDA → เร็วขึ้นเล็กน้อยเวลาย้ายข้อมูลจาก RAM → GPU
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,                # ไม่จำเป็นต้อง shuffle เพราะไม่ใช้เทรนตรง ๆ ในที่นี้
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
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
    #   - X_train, y_train, X_val, y_val, X_test, y_test
    #   - class_names: array(object) เก็บชื่อคลาส ใช้กับ Streamlit ภายหลัง
    np.savez(
        args.save_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        class_names=np.array(class_names, dtype=object),
    )
    print(f"\n[SAVE] Features saved to {args.save_path}")


if __name__ == "__main__":
    main()
