import os
import math
import time
import argparse
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

import timm
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ========================= Utils =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (N,C,H,W)."""
    def __init__(self, num_channels, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (N,C,H,W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x


# ===================== ConvNeXt Head =====================

class ConvNeXtBlock(nn.Module):
    """Simplified ConvNeXt-style block."""
    def __init__(self, dim: int):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        # pointwise convs
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        # layer scale
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        x = x + shortcut
        return x


class ConvNeXtHead(nn.Module):
    """Small ConvNeXt-style head for classification."""
    def __init__(self, in_chans: int, num_classes: int, depth: int = 2):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(ConvNeXtBlock(in_chans))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_chans, num_classes)

    def forward(self, x):
        x = self.blocks(x)          # (N,C,H,W)
        x = self.pool(x).flatten(1) # (N,C)
        x = self.fc(x)              # (N,num_classes)
        return x


# =========== ViT backbone -> feature map -> ConvNeXt ===========

class ViTConvNeXtClassifier(nn.Module):
    """
    ViT backbone (frozenได้) -> patch tokens -> reshape เป็น (C,H,W) -> ConvNeXt head.
    """
    def __init__(self, vit_name: str, num_classes: int, img_size: int = 518, freeze_vit: bool = True):
        super().__init__()
        # สร้าง ViT แบบไม่มี classifier head
        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)
        self.num_classes = num_classes

        # ดึง dim ของ feature
        self.embed_dim = getattr(self.vit, "num_features", None) or getattr(self.vit, "embed_dim")
        self.num_patches = None
        self.grid_size = None

        pe = getattr(self.vit, "patch_embed", None)
        if pe is not None:
            self.num_patches = getattr(pe, "num_patches", None)
            g = getattr(pe, "grid_size", None)
            if g is not None:
                if isinstance(g, (tuple, list)):
                    self.grid_size = g
                else:
                    try:
                        self.grid_size = (int(g[0]), int(g[1]))
                    except Exception:
                        self.grid_size = None

        # ConvNeXt head
        self.head = ConvNeXtHead(in_chans=self.embed_dim, num_classes=num_classes, depth=2)

        # freeze vit ถ้าต้องการให้เป็น feature extractor ล้วน ๆ
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

    def _tokens_to_feature_map(self, tokens: torch.Tensor):
        """
        tokens: (B, N, C)  where N is #tokens (อาจจะรวม CLS)
        return: feature map (B, C, H, W)
        """
        B, N, C = tokens.shape

        # ตัด CLS token ถ้ามี
        if self.num_patches is not None and N == self.num_patches + 1:
            tokens = tokens[:, 1:, :]
            N = self.num_patches
        elif N > 1:
            # เผื่อกรณีที่ไม่มี num_patches: ถ้า N-1 เป็น perfect square ให้เดาว่า token แรกคือ CLS
            if int(math.sqrt(N - 1)) ** 2 == (N - 1):
                tokens = tokens[:, 1:, :]
                N = N - 1

        # หาขนาด grid
        if self.grid_size is not None:
            H, W = self.grid_size
        else:
            H = W = int(math.sqrt(N))

        tokens = tokens[:, : H * W, :]
        x = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        return x

    def forward(self, x):
        # x: (B,3,H,W)
        feat = self.vit.forward_features(x)

        if isinstance(feat, dict):
            # timm DINOv2 มักมี key 'x_norm_patchtokens'
            if "x_norm_patchtokens" in feat:
                tokens = feat["x_norm_patchtokens"]  # (B,N,C)
            elif "last_hidden_state" in feat:
                tokens = feat["last_hidden_state"]
            else:
                tensors = [v for v in feat.values()
                           if isinstance(v, torch.Tensor) and v.ndim == 3]
                if not tensors:
                    raise RuntimeError("forward_features dict ไม่มี tensor 3D สำหรับ patch tokens")
                tokens = tensors[0]
        elif isinstance(feat, torch.Tensor):
            if feat.ndim == 3:
                tokens = feat
            elif feat.ndim == 4:
                # เผื่อกรณีที่ forward_features คืน feature map ตรง ๆ
                return self.head(feat)
            else:
                raise RuntimeError(f"รูปทรง feature ไม่คาดคิด: {feat.shape}")
        else:
            raise RuntimeError("forward_features คืน type ที่ไม่รองรับ")

        fmap = self._tokens_to_feature_map(tokens)  # (B,C,H,W)
        logits = self.head(fmap)
        return logits


# ===================== Transforms / Data =====================

def get_transforms(img_size: int):
    train_tf = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),            # แก้ warning palette/alpha
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        T.RandAugment(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_tf = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_tf, eval_tf


# ======================= Train / Eval =======================

@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / total
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)

    return avg_loss, acc, macro_f1, cm, report


def train_one_epoch(model, loader, optimizer, device,
                    scaler=None, use_amp=True, class_weights=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and device.type == "cuda":
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.detach().argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc


# =========================  Main  =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_small_patch14_dinov2")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--freeze_vit",
        action="store_true",
        help="Freeze ViT backbone (แนะนำให้เปิดสำหรับ dataset เล็ก)"
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vit_convnext_head")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Dataset & Transforms ---
    train_tf, eval_tf = get_transforms(args.img_size)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_set.classes
    num_classes = len(class_names)
    print("[Classes]", class_names)

    # class weights สำหรับ imbalance
    targets = np.array(train_set.targets)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * (counts + 1e-6))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("[Class Weights]", class_weights.tolist())

    # DataLoader
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
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

    # --- Model ---
    model = ViTConvNeXtClassifier(
        vit_name=args.model,
        num_classes=num_classes,
        img_size=args.img_size,
        freeze_vit=args.freeze_vit
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp and device.type == "cuda" else None
    print(f"[INFO] AMP enabled: {use_amp}, freeze_vit={args.freeze_vit}")

    best_val_f1 = -1.0
    best_ckpt = None

    # -------- Train loop --------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler, use_amp=use_amp, class_weights=class_weights
        )
        val_loss, val_acc, val_f1, cm, report = evaluate(
            model, val_loader, device, class_weights=class_weights
        )
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} | {dt:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_ckpt = os.path.join(args.save_dir, f"best_vit_convnext_{args.model}.pt")
            torch.save({
                "model": model.state_dict(),
                "classes": class_names,
                "best_val_f1": best_val_f1,
                "args": vars(args),
            }, best_ckpt)
            print(f"[SAVE] {best_ckpt} (val_macroF1={best_val_f1:.4f})")

    # -------- Test --------
    if best_ckpt is None:
        print("[WARN] No best checkpoint saved; skipping test.")
        return

    print("\n[TEST] Evaluating best model on test set ...")
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model"])

    test_loss, test_acc, test_f1, cm, report = evaluate(
        model, test_loader, device, class_weights=class_weights
    )
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} macroF1={test_f1:.4f}")
    print("[TEST] Confusion Matrix:\n", cm)
    print("[TEST] Classification Report:\n", report)


if __name__ == "__main__":
    main()
