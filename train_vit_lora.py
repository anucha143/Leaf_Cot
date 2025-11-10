# train_vit_lora.py
# ViT + LoRA training script (GPU/CPU friendly)

import os
import math
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # ใช้เวอร์ชันที่รองรับแน่นอน

import timm
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report


# ------------------------- Utils -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smart_collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


# --------------------- LoRA Components --------------------

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear

        # freeze base
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

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def apply_lora_to_vit(model: nn.Module, r=8, alpha=16, lora_dropout=0.05,
                      target_modules=("qkv", "proj")):
    replaced = 0
    for _, module in model.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear) and "qkv" in target_modules:
            module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, lora_dropout=lora_dropout)
            replaced += 1
        if hasattr(module, "proj") and isinstance(module.proj, nn.Linear) and "proj" in target_modules:
            module.proj = LoRALinear(module.proj, r=r, alpha=alpha, lora_dropout=lora_dropout)
            replaced += 1
    return replaced


def freeze_all_but_norm_and_head(model: nn.Module):
    # freeze ทั้งหมดก่อน
    for _, p in model.named_parameters():
        p.requires_grad = False

    # unfreeze head
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        for p in head.parameters():
            p.requires_grad = True
    elif hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

    # unfreeze normalization layers (ช่วยให้ fine-tune นิ่ง)
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                p.requires_grad = True


# ----------------- Mixup & Loss -----------------

def do_mixup(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, F.one_hot(y, num_classes=num_classes).float(), 1.0
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = F.one_hot(y, num_classes=num_classes).float()
    y_b = F.one_hot(y[index], num_classes=num_classes).float()
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y, lam


def soft_ce_loss(logits, soft_targets, label_smoothing=0.0):
    log_prob = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        num_classes = logits.size(-1)
        soft_targets = (1 - label_smoothing) * soft_targets + label_smoothing / num_classes
    return -(soft_targets * log_prob).sum(dim=-1).mean()


# ----------------- Scheduler -----------------

class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, g in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_step < self.warmup_steps:
                lr = base_lr * float(self.current_step) / float(max(1, self.warmup_steps))
            else:
                progress = (self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            g["lr"] = lr


# ----------------- Train / Eval -----------------

def train_one_epoch(model,
                    loader,
                    optimizer,
                    scheduler,
                    device: str,
                    epoch: int,
                    num_classes: int,
                    mixup_alpha: float = 0.2,
                    label_smoothing: float = 0.1,
                    use_amp: bool = True):
    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_alpha > 0:
            images, soft_labels, _ = do_mixup(images, labels, num_classes, alpha=mixup_alpha)
        else:
            soft_labels = F.one_hot(labels, num_classes=num_classes).float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(images)
                loss = soft_ce_loss(logits, soft_labels, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = soft_ce_loss(logits, soft_labels, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.detach().argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device: str, num_classes: int):
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / total
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return avg_loss, acc, macro_f1, cm, report


# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='vit_small_patch14_dinov2')
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--mixup', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_vit_lora')
    args = parser.parse_args()

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # สร้างโมเดล
    try:
        model = timm.create_model(args.model, pretrained=True, num_classes=3)
    except Exception:
        print(f"[WARN] Model '{args.model}' not found; fallback to 'vit_base_patch16_224.augreg_in21k'")
        model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=3)
        if args.img_size != 224:
            print("[WARN] Overriding img_size to 224 for fallback model")
            args.img_size = 224

    replaced = apply_lora_to_vit(model, r=args.lora_r, alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    print(f"[LoRA] Injected layers: {replaced}")
    freeze_all_but_norm_and_head(model)
    model.to(device)

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandAugment(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.15)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_set.classes
    num_classes = len(class_names)
    assert num_classes == 3, f"Expect 3 classes, got {num_classes} ({class_names})"
    print("[Classes]", class_names)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=True,
        collate_fn=smart_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        collate_fn=smart_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        collate_fn=smart_collate,
    )

    # optimizer
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndimension() == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': args.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = WarmupCosine(optimizer, warmup_steps, total_steps, min_lr=1e-6)

    use_amp = (device == 'cuda') and (not args.no_amp)
    print(f"[INFO] AMP enabled: {use_amp}")

    best_val_f1 = -1.0

    # -------- Train loop --------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            epoch=epoch,
            num_classes=num_classes,
            mixup_alpha=args.mixup,
            label_smoothing=args.label_smoothing,
            use_amp=use_amp,
        )
        val_loss, val_acc, val_f1, cm, report = evaluate(model, val_loader, device, num_classes)
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} | {dt:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(args.save_dir, f"best_vit_lora_{args.model}.pt")
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'class_names': class_names,
                    'args': vars(args),
                    'val_macro_f1': val_f1,
                },
                save_path,
            )
            print(f"[SAVE] {save_path} (val_macroF1={val_f1:.4f})")

    # -------- Test --------
    print("\n[TEST] Evaluating best model on test set ...")
    ckpts = [f for f in os.listdir(args.save_dir) if f.startswith("best_vit_lora_")]
    assert ckpts, "No best checkpoint found."
    best_ckpt = sorted(ckpts)[-1]

    state = torch.load(os.path.join(args.save_dir, best_ckpt), map_location=device)
    model.load_state_dict(state['model_state'])
    test_loss, test_acc, test_f1, cm, report = evaluate(model, test_loader, device, num_classes)

    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} macroF1={test_f1:.4f}")
    print("[TEST] Confusion Matrix:\n", cm)
    print("[TEST] Classification Report:\n", report)


if __name__ == "__main__":
    main()
