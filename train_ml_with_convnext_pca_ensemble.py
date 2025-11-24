# train_ml_with_convnext_pca_ensemble.py

import os
import time
import argparse

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 1. ConvNeXt-style 1D head (ต้องให้ตรงกับ streamlit_app.py)
# =========================================================

class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] (เราใช้ L=1)
        shortcut = x
        x = x.transpose(1, 2)      # [B, D, L]
        x = self.dwconv(x)
        x = x.transpose(1, 2)      # [B, L, D]

        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x + shortcut
        return x


class ConvNeXt1DHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_blocks: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ConvNeXt1DBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        x = self.proj(x)           # [B, H]
        x = x.unsqueeze(1)         # [B, 1, H]
        x = self.blocks(x)         # [B, 1, H]
        x = x[:, 0, :]             # [B, H]
        x = self.norm(x)
        x = self.fc(x)             # [B, num_classes]
        return x


# =========================================================
# 2. Utils: metric + timing + train loop for ConvNeXt
# =========================================================

def print_header(title: str):
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}\n")


def eval_sklearn_model(name: str, clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    เทรน + ประเมิน SVM / RF
    นับเวลา:
      - t_train: เวลา clf.fit()
      - t_val:   เวลา predict บน val
      - t_test:  เวลา predict บน test
    """
    print_header(f"[{name}] Training...")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    # VAL
    t0 = time.perf_counter()
    y_val_pred = clf.predict(X_val)
    t_val = time.perf_counter() - t0
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"[{name}] VAL accuracy = {val_acc:.4f}, macroF1 = {val_f1:.4f}")
    print(f"[{name}] VAL classification report:")
    print(classification_report(y_val, y_val_pred, digits=4))
    print(f"[{name}] VAL confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # TEST
    t0 = time.perf_counter()
    y_test_pred = clf.predict(X_test)
    t_test = time.perf_counter() - t0
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    print(f"[{name}] TEST accuracy = {test_acc:.4f}, macroF1 = {test_f1:.4f}")
    print(f"[{name}] TEST classification report:")
    print(classification_report(y_test, y_test_pred, digits=4))
    print(f"[{name}] TEST confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return {
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "t_train": t_train,
        "t_val": t_val,
        "t_test": t_test,
    }


def train_convnext1d(
    name: str,
    model: nn.Module,
    device: torch.device,
    X_train, y_train,
    X_val, y_val,
    epochs: int = 60,
    patience: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """
    เทรน ConvNeXt1D บน feature (ไม่ใช่ภาพ) + early stopping ตาม val macroF1

    เราจะเก็บ:
      - t_train: รวมเวลาส่วน train ทุก epoch
      - t_val:   รวมเวลาส่วน validation ทุก epoch
      - t_total: เวลารวมทั้งหมดของฟังก์ชันนี้
    """
    print_header(f"[{name}] ConvNeXt1D Training...")

    # Datasets / Loaders
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    total_train_time = 0.0
    total_val_time = 0.0

    t0_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        t0_train_epoch = time.perf_counter()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total
        total_train_time += time.perf_counter() - t0_train_epoch

        # ---------- VAL ----------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_gts = []

        t0_val_epoch = time.perf_counter()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

                val_preds.append(preds.cpu().numpy())
                val_gts.append(yb.cpu().numpy())
        total_val_time += time.perf_counter() - t0_val_epoch

        val_loss /= val_total
        val_preds = np.concatenate(val_preds)
        val_gts = np.concatenate(val_gts)
        val_acc = accuracy_score(val_gts, val_preds)
        val_f1 = f1_score(val_gts, val_preds, average="macro")

        print(
            f"[{name}] Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        # ----- early stopping -----
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = model.state_dict()
            epochs_no_improve = 0
            print(f"    -> New best model at epoch {epoch} (val_macroF1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"    -> Early stopping: no improvement for {patience} epochs. "
                    f"Best epoch = {best_epoch}."
                )
                break

    t_total = time.perf_counter() - t0_total

    # โหลด best weights กลับเข้า model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"[{name}] Training finished. "
        f"Best epoch = {best_epoch}, best val_macroF1 = {best_val_f1:.4f}. "
        f"train_time={total_train_time:.3f}s, val_time={total_val_time:.3f}s "
        f"(total={t_total:.3f}s)"
    )

    return {
        "model": model,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "t_train": total_train_time,
        "t_val": total_val_time,
        "t_total": t_total,
    }


def eval_convnext_test(name: str, model: nn.Module, device, X_test, y_test):
    """
    ประเมิน ConvNeXt1D บน test set
    คืน:
      - test_acc, test_f1
      - t_test: เวลา forward test ทั้งชุด
    """
    model.eval().to(device)
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test_t = torch.from_numpy(y_test.astype(np.int64)).to(device)

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(X_test_t)
        t_test = time.perf_counter() - t0

        preds = logits.argmax(dim=1)
        y_pred = preds.cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print_header(f"[{name}] TEST evaluation")
    print(f"[{name}] TEST accuracy = {acc:.4f}, macroF1 = {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "test_acc": acc,
        "test_f1": f1,
        "t_test": t_test,
    }


# =========================================================
# 3. Ensemble helper
# =========================================================

def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max()
    exps = np.exp(logits)
    return exps / exps.sum()


def eval_ensemble(
    name: str,
    proba_svm, proba_rf, proba_conv,
    y_true,
):
    """
    proba_*: (N, C)  ของแต่ละโมเดล
    นับเวลา t_test: เวลาในการรวม prob + argmax + metric
    """
    print_header(f"[{name}] Ensemble evaluation")

    t0 = time.perf_counter()
    proba_ens = (proba_svm + proba_rf + proba_conv) / 3.0
    y_pred = proba_ens.argmax(axis=1)
    t_test = time.perf_counter() - t0

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"[{name}] accuracy = {acc:.4f}, macroF1 = {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "acc": acc,
        "f1": f1,
        "t_test": t_test,
    }


# =========================================================
# 4. main()
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="path to vit_leaf_features.npz",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="number of PCA components",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="max epochs for ConvNeXt1D",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="early stopping patience (epochs)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="directory to save models",
    )
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------- load features ----------
    data = np.load(args.features, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"].tolist()

    print(f"[INFO] Loaded features from {args.features}")
    print(f"  X_train: {X_train.shape} y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape} y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape} y_test:  {y_test.shape}")
    print(f"  class_names: {class_names}")

    num_classes = len(class_names)
    feat_dim = X_train.shape[1]

    # save class_names for streamlit
    np.save(os.path.join(args.models_dir, "class_names.npy"), np.array(class_names))

    # =====================================================
    # 4.1 SVM / RF on raw ViT features
    # =====================================================
    print_header("SVM / RandomForest on raw ViT features")

    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    svm_metrics = eval_sklearn_model(
        "SVM (ViT features)",
        svm,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    rf_metrics = eval_sklearn_model(
        "RandomForest (ViT features)",
        rf,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    # Probabilities สำหรับ ensemble non-PCA
    proba_svm_test = svm.predict_proba(X_test)
    proba_rf_test = rf.predict_proba(X_test)

    # save models
    joblib.dump(svm, os.path.join(args.models_dir, "svm.pkl"))
    joblib.dump(rf, os.path.join(args.models_dir, "rf.pkl"))

    # =====================================================
    # 4.2 PCA + SVM / RF
    # =====================================================
    print_header("PCA + SVM / RandomForest")

    pca = PCA(n_components=args.pca_components, random_state=42)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    svm_pca = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=42,
    )
    rf_pca = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    svm_pca_metrics = eval_sklearn_model(
        "SVM (PCA+ViT features)",
        svm_pca,
        X_train_pca, y_train,
        X_val_pca, y_val,
        X_test_pca, y_test,
    )

    rf_pca_metrics = eval_sklearn_model(
        "RandomForest (PCA+ViT features)",
        rf_pca,
        X_train_pca, y_train,
        X_val_pca, y_val,
        X_test_pca, y_test,
    )

    proba_svm_pca_test = svm_pca.predict_proba(X_test_pca)
    proba_rf_pca_test = rf_pca.predict_proba(X_test_pca)

    # save PCA & models
    joblib.dump(pca, os.path.join(args.models_dir, "pca.pkl"))
    joblib.dump(svm_pca, os.path.join(args.models_dir, "svm_pca.pkl"))
    joblib.dump(rf_pca, os.path.join(args.models_dir, "rf_pca.pkl"))

    # =====================================================
    # 4.3 ConvNeXt1D (no PCA)
    # =====================================================
    conv_no_pca = ConvNeXt1DHead(
        in_dim=feat_dim,
        hidden_dim=feat_dim,
        num_classes=num_classes,
        num_blocks=2,
    )

    conv_no_pca_res = train_convnext1d(
        name="ConvNeXt1D (ViT features)",
        model=conv_no_pca,
        device=device,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        patience=args.patience,
    )

    conv_no_pca = conv_no_pca_res["model"]
    conv_no_pca_test_metrics = eval_convnext_test(
        "ConvNeXt1D (ViT features)",
        conv_no_pca,
        device,
        X_test,
        y_test,
    )

    # save checkpoint
    torch.save(
        conv_no_pca.state_dict(),
        os.path.join(args.models_dir, "convnext1d_no_pca.pth"),
    )

    # สำหรับ ensemble non-PCA: เอา logits มา softmax
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        logits_conv_test = conv_no_pca(X_test_t)
        proba_conv_test = F.softmax(logits_conv_test, dim=1).cpu().numpy()

    # Ensemble non-PCA
    ens_non_pca_metrics = eval_ensemble(
        "Ensemble (SVM+RF+ConvNeXt on ViT features)",
        proba_svm_test,
        proba_rf_test,
        proba_conv_test,
        y_test,
    )

    # =====================================================
    # 4.4 ConvNeXt1D (PCA features)
    # =====================================================
    conv_pca = ConvNeXt1DHead(
        in_dim=args.pca_components,
        hidden_dim=args.pca_components,
        num_classes=num_classes,
        num_blocks=2,
    )

    conv_pca_res = train_convnext1d(
        name="ConvNeXt1D (PCA+ViT features)",
        model=conv_pca,
        device=device,
        X_train=X_train_pca, y_train=y_train,
        X_val=X_val_pca, y_val=y_val,
        epochs=args.epochs,
        patience=args.patience,
    )

    conv_pca = conv_pca_res["model"]
    conv_pca_test_metrics = eval_convnext_test(
        "ConvNeXt1D (PCA+ViT features)",
        conv_pca,
        device,
        X_test_pca,
        y_test,
    )

    torch.save(
        conv_pca.state_dict(),
        os.path.join(args.models_dir, "convnext1d_pca.pth"),
    )

    with torch.no_grad():
        X_test_pca_t = torch.from_numpy(X_test_pca.astype(np.float32)).to(device)
        logits_conv_pca_test = conv_pca(X_test_pca_t)
        proba_conv_pca_test = F.softmax(logits_conv_pca_test, dim=1).cpu().numpy()

    # Ensemble PCA
    ens_pca_metrics = eval_ensemble(
        "Ensemble (SVM+RF+ConvNeXt on PCA+ViT)",
        proba_svm_pca_test,
        proba_rf_pca_test,
        proba_conv_pca_test,
        y_test,
    )

    # =====================================================
    # 4.5 Summary tables (Accuracy / F1 + Time)
    # =====================================================
    print_header("SUMMARY (TEST set: Accuracy / macroF1)")

    rows_metrics = [
        ("SVM",                     svm_metrics["test_acc"],          svm_metrics["test_f1"]),
        ("RandomForest",            rf_metrics["test_acc"],           rf_metrics["test_f1"]),
        ("ConvNeXt1D",              conv_no_pca_test_metrics["test_acc"], conv_no_pca_test_metrics["test_f1"]),
        ("Ensemble (no PCA)",       ens_non_pca_metrics["acc"],       ens_non_pca_metrics["f1"]),
        ("SVM + PCA",               svm_pca_metrics["test_acc"],      svm_pca_metrics["test_f1"]),
        ("RandomForest + PCA",      rf_pca_metrics["test_acc"],       rf_pca_metrics["test_f1"]),
        ("ConvNeXt1D + PCA",        conv_pca_test_metrics["test_acc"], conv_pca_test_metrics["test_f1"]),
        ("Ensemble with PCA",       ens_pca_metrics["acc"],           ens_pca_metrics["f1"]),
    ]

    print(f"{'Model':30s} | {'Test Acc':8s} | {'Test F1':8s}")
    print("-" * 60)
    for name, acc, f1_ in rows_metrics:
        print(f"{name:30s} | {acc:8.4f} | {f1_:8.4f}")

    print_header("SUMMARY (Time in seconds)")

    # สำหรับ SVM/RF/ConvNeXt: มี train / val / test
    # Ensemble: มีเฉพาะ test time ของการรวมผล
    rows_time = [
        ("SVM",                     svm_metrics["t_train"],           svm_metrics["t_val"],            svm_metrics["t_test"]),
        ("RandomForest",            rf_metrics["t_train"],            rf_metrics["t_val"],             rf_metrics["t_test"]),
        ("ConvNeXt1D",              conv_no_pca_res["t_train"],       conv_no_pca_res["t_val"],        conv_no_pca_test_metrics["t_test"]),
        ("Ensemble (no PCA)",       0.0,                              0.0,                             ens_non_pca_metrics["t_test"]),
        ("SVM + PCA",               svm_pca_metrics["t_train"],       svm_pca_metrics["t_val"],        svm_pca_metrics["t_test"]),
        ("RandomForest + PCA",      rf_pca_metrics["t_train"],        rf_pca_metrics["t_val"],         rf_pca_metrics["t_test"]),
        ("ConvNeXt1D + PCA",        conv_pca_res["t_train"],          conv_pca_res["t_val"],           conv_pca_test_metrics["t_test"]),
        ("Ensemble with PCA",       0.0,                              0.0,                             ens_pca_metrics["t_test"]),
    ]

    print(f"{'Model':30s} | {'Train(s)':9s} | {'Val(s)':9s} | {'Test(s)':9s}")
    print("-" * 80)
    for name, t_tr, t_va, t_te in rows_time:
        print(f"{name:30s} | {t_tr:9.10f} | {t_va:9.10f} | {t_te:9.10f}")

    print("\n[INFO] Training finished and all models saved to:", args.models_dir)


if __name__ == "__main__":
    main()
