# train_ml_with_convnext_pca_ensemble.py

import os
import argparse
import numpy as np
import joblib
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------------------------------------------------
# ConvNeXt-style 1D block & head (ใช้เป็น classifier บน feature vector)
# ----------------------------------------------------------------------

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        # x: [B, C, L]
        shortcut = x
        x = self.dwconv(x)            # [B, C, L]
        x = x.permute(0, 2, 1)        # [B, L, C]
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 2, 1)        # [B, C, L]
        return x + shortcut


class ConvNeXt1DHead(nn.Module):
    def __init__(self, dim_in: int, num_classes: int, depth: int = 2):
        """
        dim_in: ขนาดของ feature จาก ViT (เช่น 384 หรือขนาดหลัง PCA)
        num_classes: จำนวนคลาส (3 = monocot, dicot, other)
        depth: จำนวน ConvNeXtBlock1D
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_in)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1D(dim_in) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim_in)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        # x: [B, D]
        x = self.proj(x)        # [B, D]
        x = x.unsqueeze(-1)     # [B, D, 1] -> treat D as channels
        for blk in self.blocks:
            x = blk(x)          # [B, D, 1]
        x = x.mean(dim=-1)      # global pool -> [B, D]
        x = self.norm(x)        # [B, D]
        x = self.fc(x)          # [B, num_classes]
        return x


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def majority_vote(preds_array: np.ndarray) -> np.ndarray:
    """
    preds_array: shape (n_models, N)
    return: shape (N,)
    """
    n_models, N = preds_array.shape
    out = np.empty(N, dtype=preds_array.dtype)
    for i in range(N):
        vals, counts = np.unique(preds_array[:, i], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def print_split_metrics(name, split, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{name}] {split} accuracy = {acc:.4f}, macroF1 = {macro_f1:.4f}")
    print(f"[{name}] {split} classification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        )
    )
    print(f"[{name}] {split} confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()


def train_and_eval_sklearn(
    model, name, X_train, y_train, X_val, y_val, X_test, y_test, class_names
):
    print("=" * 80)
    print(f"[{name}] Training...")
    model.fit(X_train, y_train)

    # VAL
    y_val_pred = model.predict(X_val)
    print_split_metrics(name, "VAL", y_val, y_val_pred, class_names)

    # TEST
    y_test_pred = model.predict(X_test)
    print_split_metrics(name, "TEST", y_test, y_test_pred, class_names)

    return model, y_val_pred, y_test_pred


def train_convnext1d(
    X_train,
    y_train,
    X_val,
    y_val,
    dim_in,
    num_classes,
    device,
    class_names,
    max_epochs=50,
    batch_size=64,
    lr=1e-3,
    name="ConvNeXt1D",
):
    print("=" * 80)
    print(f"[{name}] Training... (device={device})")

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvNeXt1DHead(dim_in, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val_f1 = -1.0

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        avg_loss = total_loss / max(1, total)

        # ---- val ----
        model.eval()
        all_val_logits = []
        all_val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                all_val_logits.append(logits.cpu())
                all_val_labels.append(yb.cpu())
        all_val_logits = torch.cat(all_val_logits)
        all_val_labels = torch.cat(all_val_labels)
        y_val_pred = all_val_logits.argmax(dim=1).numpy()
        y_val_true = all_val_labels.numpy()

        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred, average="macro")

        print(
            f"[{name}] Epoch {epoch:02d}/{max_epochs} "
            f"train_loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final val metrics (ใช้ best model)
    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_t.to(device)).cpu()
    y_val_pred = logits_val.argmax(dim=1).numpy()
    y_val_true = y_val_t.numpy()
    print_split_metrics(name, "VAL", y_val_true, y_val_pred, class_names)

    return model


def predict_convnext1d(model, X, device):
    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        logits = model(X_t).cpu()
    return logits.argmax(dim=1).numpy()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True,
                        help="path to vit_leaf_features.npz")
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--conv_epochs", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---------- โหลดฟีเจอร์จาก ViT ----------
    data = np.load(args.features, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"].tolist()

    # ensure int labels
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    print("[INFO] Loaded features from", args.features)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"  class_names: {class_names}")

    num_classes = len(class_names)
    dim_in = X_train.shape[1]

    # ------------------------------------------------------------------
    # 1) โมเดลที่ไม่ใช้ PCA: SVM, RF, ConvNeXt1D
    # ------------------------------------------------------------------
    svm_non_pca = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    )
    rf_non_pca = RandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
    )

    svm_non_pca, y_val_svm, y_test_svm = train_and_eval_sklearn(
        svm_non_pca,
        "SVM (ViT features)",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_names,
    )

    rf_non_pca, y_val_rf, y_test_rf = train_and_eval_sklearn(
        rf_non_pca,
        "RandomForest (ViT features)",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_names,
    )

    conv_non_pca = train_convnext1d(
        X_train,
        y_train,
        X_val,
        y_val,
        dim_in=dim_in,
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        max_epochs=args.conv_epochs,
        name="ConvNeXt1D (ViT features)",
    )
    y_val_conv = predict_convnext1d(conv_non_pca, X_val, device)
    y_test_conv = predict_convnext1d(conv_non_pca, X_test, device)
    print_split_metrics(
        "ConvNeXt1D (ViT features)", "TEST", y_test, y_test_conv, class_names
    )

    # Ensemble (no PCA)
    print("=" * 80)
    print("Ensemble WITHOUT PCA [SVM + RF + ConvNeXt1D]")
    preds_ens_test = majority_vote(
        np.stack([y_test_svm, y_test_rf, y_test_conv], axis=0)
    )
    print_split_metrics(
        "Ensemble (no PCA)", "TEST", y_test, preds_ens_test, class_names
    )

    # ------------------------------------------------------------------
    # 2) PCA + SVM / RF / ConvNeXt1D
    # ------------------------------------------------------------------
    n_comp = min(args.pca_components, dim_in)
    print("=" * 80)
    print(f"[PCA] Fitting PCA with n_components={n_comp}")
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    svm_pca = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    )
    rf_pca = RandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
    )

    svm_pca, y_val_svm_pca, y_test_svm_pca = train_and_eval_sklearn(
        svm_pca,
        "SVM (PCA+ViT features)",
        X_train_pca,
        y_train,
        X_val_pca,
        y_val,
        X_test_pca,
        y_test,
        class_names,
    )

    rf_pca, y_val_rf_pca, y_test_rf_pca = train_and_eval_sklearn(
        rf_pca,
        "RandomForest (PCA+ViT features)",
        X_train_pca,
        y_train,
        X_val_pca,
        y_val,
        X_test_pca,
        y_test,
        class_names,
    )

    conv_pca = train_convnext1d(
        X_train_pca,
        y_train,
        X_val_pca,
        y_val,
        dim_in=n_comp,
        num_classes=num_classes,
        device=device,
        class_names=class_names,
        max_epochs=args.conv_epochs,
        name="ConvNeXt1D (PCA+ViT features)",
    )
    y_val_conv_pca = predict_convnext1d(conv_pca, X_val_pca, device)
    y_test_conv_pca = predict_convnext1d(conv_pca, X_test_pca, device)
    print_split_metrics(
        "ConvNeXt1D (PCA+ViT features)",
        "TEST",
        y_test,
        y_test_conv_pca,
        class_names,
    )

    # Ensemble (with PCA)
    print("=" * 80)
    print("Ensemble WITH PCA [SVM + RF + ConvNeXt1D]")
    preds_ens_test_pca = majority_vote(
        np.stack([y_test_svm_pca, y_test_rf_pca, y_test_conv_pca], axis=0)
    )
    print_split_metrics(
        "Ensemble (with PCA)", "TEST", y_test, preds_ens_test_pca, class_names
    )

    # ------------------------------------------------------------------
    # 3) บันทึกโมเดลทั้งหมดลงโฟลเดอร์ models/
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(svm_non_pca, "models/svm.pkl")
    joblib.dump(rf_non_pca, "models/rf.pkl")
    torch.save(conv_non_pca.state_dict(), "models/convnext1d_no_pca.pth")

    joblib.dump(pca, "models/pca.pkl")
    joblib.dump(svm_pca, "models/svm_pca.pkl")
    joblib.dump(rf_pca, "models/rf_pca.pkl")
    torch.save(conv_pca.state_dict(), "models/convnext1d_pca.pth")

    np.save("models/class_names.npy", np.array(class_names, dtype=object))

    print("Saved all models to ./models")


if __name__ == "__main__":
    main()
