# train_svm_and_rf_with_pca.py
#
# ใช้ ViT features (.npz) เพื่อเทรน:
#   - SVM (ไม่มี PCA)
#   - RandomForest (ไม่มี PCA)
#   - MLP (Neural Network บน ViT features) = ใช้เป็น "CNN" ตามใบงาน
#   - SVM + PCA
#   - RF + PCA
#   - MLP + PCA
#   - Ensemble (non-PCA) จาก SVM + RF + MLP
#   - Ensemble (PCA)    จาก SVM+PCA + RF+PCA + MLP+PCA
#
# MLP จะ train บน GPU ถ้ามี CUDA

import argparse
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# -------------------------- Utils --------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data.get("X_val", None)
    y_val = data.get("y_val", None)
    X_test = data.get("X_test", None)
    y_test = data.get("y_test", None)
    class_names = data.get("class_names", None)
    if class_names is not None:
        class_names = [str(c) for c in class_names]

    print(f"[INFO] Loaded features from {npz_path}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")
    print(f"  class_names: {class_names}")

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names


# -------------------------- MLP (ใช้แทน CNN บน features) --------------------------

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[MLPClassifier, Dict]:

    in_dim = X_train.shape[1]
    model = MLPClassifier(in_dim=in_dim, num_classes=num_classes).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # eval on val
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        print(
            f"[MLP] Epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = {"best_val_f1": best_val_f1}
    return model, stats


def mlp_predict_proba(model: MLPClassifier, X: np.ndarray, device: torch.device, batch_size: int = 128):
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ------------------------ Evaluation Helper ------------------------

def eval_sklearn_model(
    name: str,
    clf,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    class_names,
):
    print("\n" + "=" * 80)
    print(f"[{name}] Training...")
    clf.fit(X_train, y_train)

    results = {}

    def _eval(split_name, X, y):
        y_pred = clf.predict(X)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
        else:
            # ถ้าไม่มี predict_proba ให้ใช้ one-hot จาก predict แทน
            proba = np.eye(len(class_names))[y_pred]
        acc = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro")

        print(f"\n[{name}] {split_name} accuracy = {acc:.4f}, macroF1 = {macro_f1:.4f}")
        print(f"[{name}] {split_name} classification report:")
        print(classification_report(y, y_pred, target_names=class_names, digits=4))
        print(f"[{name}] {split_name} confusion matrix:")
        print(confusion_matrix(y, y_pred))

        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "y_pred": y_pred,
            "proba": proba,
        }

    results["val"] = _eval("VAL", X_val, y_val)
    results["test"] = _eval("TEST", X_test, y_test)
    return results


# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True,
                        help=".npz ที่เซฟ feature ไว้จาก extract_vit_features.py")
    parser.add_argument("--pca_components", type=int, default=50,
                        help="จำนวนมิติหลัง PCA สำหรับ models ที่ใช้ PCA")
    parser.add_argument("--mlp_epochs", type=int, default=80)
    parser.add_argument("--mlp_batch_size", type=int, default=64)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' หรือ 'cpu' (สำหรับ MLP)")
    args = parser.parse_args()

    set_seed(args.seed)

    # detect device สำหรับ MLP
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device for MLP: {device.type}")

    # -------- 1) โหลด features --------
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_features(args.features)
    num_classes = len(class_names)

    # -------- 2) Standardize features (ใช้ร่วมกันทุกโมเดล) --------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # -------- 3) โมเดลแบบไม่ใช้ PCA --------
    results = {}

    # 3.1 SVM (non-PCA)
    svm_clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=args.seed,
    )
    results["SVM"] = eval_sklearn_model("SVM (non-PCA)",
                                        svm_clf,
                                        X_train_s, y_train,
                                        X_val_s, y_val,
                                        X_test_s, y_test,
                                        class_names)

    # 3.2 RF (non-PCA)
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    results["RF"] = eval_sklearn_model("RandomForest (non-PCA)",
                                       rf_clf,
                                       X_train_s, y_train,
                                       X_val_s, y_val,
                                       X_test_s, y_test,
                                       class_names)

    # 3.3 MLP (non-PCA) = ใช้แทน "CNN" บน feature
    print("\n" + "=" * 80)
    print("[MLP (non-PCA)] Training...")
    mlp_model, mlp_stats = train_mlp(
        X_train_s, y_train,
        X_val_s, y_val,
        num_classes=num_classes,
        device=device,
        epochs=args.mlp_epochs,
        batch_size=args.mlp_batch_size,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )

    # eval MLP
    mlp_proba_val = mlp_predict_proba(mlp_model, X_val_s, device)
    mlp_proba_test = mlp_predict_proba(mlp_model, X_test_s, device)
    mlp_pred_val = mlp_proba_val.argmax(axis=1)
    mlp_pred_test = mlp_proba_test.argmax(axis=1)

    mlp_val_acc = accuracy_score(y_val, mlp_pred_val)
    mlp_val_f1 = f1_score(y_val, mlp_pred_val, average="macro")
    mlp_test_acc = accuracy_score(y_test, mlp_pred_test)
    mlp_test_f1 = f1_score(y_test, mlp_pred_test, average="macro")

    print(f"\n[MLP (non-PCA)] VAL accuracy = {mlp_val_acc:.4f}, macroF1 = {mlp_val_f1:.4f}")
    print("[MLP (non-PCA)] VAL classification report:")
    print(classification_report(y_val, mlp_pred_val, target_names=class_names, digits=4))
    print("[MLP (non-PCA)] VAL confusion matrix:")
    print(confusion_matrix(y_val, mlp_pred_val))

    print(f"\n[MLP (non-PCA)] TEST accuracy = {mlp_test_acc:.4f}, macroF1 = {mlp_test_f1:.4f}")
    print("[MLP (non-PCA)] TEST classification report:")
    print(classification_report(y_test, mlp_pred_test, target_names=class_names, digits=4))
    print("[MLP (non-PCA)] TEST confusion matrix:")
    print(confusion_matrix(y_test, mlp_pred_test))

    results["MLP"] = {
        "val": {"acc": mlp_val_acc, "macro_f1": mlp_val_f1, "y_pred": mlp_pred_val, "proba": mlp_proba_val},
        "test": {"acc": mlp_test_acc, "macro_f1": mlp_test_f1, "y_pred": mlp_pred_test, "proba": mlp_proba_test},
    }

    # -------- 4) Ensemble ของ 3 non-PCA: SVM + RF + MLP --------
    print("\n" + "=" * 80)
    print("[Ensemble non-PCA] Soft voting of SVM + RF + MLP")

    # ใช้ prob จากแต่ละโมเดล
    svm_val_proba = results["SVM"]["val"]["proba"]
    rf_val_proba = results["RF"]["val"]["proba"]
    mlp_val_proba = results["MLP"]["val"]["proba"]

    svm_test_proba = results["SVM"]["test"]["proba"]
    rf_test_proba = results["RF"]["test"]["proba"]
    mlp_test_proba = results["MLP"]["test"]["proba"]

    ens_val_proba = (svm_val_proba + rf_val_proba + mlp_val_proba) / 3.0
    ens_test_proba = (svm_test_proba + rf_test_proba + mlp_test_proba) / 3.0

    ens_val_pred = ens_val_proba.argmax(axis=1)
    ens_test_pred = ens_test_proba.argmax(axis=1)

    ens_val_acc = accuracy_score(y_val, ens_val_pred)
    ens_val_f1 = f1_score(y_val, ens_val_pred, average="macro")
    ens_test_acc = accuracy_score(y_test, ens_test_pred)
    ens_test_f1 = f1_score(y_test, ens_test_pred, average="macro")

    print(f"\n[Ensemble non-PCA] VAL accuracy = {ens_val_acc:.4f}, macroF1 = {ens_val_f1:.4f}")
    print("[Ensemble non-PCA] VAL classification report:")
    print(classification_report(y_val, ens_val_pred, target_names=class_names, digits=4))
    print("[Ensemble non-PCA] VAL confusion matrix:")
    print(confusion_matrix(y_val, ens_val_pred))

    print(f"\n[Ensemble non-PCA] TEST accuracy = {ens_test_acc:.4f}, macroF1 = {ens_test_f1:.4f}")
    print("[Ensemble non-PCA] TEST classification report:")
    print(classification_report(y_test, ens_test_pred, target_names=class_names, digits=4))
    print("[Ensemble non-PCA] TEST confusion matrix:")
    print(confusion_matrix(y_test, ens_test_pred))

    results["Ensemble_nonPCA"] = {
        "val": {"acc": ens_val_acc, "macro_f1": ens_val_f1},
        "test": {"acc": ens_test_acc, "macro_f1": ens_test_f1},
    }

    # -------- 5) โมเดลแบบใช้ PCA --------
    print("\n" + "=" * 80)
    print(f"[INFO] Applying PCA (n_components={args.pca_components}) on standardized features...")

    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    # 5.1 SVM + PCA
    svm_p_clf = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=args.seed,
    )
    results["SVM_PCA"] = eval_sklearn_model("SVM + PCA",
                                            svm_p_clf,
                                            X_train_p, y_train,
                                            X_val_p, y_val,
                                            X_test_p, y_test,
                                            class_names)

    # 5.2 RF + PCA
    rf_p_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    results["RF_PCA"] = eval_sklearn_model("RandomForest + PCA",
                                           rf_p_clf,
                                           X_train_p, y_train,
                                           X_val_p, y_val,
                                           X_test_p, y_test,
                                           class_names)

    # 5.3 MLP + PCA (ใช้ input dim = n_components)
    print("\n" + "=" * 80)
    print("[MLP + PCA] Training...")
    mlp_p_model, mlp_p_stats = train_mlp(
        X_train_p, y_train,
        X_val_p, y_val,
        num_classes=num_classes,
        device=device,
        epochs=args.mlp_epochs,
        batch_size=args.mlp_batch_size,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )

    mlp_p_proba_val = mlp_predict_proba(mlp_p_model, X_val_p, device)
    mlp_p_proba_test = mlp_predict_proba(mlp_p_model, X_test_p, device)
    mlp_p_pred_val = mlp_p_proba_val.argmax(axis=1)
    mlp_p_pred_test = mlp_p_proba_test.argmax(axis=1)

    mlp_p_val_acc = accuracy_score(y_val, mlp_p_pred_val)
    mlp_p_val_f1 = f1_score(y_val, mlp_p_pred_val, average="macro")
    mlp_p_test_acc = accuracy_score(y_test, mlp_p_pred_test)
    mlp_p_test_f1 = f1_score(y_test, mlp_p_pred_test, average="macro")

    print(f"\n[MLP + PCA] VAL accuracy = {mlp_p_val_acc:.4f}, macroF1 = {mlp_p_val_f1:.4f}")
    print("[MLP + PCA] VAL classification report:")
    print(classification_report(y_val, mlp_p_pred_val, target_names=class_names, digits=4))
    print("[MLP + PCA] VAL confusion matrix:")
    print(confusion_matrix(y_val, mlp_p_pred_val))

    print(f"\n[MLP + PCA] TEST accuracy = {mlp_p_test_acc:.4f}, macroF1 = {mlp_p_test_f1:.4f}")
    print("[MLP + PCA] TEST classification report:")
    print(classification_report(y_test, mlp_p_pred_test, target_names=class_names, digits=4))
    print("[MLP + PCA] TEST confusion matrix:")
    print(confusion_matrix(y_test, mlp_p_pred_test))

    results["MLP_PCA"] = {
        "val": {"acc": mlp_p_val_acc, "macro_f1": mlp_p_val_f1, "y_pred": mlp_p_pred_val, "proba": mlp_p_proba_val},
        "test": {"acc": mlp_p_test_acc, "macro_f1": mlp_p_test_f1, "y_pred": mlp_p_pred_test, "proba": mlp_p_proba_test},
    }

    # -------- 6) Ensemble ของ 3 PCA models --------
    print("\n" + "=" * 80)
    print("[Ensemble PCA] Soft voting of SVM+PCA + RF+PCA + MLP+PCA")

    svm_p_val_proba = results["SVM_PCA"]["val"]["proba"]
    rf_p_val_proba = results["RF_PCA"]["val"]["proba"]
    mlp_p_val_proba = results["MLP_PCA"]["val"]["proba"]

    svm_p_test_proba = results["SVM_PCA"]["test"]["proba"]
    rf_p_test_proba = results["RF_PCA"]["test"]["proba"]
    mlp_p_test_proba = results["MLP_PCA"]["test"]["proba"]

    ens_p_val_proba = (svm_p_val_proba + rf_p_val_proba + mlp_p_val_proba) / 3.0
    ens_p_test_proba = (svm_p_test_proba + rf_p_test_proba + mlp_p_test_proba) / 3.0

    ens_p_val_pred = ens_p_val_proba.argmax(axis=1)
    ens_p_test_pred = ens_p_test_proba.argmax(axis=1)

    ens_p_val_acc = accuracy_score(y_val, ens_p_val_pred)
    ens_p_val_f1 = f1_score(y_val, ens_p_val_pred, average="macro")
    ens_p_test_acc = accuracy_score(y_test, ens_p_test_pred)
    ens_p_test_f1 = f1_score(y_test, ens_p_test_pred, average="macro")

    print(f"\n[Ensemble PCA] VAL accuracy = {ens_p_val_acc:.4f}, macroF1 = {ens_p_val_f1:.4f}")
    print("[Ensemble PCA] VAL classification report:")
    print(classification_report(y_val, ens_p_val_pred, target_names=class_names, digits=4))
    print("[Ensemble PCA] VAL confusion matrix:")
    print(confusion_matrix(y_val, ens_p_val_pred))

    print(f"\n[Ensemble PCA] TEST accuracy = {ens_p_test_acc:.4f}, macroF1 = {ens_p_test_f1:.4f}")
    print("[Ensemble PCA] TEST classification report:")
    print(classification_report(y_test, ens_p_test_pred, target_names=class_names, digits=4))
    print("[Ensemble PCA] TEST confusion matrix:")
    print(confusion_matrix(y_test, ens_p_test_pred))

    results["Ensemble_PCA"] = {
        "val": {"acc": ens_p_val_acc, "macro_f1": ens_p_val_f1},
        "test": {"acc": ens_p_test_acc, "macro_f1": ens_p_test_f1},
    }

    # -------- 7) พิมพ์สรุป 8 ค่า accuracy (TEST) --------
    print("\n" + "#" * 80)
    print("# SUMMARY (TEST) – ใช้รายงาน / ทำตาราง 8 โมเดล")
    print("# 1) SVM")
    print("# 2) RF")
    print("# 3) CNN (ที่นี่ใช้ชื่อ MLP บน ViT features)")
    print("# 4) Ensemble of 3 non-PCA (SVM + RF + CNN)")
    print("# 5) SVM + PCA")
    print("# 6) RF + PCA")
    print("# 7) CNN + PCA (MLP + PCA)")
    print("# 8) Ensemble of 3 PCA (SVM+PCA + RF+PCA + CNN+PCA)")
    print("#" * 80)

    def show(label, key):
        acc = results[key]["test"]["acc"]
        f1 = results[key]["test"]["macro_f1"]
        print(f"{label:30s}  acc={acc:.4f}  macroF1={f1:.4f}")

    show("SVM", "SVM")
    show("RF", "RF")
    show("CNN (MLP non-PCA)", "MLP")
    show("Ensemble non-PCA", "Ensemble_nonPCA")
    show("SVM + PCA", "SVM_PCA")
    show("RF + PCA", "RF_PCA")
    show("CNN + PCA (MLP+PCA)", "MLP_PCA")
    show("Ensemble with PCA", "Ensemble_PCA")


if __name__ == "__main__":
    main()
