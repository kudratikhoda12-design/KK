"""
Train a CNN to classify 96x96 histopathology patches as tumor / normal,
evaluate with ROC-AUC / precision / recall / F1, and produce Grad-CAM
explanations for a handful of test patches.

Data: outputs/patches.npz, produced by extract_patches.py from real
CAMELYON16 whole-slide images (public AWS Open Data mirror).
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt

from model import HistoCNN
from gradcam import GradCAM, overlay_heatmap

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = os.environ.get("OUT_DIR", "/home/user/KK/outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PatchDataset(Dataset):
    def __init__(self, x, y, train=False):
        self.x = x  # uint8 (N, 96, 96, 3)
        self.y = y.astype(np.float32)
        if train:
            self.tf = T.Compose([
                T.ToImage(),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(90),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.tf = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, all_probs, all_labels = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(yb)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return total_loss / len(labels), auc


def predict(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE)).squeeze(1)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(yb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def evaluate_split(name, probs, labels, threshold, out_dir):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["normal", "tumor"], zero_division=0)

    text = (
        f"=== {name} (threshold={threshold:.2f}) ===\n"
        f"ROC-AUC:   {auc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-score:  {f1:.4f}\n\n"
        f"Confusion matrix (rows=true, cols=pred) [normal, tumor]:\n{cm}\n\n"
        f"{report}\n"
    )
    print(text)

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"ROC Curve ({name})"); ax[0].legend()
    ax[1].imshow(cm, cmap="Blues")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["normal", "tumor"])
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["normal", "tumor"])
    ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True"); ax[1].set_title(f"Confusion Matrix ({name})")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"roc_confusion_{name}.png"), dpi=140)
    plt.close(fig)
    return text, preds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(os.path.join(OUT_DIR, "patches.npz"))
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    x_holdout, y_holdout = data["x_holdout"], data["y_holdout"]
    print(f"train={x_train.shape} val={x_val.shape} test={x_test.shape} holdout={x_holdout.shape}")
    print(f"device={DEVICE}")

    train_ds = PatchDataset(x_train, y_train, train=True)
    val_ds = PatchDataset(x_val, y_val, train=False)
    test_ds = PatchDataset(x_test, y_test, train=False)

    holdout_ds = PatchDataset(x_holdout, y_holdout, train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    holdout_loader = DataLoader(holdout_ds, batch_size=64, shuffle=False, num_workers=2)

    model = HistoCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc, best_state = -1, None
    epochs = 20
    for epoch in range(1, epochs + 1):
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_auc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(val_auc)
        print(f"epoch {epoch:2d}/{epochs} | train loss {tr_loss:.4f} auc {tr_auc:.4f} "
              f"| val loss {val_loss:.4f} auc {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "histo_cnn.pt"))
    print(f"Best val AUC: {best_val_auc:.4f} (checkpoint restored)")

    # ---- Training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_auc"], label="train")
    axes[1].plot(history["val_auc"], label="val")
    axes[1].set_title("ROC-AUC"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=140)
    plt.close(fig)

    # ---- Pick an operating threshold on the validation set (never on test) ----
    val_probs, val_labels = predict(model, val_loader)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(val_labels, (val_probs >= t).astype(int)) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(f1s))])
    print(f"Operating threshold selected on validation set: {threshold:.2f}")

    # ---- Evaluate on the in-distribution test split AND the cross-slide holdout ----
    test_probs, test_labels = predict(model, test_loader)
    holdout_probs, holdout_labels = predict(model, holdout_loader)

    test_text, test_preds = evaluate_split("test", test_probs, test_labels, threshold, OUT_DIR)
    holdout_text, holdout_preds = evaluate_split("holdout_cross_slide", holdout_probs, holdout_labels, threshold, OUT_DIR)

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Operating threshold (tuned on val): {threshold:.2f}\n\n")
        f.write("'test' = pooled, patch-level split from the 6 slides used in training.\n")
        f.write("'holdout_cross_slide' = 2 whole slides never seen in train/val/test,\n")
        f.write("a much harder and more realistic measure of generalization.\n\n")
        f.write(test_text)
        f.write("\n")
        f.write(holdout_text)

    # ---- Grad-CAM on representative cross-slide holdout patches ----
    target_layer = model.features[3]  # last conv block (256 channels, no pooling)
    cam_engine = GradCAM(model, target_layer)

    tp = np.where((holdout_labels == 1) & (holdout_preds == 1))[0]
    tn = np.where((holdout_labels == 0) & (holdout_preds == 0))[0]
    fp = np.where((holdout_labels == 0) & (holdout_preds == 1))[0]
    fn = np.where((holdout_labels == 1) & (holdout_preds == 0))[0]
    picks = []
    for name, idx_arr in [("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn)]:
        if len(idx_arr) > 0:
            picks.append((name, idx_arr[0]))

    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6))
    if len(picks) == 1:
        axes = axes.reshape(2, 1)
    for col, (name, idx) in enumerate(picks):
        raw = x_holdout[idx]
        xb, _ = holdout_ds[idx]
        xb = xb.unsqueeze(0).to(DEVICE)
        cam = cam_engine(xb)
        overlay = overlay_heatmap(raw, cam)
        axes[0, col].imshow(raw); axes[0, col].set_title(f"{name} p={holdout_probs[idx]:.2f}"); axes[0, col].axis("off")
        axes[1, col].imshow(overlay); axes[1, col].set_title("Grad-CAM"); axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gradcam_examples.png"), dpi=140)
    plt.close(fig)

    print("Saved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
