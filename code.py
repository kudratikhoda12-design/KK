"""
Explainable Histopathology Image Classification
=================================================

A complete, single-file PyTorch pipeline for binary classification of
histopathology image patches (tumor vs. normal lymph-node tissue), with
Grad-CAM explainability.

WHAT THIS SCRIPT DOES
----------------------
1. Downloads raw whole-slide images (WSIs) + tumor annotation masks from the
   public, unauthenticated AWS Open Data mirror of the CAMELYON16 challenge
   (https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/) -- the same lymph
   node slide collection that the well-known PatchCamelyon (PCam) benchmark
   was derived from.
2. Re-derives a small PCam-style dataset of labeled 96x96 tissue patches
   directly from those raw slides: tissue detection, PCam-style
   center-region labeling from the tumor mask, and Reinhard color
   normalization to counter inter-slide staining variation.
3. Splits the data two ways:
     - a pooled, patch-level stratified train/val/test split (6 slides)
     - a slide-disjoint "cross-slide" holdout (2 further slides, never
       touched during training) -- a stricter, more realistic test of
       generalization to unseen tissue/staining.
4. Trains a compact CNN from scratch with augmentation.
5. Evaluates with ROC-AUC, precision, recall, F1, confusion matrix -- on
   both the standard test split and the cross-slide holdout.
6. Produces Grad-CAM visual explanations for representative predictions.

This file is the consolidated, readable reference implementation. The
actual project is organized as separate modules under `src/`
(`extract_patches.py`, `model.py`, `gradcam.py`, `train.py`) which is how it
was developed and run; this file mirrors that same logic in one place for
easy reading end-to-end. See `report.md` for results and `learning_notes.md`
for a topic-by-topic explanation of every technique used here.
"""

import os
import numpy as np
import cv2
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("WSI_DIR", "./data/wsi")       # raw downloaded slides
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")          # patches + results
BASE_URL = "https://camelyon-dataset.s3.amazonaws.com/CAMELYON16"

PATCH = 96                    # patch side length in pixels (PCam standard)
CENTER = 32                   # PCam-style label decided by the central 32x32 region
TISSUE_SAT_THRESH = 0.08      # HSV saturation threshold separating tissue from glass
TISSUE_MIN_FRACTION = 0.35    # fraction of a patch that must look like tissue
MAX_POS_PER_SLIDE = 1800
MAX_NEG_PER_SLIDE = 1800

# Six slides feed a pooled, patch-level stratified train/val/test split.
# Two further slides are held out completely: they never contribute a
# single patch to train/val/test, so evaluating on them measures
# generalization to tissue/staining the model has genuinely never seen.
MAIN_SLIDES = [
    ("tumor_091", True), ("normal_108", False),
    ("tumor_075", True), ("normal_042", False),
    ("tumor_082", True), ("normal_004", False),
]
HOLDOUT_SLIDES = [("tumor_084", True), ("normal_150", False)]

# Reference LAB statistics for Reinhard color normalization, estimated once
# from ~400 tissue tiles of tumor_091. Every patch (every slide, every
# split) is remapped onto this single reference so that slide-to-slide
# staining/scanner color shift is not a shortcut feature the CNN can exploit.
REF_LAB_MEAN = np.array([136.72, 156.67, 102.10], dtype=np.float32)
REF_LAB_STD = np.array([61.34, 14.79, 14.73], dtype=np.float32)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# 1. Data acquisition -- download raw WSIs + masks from public S3
# ---------------------------------------------------------------------------
def download_slides():
    """Download every slide/mask referenced by MAIN_SLIDES + HOLDOUT_SLIDES,
    if not already present. No AWS credentials are needed; the CAMELYON16
    mirror is a public, unauthenticated bucket."""
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    files = []
    for name, has_tumor in MAIN_SLIDES + HOLDOUT_SLIDES:
        files.append(f"images/{name}.tif")
        if has_tumor:
            files.append(f"masks/{name}_mask.tif")
    for rel in files:
        local = os.path.join(DATA_DIR, os.path.basename(rel))
        if os.path.exists(local):
            continue
        print(f"Downloading {rel} -> {local}")
        urllib.request.urlretrieve(f"{BASE_URL}/{rel}", local)


# ---------------------------------------------------------------------------
# 2. Patch extraction from raw whole-slide TIFFs
# ---------------------------------------------------------------------------
def pick_level(tif_path, target_downsample=8.0):
    """Pick the pyramid level whose downsample factor is closest to target.
    WSIs are stored as multi-resolution pyramids (like a zoomable map); we
    work at a downsampled level (~8x) instead of full resolution (which
    would be tens of thousands of pixels per side)."""
    tf = tifffile.TiffFile(tif_path)
    base_shape = tf.pages[0].shape
    best_i, best_diff = 0, 1e9
    for i, page in enumerate(tf.pages):
        diff = abs(base_shape[0] / page.shape[0] - target_downsample)
        if diff < best_diff:
            best_diff, best_i = diff, i
    return best_i


def load_level(tif_path, level_idx):
    return tifffile.TiffFile(tif_path).pages[level_idx].asarray()


def load_mask_matching(mask_path, target_shape):
    """Tumor masks have their own pyramid, whose downsample steps don't
    exactly line up with the image pyramid, so resize (nearest-neighbor,
    since it's a label mask) onto the image's chosen level."""
    tf = tifffile.TiffFile(mask_path)
    best_i, best_diff = 0, 1e9
    for i, page in enumerate(tf.pages):
        diff = abs(page.shape[0] - target_shape[0]) + abs(page.shape[1] - target_shape[1])
        if diff < best_diff:
            best_diff, best_i = diff, i
    mask = tf.pages[best_i].asarray()
    if mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def is_tissue(patch_rgb):
    """Glass background is low-saturation (near-white); tissue is not."""
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    return (sat > TISSUE_SAT_THRESH).mean() >= TISSUE_MIN_FRACTION


def reinhard_normalize(patch_rgb):
    """Reinhard color transfer: standardize this patch's own LAB mean/std,
    then rescale onto a single fixed reference distribution. Preserves the
    patch's *internal* relative contrast (nuclei density/pattern -- the
    diagnostic signal) while removing the *global* stain-intensity
    difference between slides/scanners (the nuisance/batch-effect signal)."""
    lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = lab.mean(axis=(0, 1))
    std = lab.std(axis=(0, 1)) + 1e-6
    lab = (lab - mean) / std * REF_LAB_STD + REF_LAB_MEAN
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_from_slide(name, has_tumor, rng):
    """Tile one slide into non-overlapping 96x96 patches, keep only tissue
    patches, and label each using the PCam convention: positive if the
    *central* 32x32 region touches any tumor-annotated pixel; negative only
    if the *entire* patch has zero tumor pixels (ambiguous boundary patches
    are dropped rather than mislabeled)."""
    img_path = os.path.join(DATA_DIR, f"{name}.tif")
    level = pick_level(img_path)
    img = load_level(img_path, level)
    print(f"[{name}] level {level} shape {img.shape}")

    mask = None
    if has_tumor:
        mask = load_mask_matching(os.path.join(DATA_DIR, f"{name}_mask.tif"), img.shape) > 0

    h, w = img.shape[:2]
    coords = [(r, c) for r in range(h // PATCH) for c in range(w // PATCH)]
    rng.shuffle(coords)
    c0 = (PATCH - CENTER) // 2

    pos, neg = [], []
    for r, c in coords:
        y0, x0 = r * PATCH, c * PATCH
        patch = img[y0:y0 + PATCH, x0:x0 + PATCH]
        if patch.shape[:2] != (PATCH, PATCH) or not is_tissue(patch):
            continue
        if mask is not None:
            m = mask[y0:y0 + PATCH, x0:x0 + PATCH]
            if m[c0:c0 + CENTER, c0:c0 + CENTER].any():
                if len(pos) < MAX_POS_PER_SLIDE:
                    pos.append(reinhard_normalize(patch))
                continue
            if m.any():
                continue  # tumor pixels exist but not in the center -> ambiguous, skip
            if len(neg) < MAX_NEG_PER_SLIDE:
                neg.append(reinhard_normalize(patch))
        else:
            if len(neg) < MAX_NEG_PER_SLIDE:
                neg.append(reinhard_normalize(patch))
        if len(pos) >= MAX_POS_PER_SLIDE and len(neg) >= MAX_NEG_PER_SLIDE:
            break

    print(f"[{name}] kept {len(pos)} positive / {len(neg)} negative patches")
    return pos, neg


def build_pool(slide_list, rng):
    all_x, all_y = [], []
    for name, has_tumor in slide_list:
        pos, neg = extract_from_slide(name, has_tumor, rng)
        all_x.extend(pos + neg)
        all_y.extend([1] * len(pos) + [0] * len(neg))
    return np.stack(all_x).astype(np.uint8), np.array(all_y, dtype=np.int64)


def balance_classes(x, y, rng):
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)])
    rng.shuffle(keep)
    return x[keep], y[keep]


def stratified_split(x, y, rng, fracs=(0.70, 0.15, 0.15)):
    idx_splits = [[], [], []]
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_train = int(fracs[0] * len(idx_c))
        n_val = int(fracs[1] * len(idx_c))
        idx_splits[0].extend(idx_c[:n_train])
        idx_splits[1].extend(idx_c[n_train:n_train + n_val])
        idx_splits[2].extend(idx_c[n_train + n_val:])
    out = []
    for idx in idx_splits:
        idx = np.array(idx)
        rng.shuffle(idx)
        out.append((x[idx], y[idx]))
    return out


def build_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    x_pool, y_pool = build_pool(MAIN_SLIDES, rng)
    x_pool, y_pool = balance_classes(x_pool, y_pool, rng)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = stratified_split(x_pool, y_pool, rng)

    x_holdout, y_holdout = build_pool(HOLDOUT_SLIDES, rng)
    x_holdout, y_holdout = balance_classes(x_holdout, y_holdout, rng)

    np.savez_compressed(
        os.path.join(OUT_DIR, "patches.npz"),
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test, x_holdout=x_holdout, y_holdout=y_holdout,
    )
    print("train:", x_train.shape, "val:", x_val.shape,
          "test:", x_test.shape, "holdout:", x_holdout.shape)


# ---------------------------------------------------------------------------
# 3. Model: a small CNN trained from scratch
# ---------------------------------------------------------------------------
class HistoCNN(nn.Module):
    """VGG-style CNN: four conv blocks (32->64->128->256 channels) + global
    average pool + a single linear head. The last block keeps its spatial
    resolution (no pooling) so Grad-CAM has a reasonably sized feature map
    (12x12) to compute a localization heatmap from."""

    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    @staticmethod
    def _block(c_in, c_out, pool=True):
        layers = [
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ---------------------------------------------------------------------------
# 4. Grad-CAM (Selvaraju et al., 2017)
# ---------------------------------------------------------------------------
class GradCAM:
    """Weights each channel of a target conv layer's activation map by the
    global-average-pooled gradient of the class score w.r.t. that channel,
    then combines and ReLUs -- highlighting the regions that most increase
    the model's predicted probability."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = self.gradients = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logit = self.model(x)
        score = logit[:, 0] if class_idx is None else logit[:, class_idx]
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        cam -= cam.min()
        return cam / cam.max() if cam.max() > 0 else cam


def overlay_heatmap(rgb_uint8, cam, alpha=0.4):
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * rgb_uint8).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Dataset / augmentation
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, x, y, train=False):
        self.x, self.y = x, y.astype(np.float32)
        base = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if train:
            # Geometric augmentation (H&E tissue has no canonical orientation)
            # plus mild color jitter for residual, patch-level color noise
            # left over after the slide-level Reinhard normalization above.
            self.tf = T.Compose([
                T.ToImage(),
                T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5), T.RandomRotation(90),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = T.Compose(base)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


# ---------------------------------------------------------------------------
# 6. Train / evaluate
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, probs_all, labels_all = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(yb)
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_all.append(yb.cpu().numpy())
    probs, labels = np.concatenate(probs_all), np.concatenate(labels_all)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return total_loss / len(labels), auc


def predict(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs_all.append(torch.sigmoid(model(xb.to(DEVICE)).squeeze(1)).cpu().numpy())
            labels_all.append(yb.numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def evaluate_split(name, probs, labels, threshold, out_dir):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["normal", "tumor"], zero_division=0)
    text = (f"=== {name} (threshold={threshold:.2f}) ===\nROC-AUC: {auc:.4f}  Precision: {precision:.4f}  "
            f"Recall: {recall:.4f}  F1: {f1:.4f}\nConfusion matrix [normal, tumor]:\n{cm}\n\n{report}\n")
    print(text)

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}"); ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title(f"ROC ({name})"); ax[0].legend()
    ax[1].imshow(cm, cmap="Blues")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["normal", "tumor"])
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["normal", "tumor"])
    ax[1].set_title(f"Confusion ({name})")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"roc_confusion_{name}.png"), dpi=140); plt.close(fig)
    return text, preds


def train_and_evaluate():
    data = np.load(os.path.join(OUT_DIR, "patches.npz"))
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    x_holdout, y_holdout = data["x_holdout"], data["y_holdout"]

    train_loader = DataLoader(PatchDataset(x_train, y_train, train=True), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(PatchDataset(x_val, y_val), batch_size=64, num_workers=2)
    test_loader = DataLoader(PatchDataset(x_test, y_test), batch_size=64, num_workers=2)
    holdout_ds = PatchDataset(x_holdout, y_holdout)
    holdout_loader = DataLoader(holdout_ds, batch_size=64, num_workers=2)

    model = HistoCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_auc, best_state = -1, None
    for epoch in range(1, 21):
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_auc)
        print(f"epoch {epoch:2d}/20 | train loss {tr_loss:.4f} auc {tr_auc:.4f} "
              f"| val loss {val_loss:.4f} auc {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, best_state = val_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "histo_cnn.pt"))

    # Operating threshold is tuned on validation only, never on test/holdout.
    val_probs, val_labels = predict(model, val_loader)
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = float(thresholds[np.argmax([f1_score(val_labels, (val_probs >= t).astype(int)) for t in thresholds])])

    test_probs, test_labels = predict(model, test_loader)
    holdout_probs, holdout_labels = predict(model, holdout_loader)
    _, _ = evaluate_split("test", test_probs, test_labels, threshold, OUT_DIR)
    _, holdout_preds = evaluate_split("holdout_cross_slide", holdout_probs, holdout_labels, threshold, OUT_DIR)

    # Grad-CAM on one true-positive / true-negative / false-positive / false-negative
    target_layer = model.features[3]
    cam_engine = GradCAM(model, target_layer)
    groups = {
        "TP": np.where((holdout_labels == 1) & (holdout_preds == 1))[0],
        "TN": np.where((holdout_labels == 0) & (holdout_preds == 0))[0],
        "FP": np.where((holdout_labels == 0) & (holdout_preds == 1))[0],
        "FN": np.where((holdout_labels == 1) & (holdout_preds == 0))[0],
    }
    picks = [(name, idx[0]) for name, idx in groups.items() if len(idx) > 0]
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6))
    for col, (name, idx) in enumerate(picks):
        xb, _ = holdout_ds[idx]
        cam = cam_engine(xb.unsqueeze(0).to(DEVICE))
        overlay = overlay_heatmap(x_holdout[idx], cam)
        axes[0, col].imshow(x_holdout[idx]); axes[0, col].set_title(f"{name} p={holdout_probs[idx]:.2f}"); axes[0, col].axis("off")
        axes[1, col].imshow(overlay); axes[1, col].set_title("Grad-CAM"); axes[1, col].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "gradcam_examples.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    download_slides()
    build_dataset()
    train_and_evaluate()
