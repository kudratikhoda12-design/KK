"""
Extract labeled 96x96 tissue patches from raw CAMELYON16 whole-slide images.

Source data: public AWS Open Data mirror of the CAMELYON16 challenge
(https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/), the same lymph-node
histopathology slides that the PatchCamelyon (PCam) benchmark was derived
from. This script re-derives a small PCam-style patch dataset directly from
raw whole-slide TIFFs + tumor annotation masks, using a slide-level
train/test split so evaluation slides are never seen during training.
"""
import os
import numpy as np
import cv2
import tifffile

DATA_DIR = os.environ.get("WSI_DIR", "/tmp/claude-0/-home-user-KK/8cde70e5-f234-515c-af31-2bccfc25299e/scratchpad/data")
OUT_DIR = os.environ.get("OUT_DIR", "/home/user/KK/outputs")
PATCH = 96
CENTER = 32  # PCam-style: label decided by the central 32x32 region
TISSUE_SAT_THRESH = 0.08      # HSV saturation threshold separating tissue from glass
TISSUE_MIN_FRACTION = 0.35    # fraction of pixels in patch that must look like tissue

# Slide-disjoint splits: every slide appears in exactly one split, so
# validation genuinely measures generalization to unseen tissue/staining
# rather than just interpolating within slides already seen in training.
SLIDES = {
    "train": [
        ("tumor_091", True), ("normal_108", False),
        ("tumor_075", True), ("normal_042", False),
    ],
    "val": [("tumor_082", True), ("normal_004", False)],
    "test": [("tumor_084", True), ("normal_150", False)],
}

MAX_POS_PER_SLIDE = 1800
MAX_NEG_PER_SLIDE = 1800


def pick_level(tif_path, target_downsample=8.0):
    """Pick the pyramid level whose downsample factor is closest to target."""
    tf = tifffile.TiffFile(tif_path)
    base_shape = tf.pages[0].shape
    best_i, best_diff = 0, 1e9
    for i, page in enumerate(tf.pages):
        ds = base_shape[0] / page.shape[0]
        diff = abs(ds - target_downsample)
        if diff < best_diff:
            best_diff, best_i = diff, i
    return best_i


def load_level(tif_path, level_idx):
    tf = tifffile.TiffFile(tif_path)
    return tf.pages[level_idx].asarray()


def load_mask_matching(mask_path, target_shape):
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
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    return (sat > TISSUE_SAT_THRESH).mean() >= TISSUE_MIN_FRACTION


# Reference LAB statistics for Reinhard color normalization, estimated once
# from ~400 tissue tiles of tumor_091. Every patch (train AND test, from every
# slide) is re-mapped onto this single reference so that per-slide staining/
# scanner color shift is no longer a shortcut feature the CNN can exploit.
REF_LAB_MEAN = np.array([136.72, 156.67, 102.10], dtype=np.float32)
REF_LAB_STD = np.array([61.34, 14.79, 14.73], dtype=np.float32)


def reinhard_normalize(patch_rgb):
    lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = lab.mean(axis=(0, 1))
    std = lab.std(axis=(0, 1)) + 1e-6
    lab = (lab - mean) / std * REF_LAB_STD + REF_LAB_MEAN
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_from_slide(name, has_tumor, rng):
    img_path = os.path.join(DATA_DIR, f"{name}.tif")
    level = pick_level(img_path, target_downsample=8.0)
    img = load_level(img_path, level)
    print(f"[{name}] level {level} shape {img.shape}")

    mask = None
    if has_tumor:
        mask_path = os.path.join(DATA_DIR, f"{name}_mask.tif")
        mask = load_mask_matching(mask_path, img.shape) > 0

    h, w = img.shape[0], img.shape[1]
    n_rows, n_cols = h // PATCH, w // PATCH

    pos, neg = [], []
    coords = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    rng.shuffle(coords)

    c0 = (PATCH - CENTER) // 2
    for r, c in coords:
        y0, x0 = r * PATCH, c * PATCH
        patch = img[y0:y0 + PATCH, x0:x0 + PATCH]
        if patch.shape[0] != PATCH or patch.shape[1] != PATCH:
            continue
        if not is_tissue(patch):
            continue

        if mask is not None:
            m = mask[y0:y0 + PATCH, x0:x0 + PATCH]
            center = m[c0:c0 + CENTER, c0:c0 + CENTER]
            if center.any():
                if len(pos) < MAX_POS_PER_SLIDE:
                    pos.append(reinhard_normalize(patch))
                continue
            if m.any():
                continue  # ambiguous tumor-boundary patch, skip
            if len(neg) < MAX_NEG_PER_SLIDE:
                neg.append(reinhard_normalize(patch))
        else:
            if len(neg) < MAX_NEG_PER_SLIDE:
                neg.append(reinhard_normalize(patch))

        if len(pos) >= MAX_POS_PER_SLIDE and len(neg) >= MAX_NEG_PER_SLIDE:
            break

    print(f"[{name}] kept {len(pos)} positive / {len(neg)} negative patches")
    return pos, neg


def build_split(split_name, rng):
    all_x, all_y = [], []
    for name, has_tumor in SLIDES[split_name]:
        pos, neg = extract_from_slide(name, has_tumor, rng)
        all_x.extend(pos + neg)
        all_y.extend([1] * len(pos) + [0] * len(neg))
    x = np.stack(all_x).astype(np.uint8)
    y = np.array(all_y, dtype=np.int64)
    idx = rng.permutation(len(y))
    return x[idx], y[idx]


def balance_classes(x, y, rng):
    """Undersample the majority class so both labels are equally represented."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)])
    rng.shuffle(keep)
    return x[keep], y[keep]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    x_train, y_train = build_split("train", rng)
    x_train, y_train = balance_classes(x_train, y_train, rng)
    x_val, y_val = build_split("val", rng)
    x_val, y_val = balance_classes(x_val, y_val, rng)
    x_test, y_test = build_split("test", rng)

    np.savez_compressed(
        os.path.join(OUT_DIR, "patches.npz"),
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )
    print("train:", x_train.shape, np.bincount(y_train))
    print("val:  ", x_val.shape, np.bincount(y_val))
    print("test: ", x_test.shape, np.bincount(y_test))
    print("Saved ->", os.path.join(OUT_DIR, "patches.npz"))


if __name__ == "__main__":
    main()
