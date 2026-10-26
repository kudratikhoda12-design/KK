"""Step 3: honest failure analysis.

Trains the best-performing few-shot method (contrastive linear probe,
k=10) once on a fixed label draw, then breaks down test accuracy by the
nuisance variables the model was never told about: body size and sensor
noise level. Also produces a confusion matrix to see which postures get
confused with which.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from src.models import Backbone
from src.data import POSTURES, BODY_SIZES, NOISE_LEVELS

torch.manual_seed(0)
np.random.seed(0)
NUM_CLASSES = len(POSTURES)
K = 10


def load_split(name):
    return np.load(f"data/{name}.npz", allow_pickle=True)


@torch.no_grad()
def extract_features(backbone, imgs_np):
    x = torch.from_numpy(imgs_np).unsqueeze(1).float()
    backbone.eval()
    feats = []
    for i in range(0, x.shape[0], 256):
        feats.append(backbone.pooled(x[i : i + 256]).numpy())
    return np.concatenate(feats, axis=0)


def sample_kshot_indices(labels, k, rng):
    idx = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(labels == c)[0]
        idx.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
    rng.shuffle(idx)
    return np.array(idx)


def main():
    pool = load_split("fewshot_pool")
    test = load_split("test_eval")

    bb = Backbone()
    bb.load_state_dict(torch.load("checkpoints/contrastive_pretrained_backbone.pt", map_location="cpu"))

    rng = np.random.default_rng(777)
    idx = sample_kshot_indices(pool["label"], K, rng)
    train_feat = extract_features(bb, pool["noisy"][idx])
    train_y = pool["label"][idx]

    test_feat = extract_features(bb, test["noisy"])
    test_y = test["label"]

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(train_feat, train_y)
    pred = clf.predict(test_feat)
    overall_acc = (pred == test_y).mean()
    print(f"overall test accuracy (contrastive linear probe, k={K}): {overall_acc:.4f}")

    results = dict(overall_accuracy=float(overall_acc), k=K, by_body_size={}, by_noise_level={}, by_posture={})

    for bs in BODY_SIZES:
        mask = test["body_size"] == bs
        acc = (pred[mask] == test_y[mask]).mean()
        results["by_body_size"][bs] = dict(accuracy=float(acc), n=int(mask.sum()))

    for nl in NOISE_LEVELS:
        mask = test["noise_level"] == nl
        acc = (pred[mask] == test_y[mask]).mean()
        results["by_noise_level"][nl] = dict(accuracy=float(acc), n=int(mask.sum()))

    for ci, posture in enumerate(POSTURES):
        mask = test_y == ci
        acc = (pred[mask] == test_y[mask]).mean()
        results["by_posture"][posture] = dict(accuracy=float(acc), n=int(mask.sum()))

    # worst joint subgroup: largest body + highest noise
    mask = (test["body_size"] == "large") & (test["noise_level"] == "high")
    results["worst_case_large_body_high_noise"] = dict(
        accuracy=float((pred[mask] == test_y[mask]).mean()), n=int(mask.sum())
    )
    mask = (test["body_size"] == "small") & (test["noise_level"] == "low")
    results["best_case_small_body_low_noise"] = dict(
        accuracy=float((pred[mask] == test_y[mask]).mean()), n=int(mask.sum())
    )

    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/failure_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # --- figures ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bs_accs = [results["by_body_size"][b]["accuracy"] for b in BODY_SIZES]
    axes[0].bar(BODY_SIZES, bs_accs, color=["#4c72b0", "#55a868", "#c44e52"])
    axes[0].set_ylim(0, 1); axes[0].set_title("Accuracy by body size"); axes[0].set_ylabel("accuracy")
    axes[0].axhline(overall_acc, linestyle="--", color="gray", label="overall")
    axes[0].legend()

    nl_accs = [results["by_noise_level"][n]["accuracy"] for n in NOISE_LEVELS]
    axes[1].bar(NOISE_LEVELS, nl_accs, color=["#4c72b0", "#55a868", "#c44e52"])
    axes[1].set_ylim(0, 1); axes[1].set_title("Accuracy by sensor noise level")
    axes[1].axhline(overall_acc, linestyle="--", color="gray", label="overall")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("reports/figures/04_failure_analysis_subgroups.png", dpi=130)

    cm = confusion_matrix(test_y, pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(POSTURES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(POSTURES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (contrastive probe, k={K})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig("reports/figures/05_confusion_matrix.png", dpi=130)
    print("saved failure analysis figures")


if __name__ == "__main__":
    main()
