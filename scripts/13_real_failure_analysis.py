"""Honest failure analysis on REAL data: breaks down the contrastive linear
probe's accuracy (k=10) by nuisance variant (flat / wedge / fetus /
bed-incline -- recorded by the dataset, withheld from the model) and by
individual held-out subject (a real, unseen body), plus a confusion
matrix."""
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
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from src.models import Backbone
from src.real_data import POSTURES, TEST_SUBJECTS, SUBJECT_INFO

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
    pool = load_split("real_fewshot_pool")
    test = load_split("real_test_eval")

    bb = Backbone()
    bb.load_state_dict(torch.load("checkpoints/real_contrastive_pretrained_backbone.pt", map_location="cpu"))

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
    overall_bacc = balanced_accuracy_score(test_y, pred)
    print(f"overall test accuracy (contrastive probe, k={K}): {overall_acc:.4f}  balanced: {overall_bacc:.4f}")

    results = dict(overall_accuracy=float(overall_acc), overall_balanced_accuracy=float(overall_bacc),
                    k=K, by_variant={}, by_subject={}, by_posture={})

    for variant in sorted(set(test["variant"].tolist())):
        mask = test["variant"] == variant
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == test_y[mask]).mean()
        results["by_variant"][variant] = dict(accuracy=float(acc), n=int(mask.sum()))

    for sid in TEST_SUBJECTS:
        mask = test["subject_id"] == sid
        acc = (pred[mask] == test_y[mask]).mean()
        age, height, weight = SUBJECT_INFO[sid]
        results["by_subject"][f"S{sid}"] = dict(
            accuracy=float(acc), n=int(mask.sum()), age=age, height_cm=height, weight_kg=weight,
        )

    for ci, posture in enumerate(POSTURES):
        mask = test_y == ci
        acc = (pred[mask] == test_y[mask]).mean()
        results["by_posture"][posture] = dict(accuracy=float(acc), n=int(mask.sum()))

    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/real_failure_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # --- figures ---
    variants = sorted(results["by_variant"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    v_accs = [results["by_variant"][v]["accuracy"] for v in variants]
    axes[0].bar(variants, v_accs, color="#4c72b0")
    axes[0].set_ylim(0, 1); axes[0].set_title("Accuracy by posture variant (real nuisance factor)")
    axes[0].axhline(overall_acc, linestyle="--", color="gray", label="overall")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    subj_labels = [f"S{sid}\n(h={SUBJECT_INFO[sid][1]}cm)" for sid in TEST_SUBJECTS]
    s_accs = [results["by_subject"][f"S{sid}"]["accuracy"] for sid in TEST_SUBJECTS]
    axes[1].bar(subj_labels, s_accs, color=["#4c72b0", "#55a868", "#c44e52"])
    axes[1].set_ylim(0, 1); axes[1].set_title("Accuracy by held-out subject (unseen body)")
    axes[1].axhline(overall_acc, linestyle="--", color="gray", label="overall")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("reports/figures/10_real_failure_analysis_subgroups.png", dpi=130)

    cm = confusion_matrix(test_y, pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(POSTURES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(POSTURES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (real data, k={K})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig("reports/figures/11_real_confusion_matrix.png", dpi=130)
    print("saved real-data failure analysis figures")


if __name__ == "__main__":
    main()
