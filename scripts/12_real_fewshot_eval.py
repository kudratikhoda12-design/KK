"""Phase 2b on REAL data: few-shot posture classification (supine/left/right),
evaluated on 3 subjects held out entirely from pretraining/denoising/label
selection -- i.e. this measures generalization to an unseen person, not
just an unseen frame. Same six methods as the synthetic study, plus
balanced accuracy (macro-averaged recall) since real posture classes are
imbalanced (supine is over-represented: 9 of 17 recorded positions)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.models import Backbone, Autoencoder, CNNFromScratch
from src.real_data import POSTURES

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"
NUM_CLASSES = len(POSTURES)
K_VALUES = [5, 10, 20]
N_TRIALS = 8


def load_split(name):
    return np.load(f"data/{name}.npz", allow_pickle=True)


def sample_kshot_indices(labels, k, rng):
    idx = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(labels == c)[0]
        chosen = rng.choice(cls_idx, size=k, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return np.array(idx)


@torch.no_grad()
def extract_features(backbone, imgs_np):
    x = torch.from_numpy(imgs_np).unsqueeze(1).float()
    backbone.eval()
    feats = []
    for i in range(0, x.shape[0], 256):
        feats.append(backbone.pooled(x[i : i + 256]).numpy())
    return np.concatenate(feats, axis=0)


def load_backbone(weights_path=None):
    bb = Backbone()
    if weights_path is not None:
        bb.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return bb


@torch.no_grad()
def denoise_images(ae, imgs_np):
    x = torch.from_numpy(imgs_np).unsqueeze(1).float()
    ae.eval()
    outs = []
    for i in range(0, x.shape[0], 256):
        recon, _ = ae(x[i : i + 256])
        outs.append(recon.squeeze(1).numpy())
    return np.concatenate(outs, axis=0)


def train_cnn_scratch(train_x, train_y, epochs=80, lr=1e-3, wd=1e-3):
    model = CNNFromScratch(num_classes=NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    xb = torch.from_numpy(train_x).unsqueeze(1).float()
    yb = torch.from_numpy(train_y).long()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def eval_cnn_scratch(model, test_x, test_y):
    model.eval()
    xb = torch.from_numpy(test_x).unsqueeze(1).float()
    logits = model(xb)
    pred = logits.argmax(1).numpy()
    return (pred == test_y).mean(), balanced_accuracy_score(test_y, pred)


def fit_logreg_and_score(train_feat, train_y, test_feat, test_y):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(train_feat, train_y)
    pred = clf.predict(test_feat)
    return (pred == test_y).mean(), balanced_accuracy_score(test_y, pred)


def main():
    pool = load_split("real_fewshot_pool")
    test = load_split("real_test_eval")

    pool_labels = pool["label"]
    test_labels = test["label"]
    pool_noisy = pool["noisy"]
    test_noisy = test["noisy"]

    bb_random = load_backbone(None)
    bb_denoise = load_backbone("checkpoints/real_denoise_pretrained_backbone.pt")
    bb_contrastive = load_backbone("checkpoints/real_contrastive_pretrained_backbone.pt")
    ae = Autoencoder()
    ae.load_state_dict(torch.load("checkpoints/real_denoising_autoencoder.pt", map_location="cpu"))

    test_feat_random = extract_features(bb_random, test_noisy)
    test_feat_denoise = extract_features(bb_denoise, test_noisy)
    test_feat_contrastive = extract_features(bb_contrastive, test_noisy)
    test_denoised_imgs = denoise_images(ae, test_noisy)

    rng_master = np.random.default_rng(123)
    rows = []

    for k in K_VALUES:
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(rng_master.integers(0, 1_000_000))
            idx = sample_kshot_indices(pool_labels, k, rng)
            train_y = pool_labels[idx]
            train_noisy = pool_noisy[idx]

            acc, bacc = fit_logreg_and_score(
                train_noisy.reshape(len(idx), -1), train_y,
                test_noisy.reshape(len(test_noisy), -1), test_labels,
            )
            rows.append(dict(method="logreg_raw", k=k, trial=trial, accuracy=acc, balanced_accuracy=bacc))

            train_denoised = denoise_images(ae, train_noisy)
            acc, bacc = fit_logreg_and_score(
                train_denoised.reshape(len(idx), -1), train_y,
                test_denoised_imgs.reshape(len(test_denoised_imgs), -1), test_labels,
            )
            rows.append(dict(method="logreg_denoised", k=k, trial=trial, accuracy=acc, balanced_accuracy=bacc))

            cnn = train_cnn_scratch(train_noisy, train_y)
            acc, bacc = eval_cnn_scratch(cnn, test_noisy, test_labels)
            rows.append(dict(method="cnn_scratch", k=k, trial=trial, accuracy=acc, balanced_accuracy=bacc))

            train_feat = extract_features(bb_random, train_noisy)
            acc, bacc = fit_logreg_and_score(train_feat, train_y, test_feat_random, test_labels)
            rows.append(dict(method="linear_probe_random", k=k, trial=trial, accuracy=acc, balanced_accuracy=bacc))

            train_feat = extract_features(bb_denoise, train_noisy)
            acc, bacc = fit_logreg_and_score(train_feat, train_y, test_feat_denoise, test_labels)
            rows.append(dict(method="linear_probe_denoise", k=k, trial=trial, accuracy=acc, balanced_accuracy=bacc))

            train_feat = extract_features(bb_contrastive, train_noisy)
            acc, bacc = fit_logreg_and_score(train_feat, train_y, test_feat_contrastive, test_labels)
            rows.append(dict(method="linear_probe_contrastive", k=k, trial=trial, accuracy=acc, balanced_accuracy=bacc))

        print(f"k={k} done")

    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/real_fewshot_raw_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    methods = sorted(set(r["method"] for r in rows))
    summary = {}
    for m in methods:
        summary[m] = {}
        for k in K_VALUES:
            accs = [r["accuracy"] for r in rows if r["method"] == m and r["k"] == k]
            baccs = [r["balanced_accuracy"] for r in rows if r["method"] == m and r["k"] == k]
            summary[m][k] = dict(
                mean=float(np.mean(accs)), std=float(np.std(accs)),
                balanced_mean=float(np.mean(baccs)), balanced_std=float(np.std(baccs)), n=len(accs),
            )
    with open("reports/results/real_fewshot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plt.figure(figsize=(7, 5))
    labels_pretty = {
        "logreg_raw": "LogReg on raw pixels",
        "logreg_denoised": "LogReg on denoised pixels",
        "cnn_scratch": "CNN from scratch",
        "linear_probe_random": "Linear probe, random-init encoder",
        "linear_probe_denoise": "Linear probe, denoising-pretrained encoder",
        "linear_probe_contrastive": "Linear probe, contrastive-pretrained encoder (ours)",
    }
    for m in methods:
        means = [summary[m][k]["balanced_mean"] for k in K_VALUES]
        stds = [summary[m][k]["balanced_std"] for k in K_VALUES]
        plt.errorbar(K_VALUES, means, yerr=stds, marker="o", capsize=3, label=labels_pretty.get(m, m))
    plt.axhline(1 / NUM_CLASSES, color="gray", linestyle=":", label="chance (33%)")
    plt.xlabel("labels per class (k)")
    plt.ylabel("balanced test accuracy (unseen subjects)")
    plt.title("Few-shot posture classification on real PmatData")
    plt.xticks(K_VALUES)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/figures/09_real_fewshot_accuracy.png", dpi=130)
    print("saved reports/figures/09_real_fewshot_accuracy.png")


if __name__ == "__main__":
    main()
