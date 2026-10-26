"""Phase 1: Denoising autoencoder.

Trains Backbone+Decoder to map noisy -> clean pressure maps (MSE loss).
Reports reconstruction error (MSE, PSNR) on a held-out validation set, saves
the trained backbone for reuse in Phase 2, and saves a qualitative figure.
"""
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

from src.models import Autoencoder

torch.manual_seed(0)
DEVICE = "cpu"


def load(split):
    d = np.load(f"data/{split}.npz", allow_pickle=True)
    return torch.from_numpy(d["noisy"]).unsqueeze(1).float(), torch.from_numpy(d["clean"]).unsqueeze(1).float()


def psnr(mse, max_val=1.0):
    mse = max(mse, 1e-10)
    return 10 * np.log10(max_val ** 2 / mse)


def main():
    noisy_tr, clean_tr = load("denoise_train")
    noisy_val, clean_val = load("denoise_val")

    model = Autoencoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs, batch_size = 40, 64
    n = noisy_tr.shape[0]
    history = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = noisy_tr[idx], clean_tr[idx]
            recon, _ = model(xb)
            loss = loss_fn(recon, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            recon_val, _ = model(noisy_val)
            val_mse = loss_fn(recon_val, clean_val).item()
            # baseline: identity (do nothing) MSE, for comparison
            identity_mse = loss_fn(noisy_val, clean_val).item()
        history.append(dict(epoch=epoch, train_mse=epoch_loss, val_mse=val_mse))
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:2d}  train_mse={epoch_loss:.5f}  val_mse={val_mse:.5f}  "
                  f"identity_mse={identity_mse:.5f}  psnr={psnr(val_mse):.2f}dB")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/denoising_autoencoder.pt")
    torch.save(model.backbone.state_dict(), "checkpoints/denoise_pretrained_backbone.pt")

    with torch.no_grad():
        recon_val, _ = model(noisy_val)
        final_val_mse = loss_fn(recon_val, clean_val).item()
        identity_mse = loss_fn(noisy_val, clean_val).item()

    results = dict(
        final_val_mse=final_val_mse,
        final_val_psnr_db=psnr(final_val_mse),
        identity_baseline_mse=identity_mse,
        identity_baseline_psnr_db=psnr(identity_mse),
        improvement_over_identity_pct=100 * (identity_mse - final_val_mse) / identity_mse,
        n_epochs=n_epochs,
        noise_model="gaussian(sigma in {0.02,0.05,0.09}) + dead-cell dropout(p in {0.01,0.03,0.06}) "
                     "+ salt spikes(p in {0.002,0.006,0.012}) + quantization({64,32,16} levels), "
                     "uniformly mixed over low/medium/high noise_level buckets",
    )
    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/denoise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("reports/results/denoise_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(json.dumps(results, indent=2))

    # qualitative figure: noisy / denoised / clean for a handful of val samples
    model.eval()
    with torch.no_grad():
        recon_val, _ = model(noisy_val[:6])
    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    for i in range(6):
        axes[0, i].imshow(noisy_val[i, 0], cmap="inferno", vmin=0, vmax=1)
        axes[1, i].imshow(recon_val[i, 0], cmap="inferno", vmin=0, vmax=1)
        axes[2, i].imshow(clean_val[i, 0], cmap="inferno", vmin=0, vmax=1)
        for r in range(3):
            axes[r, i].axis("off")
    axes[0, 0].set_title("noisy input", loc="left")
    axes[1, 0].set_title("denoised (model)", loc="left")
    axes[2, 0].set_title("clean target", loc="left")
    plt.tight_layout()
    plt.savefig("reports/figures/01_denoise_qualitative.png", dpi=130)

    # training curve
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_mse"] for h in history], label="train MSE")
    plt.plot(epochs, [h["val_mse"] for h in history], label="val MSE")
    plt.axhline(identity_mse, color="gray", linestyle="--", label="identity baseline (no denoising)")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Denoising autoencoder training")
    plt.tight_layout()
    plt.savefig("reports/figures/01_denoise_training_curve.png", dpi=130)
    print("saved figures to reports/figures/")


if __name__ == "__main__":
    main()
