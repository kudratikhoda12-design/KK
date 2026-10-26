"""Phase 2a: contrastive representation pretraining (SimCLR-style, NT-Xent)
on the *unlabeled* noisy pressure-map pool.

Two augmented views of each sample are pulled together in embedding space
and pushed apart from all other samples in the batch. The pretrained
backbone (conv feature extractor only, projection head discarded) is saved
for the few-shot linear-probe evaluation in 04_fewshot_eval.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import Backbone, ProjectionHead
from src.augment import augment_view

torch.manual_seed(0)
DEVICE = "cpu"


def nt_xent_loss(z1, z2, temperature=0.3):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = z @ z.T / temperature  # (2B, 2B)
    sim.fill_diagonal_(-1e9)

    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    d = np.load("data/pretrain_unlabeled.npz", allow_pickle=True)
    x = torch.from_numpy(d["noisy"]).unsqueeze(1).float()
    n = x.shape[0]

    backbone = Backbone().to(DEVICE)
    proj = ProjectionHead().to(DEVICE)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(proj.parameters()), lr=1e-3)

    n_epochs, batch_size = 60, 128
    history = []
    for epoch in range(n_epochs):
        backbone.train(); proj.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x[idx]
            if xb.shape[0] < 2:
                continue
            v1, v2 = augment_view(xb), augment_view(xb)
            z1 = proj(backbone.pooled(v1))
            z2 = proj(backbone.pooled(v2))
            loss = nt_xent_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= n_batches
        history.append(dict(epoch=epoch, loss=epoch_loss))
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:2d}  nt_xent_loss={epoch_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(backbone.state_dict(), "checkpoints/contrastive_pretrained_backbone.pt")

    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot([h["epoch"] for h in history], [h["loss"] for h in history])
    plt.xlabel("epoch"); plt.ylabel("NT-Xent loss"); plt.title("Contrastive pretraining")
    plt.tight_layout()
    plt.savefig("reports/figures/02_pretrain_loss_curve.png", dpi=130)
    print("saved checkpoints/contrastive_pretrained_backbone.pt")


if __name__ == "__main__":
    main()
