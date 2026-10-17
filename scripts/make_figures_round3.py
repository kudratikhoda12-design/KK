"""Round-3 figures: revised gate (generalisation gap of the frequency-aware variants) and R1 diagnostic."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round2_data as D  # noqa: E402
from make_figures_round2 import BLUE, GREY, GRID, INK2, ORANGE, PURPLE, save  # noqa: E402,F401


def fig_overfit():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    pts = []
    for r in D.jsonl("learnability_search_fa.jsonl"):
        if r["kind"] != "fae_mlp":
            continue
        lab = "A3 (reference)" if r["config"]["front"] == "tspline" else "fa (search)"
        pts.append((lab, r["train_C"], r["val_C"], np.mean(r["val_probe"]), r["val_probe"][3]))
    for r in D.jsonl("diag_fa_overfitting.jsonl"):
        if r["kind"] == "fae_mlp":
            lab = "fa compact (diagnostic)" if "compact" in r["variant"] else "fa + weight decay (diagnostic)"
            pts.append((lab, r["train_C"], r["val_C"], np.mean(r["val_probe"]), r["val_probe"][3]))
    col = {"A3 (reference)": GREY, "fa (search)": BLUE, "fa compact (diagnostic)": ORANGE,
           "fa + weight decay (diagnostic)": PURPLE}
    for lab in col:
        P = [p for p in pts if p[0] == lab]
        if P:
            a1.scatter([p[1] for p in P], [p[2] for p in P], color=col[lab], label=lab, s=30, zorder=3)
            a2.scatter([p[4] for p in P], [p[2] for p in P], color=col[lab], s=30, zorder=3)
    oracle = D.jsonl("learnability_search_fa.jsonl")[0]["val_oracle_C"]
    for a in (a1, a2):
        a.axhline(oracle - 0.03, color=ORANGE, ls="--", lw=1.2)
        a.grid(color=GRID)
    a1.plot([0.82, 0.93], [0.82, 0.93], color=INK2, lw=0.8, ls=":")
    a1.text(0.925, 0.917, "train = val", fontsize=7, color=INK2, ha="right")
    a1.set_xlabel("Train C"); a1.set_ylabel("Validation C (seed 21)")
    a1.set_title("Generalisation gap (dashed: oracle − 0.03)", fontsize=9, loc="left")
    a1.legend(fontsize=7, frameon=False, loc="lower right")
    a2.set_xlabel("Validation ω probe R²")
    a2.set_title("ω became decodable; concordance fell", fontsize=9, loc="left")
    save(fig, "fig_r4_overfitting.png")


def fig_r1():
    rs = D.jsonl("diag_r1_seeds.jsonl")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.scatter([r["epochs"] for r in rs], [r["sigma"] for r in rs], color=BLUE, label="σ̂ at early stop (R1 as registered)")
    ax.scatter([r["epochs"] for r in rs], [r["sigma_profile_train"] for r in rs], color=ORANGE, marker="s",
               label="profile σ̂ with the trained η fixed")
    ax.axhspan(0.45, 0.55, color=GRID, zorder=0)
    ax.axhline(0.5, color=INK2, lw=0.8)
    ax.set_xlabel("Early-stopping epoch"); ax.set_ylabel("σ̂ (truth 0.5; band = R1 tolerance)")
    ax.set_title("R1 on 12 seeds: σ̂ depends on when training stops", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(color=GRID)
    save(fig, "fig_r1_sigma.png")


if __name__ == "__main__":
    fig_overfit()
    fig_r1()
