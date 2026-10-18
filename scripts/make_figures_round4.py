"""Round-4 figures: survival vs reconstruction stopping (D70) and the data-information ceiling (D75)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round2_data as D  # noqa: E402
from make_figures_round2 import BLUE, GREY, GRID, INK2, ORANGE, PURPLE, save  # noqa: E402


def fig_curves():
    d = json.loads((D.REP / "diag_overfit_curves.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, (name, r) in zip(axes, d.items()):
        ep = [x["ep"] for x in r["curve"]]
        ax.plot(ep, [x["train_nll"] for x in r["curve"]], color=GREY, lw=1.2, label="train NLL")
        ax.plot(ep, [x["val_nll"] for x in r["curve"]], color=BLUE, lw=1.5, label="validation NLL")
        ax.axvline(r["epoch_min_val_nll"], color=BLUE, ls=":", lw=1)
        ax.axvline(r["epoch_min_val_objective"], color=ORANGE, ls="--", lw=1.2, label="objective-based stop")
        ax.set_title(name.split(" (")[0], fontsize=9, loc="left")
        ax.set_xlabel("Epoch"); ax.grid(color=GRID)
    axes[0].set_ylabel("AFT negative log-likelihood")
    axes[0].set_ylim(0.9, 2.0)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig_r4b_curves.png")


def fig_ceiling():
    rs = D.jsonl("diag_data_ceiling_seeds.jsonl")
    s = np.arange(len(rs))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(s, [r["oracle_C"] for r in rs], "o", color=GREY, label="oracle (true η)")
    ax.plot(s, [r["threshold"] for r in rs], "_", ms=18, mew=2, color=ORANGE, label="R2 threshold (oracle − 0.03)")
    ax.plot(s, [r["parametric_plugin_C"] for r in rs], "s", color=BLUE,
            label="near-optimal estimates + true risk function")
    ax.plot(s, [r["only_this_parameter_estimated_C"]["b"] for r in rs], "^", color=PURPLE,
            label="only b estimated (others true)")
    ax.set_xticks(s, [str(r["seed"]) for r in rs])
    ax.set_xlabel("Fresh seed (all 2000 individuals)"); ax.set_ylabel("Harrell C")
    ax.set_title("Data-information ceiling: even near-optimal estimates fall short of the R2 threshold",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22)); ax.grid(color=GRID)
    save(fig, "fig_r4b_ceiling.png")


if __name__ == "__main__":
    fig_curves()
    fig_ceiling()
