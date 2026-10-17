"""Round-2 figures (gate history, search record, confirmations, symbolic recovery, estimator study, PBC KAN-AFT).
All values come from saved records via scripts/round2_data.py."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round2_data as D  # noqa: E402

FIG = D.REP / "figures"
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0"
GREEN, PURPLE = "#2f9e5b", "#8b5cc4"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def fig_gate_history():
    g = D.gate_history()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(g))[::-1]
    for yi, r in zip(y, g):
        a1.plot([r["C"]], [yi], "o", color=BLUE, ms=6)
        a1.plot([r["threshold"]], [yi], "|", color=ORANGE, ms=14, mew=2)
    a1.axvline(g[0]["oracle"], color=GREY, lw=1, ls="--")
    a1.text(g[0]["oracle"] - 0.002, y[-1] - 0.35, "oracle C", color=GREY, ha="right", fontsize=8)
    a1.set_yticks(y, [r["label"] for r in g])
    a1.set_xlabel("Test C-index (gate seed 21)")
    a1.set_title("R2 concordance: estimate (●) vs required threshold (|)", fontsize=9, loc="left")
    a1.grid(axis="x", color=GRID)
    names = ["a", "b", "A", "ω"]
    w = 0.8 / len(g)
    for k, r in enumerate(g):
        a2.bar(np.arange(4) + (k - (len(g) - 1) / 2) * w, r["probe"], w, color=plt.cm.Blues(0.35 + 0.65 * k / (len(g) - 1)),
               label=r["label"])
    a2.axhline(0.8, color=ORANGE, lw=1.2, ls="--")
    a2.text(3.45, 0.82, "mean ≥ 0.8 required", color=ORANGE, ha="right", fontsize=8)
    a2.set_xticks(range(4), names)
    a2.set_ylim(-0.05, 1.0)
    a2.set_title("Linear-probe R² of the latent for each oscillation parameter", fontsize=9, loc="left")
    a2.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2)
    a2.grid(axis="y", color=GRID)
    save(fig, "fig_r2_gate_history.png")


def fig_search():
    rs = [r for r in D.search_records() if r["status"] == "ok"]
    steps = sorted({r["step"] for r in rs}, key=lambda s: int(s.split("-")[0][1:]))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, s in enumerate(steps):
        sub = [r for r in rs if r["step"] == s]
        ax.scatter([r["val_probe_mean"] for r in sub], [r["val_C"] - r["val_oracle_C"] for r in sub], s=28,
                   color=cols[k % 10], label=s, zorder=3)
    ax.axhline(-0.03, color=ORANGE, lw=1.2, ls="--")
    ax.axvline(0.8, color=ORANGE, lw=1.2, ls="--")
    ax.text(0.79, -0.0305, "pass region: above and right of both dashed lines", color=ORANGE, ha="right", va="top", fontsize=8)
    ax.set_xlabel("Validation mean probe R² (a, b, A, ω)")
    ax.set_ylabel("Validation C − oracle C")
    ax.set_xlim(0.38, 0.85)
    ax.set_title("Controlled search (D41 + D46): every fit, validation data only", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    ax.grid(color=GRID)
    save(fig, "fig_r2_search.png")


def fig_confirm():
    c = D.confirmations()
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    x, lab = 0, []
    for k, (name, rs) in enumerate(c.items()):
        for r in rs:
            ax.bar(x, r["test_C"] - r["threshold"], color=[BLUE, PURPLE][k])
            ax.text(x, (r["test_C"] - r["threshold"]) / 2, f"{r['probe_mean']:.2f}", ha="center", va="center",
                    fontsize=7, color="white")
            lab.append(str(r["data_seed"]))
            x += 1
    ax.axhline(0, color=ORANGE, lw=1.2)
    ax.set_xticks(range(len(lab)), lab)
    ax.set_xlabel("Fresh replication data seed (blue: D41 confirmation; purple: D46 confirmation)")
    ax.set_ylabel("Test C − threshold\n(pass needs ≥ 0)")
    ax.set_title("Independent confirmations: every replication below the C threshold (white: mean probe R²)",
                 fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID)
    save(fig, "fig_r2_confirmations.png")


def fig_symbolic():
    rs = D.symbolic_synth()
    losses = ["parametric", "bj", "ipcw", "transform"]
    comps = ["1.5 tanh(2 z1)", None, None]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
    names = sorted({s["true_component"] for r in rs for s in r["symbolic"] if s["truth_recovery_r2"] is not None})
    w = 0.2
    for k, loss in enumerate(losses):
        vals = []
        for n in names:
            v = [s["truth_recovery_r2"] for r in rs if r["loss"] == loss for s in r["symbolic"]
                 if s["true_component"] == n]
            vals.append(np.mean(v) if v else np.nan)
        a1.bar(np.arange(len(names)) + (k - 1.5) * w, vals, w, color=[BLUE, ORANGE, GREEN, PURPLE][k], label=loss)
    a1.set_ylim(0.9, 1.002)
    a1.set_xticks(range(len(names)), names, fontsize=7.5)
    a1.set_ylabel("Truth-recovery R² (mean over 3 seeds)")
    a1.set_title("Learned component vs true function", fontsize=9, loc="left")
    a1.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    a1.grid(axis="y", color=GRID)
    del comps
    for k, loss in enumerate(losses):
        sub = [r for r in rs if r["loss"] == loss]
        for j, key in enumerate(["unpruned_test", "pruned_test", "symbolic_test"]):
            v = [r[key]["harrell_c"] for r in sub]
            a2.errorbar(j + (k - 1.5) * 0.08, np.mean(v), yerr=np.std(v), fmt="o", color=[BLUE, ORANGE, GREEN, PURPLE][k],
                        label=loss if j == 0 else None, capsize=0)
    a2.set_xticks(range(3), ["spline (unpruned)", "pruned θ=1e-2", "formulas substituted"])
    a2.set_ylabel("Test Harrell C (mean ± sd, 3 seeds)")
    a2.set_title("Effect of pruning and symbolic substitution", fontsize=9, loc="left")
    a2.grid(axis="y", color=GRID)
    save(fig, "fig_kanaft_symbolic.png")


def fig_censoring():
    cv = D.censoring_variants()
    keys = ["param_lognormal", "param_weibull", "bj", "ipcw", "transform", "ipcw_oracle_condG"]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for ax, sc in zip(axes, ["CA2", "CA1", "MIS"]):
        if sc not in cv:
            continue
        b = [cv[sc]["est"][k]["bias"][0] for k in keys]
        r = [cv[sc]["est"][k]["rmse"][0] for k in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, b, color=[GREY if k == "ipcw_oracle_condG" else BLUE for k in keys])
        ax.plot(r, y, "|", color=ORANGE, ms=10, mew=2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{sc} (censored {cv[sc]['censored']:.0%})", fontsize=9, loc="left")
        ax.set_yticks(y, keys, fontsize=7.5)
        ax.grid(axis="x", color=GRID)
        ax.set_xlabel("bias of β₁ (bar), RMSE (|)")
    save(fig, "fig_censoring_variants.png")


def fig_pbc():
    rs = D.kanaft_pbc()
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)
    for ax, ep in zip(axes, ["E1", "E2"]):
        sub = [r for r in rs if r["endpoint"] == ep]
        for r in sub:
            v = [r["linear"]["test"]["ibs"], r["kan"]["test"]["ibs"], r["kan_pruned_test"]["ibs"],
                 r["kan_symbolic_test"]["ibs"]]
            ax.plot(range(4), v, "-o", color=BLUE, alpha=0.6, ms=4)
        ax.set_xticks(range(4), ["linear AFT", "KAN-AFT", "KAN pruned", "KAN formulas"], fontsize=8)
        ax.set_title(f"PBC {ep}: test IBS per split (lower is better)", fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("IBS (marginal IPCW)")
    save(fig, "fig_kanaft_pbc.png")


if __name__ == "__main__":
    fig_gate_history()
    fig_search()
    fig_confirm()
    fig_symbolic()
    fig_censoring()
    fig_pbc()
