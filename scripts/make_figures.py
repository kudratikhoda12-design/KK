"""Figures for the final report (static PNG for print/PDF). All values come from saved tables/logs.

Colour: one blue for estimates, neutral grey for reference rows (km_null, oracle_true); paired measures use
categorical slots 1-2 (blue, orange) validated with the dataviz validator (light mode, all hard gates pass).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0"
REF = {"km_null", "oracle_true"}
LABEL = {"fpca_aft": "FPCA + AFT", "fpca_cox": "FPCA + Cox", "sum_aft": "Summary + AFT", "sum_cox": "Summary + Cox",
         "sum_rsf": "Summary + RSF", "km_null": "KM null (reference)", "oracle_true": "True model (reference)"}
ORDER = ["fpca_aft", "fpca_cox", "sum_aft", "sum_cox", "sum_rsf", "km_null", "oracle_true"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

S = pd.read_csv(ROOT / "reports/tables/summary_by_model.csv")


def dotplot(ax, sub, metric, title, xlabel):
    models = [m for m in ORDER if m in set(sub.model)]
    for k, m in enumerate(models[::-1]):
        r = sub[sub.model == m].iloc[0]
        mu, sd = r[f"{metric}__mean"], r[f"{metric}__std"]
        col = GREY if m in REF else BLUE
        ax.errorbar(mu, k, xerr=0 if np.isnan(sd) else sd, fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.5,
                    capsize=0)
        ax.text(mu, k + 0.28, f"{mu:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABEL[m] for m in models[::-1]])
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_ylim(-0.6, len(models) - 0.2)


def fig_synthetic():
    syn = S[S.dataset == "synthetic"]
    settings = ["P0", "P-c0", "P-c60", "P-sparse", "P-V2", "P-lognormal"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for ax, st in zip(axes.ravel(), settings):
        dotplot(ax, syn[syn.setting == st], "curve_error", st, "Survival-curve error (lower is better)")
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.suptitle("Synthetic data (SYNTHETIC): survival-curve error vs the true curve, mean ± SD over 5 replicates",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.savefig(FIG / "fig_synthetic_curve_error.png"); plt.close(fig)

    p0 = syn[syn.setting == "P0"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    dotplot(axes[0], p0, "ibs", "IBS (lower is better)", "IBS")
    dotplot(axes[1], p0, "uno_c", "Uno's C (higher is better)", "C")
    dotplot(axes[2], p0, "dcal_p", "D-calibration p-value", "p (small = miscalibrated)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Synthetic setting P0 (SYNTHETIC): secondary metrics, mean ± SD over 5 replicates", x=0.01,
                 ha="left", fontsize=10, color=INK)
    fig.savefig(FIG / "fig_synthetic_P0_secondary.png"); plt.close(fig)


def fig_reconstruction():
    syn = S[(S.dataset == "synthetic") & (S.model == "fpca_aft")]
    settings = ["P0", "P-c0", "P-c60", "P-sparse", "P-V2", "P-lognormal"]
    syn = syn.set_index("setting").loc[[s for s in settings if s in set(syn.setting)]]
    fig, ax = plt.subplots(figsize=(7.5, 3))
    x = np.arange(len(syn))
    for off, tag, col, lab in ((-0.12, "smoother", ORANGE, "Smoother only (all coefficients)"),
                               (0.12, "fpca", BLUE, "FPCA reconstruction (selected r)")):
        mu, sd = syn[f"rec_rmse_ind_{tag}__mean"], syn[f"rec_rmse_ind_{tag}__std"]
        ax.errorbar(x + off, mu, yerr=sd, fmt="o", color=col, ms=5, elinewidth=1.5, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(syn.index)
    ax.set_ylabel("Held-out RMSE (standardised)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Reconstruction at held-out observation points (individual-weighted), mean ± SD over 5 replicates",
                 fontsize=9, loc="left", color=INK)
    fig.savefig(FIG / "fig_reconstruction.png"); plt.close(fig)


def fig_example_curves():
    import torch
    from kanfae.data.splits import split_groups
    from kanfae.data.synthetic import SynthConfig, generate
    from kanfae.functional.smoothing import evaluate
    from kanfae.models.pipelines import make_model
    torch.set_default_dtype(torch.float64)
    obs, ora = generate(SynthConfig(n=1000, seed=1000))
    tr, va, te = split_groups(obs.ids, 1000, strata=obs.delta)
    m = make_model("fpca_aft", n_components=8).fit(obs.subset(tr), obs.subset(va))
    ote = obs.subset(te)
    P = m.prepare(ote, holdout=True)
    i = int(np.argmax(P.eval_mask.any(axis=(1, 2)) & (ote.n_obs > 25)))
    fine = np.linspace(0, 1, 400)
    B = m.pre.basis(fine)
    c_fp = m.fpca.inverse_transform(m.fpca.transform(P.c))[i]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharex=True)
    for d, ax in enumerate(axes):
        mu, sd = m.pre.mean_[d], m.pre.sd_[d]
        t = ote.times[i, :ote.n_obs[i]]
        v = ote.values[i, :ote.n_obs[i], d]
        fit = P.fit_mask[i, :ote.n_obs[i], d]
        ax.plot(fine, (B @ P.c[i, d]) * sd + mu, color=ORANGE, lw=2, label="Smoother")
        ax.plot(fine, (B @ c_fp[d]) * sd + mu, color=BLUE, lw=2, label="FPCA reconstruction (r = 8)")
        ax.scatter(t[fit], v[fit], s=14, color=INK, zorder=3, label="Observed (used)")
        ax.scatter(t[~fit], v[~fit], s=22, facecolors="none", edgecolors=INK, zorder=3, label="Held out")
        ax.set_title(f"Channel x{d + 1}", fontsize=9, loc="left")
        ax.grid(color=GRID, lw=0.8); ax.set_xlabel("Window time (rescaled)")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Example test individual (SYNTHETIC, P0 replicate 0): observed points, smoother and FPCA reconstruction",
                 x=0.01, y=1.04, ha="left", fontsize=10)
    fig.savefig(FIG / "fig_example_reconstruction.png"); plt.close(fig)


def fig_calibration():
    rows = [("synthetic", "P0", "Synthetic P0"), ("pbc", "pbc_E1", "PBC E1 (death; transplant censored)"),
            ("pbc", "pbc_E2", "PBC E2 (death or transplant)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (ds, st, title) in zip(axes, rows):
        sub = S[(S.dataset == ds) & (S.setting == st)]
        for m in ORDER:
            r = sub[sub.model == m]
            if r.empty:
                continue
            r = r.iloc[0]
            px = [r[f"cal_km_q{q}__mean"] for q in (25, 50, 75)]
            py = [r[f"cal_pred_q{q}__mean"] for q in (25, 50, 75)]
            ax.plot(px, py, "o-", ms=4, lw=1, color=GREY if m in REF else BLUE, alpha=0.9)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Kaplan–Meier S(t) on test set"); ax.set_ylabel("Mean predicted S(t)")
        ax.set_title(title, fontsize=9, loc="left"); ax.grid(color=GRID, lw=0.8)
    axes[0].text(0.03, 0.92, "blue: comparators; grey: references\npoints: 25th/50th/75th percentile horizons",
                 transform=axes[0].transAxes, fontsize=7, color=INK2, va="top")
    fig.suptitle("Calibration-in-the-large (means over replicates/splits)", x=0.01, ha="left", fontsize=10)
    fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)


def fig_gate():
    vals = []
    for a in (1, 2, 3):
        txt = (ROOT / f"reports/logs/stageD_gate_attempt{a}.log").read_text()
        c = float(re.search(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+)", txt).group(1))
        orc = float(re.search(r"R2: C = ([0-9.]+), oracle C = ([0-9.]+)", txt).group(2))
        pr = [float(x) for x in re.search(r"probe R2 \(a,b,A,omega\) = \[([^\]]+)\]", txt).group(1).split()]
        vals.append((c, orc, pr))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    labels = ["Attempt 1\n(as specified)", "Attempt 2\n(+ training budget)", "Attempt 3\n(+ capacity)"]
    cs = [v[0] for v in vals]
    axes[0].plot(range(3), cs, "o", color=BLUE, ms=7)
    for k, cval in enumerate(cs):
        axes[0].text(k + 0.12, cval, f"{cval:.4f}", ha="left", va="center", fontsize=8)
    axes[0].axhline(vals[0][1], color=GREY, lw=1, ls=":")
    axes[0].text(2.45, vals[0][1] + 0.004, f"oracle C {vals[0][1]:.4f}", va="bottom", fontsize=7, color=INK2)
    axes[0].axhline(vals[0][1] - 0.03, color=ORANGE, lw=1.2)
    axes[0].text(2.45, vals[0][1] - 0.03 + 0.004, f"threshold {vals[0][1] - 0.03:.4f}", va="bottom", fontsize=7,
                 color=INK2)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(labels, fontsize=8); axes[0].set_xlim(-0.4, 3.3)
    axes[0].set_ylim(min(cs) - 0.02, vals[0][1] + 0.02)
    axes[0].set_ylabel("Test Harrell C"); axes[0].set_title("R2: functional MLP autoencoder (gate data, seed 21)",
                                                            fontsize=9, loc="left")
    axes[0].grid(axis="y", color=GRID, lw=0.8)
    names = ["a (level)", "b (slope)", "A (amplitude)", "ω (frequency)"]
    pr = vals[2][2]
    axes[1].barh(range(4), pr, color=BLUE, height=0.5)
    for k, v in enumerate(pr):
        axes[1].text(v + 0.01, k, f"{v:.2f}", va="center", fontsize=8)
    axes[1].set_yticks(range(4)); axes[1].set_yticklabels(names); axes[1].invert_yaxis()
    axes[1].axvline(0.8, color=ORANGE, lw=1.2); axes[1].text(0.81, 3.4, "0.8 (mean required)", fontsize=7, color=INK2)
    axes[1].set_xlim(0, 1); axes[1].set_xlabel("Linear-probe test R² from latent z (attempt 3)")
    axes[1].set_title("What the latent representation encodes", fontsize=9, loc="left")
    fig.savefig(FIG / "fig_gate.png"); plt.close(fig)


def fig_cmapss():
    cm = S[S.dataset == "cmapss_FD001"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (an, metric, title) in zip(axes, [("cmapss_A", "rul_official_rmse", "Option A: official test (engines ≥ s0)"),
                                             ("cmapss_B", "rul_official_rmse", "Option B: official test (last cycle)"),
                                             ("cmapss_B", "rul_test_rmse", "Option B: internal test (all landmarks)")]):
        dotplot(ax, cm[cm.analysis == an], metric, title, "RUL RMSE, cycles (lower is better)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("C-MAPSS FD001 (REAL simulated turbofan data; RUL framing, no censoring), mean ± SD over 5 engine-level splits",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(FIG / "fig_cmapss_rul.png"); plt.close(fig)
    sub = cm[cm.analysis == "cmapss_B"]
    bands = [("band_30_100", "landmark 30–99"), ("band_100_200", "100–199"), ("band_200_999", "≥ 200")]
    fig, ax = plt.subplots(figsize=(7.5, 3))
    models = [m for m in ORDER if m in set(sub.model)]
    x = np.arange(len(bands))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        r = sub[sub.model == m].iloc[0]
        mu = [r.get(f"{b}_rmse__mean", np.nan) for b, _ in bands]
        ax.plot(x - 0.4 + w * (k + 0.5), mu, "o", ms=4, color=GREY if m in REF else BLUE)
        ax.text(x[-1] - 0.4 + w * (k + 0.5), mu[-1] + 1.5, LABEL[m].split(" (")[0], rotation=90, fontsize=6,
                ha="center", va="bottom", color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([b for _, b in bands])
    ax.set_ylabel("RUL RMSE (cycles)"); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("C-MAPSS Option B: RUL RMSE by landmark band (internal test), mean over 5 splits", fontsize=9, loc="left")
    fig.savefig(FIG / "fig_cmapss_bands.png"); plt.close(fig)


def fig_pbc():
    pb = S[S.dataset == "pbc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, an, title in zip(axes, ["pbc_E1", "pbc_E2"], ["E1: death (transplant censored)", "E2: death or transplant"]):
        sub = pb[pb.analysis == an]
        models = [m for m in ORDER if m in set(sub.model)]
        for k, m in enumerate(models[::-1]):
            r = sub[sub.model == m].iloc[0]
            ax.errorbar(r["ibs__mean"], k + 0.12, xerr=r["ibs__std"], fmt="o", color=BLUE, ms=5, label="Marginal-KM IPCW (primary)" if k == 0 else None)
            ax.errorbar(r["ibs_conditional_ipcw__mean"], k - 0.12, xerr=r["ibs_conditional_ipcw__std"], fmt="s",
                        color=ORANGE, ms=4, label="Conditional IPCW (sensitivity)" if k == 0 else None)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL[m] for m in models[::-1]])
        ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("IBS (lower is better)")
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("PBC sequential data (REAL, right-censored), landmark 730 days: IBS mean ± SD over 5 splits",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(FIG / "fig_pbc_ibs.png"); plt.close(fig)
    from kanfae.data.pbc import landmark_dataset, read_pbcseq
    obs = landmark_dataset(read_pbcseq(ROOT / "data/raw/pbc/data/pbc.rda"), "E2")
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    vals, cnt = np.unique(obs.n_obs, return_counts=True)
    ax.bar(vals, cnt, color=BLUE, width=0.7)
    for v, c in zip(vals, cnt):
        ax.text(v, c + 1, str(c), ha="center", fontsize=7)
    ax.set_xlabel("Visits with day ≤ 730"); ax.set_ylabel("Patients"); ax.set_xticks(vals)
    ax.set_title(f"PBC landmark cohort (n = {len(obs)}): visits available before the landmark", fontsize=9, loc="left")
    fig.savefig(FIG / "fig_pbc_visits.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_synthetic, fig_reconstruction, fig_example_curves, fig_calibration, fig_gate, fig_cmapss, fig_pbc):
        try:
            f(); print("ok", f.__name__)
        except Exception as exc:
            print("FAILED", f.__name__, type(exc).__name__, exc)
