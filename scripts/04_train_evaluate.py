"""Step 4 - train CoxPH, DeepSurv and CoxKAN; benchmark on the held-out TEST split.

    python scripts/04_train_evaluate.py

Metrics: Harrell's C-index (+ bootstrap 95% CI), Uno's IPCW C-index, time-dependent
Brier score and Integrated Brier Score. Torch models are re-trained with 3 seeds to show
the seed variance; the seed-42 models are saved for interpretation (step 5).
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

from coxkan_churn import config as C, data, metrics, models  # noqa: E402
from coxkan_churn.plotting import INK2, SERIES, plt, save  # noqa: E402

torch.set_num_threads(4)
MODEL_DIR = C.ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
d, pp = data.prepare()
tr, va, te = d["train"], d["val"], d["test"]
print("train/val/test:", len(tr["t"]), len(va["t"]), len(te["t"]), "| features:", tr["X"].shape[1])

SPECS = {
    "CoxPH": lambda s: models.CoxPH(C.COXPH_PENALIZER),
    "DeepSurv": lambda s: models.DeepSurv(**C.DEEPSURV, seed=s),
    "CoxKAN [d,1] (additive)": lambda s: models.CoxKAN(**C.COXKAN_ADDITIVE, seed=s),
    "CoxKAN [d,4,1]": lambda s: models.CoxKAN(**C.COXKAN_DEEP, seed=s),
}
SEEDS = [42, 7, 2024]
GRID = np.arange(3, 37, 1.0)                     # evaluation horizon for Brier / IBS (months)

rows, seed_rows, fitted, risks = [], [], {}, {}
for name, make in SPECS.items():
    for s in SEEDS if name != "CoxPH" else [42]:  # CoxPH is deterministic
        t0 = time.time()
        m = make(s).fit(tr, va)
        r_te = m.predict_risk(te["X"])
        seed_rows.append(dict(model=name, seed=s, test_c=metrics.harrell_c(te["t"], te["e"], r_te),
                              secs=time.time() - t0))
        print(seed_rows[-1], flush=True)
        if s == 42:
            fitted[name], risks[name] = m, r_te

for name, m in fitted.items():
    r_tr, r_te = m.predict_risk(tr["X"]), risks[name]
    times, H0 = metrics.breslow_baseline(tr["t"], tr["e"], r_tr)
    surv = metrics.survival_curves(times, H0, r_te, GRID)
    bs = metrics.brier_scores(tr["t"], tr["e"], te["t"], te["e"], surv, GRID)
    lo, hi = metrics.bootstrap_c(te["t"], te["e"], r_te)
    sc = pd.DataFrame(seed_rows).query("model == @name").test_c
    rows.append({
        "model": name,
        "C_index_harrell": metrics.harrell_c(te["t"], te["e"], r_te),
        "C_ci_low": lo, "C_ci_high": hi,
        "C_seed_mean": sc.mean(), "C_seed_sd": sc.std() if len(sc) > 1 else 0.0,
        "C_index_uno": metrics.uno_c(tr["t"], tr["e"], te["t"], te["e"], r_te),
        "Brier_12m": bs[GRID == 12][0], "Brier_24m": bs[GRID == 24][0],
        "IBS": metrics.integrated_brier(GRID, bs),
        "val_C": getattr(m, "best_val_c", metrics.harrell_c(va["t"], va["e"], m.predict_risk(va["X"]))),
        "train_C": metrics.harrell_c(tr["t"], tr["e"], r_tr),
    })
    m.brier_curve, m.baseline = bs, (times, H0)

# Kaplan-Meier "null model" reference for the Brier score (no covariates)
km = KaplanMeierFitter().fit(tr["t"], tr["e"])
surv_null = np.tile(km.survival_function_at_times(GRID).values, (len(te["t"]), 1))
bs_null = metrics.brier_scores(tr["t"], tr["e"], te["t"], te["e"], surv_null, GRID)
rows.append(dict(model="Kaplan-Meier (no covariates)", C_index_harrell=0.5, C_index_uno=0.5,
                 Brier_12m=bs_null[GRID == 12][0], Brier_24m=bs_null[GRID == 24][0],
                 IBS=metrics.integrated_brier(GRID, bs_null)))

res = pd.DataFrame(rows)
res.round(4).to_csv(C.RES_DIR / "benchmark.csv", index=False)
pd.DataFrame(seed_rows).round(4).to_csv(C.RES_DIR / "seed_runs.csv", index=False)
print(res.round(4).to_string())

# ------------------------------------------------------------------ persist models
with open(MODEL_DIR / "coxph.pkl", "wb") as fh:
    pickle.dump(fitted["CoxPH"].cph, fh)
for name, fn in [("DeepSurv", "deepsurv.pt"), ("CoxKAN [d,1] (additive)", "coxkan_additive.pt"),
                 ("CoxKAN [d,4,1]", "coxkan_deep.pt")]:
    torch.save(fitted[name].net.state_dict(), MODEL_DIR / fn)
with open(MODEL_DIR / "preprocessor.pkl", "wb") as fh:
    pickle.dump(pp, fh)
np.savez(C.RES_DIR / "test_risks.npz", t=te["t"], e=te["e"], **{k.split()[0] + ("_deep" if "4,1" in k else ""): v for k, v in risks.items()})

# ------------------------------------------------------------------ figures
names = [n for n in SPECS]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
r = res.set_index("model").loc[names]
ax[0].barh(range(len(names)), r.C_index_harrell, color=SERIES[0], height=0.55)
ax[0].errorbar(r.C_index_harrell, range(len(names)), xerr=[r.C_index_harrell - r.C_ci_low, r.C_ci_high - r.C_index_harrell],
               fmt="none", ecolor=INK2, elinewidth=1.4)
for i, v in enumerate(r.C_index_harrell):
    ax[0].text(r.C_ci_high.iloc[i] + 0.002, i, f"{v:.4f}", va="center", fontsize=8, color=INK2)
ax[0].set_yticks(range(len(names)), names); ax[0].set_xlim(0.5, 0.66); ax[0].invert_yaxis()
ax[0].set_title("Test C-index (Harrell) with bootstrap 95% CI"); ax[0].grid(axis="y")
for i, n in enumerate(names):
    ax[1].plot(GRID, fitted[n].brier_curve, color=SERIES[i], label=n, linewidth=1.8)
ax[1].plot(GRID, bs_null, color=INK2, linestyle="--", linewidth=1.4, label="Kaplan-Meier (no covariates)")
ax[1].set_title("Time-dependent Brier score (lower = better)"); ax[1].set_xlabel("months"); ax[1].legend()
save(fig, C.FIG_DIR / "09_benchmark.png")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for i, n in enumerate(names[1:]):
    h = pd.DataFrame(fitted[n].history)
    ax[i].plot(h.epoch, h.train_loss, color=SERIES[0], label="train loss")
    ax[i].plot(h.epoch, h.val_loss, color=SERIES[1], label="val loss")
    a2 = ax[i].twinx(); a2.plot(h.epoch, h.val_c, color=SERIES[2], linewidth=1.4, label="val C")
    a2.grid(False); a2.set_ylabel("val C-index", color=INK2)
    ax[i].set_title(n); ax[i].set_xlabel("epoch"); ax[i].set_ylabel("neg. partial log-lik")
    ax[i].legend(loc="upper center", fontsize=7)
fig.tight_layout()
save(fig, C.FIG_DIR / "10_training_curves.png")

# risk-tier validation on TEST: KM curves per predicted-risk quintile (CoxKAN additive)
best = "CoxKAN [d,1] (additive)"
q = pd.qcut(risks[best], 5, labels=["Q1 lowest risk", "Q2", "Q3", "Q4", "Q5 highest risk"])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
tier = []
for i, lab in enumerate(q.categories):
    msk = np.asarray(q == lab)
    kmq = KaplanMeierFitter().fit(te["t"][msk], te["e"][msk], label=lab)
    kmq.plot_survival_function(ax=ax[0], ci_show=False, color=SERIES[i])
    tier.append(dict(tier=lab, n=int(msk.sum()), observed_prepay_rate=float(te["e"][msk].mean()),
                     km_retained_24m=float(kmq.predict(24)), km_retained_36m=float(kmq.predict(36))))
ax[0].set_title("Test-set KM curves by CoxKAN risk quintile"); ax[0].set_xlabel("months on book"); ax[0].set_ylabel("P(retained)")
pd.DataFrame(tier).round(4).to_csv(C.RES_DIR / "risk_tiers.csv", index=False)
# calibration at 24 months: predicted vs KM-observed retention per decile
m = fitted[best]
S24 = metrics.survival_curves(*m.baseline, risks[best], np.array([24.0]))[:, 0]
dec = pd.qcut(S24, 10, labels=False, duplicates="drop")
cal = []
for k in np.unique(dec):
    msk = dec == k
    cal.append((S24[msk].mean(), KaplanMeierFitter().fit(te["t"][msk], te["e"][msk]).predict(24)))
cal = np.array(cal)
ax[1].plot([0.3, 1], [0.3, 1], color=INK2, linestyle="--", linewidth=1)
ax[1].plot(cal[:, 0], cal[:, 1], "o-", color=SERIES[0], markersize=6)
ax[1].set_xlabel("predicted P(retained at 24m)"); ax[1].set_ylabel("observed KM P(retained at 24m)")
ax[1].set_title("Calibration at 24 months (test deciles)")
save(fig, C.FIG_DIR / "11_risk_tiers_calibration.png")
pd.DataFrame(cal, columns=["predicted_S24", "observed_S24"]).round(4).to_csv(C.RES_DIR / "calibration_24m.csv", index=False)
json.dump({"grid": GRID.tolist(), **{n: fitted[n].brier_curve.tolist() for n in names}, "KM": bs_null.tolist()},
          open(C.RES_DIR / "brier_curves.json", "w"))
print("done")
