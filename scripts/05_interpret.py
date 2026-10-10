"""Step 5 - interpretation: hazard ratios, PH diagnostics, CoxKAN shape functions,
symbolic formulas, permutation importance and per-customer explanations.

    python scripts/05_interpret.py      (run after 04_train_evaluate.py)
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lifelines.statistics import proportional_hazard_test  # noqa: E402

from coxkan_churn import config as C, data, metrics, models, symbolic  # noqa: E402
from coxkan_churn.plotting import INK2, SERIES, plt, save  # noqa: E402

torch.set_num_threads(4)
MD, FIG, RES = C.ROOT / "models", C.FIG_DIR, C.RES_DIR
d, pp = data.prepare()
tr, te = d["train"], d["test"]
names = list(tr["X"].columns)
Xtr = torch.tensor(tr["X"].values, dtype=torch.float32)

# ------------------------------------------------------------------ load models
cph = pickle.load(open(MD / "coxph.pkl", "rb"))
coxph = models.CoxPH(); coxph.cph = cph
kan = models.CoxKAN(**C.COXKAN_ADDITIVE); kan.net = kan.build(len(names)); kan.net.load_state_dict(torch.load(MD / "coxkan_additive.pt"))
kan_deep = models.CoxKAN(**C.COXKAN_DEEP); kan_deep.net = kan_deep.build(len(names)); kan_deep.net.load_state_dict(torch.load(MD / "coxkan_deep.pt"))
ds = models.DeepSurv(**C.DEEPSURV); ds.net = ds.build(len(names)); ds.net.load_state_dict(torch.load(MD / "deepsurv.pt"))


def pretty(f):
    return f.replace("_", " ")


# ------------------------------------------------------------------ 1. CoxPH hazard ratios
hr = coxph.hazard_ratios()
hr.index.name = "feature"
hr["abs_log_HR"] = hr["coef"].abs()
hr = hr.sort_values("abs_log_HR", ascending=False)
hr.round(4).to_csv(RES / "coxph_hazard_ratios.csv")
top5_cox = hr[hr.p < 0.05].head(5)
top5_cox.round(4).to_csv(RES / "top5_coxph.csv")

fig, ax = plt.subplots(figsize=(7.5, 8.5))
h = hr.sort_values("coef")
col = [SERIES[1] if c > 0 else SERIES[0] for c in h.coef]
ax.errorbar(h.HR, range(len(h)), xerr=[h.HR - h.HR_lower95, h.HR_upper95 - h.HR], fmt="none", ecolor=INK2, elinewidth=1.2)
ax.scatter(h.HR, range(len(h)), c=col, s=28, zorder=3)
ax.axvline(1, color=INK2, linewidth=1)
ax.set_yticks(range(len(h)), [pretty(i) for i in h.index], fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("hazard ratio (log scale): numeric = per +1 SD, binary = vs. reference")
ax.set_title("CoxPH multivariable hazard ratios for prepayment (churn)\norange = raises churn hazard, blue = lowers it")
save(fig, FIG / "12_coxph_forest.png")

# PH assumption (Schoenfeld-residual based test)
df_ph = tr["X"].copy(); df_ph["T"], df_ph["E"] = tr["t"], tr["e"]
ph = proportional_hazard_test(cph, df_ph, time_transform="rank").summary.sort_values("test_statistic", ascending=False)
ph.round(4).to_csv(RES / "ph_test.csv")

# ------------------------------------------------------------------ 2. CoxKAN shape functions
imp = pd.Series(kan.feature_importance(tr["X"].values), index=names).sort_values(ascending=False)
imp.round(5).to_csv(RES / "coxkan_importance.csv", header=["std_of_contribution"])

layer = kan.net.layers[0]


@torch.no_grad()
def phi_of(j, z):
    Z = torch.zeros(len(z), len(names)); Z[:, j] = torch.as_tensor(z, dtype=torch.float32)
    return layer.edge_activations(Z)[:, 0, j].numpy()


# "effective hazard ratio": compare a P90 vs a P10 customer on each feature (1 vs 0 for binary)
eff = []
for j, f in enumerate(names):
    x = tr["X"][f].values
    binary = set(np.unique(x)) <= {0.0, 1.0}
    lo, hi = (0.0, 1.0) if binary else np.percentile(x, [10, 90])
    kan_lhr = float(phi_of(j, np.array([hi]))[0] - phi_of(j, np.array([lo]))[0])
    cox_lhr = float(cph.params_[f] * (hi - lo))
    raw = (lambda v: v) if (binary or f not in C.NUMERIC) else (lambda v: float(data.z_to_raw(pp, f, v)))
    eff.append(dict(feature=f, contrast="1 vs 0" if binary else f"P90 ({raw(hi):.4g}) vs P10 ({raw(lo):.4g})",
                    HR_CoxKAN=np.exp(kan_lhr), HR_CoxPH=np.exp(cox_lhr), kan_importance=imp[f]))
eff = pd.DataFrame(eff)
eff["abs_log_HR_CoxKAN"] = np.log(eff.HR_CoxKAN).abs()
eff["abs_log_HR_CoxPH"] = np.log(eff.HR_CoxPH).abs()
eff = eff.sort_values("kan_importance", ascending=False)
eff.round(4).to_csv(RES / "effective_hazard_ratios.csv", index=False)

top = list(imp.index[:12])
fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
sym_rows = []
for a, f in zip(axes.ravel(), top):
    j = names.index(f)
    x = tr["X"][f].values
    binary = set(np.unique(x)) <= {0.0, 1.0}
    if binary:
        v = phi_of(j, np.array([0.0, 1.0]))
        a.bar([0, 1], v - v[0], color=[SERIES[0], SERIES[1]], width=0.5)
        a.set_xticks([0, 1], ["0", "1"])
        a.axhline(0, color=INK2, linewidth=0.8)
    else:
        lo, hi = np.percentile(x, [1, 99])
        zs = np.linspace(lo, hi, 200)
        ys = phi_of(j, zs); ys = ys - ys.mean()
        raw = data.z_to_raw(pp, f, zs)
        a.plot(raw, ys, color=SERIES[0], label="CoxKAN phi(x)")
        lin = cph.params_[f] * zs
        a.plot(raw, lin - lin.mean(), color=SERIES[1], linestyle="--", linewidth=1.4, label="CoxPH (linear)")
        a2 = a.twinx(); a2.hist(data.z_to_raw(pp, f, np.clip(x, lo, hi)), bins=40, color=INK2, alpha=0.12); a2.set_yticks([]); a2.grid(False)
        a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
        if f in C.LOG1P:
            a.set_xscale("symlog")
    a.set_title(pretty(f)); a.set_ylabel("contribution to log-hazard")
axes[0, 0].legend(fontsize=7)
fig.suptitle("CoxKAN learned shape functions (top-12 by importance) vs. CoxPH linear effect", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, FIG / "13_coxkan_shape_functions.png")

# ------------------------------------------------------------------ 3. symbolic formulas
formula_terms = []
for j, f in enumerate(names):
    x = tr["X"][f].values
    if set(np.unique(x)) <= {0.0, 1.0}:
        v = phi_of(j, np.array([0.0, 1.0]))
        sym_rows.append(dict(feature=f, function="indicator", r2=1.0, formula=f"{v[1] - v[0]:.3f}*[{f}=1]"))
        formula_terms.append((imp[f], f"{v[1] - v[0]:+.3f}*1[{f}]"))
        continue
    lo, hi = np.percentile(x, [1, 99])
    zs = np.linspace(lo, hi, 300)
    fit, _ = symbolic.fit_symbolic(zs, phi_of(j, zs))
    s = symbolic.to_string(fit, var=f"z_{f}")
    sym_rows.append(dict(feature=f, function=fit["name"], r2=fit["r2"], formula=s))
    formula_terms.append((imp[f], s.rsplit(" ", 2)[0]))
sym = pd.DataFrame(sym_rows).merge(imp.rename("importance"), left_on="feature", right_index=True).sort_values("importance", ascending=False)
sym.round(4).to_csv(RES / "symbolic_formulas.csv", index=False)
formula = " + ".join(t for _, t in sorted(formula_terms, reverse=True)[:8])
(RES / "symbolic_formula.txt").write_text("log h(t|x) = log h0(t) + " + formula + " + ... (remaining terms in symbolic_formulas.csv)\n")

# check: C-index of the symbolic (closed-form) model vs. the spline model on TEST
def symbolic_risk(X):
    total = np.zeros(len(X))
    for row in sym_rows:
        f, j = row["feature"], names.index(row["feature"])
        x = X[f].values
        if row["function"] == "indicator":
            v = phi_of(j, np.array([0.0, 1.0])); total += (v[1] - v[0]) * x
        else:
            lo, hi = np.percentile(tr["X"][f].values, [1, 99])
            zs = np.linspace(lo, hi, 300)
            fit, _ = symbolic.fit_symbolic(zs, phi_of(j, zs))
            fn = dict((n, g) for n, g, _ in symbolic.LIBRARY)[fit["name"]]
            total += fit["a"] * fn(fit["b"] * np.clip(x, lo, hi) + fit["c"]) if fit["name"] != "x" else fit["a"] * np.clip(x, lo, hi)
    return total


c_sym = metrics.harrell_c(te["t"], te["e"], symbolic_risk(te["X"]))
c_spl = metrics.harrell_c(te["t"], te["e"], kan.predict_risk(te["X"]))
pd.DataFrame([dict(model="CoxKAN additive (B-splines)", test_C=c_spl), dict(model="CoxKAN symbolic formula", test_C=c_sym)]).round(4) \
    .to_csv(RES / "symbolic_vs_spline.csv", index=False)

# ------------------------------------------------------------------ 4. permutation importance (model agnostic)
rng = np.random.default_rng(0)
perm = {}
for label, m in [("CoxPH", coxph), ("DeepSurv", ds), ("CoxKAN additive", kan), ("CoxKAN [d,4,1]", kan_deep)]:
    base = metrics.harrell_c(te["t"], te["e"], m.predict_risk(te["X"]))
    drops = {}
    for f in names:
        ds_ = []
        for _ in range(3):
            Xp = te["X"].copy(); Xp[f] = rng.permutation(Xp[f].values)
            ds_.append(base - metrics.harrell_c(te["t"], te["e"], m.predict_risk(Xp)))
        drops[f] = np.mean(ds_)
    perm[label] = pd.Series(drops)
perm = pd.DataFrame(perm)
perm.round(5).sort_values("CoxKAN additive", ascending=False).to_csv(RES / "permutation_importance.csv")

fig, ax = plt.subplots(figsize=(8, 6))
p = perm.sort_values("CoxKAN additive", ascending=False).head(12)[::-1]
w = 0.2
for i, c in enumerate(perm.columns):
    ax.barh(np.arange(len(p)) + (i - 1.5) * w, p[c] * 100, height=w, color=SERIES[i], label=c)
ax.set_yticks(range(len(p)), [pretty(i) for i in p.index], fontsize=8)
ax.set_xlabel("drop in test C-index when feature is shuffled (x100)")
ax.set_title("Permutation importance (top-12 by CoxKAN)"); ax.legend(fontsize=7)
save(fig, FIG / "14_permutation_importance.png")

top5 = pd.DataFrame({
    "rank": range(1, 6),
    "CoxPH (|log HR| per SD, p<0.05)": list(top5_cox.index),
    "CoxKAN (contribution std)": list(imp.index[:5]),
    "CoxKAN (P90/P10 HR)": list(eff.sort_values("abs_log_HR_CoxKAN", ascending=False).feature[:5]),
    "Permutation (CoxKAN)": list(perm["CoxKAN additive"].sort_values(ascending=False).index[:5]),
    "Permutation (DeepSurv)": list(perm["DeepSurv"].sort_values(ascending=False).index[:5]),
})
top5.to_csv(RES / "top5_comparison.csv", index=False)
print(top5.to_string())

# ------------------------------------------------------------------ 5. deep CoxKAN edge map + local explanations
E1 = kan_deep.net.layers[0].edge_importance(Xtr).numpy()           # (4, d)
fig, ax = plt.subplots(figsize=(13, 2.6))
order = np.argsort(-E1.sum(0))
im = ax.imshow(E1[:, order], aspect="auto", cmap="Blues")
ax.set_xticks(range(len(names)), [pretty(names[i]) for i in order], rotation=90, fontsize=7)
ax.set_yticks(range(4), [f"hidden {k + 1}" for k in range(4)]); ax.grid(False)
fig.colorbar(im, label="edge activation std")
ax.set_title("CoxKAN [d,4,1]: strength of every input->hidden edge (pruning candidates are near-white)")
save(fig, FIG / "15_coxkan_deep_edges.png")

r_te = kan.predict_risk(te["X"])
r_tr = kan.predict_risk(tr["X"])
times, H0 = metrics.breslow_baseline(tr["t"], tr["e"], r_tr)
grid = np.arange(0, 37, 1.0)
idx = [int(np.argsort(r_te)[int(q * (len(r_te) - 1))]) for q in (0.05, 0.5, 0.95)]
S = metrics.survival_curves(times, H0, r_te[idx], grid)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for k, (i, lab) in enumerate(zip(idx, ["low-risk customer (P5)", "median customer (P50)", "high-risk customer (P95)"])):
    ax[0].plot(grid, S[k], color=SERIES[k], label=f"{lab}: S(24m)={S[k][24]:.2f}")
ax[0].set_title("Predicted retention curves S(t|x) - CoxKAN"); ax[0].set_xlabel("months on book")
ax[0].set_ylabel("P(still a customer)"); ax[0].legend(); ax[0].set_ylim(0, 1.02)
with torch.no_grad():
    contrib = layer.edge_activations(torch.tensor(te["X"].values[[idx[2]]], dtype=torch.float32))[0, 0].numpy()
    ref = layer.edge_activations(Xtr)[:, 0].mean(0).numpy()
c = pd.Series(contrib - ref, index=names)
c = c.reindex(c.abs().sort_values(ascending=False).index[:10])[::-1]
ax[1].barh(range(len(c)), c.values, color=[SERIES[1] if v > 0 else SERIES[0] for v in c.values], height=0.6)
raw_row = te["raw"].iloc[idx[2]]
ax[1].set_yticks(range(len(c)), [f"{pretty(f)} = {raw_row[f] if f in raw_row else te['X'][f].iloc[idx[2]]:.4g}" if f in raw_row else pretty(f) for f in c.index], fontsize=8)
ax[1].axvline(0, color=INK2, linewidth=0.8)
ax[1].set_title("Why is the P95 customer high-risk? (log-hazard vs. average)")
save(fig, FIG / "16_customer_explanations.png")
print(open(RES / "symbolic_formula.txt").read())
print(pd.read_csv(RES / "symbolic_vs_spline.csv"))

# ------------------------------------------------------------------ 6. paired bootstrap of C-index differences
z = np.load(RES / "test_risks.npz")
t_, e_ = z["t"], z["e"]
pairs = [("CoxKAN", "CoxPH"), ("CoxKAN_deep", "CoxPH"), ("DeepSurv", "CoxPH"), ("DeepSurv", "CoxKAN"), ("DeepSurv", "CoxKAN_deep")]
rng = np.random.default_rng(1)
boots = [rng.integers(0, len(t_), len(t_)) for _ in range(300)]
out = []
for a, b in pairs:
    diffs = [metrics.harrell_c(t_[i], e_[i], z[a][i]) - metrics.harrell_c(t_[i], e_[i], z[b][i]) for i in boots]
    out.append(dict(comparison=f"{a} - {b}", delta_C=metrics.harrell_c(t_, e_, z[a]) - metrics.harrell_c(t_, e_, z[b]),
                    ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5),
                    p_one_sided=float(np.mean(np.array(diffs) <= 0))))
pd.DataFrame(out).round(4).to_csv(RES / "paired_bootstrap.csv", index=False)
print(pd.DataFrame(out).round(4))
