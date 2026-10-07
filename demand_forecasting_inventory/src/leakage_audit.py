"""Phases 10 and 15 - automated leakage audit.

Every check returns (name, PASS/FAIL, detail).  ``run_audit`` executes all of them on the real prepared data and writes
reports/leakage_audit.md and outputs/tables/leakage_audit.csv; the pytest suite asserts that nothing fails.  If any check FAILS the
affected analysis must be stopped, fixed and re-run - results from a failing audit are never used.

The strongest tests are *perturbation tests*: change the future (everything after a cut-off day) in a drastic way and verify
that nothing computed for origins up to the cut-off changes.  A leak cannot survive such a test.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import stat_models as sm
from .forecast_core import Context, ForecastBook
from .splits import Split

Result = tuple[str, bool, str]


def _ok(name: str, cond: bool, detail: str) -> Result:
    return (name, bool(cond), detail)


# --------------------------------------------------------------------------------------
# 1. Chronology / splitting
# --------------------------------------------------------------------------------------
def check_split_chronology(split: Split) -> list[Result]:
    a, b, N = split.train_end, split.val_end, split.n_days
    d = split.dates
    res = [
        _ok("split is chronological", 0 < a < b < N and d[a - 1] < d[a] and d[b - 1] < d[b],
            f"train {d[0].date()}..{d[a-1].date()} | val {d[a].date()}..{d[b-1].date()} | test {d[b].date()}..{d[-1].date()}"),
        _ok("periods are disjoint and exhaustive", (a - 0) + (b - a) + (N - b) == N, f"{a} + {b - a} + {N - b} = {N} days"),
    ]
    for H in C.HORIZONS:
        vo, to = split.eval_origins("validation", H), split.eval_origins("test", H)
        res.append(_ok(f"evaluation windows stay inside their period (H={H})",
                       (vo + 1 >= a).all() and (vo + H <= b - 1).all() and (to + 1 >= b).all() and (to + H <= N - 1).all(),
                       f"val origins {vo.min()}..{vo.max()}, test origins {to.min()}..{to.max()}"))
        tr = split.train_origins(H, a)
        res.append(_ok(f"training targets end before validation starts (H={H})", (tr + H).max() <= a - 1,
                       f"max target day {int((tr + H).max())} <= {a - 1}"))
        tr2 = split.train_origins(H, b)
        res.append(_ok(f"final-fit targets end before test starts (H={H})", (tr2 + H).max() <= b - 1, f"max target day {int((tr2 + H).max())} <= {b - 1}"))
    return res


def check_no_random_splitting(src_dir: Path | None = None) -> list[Result]:
    src_dir = src_dir or Path(__file__).parent
    bad = re.compile(r"train_test_split|shuffle\s*=\s*True|ShuffleSplit|KFold\(|\.sample\(frac")
    hits = []
    for p in sorted(src_dir.glob("*.py")):
        if p.name in ("leakage_audit.py", "explain.py"):               # explain.py samples rows only for SHAP plots (no fitting)
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if bad.search(line) and not line.lstrip().startswith("#"):
                hits.append(f"{p.name}:{i}")
    return [_ok("no random splitting / shuffling API used for fitting or evaluation", not hits, "no occurrences" if not hits else ", ".join(hits))]


# --------------------------------------------------------------------------------------
# 2. Feature leakage: perturbation + explicit-definition tests
# --------------------------------------------------------------------------------------
def _perturbed_context_arrays(ctx: Context, cut: int, rng: np.random.Generator):
    """Y and price with EVERYTHING after day `cut` replaced by wild values."""
    Y2, P2 = ctx.Y.copy(), ctx.price.copy()
    Y2[:, cut + 1:] = rng.integers(0, 500, size=Y2[:, cut + 1:].shape)
    P2[:, cut + 1:] = rng.uniform(0.1, 50.0, size=P2[:, cut + 1:].shape)
    return Y2, P2


def check_feature_perturbation(ctx: Context, H: int = 7, n_cuts: int = 3, seed: int = C.SEED) -> list[Result]:
    rng = np.random.default_rng(seed)
    base = F.build_features(ctx.Y, ctx.price, ctx.dates, ctx.snap_full, ctx.event_full, ctx.christmas_full, H)
    known = F.FEATURE_GROUPS["known_ahead"]
    non_future = [f for f in F.FEATURES if f not in known]
    res = []
    cuts = rng.integers(ctx.split.train_end - 200, ctx.N - 40, size=n_cuts)
    all_equal_hist, all_equal_known = True, True
    for cut in cuts:
        Y2, P2 = _perturbed_context_arrays(ctx, int(cut), rng)
        pert = F.build_features(Y2, P2, ctx.dates, ctx.snap_full, ctx.event_full, ctx.christmas_full, H)
        m = base["origin"] <= cut
        for f in non_future:
            if not np.array_equal(base.loc[m, f].to_numpy(), pert.loc[m, f].to_numpy()):
                all_equal_hist = False
                res.append(_ok(f"history feature '{f}' unaffected by future data", False, f"changed when data after day {cut} was perturbed"))
        for f in known:                                                    # calendar-only: unaffected for ALL rows
            if not np.array_equal(base[f].to_numpy(), pert[f].to_numpy()):
                all_equal_known = False
                res.append(_ok(f"known-ahead feature '{f}' independent of sales/price", False, "changed with perturbed sales"))
    if all_equal_hist:
        res.append(_ok("history/calendar-origin features at origin <= cut-off are unchanged when ALL later sales and prices are perturbed",
                       True, f"{len(non_future)} features x {n_cuts} random cut-offs, {len(base):,} rows each: identical"))
    if all_equal_known:
        res.append(_ok("known-ahead (window) features depend only on the published calendar", True,
                       f"{len(known)} features identical under random sales/price perturbation: {', '.join(known)}"))
    return res


def check_feature_definitions(ctx: Context, H: int = 7, n: int = 400, seed: int = C.SEED) -> list[Result]:
    df = F.build_features(ctx.Y, ctx.price, ctx.dates, ctx.snap_full, ctx.event_full, ctx.christmas_full, H)
    rng = np.random.default_rng(seed)
    bad = []
    for _ in range(n):
        s = int(rng.integers(ctx.n_series)); t = int(rng.integers(F.FIRST_ORIGIN, ctx.N))
        row = df[(df["series_idx"] == s) & (df["origin"] == t)].iloc[0]
        y = ctx.Y[s]
        exp = {"lag_1": y[t], "lag_7": y[t - 6], "lag_14": y[t - 13], "lag_21": y[t - 20], "lag_28": y[t - 27],
               "rolling_mean_7": y[t - 6:t + 1].mean(), "rolling_mean_14": y[t - 13:t + 1].mean(), "rolling_mean_28": y[t - 27:t + 1].mean(),
               "rolling_std_7": y[t - 6:t + 1].std(ddof=1), "rolling_std_28": y[t - 27:t + 1].std(ddof=1),
               "snap_days_window": ctx.snap_full[s, t + 1:t + H + 1].sum(), "event_days_window": ctx.event_full[t + 1:t + H + 1].sum()}
        if t + H <= ctx.N - 1:
            exp["target"] = y[t + 1:t + H + 1].sum()
        for k, v in exp.items():
            if not np.isclose(row[k], v, rtol=0, atol=1e-9):
                bad.append((s, t, k))
    return [_ok("lags, rolling windows (end at the origin), window counts and targets equal explicit loops", not bad,
                f"{n} random rows checked, mismatches: {len(bad)}")]


def check_rolling_window_alignment() -> list[Result]:
    """rolling_mean_7(t) must equal mean(y[t-6..t]) and must NOT equal a centred window (t-3..t+3)."""
    rng = np.random.default_rng(1)
    y = rng.poisson(4, 60).astype(float)
    t = 30
    got = F.trailing_stat(y, 7, "mean")[t]
    past = y[t - 6:t + 1].mean()
    centred = y[t - 3:t + 4].mean()
    return [_ok("rolling_mean_7(t) uses t-6..t (past only), not a centred window", np.isclose(got, past) and not np.isclose(got, centred),
                f"value {got:.4f} == past-window mean {past:.4f}; centred-window mean would be {centred:.4f}")]


# --------------------------------------------------------------------------------------
# 3. Model-level causality (forecasts at origin t use only data <= t) and fit-window isolation
# --------------------------------------------------------------------------------------
def check_ets_causality(ctx: Context, series: int = 0, seed: int = C.SEED) -> list[Result]:
    rng = np.random.default_rng(seed)
    y = ctx.Y[series].copy()
    a = ctx.split.train_end
    chosen, _ = sm.choose_ets_models(y[:a])
    cut = a + 120
    y2 = y.copy()
    y2[cut + 1:] = rng.integers(0, 300, size=len(y2[cut + 1:]))
    res = []
    for name, f in chosen.items():
        p1, p2 = sm.ets_rolling_paths(y, f), sm.ets_rolling_paths(y2, f)
        res.append(_ok(f"{name}: forecasts at origins <= cut-off unchanged by later data", np.allclose(p1[: cut + 1], p2[: cut + 1], equal_nan=True),
                       f"max abs diff {np.nanmax(np.abs(p1[: cut + 1] - p2[: cut + 1])):.2e}"))
    # fit isolation: parameters estimated on y[:a] do not depend on y[a:]
    c1, _ = sm.choose_ets_models(y[:a])
    y3 = y.copy(); y3[a:] = rng.integers(0, 300, size=len(y3[a:]))
    c2, _ = sm.choose_ets_models(y3[:a])
    same = all(np.isclose(c1[k]["params"].get("smoothing_level", 0), c2[k]["params"].get("smoothing_level", 0)) for k in c1)
    res.append(_ok("ETS parameters estimated on the training window are independent of later data", same, "smoothing parameters identical"))
    return res


def check_sarima_causality(seed: int = C.SEED) -> list[Result]:
    """Short synthetic weekly-seasonal series (full-size SARIMA perturbation test would be slow)."""
    rng = np.random.default_rng(seed)
    n = 260
    season = np.tile([5, 6, 7, 6, 8, 12, 10], n // 7 + 1)[:n]
    y = rng.poisson(season).astype(float)
    fit_end, cut = 200, 230
    res0, ok = sm._fit_sarima(y[:fit_end], sm.SARIMA_CANDIDATES["S1:(1,0,1)(1,0,1)7+c"])
    y2 = y.copy(); y2[cut + 1:] = rng.integers(0, 100, size=n - cut - 1)
    p1 = sm.roll_sarima(res0, y, fit_end - 1, n - 2)
    p2 = sm.roll_sarima(res0, y2, fit_end - 1, n - 2)
    m = cut - (fit_end - 1) + 1
    return [_ok("SARIMA: forecasts at origins <= cut-off unchanged by later data (state updated one day at a time)",
                np.allclose(p1[:m], p2[:m]), f"max abs diff {np.abs(p1[:m] - p2[:m]).max():.2e}")]


def check_xgb_prediction_causality(ctx: Context, H: int = 7, seed: int = C.SEED) -> list[Result]:
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    m = xgb.XGBRegressor(); m.load_model(C.OUT_MOD / f"xgb_H{H}_final_train_val.json")
    cut = ctx.split.val_end + 100
    Y2, P2 = _perturbed_context_arrays(ctx, cut, rng)
    d1 = F.build_features(ctx.Y, ctx.price, ctx.dates, ctx.snap_full, ctx.event_full, ctx.christmas_full, H)
    d2 = F.build_features(Y2, P2, ctx.dates, ctx.snap_full, ctx.event_full, ctx.christmas_full, H)
    msk = d1["origin"] <= cut
    p1, p2 = m.predict(d1.loc[msk, F.FEATURES]), m.predict(d2.loc[msk, F.FEATURES])
    return [_ok("XGBoost predictions at origins <= cut-off unchanged by later data", np.array_equal(p1, p2),
                f"{int(msk.sum()):,} rows, max abs diff {np.abs(p1 - p2).max():.2e}")]


def check_xgb_training_rows(ctx: Context) -> list[Result]:
    res = []
    for H in C.HORIZONS:
        ds = F.dataset_for_horizon(ctx, H)
        tr = F.rows_for_fit(ds, H, ctx.split.train_end)
        fin = F.rows_for_fit(ds, H, ctx.split.val_end)
        res.append(_ok(f"XGBoost H={H}: tuning rows contain no validation/test information", (tr["origin"] + H).max() <= ctx.split.train_end - 1,
                       f"{len(tr):,} rows, last target day {int((tr['origin'] + H).max())} < validation start {ctx.split.train_end}"))
        res.append(_ok(f"XGBoost H={H}: final-fit rows contain no test information", (fin["origin"] + H).max() <= ctx.split.val_end - 1,
                       f"{len(fin):,} rows, last target day {int((fin['origin'] + H).max())} < test start {ctx.split.val_end}"))
    return res


def check_tuning_uses_validation_only(tuning_log: pd.DataFrame | None) -> list[Result]:
    if tuning_log is None:
        return []
    return [_ok("hyper-parameter search scored on validation rows only (log has validation metrics, no test metrics)",
                not any(c.startswith("test_") for c in tuning_log.columns), "columns: " + ", ".join(tuning_log.columns[:8]) + " ...")]


# --------------------------------------------------------------------------------------
# 4. Inventory policy information set
# --------------------------------------------------------------------------------------
def check_inventory_policy_causality(ctx: Context, book: ForecastBook | None, sigma_df: pd.DataFrame | None, seed: int = C.SEED) -> list[Result]:
    from . import inventory as inv
    from . import inventory_experiments as ie
    res = []
    inputs = ie.prepare_inputs(ctx)
    # policy A: decision arrays at test day j use trailing statistics of Y up to that day -> perturb the future
    rng = np.random.default_rng(seed)
    j0 = 100
    t0 = int(inputs["t_days"][j0])
    Y2 = ctx.Y.copy(); Y2[:, t0 + 1:] = rng.integers(0, 400, size=Y2[:, t0 + 1:].shape)
    m28 = np.vstack([F.trailing_stat(ctx.Y[s], 28, "mean") for s in range(ctx.n_series)])[:, inputs["t_days"]]
    m28p = np.vstack([F.trailing_stat(Y2[s], 28, "mean") for s in range(ctx.n_series)])[:, inputs["t_days"]]
    s28 = np.vstack([F.trailing_stat(ctx.Y[s], 28, "std") for s in range(ctx.n_series)])[:, inputs["t_days"]]
    s28p = np.vstack([F.trailing_stat(Y2[s], 28, "std") for s in range(ctx.n_series)])[:, inputs["t_days"]]
    res.append(_ok("Policy A reorder-point inputs on day j use data up to day j only", np.array_equal(m28[:, : j0 + 1], m28p[:, : j0 + 1]) and
                   np.array_equal(s28[:, : j0 + 1], s28p[:, : j0 + 1]), f"trailing 28-day mean/std unchanged for days 0..{j0} after perturbing later demand"))
    # simulation: perturb demand after day j0 with fixed decision arrays -> identical orders up to j0 (also unit-tested)
    s = 0
    rop = np.full(len(inputs["t_days"]), 5.0); S = rop + 20
    d1 = inputs["demand"][s].astype(float); d2 = d1.copy(); d2[j0 + 1:] = rng.integers(0, 50, size=len(d2) - j0 - 1)
    l1, l2 = inv.simulate(d1, rop, S, 7, 10), inv.simulate(d2, rop, S, 7, 10)
    res.append(_ok("simulation decisions up to day j are independent of demand after day j",
                   np.array_equal(l1.orders_placed[: j0 + 1], l2.orders_placed[: j0 + 1]) and np.array_equal(l1.inventory_position[: j0 + 1], l2.inventory_position[: j0 + 1]),
                   f"orders and inventory position identical on days 0..{j0}"))
    if book is not None:
        res += check_sigma_validation_only(ctx, book)
        t_eval = ctx.split.eval_origins("validation", 28)
        res.append(_ok("validation residual windows end before the test period starts", (t_eval + 28).max() <= ctx.split.val_end - 1,
                       f"last validation window ends on day {int((t_eval + 28).max())} < {ctx.split.val_end}"))
    return res


def check_sigma_validation_only(ctx: Context, book: ForecastBook, seed: int = C.SEED) -> list[Result]:
    """Perturbation test: wreck every TEST-period actual and every TEST-phase forecast; sigma_L (Policy B) must not change."""
    import copy
    from dataclasses import replace

    from .uncertainty import sigma_table
    rng = np.random.default_rng(seed)
    models = [m for m in ("SARIMA", "XGBoost", "SeasonalNaive") if m in book.F["val"]]
    s1 = sigma_table(book, models, horizons=(3, 7, 28))
    Y2 = ctx.Y.copy()
    Y2[:, ctx.split.val_end:] = rng.integers(0, 500, size=Y2[:, ctx.split.val_end:].shape)
    book2 = ForecastBook(replace(ctx, Y=Y2), F=copy.deepcopy(book.F))
    for m in book2.F["test"]:
        book2.F["test"][m] = book2.F["test"][m] * 7.0 + 13.0
    s2 = sigma_table(book2, models, horizons=(3, 7, 28))
    same = np.allclose(s1[["sigma_rmse", "sigma_std", "bias"]].to_numpy(), s2[["sigma_rmse", "sigma_std", "bias"]].to_numpy())
    return [_ok("sigma_L (Policy B safety stock) is invariant to test-period actuals and test-phase forecasts", same,
                f"{len(s1)} (model, horizon, series) sigmas identical after perturbing all data from day {ctx.split.val_end} on")]


def check_forecast_book_alignment(book: ForecastBook) -> list[Result]:
    """All models are evaluated on identical origins and targets."""
    ok = True
    for ph in ("val", "test"):
        shapes = {m: arr.shape for m, arr in book.F[ph].items()}
        ok &= len(set(shapes.values())) == 1
    n_nan = {m: int(np.isnan(arr).sum()) for ph in ("val", "test") for m, arr in book.F[ph].items() if np.isnan(arr).any()}
    return [_ok("all models forecast the same origins (identical array shapes in val and test)", ok, f"{len(book.F['val'])} models"),
            _ok("no NaN forecasts in any model/phase", not n_nan, "none" if not n_nan else str(n_nan))]


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def run_audit(ctx: Context, book: ForecastBook | None = None, sigma_df: pd.DataFrame | None = None,
              tuning_log: pd.DataFrame | None = None, include_models: bool = True, write: bool = True) -> pd.DataFrame:
    results: list[Result] = []
    results += check_split_chronology(ctx.split)
    results += check_no_random_splitting()
    results += check_rolling_window_alignment()
    results += check_feature_definitions(ctx)
    results += check_feature_perturbation(ctx)
    results += check_xgb_training_rows(ctx)
    results += check_tuning_uses_validation_only(tuning_log)
    if include_models:
        results += check_ets_causality(ctx)
        results += check_sarima_causality()
        if (C.OUT_MOD / "xgb_H7_final_train_val.json").exists():
            results += check_xgb_prediction_causality(ctx)
    if book is not None:
        results += check_forecast_book_alignment(book)
    results += check_inventory_policy_causality(ctx, book, sigma_df)
    df = pd.DataFrame(results, columns=["check", "passed", "detail"])
    df["status"] = np.where(df["passed"], "PASS", "FAIL")
    if write:                                                           # unit tests call with write=False so they never overwrite pipeline outputs
        df[["check", "status", "detail"]].to_csv(C.OUT_TAB / "leakage_audit.csv", index=False)
        write_audit_report(df)
    return df


def write_audit_report(df: pd.DataFrame) -> None:
    n_fail = int((~df["passed"]).sum())
    L = ["# Leakage audit (Phases 10 and 15)", "",
         f"**Result: {'ALL ' + str(len(df)) + ' CHECKS PASSED' if n_fail == 0 else str(n_fail) + ' CHECK(S) FAILED - results must not be used'}.** "
         "Generated by `src/leakage_audit.py` on the real data; re-run by `pytest tests/test_leakage.py`.", "",
         "## What is audited and how",
         "* **Chronology:** no random split anywhere; every evaluation window lies inside its period; training targets end before the next period starts.",
         "* **Feature leakage:** (i) explicit-definition checks (lag_k(t) = y[t+1-k], rolling window = y[t-w+1..t]); (ii) *perturbation tests* - all sales and "
         "prices after a random cut-off day are replaced by wild values and every history/calendar feature of origins up to the cut-off must be identical.",
         "* **Known-ahead features** (counts of SNAP days / events / Christmas inside the forecast window) are the only features whose timestamp is later than "
         "the origin. They are deterministic functions of the *published calendar*; the perturbation test shows they are invariant to sales and prices.",
         "* **Model causality:** ETS / SARIMA / XGBoost forecasts at origin t are unchanged when data after t is perturbed.",
         "* **Tuning and uncertainty:** hyper-parameters are chosen on validation rows only; sigma_L is computed from validation residuals only.",
         "* **Inventory policy:** decisions on day t use only information available at the end of day t (perturbation test of policy inputs and of the simulator).", "",
         "## Results", "",
         df[["check", "status", "detail"]].to_markdown(index=False)]
    (C.REPORTS / "leakage_audit.md").write_text("\n".join(L) + "\n")
