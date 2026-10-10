"""End-to-end run:  python -m credit_risk.pipeline [--champion rf|lr|hgb] [--quick]

Protocol (the hold-out test split is used exactly once, after every choice is frozen):

1. stratified 75/25 train/test split of the 1-year-horizon firms
2. per model: hyper-parameters by 5-fold CV on the training split (SQC screening is a
   pipeline step, so it is re-fitted inside every fold)
3. out-of-fold raw scores -> choose the PD calibration method by cross-fitted Brier
4. cost-sensitive threshold on out-of-fold calibrated PDs (training firms only)
5. refit on all training firms, score the test portfolio, build the EL-ranked watchlist
6. robustness: other forecast horizons and repeated random splits
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from . import calibration, config, expected_loss as el, plots, threshold as thr
from .data import load_horizon
from .metrics import reliability_table, stratified_bootstrap_ci, summarize
from .models import FACTORIES, PARAM_GRIDS
from .screening import SQCRatioScreener

MODELS = ("lr", "rf", "hgb")
SCOPE = ("lr", "rf")  # declared project scope; hgb is a reference challenger


def altman_z(X: pd.DataFrame, fill: float | None = None) -> tuple[np.ndarray, float]:
    """Altman Z'' (emerging-market / non-manufacturer) from WC/TA, RE/TA, EBIT/TA, BVE/TL. Higher = safer."""
    z = 6.56 * X["Attr3"] + 3.26 * X["Attr6"] + 6.72 * X["Attr7"] + 1.05 * X["Attr8"]
    fill = float(z.median()) if fill is None else fill
    return z.fillna(fill).to_numpy(), fill


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.reset_index().to_dict(orient="records"))
    return o


def tune(name, Xtr, ytr, quick: bool):
    """Grid search on the training split. Returns (fitted-params dict, CV table)."""
    grid = PARAM_GRIDS[name]
    if quick:
        grid = {k: v[:1] for k, v in grid.items()}
    base = FACTORIES[name](n_estimators=200) if name == "rf" else FACTORIES[name]()
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=False)
    gs.fit(Xtr, ytr)
    table = pd.DataFrame(gs.cv_results_)
    table = table[[c for c in table.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "rank_test_score"]]
    table = table.sort_values("rank_test_score")
    return gs.best_params_, float(gs.best_score_), float(table.iloc[0]["std_test_score"]), table


def build_model(name, params, quick: bool):
    est = FACTORIES[name]()
    est.set_params(**params)
    if name == "rf":
        est.set_params(clf__n_estimators=150 if quick else 500)
    return est


def evaluate_boot(y, p, n_boot):
    lo, hi = stratified_bootstrap_ci(y, p, n_boot=n_boot)
    return {**summarize(y, p), "auc_ci95": [lo, hi]}


def run(champion: str | None = None, quick: bool = False, out_dir=config.REPORT_DIR, skip_robustness: bool = False):
    warnings.filterwarnings("ignore")
    cfg = config.PipelineConfig()
    n_boot = 100 if quick else cfg.n_bootstrap
    out_dir = type(config.REPORT_DIR)(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {"environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "numpy": np.__version__, "pandas": pd.__version__, "quick_mode": quick},
               "config": {"seed": config.RANDOM_STATE, "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
                          "sqc": vars(cfg.sqc), "loss": {k: v for k, v in vars(cfg.loss).items()}}}

    # ---------------------------------------------------------------- data
    X, y = load_horizon(config.PRIMARY_HORIZON)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    R["data"] = {"horizon_file": f"{config.PRIMARY_HORIZON}year.arff", "firms": len(X), "defaults": int(y.sum()),
                 "default_rate": float(y.mean()), "train_firms": len(Xtr), "train_defaults": int(ytr.sum()),
                 "test_firms": len(Xte), "test_defaults": int(yte.sum())}
    _log(f"data: {R['data']}")

    # ----------------------------------------------------- SQC screening report
    screener = SQCRatioScreener(tail_prob=cfg.sqc.tail_prob, max_missing=cfg.sqc.max_missing, fdr=cfg.sqc.fdr,
                                max_corr=cfg.sqc.max_corr).fit(Xtr, ytr)
    sqc_report = screener.report_.sort_values(["status", "lift"], ascending=[True, False])
    sqc_report.to_csv(out_dir / "sqc_screening.csv", index=False)
    plots.sqc_plot(screener.report_, fig_dir / "sqc_top_ratios.png")
    R["sqc"] = {"ratios_in": int(X.shape[1]), "ratios_kept": int(len(screener.kept_)),
                "status_counts": screener.report_["status"].value_counts().to_dict()}
    _log(f"SQC screening kept {len(screener.kept_)}/{X.shape[1]} ratios: {R['sqc']['status_counts']}")

    # ------------------------------------------------- tune, OOF scores, calibration
    best_params, tuned, oof_raw, oof_pd, calib_choice, calib_tables, final_models = {}, {}, {}, {}, {}, {}, {}
    outer = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE + 1)
    for name in MODELS:
        t0 = time.time()
        params, cv_auc, cv_sd, table = tune(name, Xtr, ytr, quick)
        best_params[name] = params
        tuned[name] = {"cv_auc": cv_auc, "cv_auc_sd": cv_sd, "best_params": params}
        table.to_csv(out_dir / f"tuning_{name}.csv", index=False)
        est = build_model(name, params, quick)
        oof_raw[name] = cross_val_predict(est, Xtr, ytr, cv=outer, method="predict_proba")[:, 1]
        calib_choice[name], calib_tables[name], cal_oof = calibration.select_method(oof_raw[name], ytr)
        oof_pd[name] = cal_oof[calib_choice[name]]
        final_models[name] = est.fit(Xtr, ytr)
        _log(f"{name}: CV AUC {cv_auc:.4f} (±{cv_sd:.4f}) params={params} calibration={calib_choice[name]} [{time.time() - t0:.0f}s]")
    R["tuning"] = tuned
    R["calibration_selection"] = {n: calib_tables[n] for n in MODELS}
    R["calibration_choice"] = calib_choice

    champion = champion or max(SCOPE, key=lambda n: tuned[n]["cv_auc"])
    R["champion"] = {"model": champion, "rule": "best training-CV AUC among the declared scope (LR, RF)"
                     if champion in SCOPE else "manual override"}
    _log(f"champion = {champion}")

    # ---------------------------------------------------------- hold-out test (once)
    test_raw, test_pd = {}, {}
    for name in MODELS:
        test_raw[name] = final_models[name].predict_proba(Xte)[:, 1]
        test_pd[name] = calibration.ScoreCalibrator(calib_choice[name]).fit(oof_raw[name], ytr).predict(test_raw[name])
    z_tr, fill = altman_z(Xtr)
    z_te, _ = altman_z(Xte, fill)
    test_metrics = {n: evaluate_boot(yte, test_pd[n], n_boot) for n in MODELS}
    test_metrics["rf_uncalibrated"] = evaluate_boot(yte, test_raw["rf"], n_boot)
    test_metrics["altman_z"] = {"auc_roc": float(roc_auc_score(yte, -z_te)),
                                "auc_ci95": list(stratified_bootstrap_ci(yte, -z_te, n_boot=n_boot))}
    R["test_metrics"] = test_metrics
    R["train_oof_metrics"] = {n: summarize(ytr, oof_pd[n]) for n in MODELS}
    for n in (*MODELS, "altman_z"):
        _log(f"TEST {n:9s} AUC={test_metrics[n]['auc_roc']:.4f}  CI95={np.round(test_metrics[n]['auc_ci95'], 3)}")
    plots.roc_plot(yte, {"lr": test_pd["lr"], "rf": test_pd["rf"], "hgb": test_pd["hgb"], "altman": -z_te},
                   {"lr": test_metrics["lr"]["auc_roc"], "rf": test_metrics["rf"]["auc_roc"],
                    "hgb": test_metrics["hgb"]["auc_roc"], "altman": test_metrics["altman_z"]["auc_roc"]},
                   fig_dir / "roc_test.png")
    plots.calibration_plot(yte, {"RF uncalibrated": reliability_table(yte, test_raw["rf"]),
                                 f"RF calibrated ({calib_choice['rf']})": reliability_table(yte, test_pd["rf"]),
                                 f"LR ({calib_choice['lr']})": reliability_table(yte, test_pd["lr"])},
                           fig_dir / "calibration_test.png")

    # ----------------------------------------------- cost-sensitive threshold
    loss = cfg.loss
    lgd_tr, ead_tr = el.estimate_lgd(Xtr, loss), el.estimate_ead(Xtr, loss)
    lgd_te, ead_te = el.estimate_lgd(Xte, loss), el.estimate_ead(Xte, loss)
    cfn_tr, cfp_tr = lgd_tr * ead_tr, loss.margin * ead_tr
    cfn_te, cfp_te = lgd_te * ead_te, loss.margin * ead_te
    p_oof, p_te = oof_pd[champion], test_pd[champion]
    t_star, (t_lo, t_hi) = thr.bootstrap_threshold(p_oof, ytr, cfn_tr, cfp_tr, n_boot=60 if quick else 300)
    curve = thr.cost_curve(p_oof, ytr, cfn_tr, cfp_tr)
    t_youden = thr.youden_threshold(p_oof, ytr)
    bayes_flags = p_te >= thr.bayes_threshold(lgd_te, loss.margin)
    rules = {
        "cost-optimal global threshold": thr.evaluate_rule(p_te, yte, cfn_te, cfp_te, t_star),
        "per-firm Bayes rule (PD >= m/(m+LGD))": {"threshold": None, **thr.evaluate_flags(bayes_flags, yte, cfn_te, cfp_te)},
        "naive threshold 0.5": thr.evaluate_rule(p_te, yte, cfn_te, cfp_te, 0.5),
        "Youden-J threshold": thr.evaluate_rule(p_te, yte, cfn_te, cfp_te, t_youden),
        "flag nobody": thr.evaluate_flags(np.zeros(len(yte), bool), yte, cfn_te, cfp_te),
        "flag everybody": thr.evaluate_flags(np.ones(len(yte), bool), yte, cfn_te, cfp_te),
    }
    base_cost = rules["flag nobody"]["cost"]
    for r in rules.values():
        r["cost_saving_vs_flag_nobody"] = 1 - r["cost"] / base_cost
    sens = []
    for m in (0.01, 0.02, 0.03, 0.05, 0.08):
        cfp_m_tr, cfp_m_te = m * ead_tr, m * ead_te
        t_m, _ = thr.bootstrap_threshold(p_oof, ytr, cfn_tr, cfp_m_tr, n_boot=20 if quick else 100)
        opt, naive = (thr.evaluate_rule(p_te, yte, cfn_te, cfp_m_te, t) for t in (t_m, 0.5))
        sens.append({"margin": m, "cost_ratio_fn_to_fp": float(np.mean(cfn_te) / np.mean(cfp_m_te)),
                     "optimal_threshold": t_m, "flagged_share": opt["flagged_share"], "recall": opt["recall"],
                     "cost_saving_vs_0.5": 1 - opt["cost"] / naive["cost"]})
    R["threshold"] = {"champion": champion, "t_star": t_star, "t_star_ci90_bootstrap": [t_lo, t_hi],
                      "t_youden": t_youden, "bayes_threshold_at_mean_lgd": float(loss.margin / (loss.margin + lgd_te.mean())),
                      "test_rules": rules, "margin_sensitivity": sens,
                      "proxy_summary": {"mean_lgd": float(lgd_te.mean()), "mean_lgd_defaulters_train": float(lgd_tr[ytr == 1].mean()),
                                        "median_ead": float(np.median(ead_te))}}
    plots.cost_curve_plot(curve, t_star, {"Youden J": t_youden, "Bayes @ mean LGD": R["threshold"]["bayes_threshold_at_mean_lgd"]},
                          fig_dir / "cost_threshold.png")
    _log(f"cost-optimal threshold {t_star:.4f} (90% boot {t_lo:.3f}-{t_hi:.3f}); flagged {rules['cost-optimal global threshold']['flagged_share']:.1%}, "
         f"recall {rules['cost-optimal global threshold']['recall']:.1%}, saving vs flag-nobody {rules['cost-optimal global threshold']['cost_saving_vs_flag_nobody']:.1%}")

    # ------------------------------------------------- EL watchlist + back-test
    wl = el.build_watchlist(Xte, p_te, t_star, yte, loss)
    wl.to_csv(out_dir / "watchlist_full.csv", index=False)
    wl.head(cfg.watchlist_top_n).to_csv(out_dir / f"watchlist_top{cfg.watchlist_top_n}.csv", index=False)
    plots.watchlist_plot(wl, fig_dir / "watchlist_top.png", cfg.watchlist_top_n)
    realized = yte * lgd_te * ead_te
    el_te = el.expected_loss(p_te, lgd_te, ead_te)
    fr = cfg.capture_fractions
    rankings = {"EL rank": el_te, "PD rank": p_te, "EAD only": ead_te}
    capture = {k: el.loss_capture(realized, s, fr) for k, s in rankings.items()}
    d_capture = {k: el.default_capture(yte, s, fr) for k, s in rankings.items()}
    rng = np.random.default_rng(config.RANDOM_STATE)
    boot = {q: [] for q in fr}
    for _ in range(n_boot):
        idx = rng.integers(0, len(yte), len(yte))
        if realized[idx].sum() == 0:
            continue
        a, b = (el.loss_capture(realized[idx], s[idx], fr) for s in (el_te, p_te))
        for q in fr:
            boot[q].append(a[q] - b[q])
    top_el, top_pd = set(wl.head(cfg.watchlist_top_n)["firm_id"]), set(wl.nsmallest(cfg.watchlist_top_n, "pd_rank")["firm_id"])
    R["watchlist"] = {
        "portfolio_firms": len(wl), "total_expected_loss": float(wl["expected_loss"].sum()),
        "total_realized_loss": float(realized.sum()), "pd_total_calibration": {"sum_pd": float(p_te.sum()), "defaults": int(yte.sum())},
        "loss_capture": capture, "default_capture": d_capture,
        "el_minus_pd_capture_ci95": {q: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for q, v in boot.items() if v},
        f"top{cfg.watchlist_top_n}_overlap_el_vs_pd": len(top_el & top_pd),
        f"top{cfg.watchlist_top_n}_defaults": int(wl.head(cfg.watchlist_top_n)["realized_default"].sum()),
        "top_decile_el_share": float(wl.head(int(np.ceil(0.1 * len(wl))))["expected_loss"].sum() / wl["expected_loss"].sum()),
        "grade_table": wl.groupby("grade").agg(firms=("pd", "size"), mean_pd=("pd", "mean"), observed_default_rate=("realized_default", "mean"),
                                                expected_loss=("expected_loss", "sum")).reset_index(),
    }
    curves = {}
    grid_q = np.linspace(0.0, 1.0, 101)
    for label, s in (("EL rank", el_te), ("PD rank", p_te)):
        curves[label] = (grid_q, [0.0] + list(el.loss_capture(realized, s, grid_q[1:]).values()))
    curves["Random"] = (grid_q, grid_q)
    plots.capture_plot(curves, fig_dir / "watchlist_capture.png")
    _log(f"loss capture @10%: EL-rank {capture['EL rank'][0.1]:.1%} vs PD-rank {capture['PD rank'][0.1]:.1%} vs random 10%")

    # ------------------------------------------------------------ robustness
    if not skip_robustness:
        R["horizons"], R["repeated_splits"] = robustness(best_params, quick, fig_dir, out_dir)

    R["model_card"] = {"feature_names": list(final_models[champion].named_steps["sqc"].get_feature_names_out())}
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(R), indent=2))
    _log(f"wrote {out_dir / 'metrics.json'}")
    return R


def robustness(best_params, quick: bool, fig_dir, out_dir):
    """Same protocol, fixed (primary-horizon) hyper-parameters: other horizons + repeated splits."""
    rows = {}
    for h in config.ALL_HORIZONS:
        Xh, yh = load_horizon(h)
        a, b, ya, yb = train_test_split(Xh, yh, test_size=config.TEST_SIZE, stratify=yh, random_state=config.RANDOM_STATE)
        row = {"firms": len(Xh), "defaults": int(yh.sum()), "default_rate": float(yh.mean())}
        for n in MODELS:
            row[n] = float(roc_auc_score(yb, build_model(n, best_params[n], quick).fit(a, ya).predict_proba(b)[:, 1]))
        zt, fill = altman_z(a)
        row["altman"] = float(roc_auc_score(yb, -altman_z(b, fill)[0]))
        rows[6 - h] = row  # file k = ratios observed (6-k) years before the bankruptcy outcome
        _log(f"horizon {6 - h}y ahead: " + ", ".join(f"{k}={row[k]:.3f}" for k in (*MODELS, "altman")))
    table = pd.DataFrame(rows).T.sort_index()
    table.index.name = "years_ahead"
    plots.horizon_plot(table, fig_dir / "auc_by_horizon.png")
    table.to_csv(out_dir / "horizon_robustness.csv")

    X, y = load_horizon(config.PRIMARY_HORIZON)
    n_rep = 3 if quick else 10
    reps = {n: [] for n in (*MODELS, "altman")}
    for r in range(n_rep):
        a, b, ya, yb = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=1000 + r)
        for n in MODELS:
            reps[n].append(float(roc_auc_score(yb, build_model(n, best_params[n], quick).fit(a, ya).predict_proba(b)[:, 1])))
        zt, fill = altman_z(a)
        reps["altman"].append(float(roc_auc_score(yb, -altman_z(b, fill)[0])))
    summary = {n: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(np.min(v)), "max": float(np.max(v)), "all": v}
               for n, v in reps.items()}
    _log("repeated splits AUC: " + ", ".join(f"{n}={s['mean']:.3f}±{s['sd']:.3f}" for n, s in summary.items()))
    return _jsonable(table), summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", choices=MODELS, default=None, help="override the champion (default: best CV AUC of LR/RF)")
    ap.add_argument("--quick", action="store_true", help="tiny grids/trees for a smoke test")
    ap.add_argument("--out", default=str(config.REPORT_DIR), help="output directory")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()
    run(args.champion, args.quick, args.out, args.skip_robustness)


if __name__ == "__main__":
    main()
