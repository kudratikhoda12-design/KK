# Expected Loss-Based Credit Risk Scorecard

A corporate credit-risk engine built end to end on real bankruptcy data: **SQC control-chart ratio screening**, logistic regression and random forest PD models, **calibrated PDs**, a **cost-sensitive decision threshold**, and an **Expected Loss-ranked watchlist** for portfolio prioritisation.

**Full write-up (21 pages, raw data to final result): [`reports/Credit_Risk_Scorecard_Report.pdf`](reports/Credit_Risk_Scorecard_Report.pdf)**

## Results (hold-out portfolio: 1,463 firms, 102 defaults, one-year horizon)

| Model | AUC-ROC | 95% CI |
|---|---|---|
| Altman Z'' score (benchmark) | 0.759 | 0.701 - 0.817 |
| Logistic regression | 0.856 | 0.808 - 0.897 |
| **Random forest (champion)** | **0.923** | 0.890 - 0.950 |
| Gradient boosting (challenger, outside RF/LR scope) | 0.955 | 0.930 - 0.975 |

- Over 10 random 75/25 splits the random forest averages 0.923 (sd 0.013) and never reached 0.947 (best 0.944). Gradient boosting did in 9 of 10. Quote the number that matches the model you describe; the report has two resume wordings that are backed by this evidence.
- SQC screening keeps 44 of 64 ratios. The cost-optimal threshold (PD >= 0.085, not 0.5) flags 17% of firms and catches 84% of defaulters.
- Ranking by Expected Loss puts 88% of realised loss in the top 10% of firms (PD-only ranking: 84%). That difference is **not** statistically established on this hold-out (bootstrap interval includes zero).

**LGD and EAD are proxies** built from balance-sheet ratios because the public data has no loan-level information, so EL and cost figures illustrate the method rather than forecast a real bank. Details and other limitations are in Section 11 of the report.

## Pipeline

```
raw ARFF -> dedupe/audit -> stratified 75/25 split
  -> SQC screening (healthy-firm control limits, detection vs false-alarm test, BH-FDR, redundancy pruning)
  -> LR / RF (+ GBM challenger), tuned by 5-fold CV on the training split
  -> cross-fitted PD calibration (Platt / isotonic by Brier score)
  -> cost-sensitive threshold on out-of-fold PDs (miss = LGD x EAD, false alarm = margin x EAD)
  -> Expected Loss = PD x LGD x EAD -> ranked watchlist -> loss-capture back-test
  -> robustness: five forecast horizons, ten random splits
```

Every preprocessing step is a scikit-learn pipeline stage, so it is re-fitted inside each CV fold and never sees validation or hold-out firms.

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
python -m credit_risk.data        # download UCI archive, verify SHA-256
python -m credit_risk.pipeline    # full run (~10 min), writes reports/
python -m credit_risk.report      # rebuild the PDF from reports/metrics.json
python -m pytest                  # 35 unit tests; RUN_SLOW=1 adds an end-to-end smoke test
```

`python -m credit_risk.pipeline --champion hgb` drives calibration, thresholds and the watchlist with the boosting model instead of the RF/LR champion.

## Layout

| Path | Purpose |
|---|---|
| `src/credit_risk/screening.py` | SQC control-chart screener (scikit-learn transformer) |
| `src/credit_risk/models.py` | LR / RF / GBM pipelines and search grids |
| `src/credit_risk/calibration.py` | cross-fitted calibration, base-rate shift |
| `src/credit_risk/threshold.py` | cost curve, bootstrap-stable optimal threshold, Bayes rule |
| `src/credit_risk/expected_loss.py` | LGD / EAD proxies, EL, scorecard points, grades, watchlist |
| `src/credit_risk/pipeline.py`, `report.py`, `plots.py` | orchestration, PDF, figures |
| `src/credit_risk/config.py` | every assumption and tunable in one place |
| `reports/` | `metrics.json`, `watchlist_full.csv`, `watchlist_top25.csv`, `sqc_screening.csv`, tuning tables, figures, PDF |
| `tests/` | unit tests (screener, thresholds, EL, calibration, metrics) |

## Data

[Polish companies bankruptcy data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), UCI Machine Learning Repository, CC BY 4.0. Zieba, M., Tomczak, S. K., Tomczak, J. M. (2016), *Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction*, Expert Systems with Applications 58, 93-101. The raw files are downloaded on demand and are not committed.
