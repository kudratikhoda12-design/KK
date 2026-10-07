# Technical reviewer guide

A map for someone who wants to check this project quickly. Paths are relative to the repository root.

**The claim being made:** BTCUSDT's next-hour direction is slightly predictable out of sample (test AUC 0.545, 95% CI 0.538–0.552), but the edge is about 0.19 bp per trade side against roughly 7 bp of cost, so the strategy lost 92% net on an untouched 2024–2026 test. The negative economic result is the point.

---

## If you have 2 minutes

1. `README.md` — the summary table at the top.
2. `reports/Final_Gravia_Project_Report.pdf` — Section 1 (Executive Summary) and Table 5 (backtest economics).
3. `reports/Gravia_Project_One_Page_Summary.pdf` — the one-page version.

## If you have 10 minutes

| Look at | What to check |
|---|---|
| `reports/Final_Gravia_Project_Report.pdf`, Sections 5, 8, 11, 12 | Leakage control, validation design, backtest, why prediction did not become profit |
| `src/features.py` → `minute_features`, `build_dataset` | Features use only candles closed by *t* (`mfeat.shift(1)`); entry at the open of *t*+1, exit at *t*+61 |
| `src/validation.py` → `walk_forward_folds` | Expanding folds; the purge is `train = (idx < start) & (data["exit_time"] <= start)` |
| `src/backtest.py` → `run_backtest` | Costs charged on |Δposition|; positions are not carried across missing hours; funding applied over (entry, exit] |
| `reports/results.json` → `test` | Every headline number of the test period |
| `reports/selection.json` | The frozen choice (LightGBM config 1, k = 1.5, long/flat) and the reason string |

## If you want to check that the test was not tuned

* `reports/research_design_preregistration.md` — the design, written before any data was downloaded (commit `b312d58`), and the deviation log (§12) listing every later change.
* `git log --oneline` — `231634d` commits `selection.json` (frozen 21:23 UTC) and contains no test output; `bd2963f` adds the test results. The single TEST row in `reports/tables/experiment_log.csv` is logged at 21:24 UTC.
* `src/pipeline.py` → `validation_stage` asserts that no fold touches 2024; `scripts/run_pipeline.py` → `stage_test` refuses to run without `selection.json`.
* `reports/tables/experiment_log.csv` — 44 logged runs (6 models, 10 signal settings, 3 baselines, 1 test, 23 robustness, 1 ETH).

## If you want to check leakage

* `src/features.py` → `perturbation_leakage_check`: replaces all data at or after *t* with noise and recomputes; result in `reports/tables/leakage_perturbation_real_data.csv` (0 of 16 features changed at 24 real times; the target changed every time).
* `tests/test_features_leakage.py` — alignment, execution lag, backward-looking windows, no filling.
* `src/models.py` → `make_lr` — scaling inside a scikit-learn `Pipeline`, so it is fitted per training window.

## If you want to inspect data handling

* `src/data.py` → `detect_time_unit`, `read_kline_zip` — milliseconds vs microseconds detected per file (Binance switched in 2025).
* `src/quality.py` → `audit`, `gap_analysis`, `build_processed` — the 18 checks, phase-shifted candle detection, and a regular 1-minute grid with gaps left empty.
* `reports/tables/BTCUSDT_audit_dq_table.csv`, `BTCUSDT_audit_gaps_top25.csv`, `BTCUSDT_audit_missing_by_year.csv`.
* `tests/test_data_quality.py` — includes `test_phase_shifted_candles_become_missing_not_snapped`.

## If you want to understand the economics

* `src/backtest.py` (`positions_from_probs`, `run_backtest`, `funding_between`, `capacity_check`) and `src/risk.py` → `performance`.
* `reports/tables/test_benchmarks.csv` — strategy vs buy-and-hold vs 1-hour momentum.
* `reports/tables/test_calibration.csv` — mean next-hour return by predicted-probability decile (−1.3 to +1.9 bp).
* `reports/tables/stress_costs.csv` — cost ×0/×1/×2/×3 and extra slippage; `robustness_spot_fee_check.csv` — the pre-registered spot-fee variant.
* Break-even cost: `gross_sum / (cost_sum / 0.0007)` from `results.json` → `test.performance` = 0.06998 / 3,668 = 0.19 bp per side.

## If you want to inspect robustness and risk

* `src/pipeline.py` → `robustness_stage`, `permutation_test`; output `reports/tables/robustness_variants.csv` (23 rows) and `results.json` → `robustness.permutation`.
* `src/stress_test.py`; outputs `reports/tables/stress_*.csv`.
* `src/risk.py` (`var_es`, `kupiec_pof`, `bootstrap_paths`) and `src/stats.py` (`stationary_bootstrap_indices`, `hac_mean_test`, `holm`); outputs `reports/tables/risk_*.csv`.
* ETH replication: `scripts/run_pipeline.py` → `stage_second_asset`; outputs `reports/tables/second_asset_ETHUSDT_*.csv`.

## If you want to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                          # 26 tests, no data needed
python scripts/smoke_test_synthetic.py       # every stage on synthetic data, temporary folder only
```

The full pipeline needs the Binance data first; see `reproduction/README.md`. A clean extraction of the delivery package, run end to end on the study's input files, reproduced every value in `results.json` and every pipeline table exactly (checked on 7 October 2026). Notebooks `02`–`10` display the saved outputs of each stage without re-running anything.

## Known limitations (stated up front)

* Spread (1 bp) and slippage (1 bp) are assumptions; only the 5 bp fee is sourced. Break-even is below the fee alone.
* LightGBM passed the pre-registered selection rule exactly at its threshold (7 of 10 folds), and on the test it lost more than logistic regression.
* The permutation test uses logistic regression, not the selected LightGBM.
* No prediction-market, order-book, fill or wallet data was used; no live trading.
* The credit-risk module (`credit_module/`) is the author's earlier project; its numbers come from its original report, and its LightGBM part is not reproducible from this repository.
