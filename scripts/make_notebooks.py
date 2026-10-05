"""Generate the market-module notebooks (02-10) from one specification.

Each notebook: question -> why -> method -> run the stage -> show its saved
tables/figures. Executing them in order reproduces the whole analysis:

    python scripts/make_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/0[2-9]*.ipynb notebooks/10_*.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)
import run_pipeline as rp
from src import config
T, F = config.TABLES_DIR, config.FIGURES_DIR
SYMBOL = config.SYMBOL
RUN_STAGE = True   # set False to only display previously saved outputs
def show(name, **kw):
    display(Markdown(f"**{name}**")); display(pd.read_csv(T / name, **kw))
def fig(name):
    display(Image(filename=str(F / name)))
def results(section):
    return json.loads((config.REPORTS_DIR / "results.json").read_text()).get(section, {})"""

NOTEBOOKS = [
    ("02_market_data_health", "audit", """# 02 - Market data: acquisition and data-health audit

**Question:** Is the Binance BTCUSDT 1-minute data complete, consistent and correctly timestamped, and what must be treated before research?

**Why:** Every later conclusion inherits the data's problems. Known traps: the switch from millisecond to microsecond timestamps on 2025-01-01, exchange-downtime gaps, revised archive files and duplicate or conflicting rows.

**Method:** `src/quality.py` checks schema, timestamps, frequency, OHLC consistency, volume, duplicates, missing values and outliers. Treatments are rule-based and fixed in advance:
- exact duplicate rows are kept once;
- conflicting duplicates and impossible candles become missing;
- gaps are never filled;
- extreme but consistent moves are kept and flagged.

Data must already be downloaded; see `data/README.md`.""",
     ["SYMBOL + '_audit_dq_table.csv'", "SYMBOL + '_audit_treatments.csv'", "SYMBOL + '_audit_missing_by_year.csv'",
      "SYMBOL + '_audit_gaps_top25.csv'", "SYMBOL + '_audit_outliers_top50.csv'", "SYMBOL + '_audit_file_summary.csv'"],
     [], "audit"),
    ("03_market_eda", "eda", """# 03 - Exploratory data analysis

Each table or figure answers one question:
1. How are returns distributed at 1 min / 15 min / 1 h / 1 day, and how heavy are the tails?
2. Are hourly returns autocorrelated, i.e. predictable from their own past? Does volatility cluster? *(development data only)*
3. How are volume and returns related, both in the same hour and the next hour? *(development data only)*
4. Is there intraday seasonality in volatility and volume?
5. Which volatility and trend regimes occurred, and when were the extreme hours?

**Rule:** anything relating the past to *future* returns uses development data (before 2024) only, so the test period stays untouched.""",
     ["'eda_return_distribution.csv'", "'eda_dependence_dev.csv'", "'eda_volume_return_dev.csv'",
      "'eda_intraday_dev.csv'", "'eda_regimes_by_year.csv'", "'eda_extreme_hours.csv'"],
     ["eda_price_volatility.png", "eda_hourly_return_qq.png", "eda_acf_dev.png", "eda_intraday_dev.png"], "eda"),
    ("04_hypothesis_features", "features", """# 04 - Hypotheses, features and leakage audit

The hypotheses (H1 momentum/reversal, H2 volume/order flow, H3 regime dependence, H4 combination, H5 economic value) were **pre-registered before the data was seen**; see `reports/research_design_preregistration.md`.

**Features:** 16 scale-free features in five groups (returns, momentum, volatility, volume/order flow, candle). Each is computed only from candles that closed by the decision time *t*.

**Leakage audit:**
1. A feature-timing table: feature → latest information used → prediction time → execution time.
2. A perturbation test on the real data: replace everything at or after *t* with random values and recompute. A feature that changes would have been looking into the future.

**H1/H2 test:** Spearman rank correlation between each feature and the next-hour return on development data, a block-bootstrap p-value, and Holm correction across all 16 features.""",
     ["'feature_timing_table.csv'", "'leakage_perturbation_real_data.csv'", "'h1_h2_univariate_dev.csv'",
      "'feature_summary.csv'"], [], "features"),
    ("05_modeling", "validation", """# 05 - Baselines and models (logistic regression vs LightGBM)

**Models** predict P(next-60-minute return > 0):
- base rate (no information);
- logistic regression with C ∈ {0.01, 0.1, 1};
- two conservative LightGBM configurations.

**Baselines for trading:** buy-and-hold, and the sign of the last 1 h or 24 h return.

**Pre-registered simplicity rule:** LightGBM is used only if its mean validation AUC beats logistic regression by more than 0.005 **and** its log-loss is lower in at least 70% of folds.

Predictive metrics (AUC, log-loss, Brier, accuracy, precision, recall) do **not** show profitability on their own; that is decided by the backtest.""",
     ["'validation_models.csv'", "'validation_baselines.csv'"], [], "validation"),
    ("06_walk_forward_validation", None, """# 06 - Walk-forward validation and signal selection

**Why not random K-fold?** It trains on the future and puts near-identical neighbouring hours on both sides of the split, so it overstates skill.

**Design:**
- expanding training window;
- 6-month validation blocks from 2019H1 to 2023H2;
- purging of training rows whose label overlaps the block.

The threshold k and the long/short vs long/flat choice are selected here, on validation only, by mean net Sharpe across folds at base costs, among configurations trading at least 100 times a year. **The test period (2024 onwards) is not touched.**""",
     ["'validation_folds.csv'", "'validation_signal_grid.csv'"], [], None),
    ("07_backtesting", "test", """# 07 - Final test: frozen strategy on the untouched period (2024-01 onwards)

Run **once** with the frozen selection in `reports/selection.json`. The model is refitted every 6 months on all earlier data; no parameter changes.

**Event timing:** decision at *t* (candles closed by *t*), execution at the open of *t*+1 min, exit at the open of *t*+61 min.

**Costs per side:** 5 bp fee (sourced) + 1 bp half-spread (assumed) + 1 bp slippage (assumed), charged on every change in position.

Gross and net results are shown side by side and compared with buy-and-hold and a momentum baseline.""",
     ["'test_benchmarks.csv'", "'test_per_block.csv'", "'test_calibration.csv'", "'test_confusion.csv'",
      "'test_importance.csv'", "'test_drift.csv'", "'test_backtest_sample_rows.csv'"],
     ["test_equity.png", "test_calibration.png", "test_decile_returns.png"], "test"),
    ("08_risk_analysis", "risk", """# 08 - Risk analysis: VaR, Expected Shortfall, VaR backtest, Monte Carlo

- **VaR / ES:** 1-day horizon, 95% and 99%, on daily strategy returns. Three methods: historical, parametric normal and Student-t. The comparison shows how much a normal assumption understates fat tails.
- **Kupiec test:** a rolling one-year historical VaR forecast, with exceptions counted against the expected rate. This mirrors the PD backtest in the credit module.
- **Monte Carlo:** a stationary block bootstrap of real daily returns, giving one-year outcomes and drawdowns.

Why not the credit module's Gaussian one-factor model? Default dependence is unobserved, but strategy returns are observed directly, so resampling them keeps their real fat tails and volatility clustering.""",
     ["'risk_var_es_daily.csv'", "'risk_kupiec.csv'", "'risk_bootstrap_mc.csv'"], ["risk_mc_one_year.png"], "risk"),
    ("09_stress_testing", "stress", """# 09 - Stress testing

How do P&L and risk change when conditions get worse?
- **Costs:** ×0 (gross), ×1, ×2, ×3, plus extra slippage of 2 bp and 5 bp.
- **Volatility shock:** every move doubled.
- **Signal degradation:** 10%, 25% or 50% of signals flipped.
- **Historical:** the 10 worst BTC days in the test period.
- **Hypothetical gap:** an instant −10%, −20% or −30% move while positioned.
- **Regimes:** volatility and trend.""",
     ["'stress_costs.csv'", "'stress_volatility.csv'", "'stress_signal_degradation.csv'", "'stress_worst_btc_days.csv'",
      "'stress_gap_shock.csv'", "'stress_regime_vol.csv'", "'stress_regime_trend.csv'"], ["stress_costs.png"], "stress"),
    ("10_robustness", "robustness", """# 10 - Robustness: trying to kill the signal

Every variant is run on the test period **after** the frozen result is recorded. The variants are reported, never used to change the selection. They cover:
- thresholds and position modes;
- removing each feature group;
- swapping the model;
- execution lag of 0 or 5 min;
- decision times offset to :15, :30 and :45;
- horizons of 15 and 30 min;
- a label-permutation test.

The full experiment count, including failures, is in `reports/tables/experiment_log.csv` and feeds the Deflated Sharpe Ratio.""",
     ["'robustness_variants.csv'", "'experiment_log.csv'"], ["robustness_variants.png"], "robustness"),
]


def build(name, stage, intro, tables, figures, run_stage):
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(SETUP)]
    if run_stage:
        cells.append(nbf.v4.new_code_cell(f"if RUN_STAGE:\n    rp.stage_{run_stage}(SYMBOL)"))
    for t in tables:
        cells.append(nbf.v4.new_code_cell(f"show({t})"))
    for f in figures:
        cells.append(nbf.v4.new_code_cell(f"fig('{f}')"))
    if run_stage == "robustness":
        cells.append(nbf.v4.new_code_cell("results('robustness').get('permutation')"))
    cells.append(nbf.v4.new_markdown_cell(
        "**Interpretation:** see the matching section of `reports/final_report.md`, written from these outputs."))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")


if __name__ == "__main__":
    for spec in NOTEBOOKS:
        build(*spec[:5], spec[5])
    print("notebooks written")
