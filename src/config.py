"""Pre-registered research configuration.

Every parameter that could be tuned to make a backtest look better is fixed
here BEFORE the real data has been inspected (see
reports/research_design_preregistration.md and the git history of this file).
Any later change must be logged as a deviation in the final report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # zip files exactly as downloaded from Binance
INTERIM_DIR = DATA_DIR / "interim"    # combined, typed, NOT cleaned
PROCESSED_DIR = DATA_DIR / "processed"  # audited 1-minute grid used for research
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = ROOT / "figures"

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
SECOND_ASSET = "ETHUSDT"        # used ONLY after the BTC analysis is frozen
INTERVAL = "1m"
DATA_START_MONTH = "2017-08"    # first BTCUSDT spot month on Binance
DATA_END_MONTH = "2026-09"      # last complete month at project time (Oct 2026)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

# --------------------------------------------------------------------------
# Chronological split (no random splitting anywhere in the market module)
# --------------------------------------------------------------------------
FIRST_VALIDATION_START = pd.Timestamp("2019-01-01", tz="UTC")
DEV_END = pd.Timestamp("2024-01-01", tz="UTC")       # exclusive: dev = data before this
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")    # final test, untouched until design frozen
VALIDATION_BLOCK_MONTHS = 6     # walk-forward validation blocks inside development
TEST_RETRAIN_MONTHS = 6         # frozen model is re-fitted every 6 months in the test period

# --------------------------------------------------------------------------
# Target and timing
# --------------------------------------------------------------------------
PRIMARY_HORIZON_MIN = 60              # holding period h
ROBUSTNESS_HORIZONS_MIN = (15, 30)    # only these two alternatives are ever tested
EXECUTION_LAG_MIN = 1                 # trade at the open of the minute AFTER the decision minute
ROBUSTNESS_LAGS_MIN = (0, 5)
DECISION_OFFSET_MIN = 0               # decisions at HH:00 UTC
ROBUSTNESS_OFFSETS_MIN = (15, 30, 45)

# A rolling feature is only computed if at least this share of minutes in its
# window is present; otherwise the observation is dropped (never forward-filled).
MIN_WINDOW_COVERAGE = 0.90

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
RANDOM_SEED = 42
LR_C_GRID = (0.01, 0.1, 1.0)
LGBM_CONFIGS = (
    dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=500,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
    dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=1000,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0),
)
# Simplicity rule: LightGBM replaces logistic regression only if it improves
# mean validation AUC by more than this AND has lower log-loss in at least
# LGBM_MIN_FOLD_WIN_SHARE of the validation folds.
LGBM_MIN_AUC_GAIN = 0.005
LGBM_MIN_FOLD_WIN_SHARE = 0.70

# --------------------------------------------------------------------------
# Signal construction (selected on validation folds only)
# --------------------------------------------------------------------------
# Long if p > 0.5 + k*sd, short if p < 0.5 - k*sd, else flat, where sd is the
# standard deviation of the model's predictions on its own training window
# (past information only).
THRESHOLD_K_GRID = (0.0, 0.25, 0.5, 1.0, 1.5)
POSITION_MODES = ("long_short", "long_flat")
MIN_TRADES_PER_YEAR = 100       # a configuration trading less is not eligible

# --------------------------------------------------------------------------
# Costs: fraction of notional, per side (one buy or one sell)
# --------------------------------------------------------------------------
FEE_PERP_TAKER = 0.0005   # SOURCED: Binance USD-M regular-tier taker fee (2026 schedule)
FEE_SPOT_TAKER = 0.0010   # SOURCED: Binance spot regular-tier fee (2026 schedule)
HALF_SPREAD = 0.0001      # ASSUMPTION: no historical spot order book in the dataset
SLIPPAGE = 0.0001         # ASSUMPTION: small notional, market order
BASE_COST_PER_SIDE = FEE_PERP_TAKER + HALF_SPREAD + SLIPPAGE   # 7 bp
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 3.0)   # 0 = gross, shown only for comparison
EXTRA_SLIPPAGE_SCENARIOS = (0.0002, 0.0005)

# --------------------------------------------------------------------------
# Statistics and risk
# --------------------------------------------------------------------------
ALPHA = 0.05
HOURS_PER_YEAR = 24 * 365.25
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK_HOURS = 168     # mean block length (1 week) for hourly series
BOOTSTRAP_BLOCK_DAYS = 10       # mean block length for daily series
VAR_LEVELS = (0.95, 0.99)
PERMUTATION_N = 200


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
