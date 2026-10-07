"""Central configuration: paths, seeds, split fractions, horizons and (hypothetical) cost assumptions.

Every tunable constant of the project lives here so that results are reproducible and the
assumptions are visible in one place.  Nothing in this file is derived from validation/test
performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_MOD = ROOT / "outputs" / "models"
REPORTS = ROOT / "reports"

RAW_FILES = {
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
    "sales_validation": "sales_train_validation.csv",
    "sales_evaluation": "sales_train_evaluation.csv",
}

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED = 42
N_JOBS = 4

# --------------------------------------------------------------------------------------
# Dataset provenance (official Kaggle M5 requires credentials -> legitimate public mirrors)
# --------------------------------------------------------------------------------------
PRIMARY_MIRROR = "kashif/M5"
CROSSCHECK_MIRROR = "denephew/M5_Forecasting"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
HF_API = "https://huggingface.co/api/datasets/{repo}?blobs=true"

# --------------------------------------------------------------------------------------
# Chronological split (fractions of the available modelling window)
# --------------------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15            # test = remainder (~15 %)

# --------------------------------------------------------------------------------------
# Forecasting set-up
# --------------------------------------------------------------------------------------
SEASONAL_PERIOD = 7
PRIMARY_HORIZONS = (7, 14, 28)
# H = 3 is NOT a headline horizon: it is trained only because a 3-day lead-time demand
# forecast is needed by the inventory layer (lead time L = 3).
HORIZONS = (3, 7, 14, 28)
MAX_HORIZON = max(HORIZONS)
MA_WINDOWS = (7, 14, 28)
MIN_HISTORY = 28           # days of history needed before the first usable forecast origin

# --------------------------------------------------------------------------------------
# Subset selection (uses TRAIN-period information only; see subset_selection.py)
# --------------------------------------------------------------------------------------
N_STORES_PER_STATE = 1
ITEMS_PER_TIER_PER_STORE = 2
# percentile bands of train-period mean daily sales inside the pooled eligible pool
TIER_BANDS = {"low": (0.10, 0.35), "medium": (0.45, 0.65), "high": (0.85, 0.97)}
TIER_ORDER = ("high", "medium", "low")
# Syntetos-Boylan-Croston demand categorisation cut-offs
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49

# --------------------------------------------------------------------------------------
# Inventory model
# --------------------------------------------------------------------------------------
LEAD_TIMES = (3, 7, 14)                              # HYPOTHETICAL supplier lead times (days)
SERVICE_LEVELS = (0.90, 0.95, 0.98)
Z_VALUES = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054}   # as prescribed by the project brief
DAYS_PER_YEAR = 365


@dataclass(frozen=True)
class CostAssumptions:
    """HYPOTHETICAL cost structure.  None of these numbers are observed business costs.

    unit_cost_ratio        : unit purchase cost = ratio x last observed selling price
    holding_rate           : annual holding cost = rate x unit cost (capital + space + shrink)
    order_cost             : fixed cost per replenishment order line (USD)
    stockout_multipliers   : lost-sales penalty per unit short = multiplier x unit margin
                              (margin = price - unit cost).  LOW < 1 means part of the demand is
                              recovered (substitution / back-order); HIGH > 1 adds goodwill loss.
    """

    unit_cost_ratio: float = 0.70
    holding_rate: float = 0.25
    order_cost: float = 1.00
    stockout_multipliers: tuple[tuple[str, float], ...] = (("LOW", 0.25), ("MEDIUM", 1.0), ("HIGH", 3.0))

    def multipliers(self) -> dict[str, float]:
        return dict(self.stockout_multipliers)


COSTS = CostAssumptions()
SCENARIOS = tuple(COSTS.multipliers().keys())

# --------------------------------------------------------------------------------------
# Prediction intervals / uncertainty
# --------------------------------------------------------------------------------------
INTERVAL_LEVELS = (0.80, 0.90)

# --------------------------------------------------------------------------------------
# XGBoost tuning budget (random search on the validation period ONLY)
# --------------------------------------------------------------------------------------
XGB_SEARCH_ITER = 24
XGB_MAX_ESTIMATORS = 600
XGB_EARLY_STOPPING = 40
XGB_SHAP_MAX_ROWS = 4000


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROC, OUT_FIG, OUT_TAB, OUT_MOD, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
