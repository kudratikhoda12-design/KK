"""Single place for every tunable choice and every modelling assumption.

The public dataset has no loan-level exposure or recovery data, so EAD and LGD are
*proxies* built from balance-sheet ratios (see ``expected_loss.py``). All of their
parameters live here so they can be challenged or replaced with real portfolio data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "polish"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# UCI "Polish companies bankruptcy data" (CC BY 4.0), 64 financial ratios per firm.
DATASET_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
DATASET_SHA256 = "17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9"

# "5year.arff" = ratios from the 5th year of the forecast window, label = bankrupt
# within 1 year -> the standard 1-year-horizon PD set-up. The other files are used
# only for the robustness table.
PRIMARY_HORIZON = 5
ALL_HORIZONS = (1, 2, 3, 4, 5)

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5


@dataclass(frozen=True)
class SQCConfig:
    tail_prob: float = 0.025   # false-alarm probability per tail of the control chart (probability limits)
    max_missing: float = 0.40  # drop ratios with more missing values than this
    fdr: float = 0.05          # Benjamini-Hochberg q for "defaulters signal more than healthy firms"
    max_corr: float = 0.95     # |Spearman| above which the weaker of two ratios is dropped


@dataclass(frozen=True)
class LossConfig:
    """Proxy loss model. Replace with portfolio data when available."""
    bank_share_of_liabilities: float = 0.10  # EAD = share * total liabilities
    tl_to_ta_clip: tuple[float, float] = (0.05, 3.0)  # tame data-entry outliers (raw range -430..72)
    recovery_haircut: float = 0.40   # recovery = haircut * asset coverage (TA/TL), capped at 1
    lgd_floor: float = 0.15
    lgd_cap: float = 0.90
    margin: float = 0.03             # foregone net margin on a good client wrongly flagged (cost of a false alarm)


@dataclass(frozen=True)
class PipelineConfig:
    sqc: SQCConfig = SQCConfig()
    loss: LossConfig = LossConfig()
    watchlist_top_n: int = 25
    capture_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    n_bootstrap: int = 1000
