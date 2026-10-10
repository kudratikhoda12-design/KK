"""Expected Loss = PD x LGD x EAD, risk scores and the EL-ranked watchlist.

The public dataset has no loan-level exposure or recovery data, so EAD and LGD are
transparent *proxies* derived from the balance sheet (parameters in ``config.LossConfig``):

* EAD  = bank_share_of_liabilities x total liabilities, where
         total liabilities = (TL/TA, Attr2) x (total assets = exp(Attr29)).
         Monetary units are those of the source data; only relative size matters here.
* LGD  = 1 - min(1, recovery_haircut x asset coverage), asset coverage = TA/TL,
         floored/capped. A more levered borrower leaves less collateral per unit of claim.

Neither proxy uses the default label, so they cannot leak the outcome. Replace both with
portfolio data when available - the rest of the engine only needs the two vectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LossConfig

TL_TA, LN_TOTAL_ASSETS = "Attr2", "Attr29"


def _tl_to_ta(X: pd.DataFrame, cfg: LossConfig) -> np.ndarray:
    lo, hi = cfg.tl_to_ta_clip
    ratio = X[TL_TA].to_numpy(float)
    return np.clip(np.where(np.isnan(ratio), np.nanmedian(ratio), ratio), lo, hi)


def estimate_ead(X: pd.DataFrame, cfg: LossConfig = LossConfig()) -> np.ndarray:
    total_assets = np.exp(X[LN_TOTAL_ASSETS].fillna(X[LN_TOTAL_ASSETS].median()).to_numpy(float))
    return cfg.bank_share_of_liabilities * _tl_to_ta(X, cfg) * total_assets


def estimate_lgd(X: pd.DataFrame, cfg: LossConfig = LossConfig()) -> np.ndarray:
    coverage = 1.0 / _tl_to_ta(X, cfg)
    recovery = np.minimum(1.0, cfg.recovery_haircut * coverage)
    return np.clip(1.0 - recovery, cfg.lgd_floor, cfg.lgd_cap)


def expected_loss(pd_, lgd, ead) -> np.ndarray:
    return np.asarray(pd_, float) * np.asarray(lgd, float) * np.asarray(ead, float)


def pd_to_score(pd_, base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0) -> np.ndarray:
    """Classic scorecard scaling: `base_score` at good:bad odds of `base_odds`, +`pdo` points doubles the odds."""
    pd_ = np.clip(np.asarray(pd_, float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    return base_score - factor * np.log(base_odds) + factor * np.log((1 - pd_) / pd_)


GRADE_EDGES = (0.01, 0.025, 0.05, 0.10, 0.25)
GRADE_LABELS = ("A", "B", "C", "D", "E", "F")


def risk_grade(pd_) -> np.ndarray:
    return np.array(GRADE_LABELS, dtype=object)[np.searchsorted(GRADE_EDGES, np.asarray(pd_, float), side="right")]


def build_watchlist(X: pd.DataFrame, pd_, threshold: float, y=None, cfg: LossConfig = LossConfig()) -> pd.DataFrame:
    """All firms ranked by Expected Loss (largest first). ``firm_id`` is the row id in the loaded data.

    ``realized_default`` is included only for back-testing; it is unknown in live use.
    """
    pd_ = np.asarray(pd_, float)
    lgd, ead = estimate_lgd(X, cfg), estimate_ead(X, cfg)
    el = expected_loss(pd_, lgd, ead)
    wl = pd.DataFrame({
        "firm_id": X.index.to_numpy(), "pd": pd_, "score": pd_to_score(pd_), "grade": risk_grade(pd_),
        "lgd": lgd, "ead": ead, "expected_loss": el, "flagged": pd_ >= threshold,
    })
    wl["pd_rank"] = wl["pd"].rank(ascending=False, method="first").astype(int)
    if y is not None:
        wl["realized_default"] = np.asarray(y).astype(int)
    wl = wl.sort_values("expected_loss", ascending=False, kind="stable").reset_index(drop=True)
    wl.insert(0, "el_rank", np.arange(1, len(wl) + 1))
    wl["el_share"] = wl["expected_loss"] / wl["expected_loss"].sum()
    wl["cum_el_share"] = wl["el_share"].cumsum()
    return wl


def loss_capture(realized_loss, ranking_score, fractions) -> dict[float, float]:
    """Share of total realised loss found in the top q% of firms when ranked by ``ranking_score``."""
    realized_loss, ranking_score = np.asarray(realized_loss, float), np.asarray(ranking_score, float)
    order = np.argsort(-ranking_score, kind="stable")
    cum = np.cumsum(realized_loss[order])
    total = cum[-1]
    n = len(order)
    return {q: float(cum[max(int(np.ceil(q * n)) - 1, 0)] / total) if total > 0 else float("nan") for q in fractions}


def default_capture(y, ranking_score, fractions) -> dict[float, float]:
    return loss_capture(np.asarray(y, float), ranking_score, fractions)
