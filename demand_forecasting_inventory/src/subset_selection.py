"""Phase 6 - reproducible subset selection.

Algorithm (documented in reports/subset_selection.md; **no validation/test sales or forecast
performance is used**)::

    1. Split: chronological 70/15/15 split of the 1,941 observed days (splits.py).
    2. Stores: in each state (CA, TX, WI) take the store with the largest TRAIN-period unit sales.
    3. Eligibility (structural data-validity filter, TRAIN period only): the item-store must be *listed*
       (price row present) on every TRAIN day starting at d_1.  Zero sales of an unlisted item are not
       demand, so launches during training are excluded.  A post-hoc check (not a selection input) verifies
       that the selected series are also listed on every validation/test day (no delisting), because the
       pipeline has no assortment model; in this dataset that holds for 100 % of the eligible pool.
    4. Tiers: eligible series of the three stores are pooled and ranked by TRAIN mean daily sales.
       Percentile bands (config.TIER_BANDS) define low / medium / high demand tiers.
    5. Sampling: for each store x tier draw ITEMS_PER_TIER_PER_STORE series at random (numpy default_rng(SEED)),
       preferring different categories and never re-using an item that was already drawn.
    6. Descriptive statistics (ADI, CV^2, Syntetos-Boylan class) are computed on TRAIN data and stored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data_loading as dl
from .splits import Split


def sb_class(adi: float, cv2: float) -> str:
    """Syntetos-Boylan-Croston categorisation of a demand series."""
    if adi < C.ADI_CUTOFF:
        return "smooth" if cv2 < C.CV2_CUTOFF else "erratic"
    return "intermittent" if cv2 < C.CV2_CUTOFF else "lumpy"


def demand_profile(y: np.ndarray) -> dict:
    """Mean/median/std/CV/min/max/zero share and ADI / CV^2 of a daily demand vector."""
    y = np.asarray(y, dtype=float)
    nz = y[y > 0]
    mean = float(y.mean())
    std = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
    adi = float(len(y) / max(len(nz), 1))
    cv2 = float((nz.std(ddof=0) / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    return {
        "mean": mean, "median": float(np.median(y)), "std": std, "cv": std / mean if mean > 0 else float("nan"),
        "min": float(y.min()), "max": float(y.max()), "zero_share": float((y == 0).mean()),
        "adi": adi, "cv2": cv2, "sb_class": sb_class(adi, cv2),
    }


def select_stores(raw: dl.RawData, split: Split) -> list[str]:
    """One store per state: the largest total TRAIN-period unit sales."""
    tr = raw.Y[:, : split.train_end].sum(axis=1)
    df = pd.DataFrame({"store": raw.ids["store_id"].astype(str).to_numpy(),
                       "state": raw.ids["state_id"].astype(str).to_numpy(), "units": tr})
    tot = df.groupby(["state", "store"])["units"].sum().reset_index()
    chosen = (tot.sort_values(["state", "units"], ascending=[True, False])
                 .groupby("state").head(C.N_STORES_PER_STATE))
    return sorted(chosen["store"].tolist())


def select_subset(raw: dl.RawData, split: Split, seed: int = C.SEED) -> tuple[pd.DataFrame, dict]:
    """Return (selection table, log of candidate counts)."""
    log: dict = {}
    stores = select_stores(raw, split)
    log["stores"] = stores
    store_ids = raw.ids["store_id"].astype(str).to_numpy()
    listed = raw.listed_matrix()
    listed_all_train_days = listed[:, : split.train_end].all(axis=1)        # step 3 (TRAIN information only)
    in_store = np.isin(store_ids, stores)
    eligible = in_store & listed_all_train_days
    log["n_series_in_selected_stores"] = int(in_store.sum())
    log["n_eligible_listed_all_train_days"] = int(eligible.sum())
    log["eligible_by_store"] = {s: int((eligible & (store_ids == s)).sum()) for s in stores}
    # post-hoc validity diagnostic over the whole eligible pool (NOT used to select anything)
    still_listed = listed[:, split.train_end:].all(axis=1)
    log["n_eligible_also_listed_after_train"] = int((eligible & still_listed).sum())
    log["share_eligible_also_listed_after_train"] = float((eligible & still_listed).sum() / max(eligible.sum(), 1))

    train_mean = raw.Y[:, : split.train_end].mean(axis=1)
    pool = np.flatnonzero(eligible)
    pool_means = train_mean[pool]
    q = {tier: (float(np.quantile(pool_means, lo)), float(np.quantile(pool_means, hi)))
         for tier, (lo, hi) in C.TIER_BANDS.items()}
    log["tier_thresholds_mean_daily_sales"] = q
    log["pool_mean_quantiles"] = {f"p{int(p*100)}": float(np.quantile(pool_means, p)) for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99)}

    rng = np.random.default_rng(seed)
    used_items: set[str] = set()
    cats = raw.ids["cat_id"].astype(str).to_numpy()
    items = raw.ids["item_id"].astype(str).to_numpy()
    chosen_rows: list[tuple[int, str]] = []
    cell_counts = {}
    for store in stores:
        for tier in C.TIER_ORDER:
            lo, hi = q[tier]
            cand = [r for r in pool if store_ids[r] == store and lo <= train_mean[r] <= hi and items[r] not in used_items]
            cell_counts[f"{store}|{tier}"] = len(cand)
            order = list(rng.permutation(cand))
            picks: list[int] = []
            for r in order:                                                      # first pick: first in random order
                picks.append(int(r)); break
            for r in order:                                                      # second pick: prefer a different category
                if len(picks) >= C.ITEMS_PER_TIER_PER_STORE:
                    break
                if int(r) not in picks and cats[r] not in {cats[p] for p in picks}:
                    picks.append(int(r))
            for r in order:                                                      # fallback: any remaining candidate
                if len(picks) >= C.ITEMS_PER_TIER_PER_STORE:
                    break
                if int(r) not in picks:
                    picks.append(int(r))
            if len(picks) < C.ITEMS_PER_TIER_PER_STORE:
                raise RuntimeError(f"not enough candidates for {store}/{tier}")
            for r in picks:
                used_items.add(items[r])
                chosen_rows.append((r, tier))
    log["candidates_per_cell"] = cell_counts

    chosen_idx = np.array([r for r, _ in chosen_rows])
    if not still_listed[chosen_idx].all():
        raise RuntimeError("a selected series is delisted after the training period - the pipeline has no "
                           "assortment model; extend the eligibility rule or handle delistings explicitly")
    log["selected_all_listed_after_train"] = True

    price_day = raw.price_matrix()
    rows = []
    for r, tier in chosen_rows:
        y_tr = raw.Y[r, : split.train_end]
        prof = demand_profile(y_tr)
        meta = raw.ids.iloc[r]
        px = price_day[r, : split.train_end]
        rows.append({
            "series_id": f"{meta['item_id']}_{meta['store_id']}",
            "raw_row": int(r), "item_id": str(meta["item_id"]), "store_id": str(meta["store_id"]),
            "state_id": str(meta["state_id"]), "dept_id": str(meta["dept_id"]), "cat_id": str(meta["cat_id"]),
            "tier": tier, **{f"train_{k}": v for k, v in prof.items()},
            "train_mean_price": float(np.nanmean(px)), "last_train_price": float(px[-1]),
        })
    sel = pd.DataFrame(rows)
    sel["tier"] = pd.Categorical(sel["tier"], categories=list(C.TIER_ORDER), ordered=True)
    sel = sel.sort_values(["store_id", "tier", "item_id"]).reset_index(drop=True)
    sel["tier"] = sel["tier"].astype(str)
    sel["intermittency"] = np.where(sel["train_adi"] >= C.ADI_CUTOFF, "intermittent", "regular")
    sel.insert(0, "series_idx", np.arange(len(sel)))
    return sel, log


def store_outage_matrix(raw: dl.RawData) -> pd.DataFrame:
    """(n_days x n_stores) boolean: store-wide closure/outage (total sales < 5 % of a centred 29-day rolling median).

    DIAGNOSTIC only (uses same-day sales); never used as a model feature.
    """
    stores = sorted(raw.ids["store_id"].astype(str).unique())
    sid = raw.ids["store_id"].astype(str).to_numpy()
    tot = pd.DataFrame({s: raw.Y[sid == s].sum(axis=0) for s in stores}, index=raw.dates)
    med = tot.rolling(29, center=True, min_periods=10).median()
    return tot < 0.05 * med
