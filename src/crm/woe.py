"""Weight-of-evidence binning with monotonicity constraints (scorecard style).

WOE_b = ln(%bad_b / %good_b), so a higher WOE means riskier and every fitted
logistic coefficient on a WOE variable is expected to be positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SMOOTH = 0.5


def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.crosstab(bins, y)
    t = t.reindex(columns=[0, 1], fill_value=0)
    good, bad = t[0] + SMOOTH, t[1] + SMOOTH
    pg, pb = good / good.sum(), bad / bad.sum()
    out = pd.DataFrame({"n": t[0] + t[1], "bad": t[1]})
    out["dr"] = out["bad"] / out["n"]
    out["woe"] = np.log(pb / pg)
    out["iv"] = (pb - pg) * out["woe"]
    return out


class NumericBinner:
    def __init__(self, n_fine=20, min_share=0.05):
        self.n_fine, self.min_share = n_fine, min_share

    def fit(self, x: pd.Series, y: pd.Series):
        x = x.astype(float)
        nn = x.notna()
        edges = np.unique(np.nanquantile(x[nn], np.linspace(0, 1, self.n_fine + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        sign = np.sign(stats.spearmanr(x[nn], y[nn]).statistic) or 1.0
        # Merge adjacent fine bins until default rates are monotone in the
        # direction of the Spearman trend and every bin holds >= min_share.
        while True:
            b = np.clip(np.searchsorted(edges, x[nn].to_numpy(), side="left") - 1, 0, len(edges) - 2)
            g = pd.DataFrame({"b": b, "y": y[nn]}).groupby("b")["y"].agg(["mean", "size"])
            g = g.reindex(range(len(edges) - 1)).fillna({"size": 0})
            if len(edges) <= 2:
                break
            share = g["size"] / nn.sum()
            diffs = sign * np.diff(g["mean"].values)
            viol = np.where((diffs < 0) | np.isnan(diffs))[0]
            small = np.where(share.values < self.min_share)[0]
            if len(viol) == 0 and len(small) == 0:
                break
            if len(small):
                i = small[0]
                j = i if i < len(edges) - 2 else i - 1   # merge with right neighbour
            else:
                j = viol[0]
            edges = np.delete(edges, j + 1)
        self.edges = edges
        self.sign = sign
        binned = self._bin(x)
        self.table = _woe_table(binned, y)
        miss_n = int((~nn).sum())
        # Rare missing bins borrow the overall WOE (0) to avoid noise.
        if miss_n and miss_n / len(x) < 0.005:
            self.table.loc[-1, "woe"] = 0.0
        self.iv = float(self.table["iv"].sum())
        self.map = self.table["woe"].to_dict()
        return self

    def _bin(self, x: pd.Series) -> pd.Series:
        """Integer bin codes (fast); -1 = missing. Labels kept in self.labels."""
        v = x.to_numpy(float)
        codes = np.searchsorted(self.edges, v, side="left") - 1
        codes = np.clip(codes, 0, len(self.edges) - 2)
        codes[np.isnan(v)] = -1
        return pd.Series(codes, index=x.index)

    @property
    def labels(self) -> dict:
        lab = {i: f"({self.edges[i]:g}, {self.edges[i + 1]:g}]" for i in range(len(self.edges) - 1)}
        lab[-1] = "MISSING"
        return lab

    def transform(self, x: pd.Series) -> np.ndarray:
        return self._bin(x).map(self.map).fillna(0.0).to_numpy(float)


class CategoricalBinner:
    def __init__(self, min_share=0.005):
        self.min_share = min_share

    def fit(self, x: pd.Series, y: pd.Series):
        x = x.astype(str)
        share = x.value_counts(normalize=True)
        self.labels = {}
        self.keep = set(share[share >= self.min_share].index)
        self.table = _woe_table(self._bin(x), y)
        self.iv = float(self.table["iv"].sum())
        self.map = self.table["woe"].to_dict()
        return self

    def _bin(self, x: pd.Series) -> pd.Series:
        x = x.astype(str)
        return x.where(x.isin(self.keep), "OTHER")

    def transform(self, x: pd.Series) -> np.ndarray:
        return self._bin(x).map(self.map).fillna(0.0).to_numpy(float)


class WOETransformer:
    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric, self.categorical = numeric, categorical

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y), index=X.index)
        self.binners = {}
        for c in self.numeric:
            self.binners[c] = NumericBinner().fit(X[c], y)
        for c in self.categorical:
            self.binners[c] = CategoricalBinner().fit(X[c], y)
        self.iv = pd.Series({c: b.iv for c, b in self.binners.items()}).sort_values(ascending=False)
        return self

    def transform(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or list(self.binners)
        return pd.DataFrame({c: self.binners[c].transform(X[c]) for c in cols}, index=X.index)
