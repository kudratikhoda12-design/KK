"""Shared forecasting infrastructure: context, true H-day sums and the ForecastBook container.

Everything that evaluates or consumes forecasts (accuracy tables, DM tests, uncertainty, inventory simulation) reads
from one :class:`ForecastBook`, which guarantees that all models are scored on *identical* origins and targets.

Shapes
------
* ``paths``  (n_series, n_days, MAX_H)           daily forecast path from every origin t (statistical models)
* ``sums``   (n_series, n_origins, len(HORIZONS)) forecast of sum(y[t+1..t+H]) for the origins of a phase
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import data_prep as dp
from .splits import Split, make_split

H_INDEX = {h: i for i, h in enumerate(C.HORIZONS)}


@dataclass
class Context:
    ids: list[str]
    sel: pd.DataFrame
    panel: pd.DataFrame
    Y: np.ndarray                 # (n_series, N) float64 observed sales
    price: np.ndarray             # (n_series, N) float64 daily price
    dates: pd.DatetimeIndex       # N observed dates
    split: Split
    calendar: pd.DataFrame        # full calendar (N + 28 days) with filled events
    snap_full: np.ndarray         # (n_series, N + 28) state-specific SNAP flag
    event_full: np.ndarray        # (N + 28,) 1 if event_name_1 != 'None'
    christmas_full: np.ndarray    # (N + 28,)

    @property
    def n_series(self) -> int:
        return self.Y.shape[0]

    @property
    def N(self) -> int:
        return self.Y.shape[1]


def load_context() -> Context:
    """Load the prepared panel and align every array on the canonical series order."""
    panel, sel = dp.load_prepared()
    ids = sel["series_id"].tolist()
    Y = dp.panel_matrix(panel, ids, "sales").astype(float)
    price = dp.panel_matrix(panel, ids, "sell_price").astype(float)
    dates = pd.DatetimeIndex(np.sort(panel["date"].unique()))
    split = make_split(dates)
    from . import data_loading as dl            # local import: avoids loading raw data unless the calendar is needed
    cal = pd.read_csv(C.DATA_RAW / C.RAW_FILES["calendar"])
    cal["date"] = pd.to_datetime(cal["date"])
    for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cal[c] = cal[c].fillna("None")
    if not (cal["date"].iloc[: len(dates)].to_numpy() == dates.to_numpy()).all():
        raise ValueError("calendar and panel dates are misaligned")
    state = sel["state_id"].to_numpy()
    snap_full = np.vstack([cal[dl.STATE_SNAP[s]].to_numpy() for s in state]).astype(float)
    event_full = (cal["event_name_1"] != "None").to_numpy().astype(float)
    christmas_full = (cal["event_name_1"] == "Christmas").to_numpy().astype(float)
    return Context(ids, sel, panel, Y, price, dates, split, cal, snap_full, event_full, christmas_full)


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
def truth_sums(Y: np.ndarray, H: int) -> np.ndarray:
    """T[s, t] = sum_{k=1..H} Y[s, t+k]  (NaN when the window runs past the last observed day)."""
    n_s, N = Y.shape
    c = np.concatenate([np.zeros((n_s, 1)), np.cumsum(Y, axis=1)], axis=1)       # c[:, j] = sum Y[:j]
    out = np.full((n_s, N), np.nan)
    t = np.arange(N - H)                                                         # t + H <= N - 1
    out[:, t] = c[:, t + H + 1] - c[:, t + 1]
    return out


def paths_to_sums(paths: np.ndarray) -> np.ndarray:
    """(..., MAX_H) daily path -> (..., len(HORIZONS)) sums over the first H steps."""
    cs = np.cumsum(paths, axis=-1)
    return np.stack([cs[..., h - 1] for h in C.HORIZONS], axis=-1)


# --------------------------------------------------------------------------------------
# Forecast book
# --------------------------------------------------------------------------------------
PHASES = ("val", "test")


@dataclass
class ForecastBook:
    ctx: Context
    F: dict = field(default_factory=lambda: {"val": {}, "test": {}})

    def origins(self, phase: str) -> np.ndarray:
        return self.ctx.split.all_origins("validation" if phase == "val" else "test")

    def add_sums(self, model: str, phase: str, sums_all_origins: np.ndarray) -> None:
        """``sums_all_origins``: (n_series, n_origins(phase), len(HORIZONS))."""
        exp = (self.ctx.n_series, len(self.origins(phase)), len(C.HORIZONS))
        if sums_all_origins.shape != exp:
            raise ValueError(f"{model}/{phase}: expected {exp}, got {sums_all_origins.shape}")
        self.F[phase][model] = np.asarray(sums_all_origins, float)

    def add_paths(self, model: str, phase: str, paths: np.ndarray) -> None:
        """``paths``: (n_series, N, MAX_H) daily forecast path for every day-origin (clipped at 0 here)."""
        orig = self.origins(phase)
        self.add_sums(model, phase, paths_to_sums(np.clip(paths[:, orig, :], 0.0, None)))

    def models(self, phase: str) -> list[str]:
        return list(self.F[phase].keys())

    def get(self, model: str, phase: str, H: int) -> np.ndarray:
        """(n_series, n_origins) forecast of the H-day sum from every origin of the phase."""
        return self.F[phase][model][:, :, H_INDEX[H]]

    # ---- evaluation views --------------------------------------------------------------------
    def eval_arrays(self, model: str, phase: str, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(origins, y_true, y_pred) restricted to origins whose complete H-day window lies in the period."""
        period = "validation" if phase == "val" else "test"
        t_eval = self.ctx.split.eval_origins(period, H)
        all_o = self.origins(phase)
        idx = t_eval - all_o[0]
        y = truth_sums(self.ctx.Y, H)[:, t_eval]
        f = self.get(model, phase, H)[:, idx]
        return t_eval, y, f

    def long_table(self, phase: str, models: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for m in models or self.models(phase):
            for H in C.HORIZONS:
                t, y, f = self.eval_arrays(m, phase, H)
                s_idx = np.repeat(np.arange(self.ctx.n_series), len(t))
                rows.append(pd.DataFrame({
                    "model": m, "phase": phase, "H": H, "series_idx": s_idx,
                    "series_id": np.array(self.ctx.ids)[s_idx], "origin": np.tile(t, self.ctx.n_series),
                    "y_true": y.ravel(), "y_pred": f.ravel()}))
        return pd.concat(rows, ignore_index=True)

    # ---- persistence -----------------------------------------------------------------------------
    def save(self, path=None) -> None:
        path = path or (C.DATA_PROC / "forecasts.npz")
        payload = {f"{ph}|{m}": arr for ph in PHASES for m, arr in self.F[ph].items()}
        np.savez_compressed(path, **payload)

    def load(self, path=None) -> "ForecastBook":
        path = path or (C.DATA_PROC / "forecasts.npz")
        z = np.load(path)
        for key in z.files:
            ph, m = key.split("|", 1)
            self.F[ph][m] = z[key]
        return self
