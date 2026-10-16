"""Observed / Oracle data containers (docs/design.md sections 1.2 and 5.4).

`Observed` is the only object training code may accept. `Oracle` holds hidden truth for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np


@dataclass(frozen=True)
class Observed:
    ids: np.ndarray            # (N,) individual identifiers
    times: np.ndarray          # (N, L) in [0, 1]; padding = 0
    values: np.ndarray         # (N, L, D); masked/padded entries = 0
    mask: np.ndarray           # (N, L, D) bool; padding False
    n_obs: np.ndarray          # (N,) number of real observation times
    y_obs: np.ndarray          # (N,) min(T*, C) > 0
    delta: np.ndarray          # (N,) bool event indicator
    u: np.ndarray | None = None              # (N, p) baseline covariates
    channel_names: tuple = ()
    meta: dict = field(default_factory=dict)  # non-outcome metadata (e.g. landmark, source)

    def __len__(self):
        return len(self.ids)

    def subset(self, idx) -> "Observed":
        idx = np.asarray(idx)
        return Observed(self.ids[idx], self.times[idx], self.values[idx], self.mask[idx], self.n_obs[idx],
                        self.y_obs[idx], self.delta[idx], None if self.u is None else self.u[idx],
                        self.channel_names, dict(self.meta))


ORACLE_FIELDS = ("T_true", "C", "E", "theta", "eta_true")


@dataclass(frozen=True)
class Oracle:
    ids: np.ndarray
    T_true: np.ndarray         # (N,) true residual event times (hidden when censored)
    C: np.ndarray              # (N,) censoring times (inf when no censoring)
    E: np.ndarray              # (N,) exponential (non-administrative) censoring component
    theta: np.ndarray          # (N, 5) latent parameters (a, b, A, omega, phi)
    eta_true: np.ndarray       # (N,)
    sigma_true: float
    family: str
    t_max: float               # 0.9 quantile of T* in the pilot sample (survival-curve error horizon)
    tau_admin: float
    mu: float
    theta_names: tuple = ("a", "b", "A", "omega", "phi")

    def subset(self, idx) -> "Oracle":
        idx = np.asarray(idx)
        return Oracle(self.ids[idx], self.T_true[idx], self.C[idx], self.E[idx], self.theta[idx],
                      self.eta_true[idx], self.sigma_true, self.family, self.t_max, self.tau_admin, self.mu)

    def survival(self, t) -> np.ndarray:
        """True S(t | theta_i) on grid t -> (N, T)."""
        import torch

        from ..survival.aft import aft_survival
        return aft_survival(torch.tensor(np.asarray(t, float)), torch.tensor(self.eta_true, dtype=torch.float64),
                            self.sigma_true, self.family).numpy()


def to_training_arrays(obs) -> dict:
    """Training-facing conversion. Accepts only `Observed` (criterion G5)."""
    if not isinstance(obs, Observed):
        raise TypeError(f"training code accepts only Observed, got {type(obs).__name__}")
    return {f.name: getattr(obs, f.name) for f in fields(obs)}


def make_observed_outcomes(T_true, C):
    """Y = min(T*, C), delta = 1{T* <= C} (criteria G1, G2)."""
    T_true, C = np.asarray(T_true, float), np.asarray(C, float)
    return np.minimum(T_true, C), T_true <= C
