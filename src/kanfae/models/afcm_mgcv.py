"""D168: the reference AFCM fitted with mgcv in the project-local R (scripts/afcm_mgcv.R), as specified by the paper
(Cui, Crainiceanu & Leroux 2021) and the author's public mgcv vignette: one ti(tmat, X_d, by = lmat, bs = c("cr","cr"),
k = c(10, 10), mc = c(FALSE, TRUE)) per channel, quantile-transformed covariate, cox.ph(), REML.

This module only prepares identical inputs, calls R and reads the result back; it does not re-implement the fit.
Quantile transformation: for channel d and grid time t_g, x~ = Q_dg(x), the TRAINING empirical CDF with plotting
positions (i - 0.5)/n and linear interpolation between order statistics, constant beyond the extremes (the same map as
kanfae.models.afcm.AFCM with xin="quantile"); training and test curves go through the same map.
The fitted surface F_d(t, x~) is returned by R on the model grid in t and a fine grid of x~ in [0, 1]; effect(d, x, t)
maps x through Q_d(t) and interpolates linearly in x~ at the nearest grid time (all shape metrics evaluate on the grid).
Survival prediction: the R linear predictors with the Breslow baseline estimated on the training linear predictors
(kanfae.survival.cox), the same construction as every other Cox model in the project.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..survival.cox import breslow_baseline, cox_survival

ROOT = Path(__file__).resolve().parents[3]
RSCRIPT = ROOT / ".rlocal/env/bin/Rscript"
RFILE = ROOT / "scripts/afcm_mgcv.R"
GRID = np.linspace(0.0, 1.0, 50)


def trap_weights(grid):
    w = np.full(len(grid), grid[1] - grid[0])
    w[[0, -1]] *= 0.5
    return w


class QuantileMap:
    """Training empirical CDF of each channel at each grid time."""

    def __init__(self, xg_train):
        self.qs = np.sort(np.asarray(xg_train, float), axis=0)               # (n, G, D)
        n = self.qs.shape[0]
        self.qp = (np.arange(1, n + 1) - 0.5) / n

    def __call__(self, d, x, g):
        x, g = np.broadcast_arrays(np.asarray(x, float), np.asarray(g))
        out = np.empty(x.shape)
        for gi in np.unique(g):
            m = g == gi
            out[m] = np.interp(x[m], self.qs[:, gi, d], self.qp)
        return out

    def transform(self, xg):
        n, G, D = xg.shape
        g = np.broadcast_to(np.arange(G)[None, :], (n, G))
        return np.stack([self(d, xg[:, :, d], g) for d in range(D)], axis=2)


class AFCMReference:
    def __init__(self, k_x=10, k_t=10, n_xgrid=401, timeout=3600, keep_dir=None):
        self.k_x, self.k_t, self.n_xgrid, self.timeout, self.keep_dir = k_x, k_t, n_xgrid, timeout, keep_dir

    def fit(self, train, test=None):
        """train/test: dicts with xg (n, G, D) on GRID, z (n, p); train also y, d. Returns self (self.info holds the
        R diagnostics; self.info['status'] != 'ok' if the fit failed)."""
        xg = np.asarray(train["xg"], float)
        self.qmap = QuantileMap(xg)
        self.D = xg.shape[2]
        self.xgrid = np.linspace(0.0, 1.0, self.n_xgrid)
        ctx = tempfile.TemporaryDirectory() if self.keep_dir is None else None
        wd = Path(ctx.name if ctx else self.keep_dir)
        wd.mkdir(parents=True, exist_ok=True)
        sv = lambda name, a: np.savetxt(wd / name, np.atleast_2d(a) if np.ndim(a) > 1 else np.asarray(a)[:, None],  # noqa: E731
                                        delimiter=",", fmt="%.17g")
        sv("tgrid.csv", GRID)
        sv("lmat.csv", trap_weights(GRID))
        sv("xgrid.csv", self.xgrid)
        xq = self.qmap.transform(xg)
        for d in range(self.D):
            sv(f"xq_tr_{d + 1}.csv", xq[:, :, d])
        sv("z_tr.csv", np.asarray(train["z"], float))
        sv("yd_tr.csv", np.c_[train["y"], train["d"]])
        if test is not None:
            xqt = self.qmap.transform(np.asarray(test["xg"], float))
            for d in range(self.D):
                sv(f"xq_te_{d + 1}.csv", xqt[:, :, d])
            sv("z_te.csv", np.asarray(test["z"], float))
        (wd / "meta.json").write_text(json.dumps(dict(D=int(self.D), k_x=self.k_x, k_t=self.k_t)))
        t0 = time.time()
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        p = subprocess.run([str(RSCRIPT), str(RFILE), str(wd)], capture_output=True, text=True, timeout=self.timeout, env=env)
        self.seconds_total = time.time() - t0
        res = wd / "result.json"
        info = json.loads(res.read_text()) if res.exists() else dict(status="failed", error=(p.stderr or p.stdout)[-3000:])
        if ctx:
            ctx.cleanup()
        self.info = {k: v for k, v in info.items() if k not in ("surfaces", "eta_tr", "eta_te", "xgrid", "tgrid")}
        if info.get("status") != "ok":
            return self
        self.surfaces = np.array(info["surfaces"], float)                   # (D, G, n_xgrid)
        self.eta_tr = np.asarray(info["eta_tr"], float)
        self.eta_te = None if info.get("eta_te") is None else np.asarray(info["eta_te"], float)
        self.baseline = breslow_baseline(self.eta_tr, np.asarray(train["y"], float), np.asarray(train["d"], bool))
        self.finite = bool(np.all(np.isfinite(self.surfaces)) and np.all(np.isfinite(self.eta_tr)) and
                           (self.eta_te is None or np.all(np.isfinite(self.eta_te))))
        return self

    @property
    def ok(self):
        return self.info.get("status") == "ok"

    def effect(self, d, x, t):
        """F_d(x, t) on arrays x, t (broadcast): x through the training quantile map at the nearest grid time."""
        x = np.asarray(x, float)
        t = np.broadcast_to(np.asarray(t, float), x.shape)
        g = np.clip(np.rint(t * (len(GRID) - 1)).astype(int), 0, len(GRID) - 1)
        xq = self.qmap(d, x, g)
        out = np.empty(x.shape)
        for gi in np.unique(g):
            m = g == gi
            out[m] = np.interp(xq[m], self.xgrid, self.surfaces[d, gi])
        return out

    def predict_risk_test(self):
        return self.eta_te

    def predict_survival_test(self, times):
        return cox_survival(self.eta_te, self.baseline[None], times)

    def mean_surface_over_curves(self, d, xg):
        """max over grid times of |mean_i F_d(t_g, X_i(t_g))| relative to the RMS of F_d over the same points (A5)."""
        n, G = xg.shape[:2]
        tt = np.broadcast_to(GRID[None, :], (n, G))
        F = self.effect(d, xg[:, :, d], tt)
        return float(np.max(np.abs(F.mean(0))) / max(np.sqrt(np.mean(F ** 2)), 1e-300))
