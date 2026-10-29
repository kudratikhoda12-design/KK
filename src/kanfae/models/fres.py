"""Landmark-anchored residual functional CoxKAN, b2s_res_cox (Round 15, D133).

Each channel is split into its last observed value before the landmark and the trajectory residual:
  eta = sum_d psi_d(X_d,last) + sum_k psi_k(z_k) + sum_d gate_d * sum_m int g_dm(X_d(t) - X_d,last) h_dm(t) dt,
where psi are static KAN edges (the landmark part, like static_coxkan on last values + z) and the second sum is a B2-S
functional layer on the residual curves. This is the existing FKANNet "b2s" kind applied to transformed inputs
(xg -> xg - last, z -> [last, z]), so the training, ensembling, pruning and symbolic machinery are unchanged.

Validation gate (gate_d in {0, 1}): on the tuned seed-0 member, the residual channels are pruned as wholes by their
integrated score (D120 channel score, residual layer only) at theta_rel in THETAS_D120 plus "all off" (the pure
landmark model), each followed by the same 200-epoch masked fine-tune; the parsimonious D120 rule picks the largest
theta (all off counts as the largest) whose validation loss is within `tol` (0.01, mean NLPL per event) of the
minimum. The chosen residual mask is applied to every ensemble member, each then fine-tuned for 200 epochs with the
mask fixed. Residual trajectory effects therefore enter only where validation loss supports them.

Two-way gate (D141, Round 17; gate_last_value=True, anchored variant only, Cox or AFT loss): the last-value edges
psi_d(X_d,last) are gated as well, and the plain B2-S ensemble fitted on the same split (`ref`, the b2s_cox / b2s_aft
comparator) is itself a candidate. Candidates (all on the tuned seed-0 anchored member, each with the 200-epoch masked
fine-tune; identical masks are fitted once):
  joint   for theta in THETAS_D120: trajectory channels pruned by the integrated score at theta (as D133) AND each
          last-value edge pruned if its score, normalised by the largest last-value edge score, is below theta;
  static  for theta in THETAS_D120: every trajectory channel off, last-value edges pruned at theta.
Rule: (1) among the joint states that keep at least one trajectory channel, the D120 parsimonious rule (tol, largest
theta); a static state replaces it only if its validation loss is strictly lower (the Round-16 non-cliff guard); (2)
the chosen anchored state is used only if its validation loss is below the comparator's seed-0 validation loss minus
`last_tol`; otherwise the model IS the comparator ensemble (exact reduction to b2s_cox / b2s_aft).

Confirmation split (D144, Round 18; `confirm=` in fit, default None = the Round-17 rule): the caller splits its
validation subjects into val_select (passed as `val`: early stopping, (lr, wd) tuning, candidate fine-tunes and the
step-(1) choice, for the anchored model AND the comparator `ref`) and val_confirm (passed as `confirm`, never used
before step (2)). Step (2) then compares, on val_confirm only, the step-(1) anchored candidate with the comparator's
seed-0 member (one model against one model): anchored iff loss_confirm(anchored) < loss_confirm(comparator) - last_tol.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..survival.cox import cox_nll
from .fkan import THETAS_D120, FKANEnsemble, FKANSurv

ALL_OFF = float("inf")


def residual_pack(pk, residual=True):
    """{xg, last, z, y, d} -> the b2s inputs {xg: xg - last (residual=True; the full curve xg otherwise),
    z: [last, z], y, d}. residual=False is the "anchored" variant (D133 dry run): last-value edges plus a gated B2-S
    on the full trajectory, which nests both b2s_cox and the pure landmark model."""
    z = pk["z"] if pk["z"] is not None and pk["z"].shape[1] else np.zeros((len(pk["last"]), 0))
    xg = pk["xg"] - pk["last"][:, None, :] if residual else pk["xg"]
    return dict(xg=xg, z=np.hstack([pk["last"], z]), y=pk["y"], d=pk["d"])


def _gate(m, theta, train, epochs):
    m = copy.deepcopy(m)
    if theta == ALL_OFF:
        m.net.g.mask[:] = 0.0
    elif theta > 0:
        m.prune(theta, "integrated", only=("g",))
    m.fine_tune(epochs)
    m._finalise(train)
    return m


def choose_gate(losses, thetas, tol=0.01, rule="parsimonious"):
    """Index of the chosen gate setting. "parsimonious" (D133): the largest theta (all off counts as the largest) whose
    loss is within tol of the minimum. "noncliff" (D138, Round 16): the same rule restricted to the settings that keep
    at least one functional channel (theta in THETAS_D120; their tolerance stays tol), and "all off" only if its loss is
    strictly below the best of those settings (tolerance 0), so the gate removes every trajectory channel only when
    the trajectory part makes the validation loss worse."""
    if rule == "parsimonious":
        vmin = min(losses)
        return max((i for i, v in enumerate(losses) if v <= vmin + tol), key=lambda i: thetas[i])
    on = [i for i, th in enumerate(thetas) if th != ALL_OFF]
    off = [i for i, th in enumerate(thetas) if th == ALL_OFF]
    vmin = min(losses[i] for i in on)
    k = max((i for i in on if losses[i] <= vmin + tol), key=lambda i: thetas[i])
    if off and losses[off[0]] < vmin:
        return off[0]
    return k


def _lv_prune(m, theta, D):
    """Mask the last-value edges (static inputs 0..D-1) whose score, normalised by the largest last-value edge
    score, is below theta (D141)."""
    _, s = m.edge_scores()["stat"]
    s = s[:D, 0]
    s = s / max(float(s.max()), 1e-12)
    for d in range(D):
        if s[d] < theta:
            m.net.stat.mask[d, 0] = 0.0


def choose_twoway(grid, tol=0.01, last_tol=0.0, ref_loss=None):
    """D141 two-way rule on candidate records {kind: "joint"|"static", theta, val_loss, traj_on}. Returns the index of
    the chosen anchored state, or None for the pure-trajectory comparator (see the module docstring)."""
    joint = [i for i, g in enumerate(grid) if g["kind"] == "joint" and sum(g["traj_on"]) > 0]
    static = [i for i, g in enumerate(grid) if g["kind"] == "static"]
    vmin = min(grid[i]["val_loss"] for i in joint)
    k = max((i for i in joint if grid[i]["val_loss"] <= vmin + tol), key=lambda i: grid[i]["theta"])
    if static:
        s = min(static, key=lambda i: (grid[i]["val_loss"], -grid[i]["theta"]))
        if grid[s]["val_loss"] < vmin:
            k = s
    if ref_loss is not None and not grid[k]["val_loss"] < ref_loss - last_tol:
        return None
    return k


def heldout_loss(m, pk):
    """Unpenalised loss of a fitted FKANSurv on a held-out set {xg, z, y, d} (inputs as the model sees them): mean NLPL
    per event (Cox) or mean NLL per subject (AFT), the units of val_loss (D144)."""
    X, Z = m._tensors(pk["xg"], pk["z"])
    with torch.no_grad():
        return float(m._loss(m.net(X, Z), torch.tensor(pk["y"]), torch.tensor(pk["d"])))


class ResFKAN:
    """b2s_res_cox: FKANEnsemble (D112 protocol) on residual inputs, then the validation gate."""

    def __init__(self, thetas=THETAS_D120, tol=0.01, fine_tune_epochs=200, lrs=(1e-3, 3e-3), wds=(1e-4, 1e-3),
                 seeds=(0, 1, 2), residual=True, gate="split", cv_folds=5, gate_rule="parsimonious", loss="cox",
                 gate_last_value=False, last_tol=0.0, **kw):
        self.thetas, self.tol, self.ft, self.residual = tuple(thetas) + (ALL_OFF,), tol, fine_tune_epochs, residual
        self.gate, self.cv_folds, self.kw, self.gate_rule = gate, cv_folds, kw, gate_rule
        self.gate_last_value, self.last_tol, self.loss = gate_last_value, last_tol, loss
        self.ens_kw = dict(lrs=lrs, wds=wds, seeds=seeds, kind="b2s", loss=loss, **kw)
        self.ref, self.reduced = None, False

    def fit(self, train, val, ref=None, confirm=None):
        """ref (two-way gate only): the plain B2-S ensemble already fitted on the same split (the comparator); fitted
        here with the same settings if None. confirm (two-way gate only, D144): held-out confirmation subjects
        {xg, last, z, y, d}, used only for the step-(2) decision; None keeps the Round-17 rule."""
        self._confirm = confirm
        tr, va = residual_pack(train, self.residual), residual_pack(val, self.residual)
        self.D = train["last"].shape[1]
        ens = FKANEnsemble(**self.ens_kw).fit(tr, va)
        self.lr, self.wd, self.tuning = ens.lr, ens.wd, ens.tuning
        if self.gate_last_value:
            if self.residual:
                raise ValueError("the two-way gate is defined for the anchored variant (residual=False)")
            if ref is None:
                ref = FKANEnsemble(**self.ens_kw).fit({k: train[k] for k in ("xg", "z", "y", "d")},
                                                      {k: val[k] for k in ("xg", "z", "y", "d")})
            return self._fit_twoway(ens, tr, ref)
        cand, self.gate_grid = [], []
        for th in self.thetas:
            m = _gate(ens.members[0], th, tr, self.ft)
            cand.append(m)
            self.gate_grid.append(dict(theta=th if th != ALL_OFF else "all_off", val_loss=float(m.val_loss),
                                       residual_on=[int((m.net.g.mask[d] > 0).any()) for d in range(self.D)]))
        score = "val_loss"
        if self.gate == "cv":                                       # D134: cross-validated gate
            cvs = self._cv_scores(tr, va, ens)
            for g, s in zip(self.gate_grid, cvs):
                g["cv_loss"] = s
            score = "cv_loss"
        k = choose_gate([g[score] for g in self.gate_grid], self.thetas, self.tol, self.gate_rule)
        self.gate_grid[k]["chosen"] = True
        mask = cand[k].net.g.mask.clone()
        self.residual_on = self.gate_grid[k]["residual_on"]
        self.members = [cand[k]]
        for m in ens.members[1:]:
            m = copy.deepcopy(m)
            m.net.g.mask[:] = mask
            m.fine_tune(self.ft)
            m._finalise(tr)
            self.members.append(m)
        self._tr = None
        return self

    def _fit_twoway(self, ens, tr, ref):
        m0, D = ens.members[0], self.D
        states = []
        for kind in ("joint", "static"):
            for th in self.thetas[:-1]:
                m = copy.deepcopy(m0)
                if kind == "joint":
                    if th > 0:
                        m.prune(th, "integrated", only=("g",))
                else:
                    m.net.g.mask[:] = 0.0
                if th > 0:
                    _lv_prune(m, th, D)
                states.append((kind, th, m))
        fitted, self.gate_grid, cand = {}, [], []
        for kind, th, m in states:
            key = (tuple(m.net.g.mask.flatten().tolist()), tuple(m.net.stat.mask[:D, 0].tolist()))
            if key not in fitted:
                m.fine_tune(self.ft)
                m._finalise(tr)
                fitted[key] = m
            m = fitted[key]
            cand.append(m)
            self.gate_grid.append(dict(kind=kind, theta=th, val_loss=float(m.val_loss),
                                       traj_on=[int((m.net.g.mask[d] > 0).any()) for d in range(D)],
                                       last_on=[int(m.net.stat.mask[d, 0] > 0) for d in range(D)]))
        self.n_fine_tunes = len(fitted)
        self.ref, self.ref_loss = ref, float(ref.members[0].val_loss)
        k_anch = choose_twoway(self.gate_grid, self.tol)                  # the anchored state, before step (2)
        if self._confirm is None:
            k = choose_twoway(self.gate_grid, self.tol, self.last_tol, self.ref_loss)
        else:                                                              # D144: one-vs-one on val_confirm
            c = self._confirm
            self.confirm_loss_anchored = heldout_loss(cand[k_anch], residual_pack(c, self.residual))
            self.confirm_loss_ref = heldout_loss(ref.members[0], {q: c[q] for q in ("xg", "z", "y", "d")})
            k = k_anch if self.confirm_loss_anchored < self.confirm_loss_ref - self.last_tol else None
        self.anchored_choice = dict(self.gate_grid[k_anch], index=k_anch)
        self.reduced = k is None
        self.residual_on = [0] * D if self.reduced else self.gate_grid[k]["traj_on"]
        self.last_on = [0] * D if self.reduced else self.gate_grid[k]["last_on"]
        self._ens, self._k_anch, self._cand_anch = ens, k_anch, cand[k_anch]
        self.members = None
        if not self.reduced:
            self.gate_grid[k]["chosen"] = True
            self.members = self._finish(ens, cand[k], tr)
        self._tr = tr
        return self

    def _finish(self, ens, m0, tr):
        gm, sm = m0.net.g.mask.clone(), m0.net.stat.mask.clone()
        out = [m0]
        for m in ens.members[1:]:
            m = copy.deepcopy(m)
            m.net.g.mask[:] = gm
            m.net.stat.mask[:] = sm
            m.fine_tune(self.ft)
            m._finalise(tr)
            out.append(m)
        return out

    def anchored_alternative(self):
        """Two-way gate: the anchored ensemble at the anchored state chosen in step (1), whatever step (2) decided
        (for development analysis: lets every last_tol be evaluated from one fit)."""
        alt = copy.copy(self)
        alt.reduced = False
        if self.members is None:                       # step (2) chose the comparator: finish the anchored state
            alt.members = self._finish(self._ens, self._cand_anch, self._tr)
        alt.residual_on, alt.last_on = self.anchored_choice["traj_on"], self.anchored_choice["last_on"]
        return alt

    def _cv_scores(self, tr, va, ens):
        """D134 CV gate: pooled training + validation subjects, `cv_folds` folds (seed 0, stratified on the event);
        per fold the seed-0 model at the tuned (lr, wd) is trained on the other folds for the tuned model's best epoch
        count (no early stopping), each gate setting is fine-tuned `ft` epochs, and the held-out fold's NLPL per event
        is recorded. Returns the fold-mean NLPL per gate setting (order of self.thetas)."""
        from sklearn.model_selection import StratifiedKFold
        pool = {k: np.concatenate([tr[k], va[k]]) for k in ("xg", "z", "y", "d")}
        best_ep = int(np.argmin(ens.members[0].history)) + 1
        folds = StratifiedKFold(self.cv_folds, shuffle=True, random_state=0).split(pool["y"], pool["d"])
        sc = np.zeros(len(self.thetas))
        for a, b in folds:
            ftr = {k: v[a] for k, v in pool.items()}
            m = FKANSurv(**{**self.kw, "kind": "b2s", "loss": "cox", "lr": self.lr, "wd": self.wd, "seed": 0,
                            "max_epochs": best_ep, "patience": best_ep, "early_stopping": False}).fit(ftr, None)
            for j, th in enumerate(self.thetas):
                mm = _gate(m, th, ftr, self.ft)
                eta = torch.tensor(mm.eta(pool["xg"][b], pool["z"][b]))
                sc[j] += float(cox_nll(eta, torch.tensor(pool["y"][b]), torch.tensor(pool["d"][b]), "efron")) / self.cv_folds
        return sc.tolist()

    # prediction on {xg, last, z}
    def _inp(self, xg, last, z):
        p = residual_pack(dict(xg=xg, last=last, z=z, y=None, d=None), self.residual)
        return p["xg"], p["z"]

    def eta(self, xg, last, z):
        if self.reduced:
            return self.ref.eta(xg, z)
        X, Z = self._inp(xg, last, z)
        return np.mean([m.eta(X, Z) for m in self.members], 0)

    def predict_risk(self, xg, last, z):
        if self.reduced:
            return self.ref.predict_risk(xg, z)
        if self.loss == "cox":
            return self.eta(xg, last, z)
        X, Z = self._inp(xg, last, z)
        return np.mean([m.predict_risk(X, Z) for m in self.members], 0)

    def predict_survival(self, xg, last, z, times):
        if self.reduced:
            return self.ref.predict_survival(xg, z, times)
        X, Z = self._inp(xg, last, z)
        return np.mean([m.predict_survival(X, Z, times) for m in self.members], 0)

    def last_value_effect(self, d, x):
        """Member-mean static edge psi_d(x) of the last value of channel d, centred, on the original scale."""
        import torch
        out = []
        for m in self.members:
            inp = torch.zeros((len(x), m.net.stat.n_in))
            inp[:, d] = torch.tensor((np.asarray(x, float) - m.zm[d]) / m.zs[d])
            with torch.no_grad():
                out.append(m.net.stat.edge_activations(inp)[:, d, 0].numpy())
        e = np.mean(out, 0)
        return e - e.mean()

    def residual_effect(self, d, r, t):
        """Member-mean residual surface sum_m g_dm(r) h_dm(t) of channel d (r = X_d(t) - X_d,last)."""
        return np.mean([m.effect(d, r, t) for m in self.members], 0)


def set_validation(m, pk):
    """Replace a fitted FKANSurv's validation data (the data its fine-tune early-stops on) by pk {xg, z, y, d}, given
    as the model sees its inputs (D147)."""
    Xva, Zva = m._tensors(pk["xg"], pk["z"])
    m.data.update(Xva=Xva, Zva=Zva, yva=torch.tensor(pk["y"]), dva=torch.tensor(pk["d"]), sva=None)


class ConfirmRefitAnchor:
    """D147 (Round 19): the Round-18 confirmation-split decision, then a full-validation estimator.

    fit(train, val_select, val_confirm, ref_select, ref_full): ref_select is the plain B2-S ensemble tuned and early-stopped
    on val_select (it takes part in the step-(2) decision only); ref_full is the usual B2-S ensemble tuned and
    early-stopped on the whole validation set (val_select + val_confirm). Steps (1)-(2) are exactly ResFKAN's two-way gate
    with confirm= (val_confirm untouched before the decision). Then:
      not anchored -> the estimator IS ref_full (its predictions are returned unchanged);
      anchored     -> the chosen state's masks (trajectory channels and last-value edges of the step-(1) candidate) are
                      applied to an anchored ensemble that uses the whole validation set, then each member is fine-tuned
                      with the mask fixed, early-stopping on the whole validation set:
                      refit="retrain": a new anchored ensemble, (lr, wd) tuned and early-stopped on the whole validation
                                       set (D112 protocol), masked and fine-tuned;
                      refit="finetune": the step-(1) anchored ensemble (trained on val_select) fine-tuned once more
                                       with the mask, early-stopping on the whole validation set.
    The test split is never passed in."""

    def __init__(self, loss="cox", margin=0.0, refit="retrain", fine_tune_epochs=200, **kw):
        if refit not in ("retrain", "finetune"):
            raise ValueError(refit)
        self.loss, self.margin, self.refit, self.ft, self.kw = loss, margin, refit, fine_tune_epochs, kw

    def fit(self, train, val_select, val_confirm, ref_select, ref_full):
        self.gate = ResFKAN(residual=False, gate_last_value=True, last_tol=self.margin, loss=self.loss,
                            fine_tune_epochs=self.ft, **self.kw).fit(train, val_select, ref=ref_select, confirm=val_confirm)
        self.reduced, self.ref_full = self.gate.reduced, ref_full
        self.val_full = {k: np.concatenate([val_select[k], val_confirm[k]]) for k in ("xg", "last", "z", "y", "d")}
        self._train = train
        self.members = None if self.reduced else self.refit_members(self.refit)
        return self

    def refit_members(self, how):
        """The full-validation anchored ensemble at the step-(1) state (also used, whatever step (2) decided, by the
        development analysis)."""
        g = self.gate
        tr, vf = residual_pack(self._train, False), residual_pack(self.val_full, False)
        c = g._cand_anch
        gm, sm = c.net.g.mask.clone(), c.net.stat.mask.clone()
        if how == "retrain":
            base = FKANEnsemble(**g.ens_kw).fit(tr, vf).members
        else:
            base = g.anchored_alternative().members
        out = []
        for m in base:
            m = copy.deepcopy(m)
            m.net.g.mask[:] = gm
            m.net.stat.mask[:] = sm
            if how == "finetune":
                set_validation(m, vf)
            m.fine_tune(self.ft)
            m._finalise(tr)
            out.append(m)
        return out

    def _inp(self, xg, last, z):
        p = residual_pack(dict(xg=xg, last=last, z=z, y=None, d=None), False)
        return p["xg"], p["z"]

    def predict_survival(self, xg, last, z, times, members=None):
        members = self.members if members is None else members
        if members is None:
            return self.ref_full.predict_survival(xg, z, times)
        X, Z = self._inp(xg, last, z)
        return np.mean([m.predict_survival(X, Z, times) for m in members], 0)

    def predict_risk(self, xg, last, z, members=None):
        members = self.members if members is None else members
        if members is None:
            return self.ref_full.predict_risk(xg, z)
        X, Z = self._inp(xg, last, z)
        return np.mean([m.predict_risk(X, Z) for m in members], 0)
