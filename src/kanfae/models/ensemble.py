"""Revision 4f (D100): multi-seed risk averaging. An ensemble of k NeuralAFT fits that differ only in the training seed
(0..k-1). Risk is the mean of the members' risk scores (risk = -eta, so this is -mean eta); survival curves are the
mean of the members' curves. With k = 1 the ensemble is exactly the single seed-0 model."""
import numpy as np

from .pipelines import NeuralAFT


class SeedEnsembleAFT:
    def __init__(self, kind="fae_mlp", ensemble_seeds=1, **cfg):
        self.kind, self.k, self.cfg = kind, int(ensemble_seeds), cfg
        self.members = []

    @classmethod
    def from_members(cls, members):
        """Build from already fitted members (seeds 0..k-1, in order)."""
        e = cls(members[0].kind, len(members), **members[0].cfg)
        e.members = list(members)
        return e

    def fit(self, train, val, seed=0):
        # seed offsets the member seeds (members use seed, seed+1, ..., seed+k-1); the registered runs use seed = 0
        self.members = [NeuralAFT(self.kind, **self.cfg).fit(train, val, seed=seed + j) for j in range(self.k)]
        self.info = {"members": [m.info for m in self.members]}
        self.n_params = sum(m.n_params for m in self.members)
        self.fit_seconds = sum(m.fit_seconds for m in self.members)
        return self

    def prepare(self, obs, holdout=False):
        return [m.prepare(obs, holdout=holdout) for m in self.members]

    def predict_risk(self, obs, P=None):
        P = P or [None] * len(self.members)
        return np.mean([m.predict_risk(obs, P=p) for m, p in zip(self.members, P)], axis=0)

    def predict_survival(self, obs, times, P=None):
        P = P or [None] * len(self.members)
        return np.mean([m.predict_survival(obs, times, P=p) for m, p in zip(self.members, P)], axis=0)

    def latent(self, obs, P=None):
        """Seed-0 member's latent (each member's latent is available via .members)."""
        return self.members[0].latent(obs)
