"""Individual-level (group-level) splits (K1, K3)."""
from __future__ import annotations

import numpy as np


def split_groups(groups, seed, fracs=(0.6, 0.2, 0.2), strata=None):
    """Split unique group IDs into train/val/test, stratified on a per-group label if given.

    Returns index arrays into `groups` (all rows of a group go to the same part)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    if strata is None:
        lab = {g: 0 for g in uniq}
    else:
        strata = np.asarray(strata)
        lab = {g: strata[np.nonzero(groups == g)[0][0]] for g in uniq}
    parts = [[], [], []]
    for s in sorted(set(lab.values())):
        ids = np.array([g for g in uniq if lab[g] == s])
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
        parts[0] += list(ids[:n_tr]); parts[1] += list(ids[n_tr:n_tr + n_va]); parts[2] += list(ids[n_tr + n_va:])
    sets = [set(p) for p in parts]
    return tuple(np.nonzero(np.isin(groups, list(s)))[0] for s in sets)
