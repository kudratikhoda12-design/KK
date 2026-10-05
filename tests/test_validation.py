import pandas as pd

from src.features import build_dataset
from src.validation import walk_forward_folds


def test_folds_are_chronological_and_purged(small_grid):
    d = build_dataset(small_grid)
    folds = walk_forward_folds(d, pd.Timestamp("2021-01-10", tz="UTC"),
                               pd.Timestamp("2021-01-21", tz="UTC"), block_months=1)
    assert folds
    for f in folds:
        tr, ev = d.index[f.train_mask], d.index[f.eval_mask]
        assert tr.max() < ev.min()                               # train strictly before eval
        assert (d.loc[f.train_mask, "exit_time"] <= ev.min()).all()   # purged: no label overlap
