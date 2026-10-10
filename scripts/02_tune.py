"""Step 2 - small grid search on the VALIDATION split (test set untouched).

    python scripts/02_tune.py
Writes reports/results/tuning.csv. Chosen settings are copied into config.py.
"""
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from coxkan_churn import config as C, data, models  # noqa: E402

torch.set_num_threads(4)
d, _ = data.prepare()
rows = []

grids = {
    "CoxKAN [d,1]": (models.CoxKAN, dict(width=[[None, 1]], grid=[3, 5, 8], lr=[1e-2, 3e-2], lamb_l1=[0.0, 1e-3])),
    "CoxKAN [d,4,1]": (models.CoxKAN, dict(width=[[None, 4, 1]], grid=[3, 5], lr=[3e-3, 1e-2], lamb_l1=[0.0, 1e-3])),
    "DeepSurv": (models.DeepSurv, dict(hidden=[[32], [64, 32], [128, 64]], dropout=[0.1, 0.3], lr=[1e-3, 3e-3])),
}
for label, (cls, grid) in grids.items():
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        params = dict(zip(keys, vals))
        t0 = time.time()
        m = cls(**{k: (list(v) if isinstance(v, list) else v) for k, v in params.items()},
                epochs=600, weight_decay=1e-4 if cls is models.DeepSurv else 0.0).fit(d["train"], d["val"])
        rows.append(dict(model=label, **{k: str(v) for k, v in params.items()},
                         val_c=round(m.best_val_c, 4), epochs=m.history[-1]["epoch"], secs=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)

C.RES_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).reindex(columns=["model", "width", "grid", "lamb_l1", "hidden", "dropout", "lr", "val_c", "epochs", "secs"]).sort_values(["model", "val_c"], ascending=[True, False]).to_csv(C.RES_DIR / "tuning.csv", index=False)
