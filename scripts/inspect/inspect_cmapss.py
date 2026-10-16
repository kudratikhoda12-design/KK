"""Stage F step 2: inspect C-MAPSS before writing any loader (protocol 4.2)."""
import sys, json
import numpy as np, pandas as pd
RAW = sys.argv[1]
cols = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
out = {}
for fd in ["FD001", "FD003"]:
    tr = pd.read_csv(f"{RAW}/train_{fd}.txt", sep=r"\s+", header=None, names=cols)
    te = pd.read_csv(f"{RAW}/test_{fd}.txt", sep=r"\s+", header=None, names=cols)
    rul = pd.read_csv(f"{RAW}/RUL_{fd}.txt", sep=r"\s+", header=None)[0].to_numpy()
    life = tr.groupby("unit").cycle.max()
    tlen = te.groupby("unit").cycle.max()
    consec = all((g.cycle.diff().dropna() == 1).all() for _, g in tr.groupby("unit"))
    sd = tr[[f"s{i}" for i in range(1, 22)] + ["os1", "os2", "os3"]].std()
    info = dict(train_rows=len(tr), train_units=int(tr.unit.nunique()), test_units=int(te.unit.nunique()),
                rul_rows=len(rul), cycles_consecutive=bool(consec), missing=int(tr.isna().sum().sum()),
                train_life=dict(min=int(life.min()), q25=float(life.quantile(.25)), median=float(life.median()),
                                max=int(life.max())),
                test_length=dict(min=int(tlen.min()), median=float(tlen.median()), max=int(tlen.max()),
                                 n_below_30=int((tlen < 30).sum())),
                test_true_rul=dict(min=int(rul.min()), median=float(np.median(rul)), max=int(rul.max())),
                test_total_life=dict(min=int((tlen.to_numpy() + rul).min()), max=int((tlen.to_numpy() + rul).max())),
                near_constant_sd_lt_1em6=[c for c, v in sd.items() if v < 1e-6],
                low_sd=[f"{c}:{v:.2e}" for c, v in sd.items() if v < 1e-3])
    out[fd] = info
    print(fd, json.dumps(info, indent=1))
json.dump(out, open("reports/logs/cmapss_inspection.json", "w"), indent=1)
