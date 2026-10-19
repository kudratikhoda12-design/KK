"""D108 Round-9 analysis: KAN-fairness contrasts F1-F3 (and S3 - S2 within each arm) from reports/round9/*.jsonl
(primary = phase "ensemble"), plus the determinism check of the repeated mlp/kan S2 cells against the H1 records.
Writes reports/round9_summary.json and reports/tables/r9_*.csv."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_step3 import paired

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
METRICS = ["curve_error", "ibs", "harrell_c", "uno_c", "dcal_p", "reconstruction_test"]
SURV = ["curve_error", "ibs", "harrell_c", "uno_c", "dcal_p"]
CONTRASTS = [("F1", "kan_S3 − mlp_S3 (D31 matching, tuned S3)", ("mlp", "S3"), ("kan", "S3")),
             ("F2", "kan_shared − mlp, S2", ("mlp", "S2"), ("kan_shared", "S2")),
             ("F2", "kan_shared − mlp, S3", ("mlp", "S3"), ("kan_shared", "S3")),
             ("F2", "kan_shared − kan, S2 (bottleneck effect within KAN)", ("kan", "S2"), ("kan_shared", "S2")),
             ("F2", "kan_shared − kan, S3 (bottleneck effect within KAN)", ("kan", "S3"), ("kan_shared", "S3")),
             ("F3", "kan_unb − mlp_wide, S2", ("mlp_wide", "S2"), ("kan_unb", "S2")),
             ("F3", "kan_unb − mlp_wide, S3", ("mlp_wide", "S3"), ("kan_unb", "S3"))]
ARMS = ["mlp", "kan", "kan_shared", "kan_unb", "mlp_wide"]


def load():
    rows = []
    for f in sorted(glob.glob(str(REP / "round9/*.jsonl"))):
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def main():
    df = load()
    (REP / "tables").mkdir(exist_ok=True)
    out = {"n_records": int(len(df))}
    st = df.groupby(["arm", "strategy", "phase", "status"]).size().reset_index(name="n")
    st.to_csv(REP / "tables/r9_status.csv", index=False)
    out["status"] = st.to_dict("records")
    E = df[(df.phase == "ensemble") & (df.status == "ok")]
    cells = E.groupby(["arm", "strategy"])[METRICS].agg(["mean", "std", "count"]).reset_index()
    cells.columns = ["_".join(c).strip("_") for c in cells.columns]
    cells["n_params"] = [int(E[(E.arm == a) & (E.strategy == s)].n_params.iloc[0]) for a, s in zip(cells.arm, cells.strategy)]
    cells.to_csv(REP / "tables/r9_cells.csv", index=False)
    sel = E.assign(cfg=E.config.map(lambda c: ", ".join(f"{k} {v:g}" for k, v in json.loads(c).items() if k in ("lr", "wd"))))
    sel.groupby(["arm", "strategy"])["cfg"].apply(list).reset_index().to_csv(REP / "tables/r9_tuning.csv", index=False)
    get = lambda a, s: E[(E.arm == a) & (E.strategy == s)].set_index("replicate")  # noqa: E731
    rows = []
    for fid, lab, (aa, sa), (ab, sb) in CONTRASTS + [("S", f"{a}: S3 − S2", (a, "S2"), (a, "S3")) for a in ARMS]:
        A, B = get(aa, sa), get(ab, sb)
        reps = sorted(set(A.index) & set(B.index))
        if not reps:
            continue
        for m in METRICS:
            rows.append(dict(id=fid, contrast=lab, metric=m, **paired(A.loc[reps, m], B.loc[reps, m])))
    C = pd.DataFrame(rows)
    C.to_csv(REP / "tables/r9_contrasts.csv", index=False)
    out["contrasts"] = rows
    # determinism: the repeated S2 cells vs the H1 records (survival metrics, ensemble and per-seed records)
    h1 = pd.DataFrame([json.loads(l) for l in (REP / "step3/H1.jsonl").read_text().splitlines() if l.strip()])
    chk = []
    for arm, model in (("mlp", "fae_mlp"), ("kan", "fae_kan")):
        for ph in ("ensemble", "final"):
            a = h1[(h1.model == model) & (h1.phase == ph)].sort_values(["replicate", "seed"])
            b = df[(df.arm == arm) & (df.strategy == "S2") & (df.phase == ph)].sort_values(["replicate", "seed"])
            same = len(a) == len(b) and all(np.array_equal(a[m].to_numpy(float), b[m].to_numpy(float)) for m in SURV)
            chk.append(dict(arm=arm, phase=ph, n_h1=len(a), n_r9=len(b), bit_identical=bool(same)))
    pd.DataFrame(chk).to_csv(REP / "tables/r9_determinism.csv", index=False)
    out["determinism_vs_H1"] = chk
    (REP / "round9_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(pd.DataFrame(chk).to_string())
    print(C[C.metric.isin(["curve_error", "harrell_c", "reconstruction_test"])][
        ["id", "contrast", "metric", "mean_diff", "ci_lo", "ci_hi", "p_t", "n_pos", "n_neg", "p_wilcoxon"]].round(4).to_string())


if __name__ == "__main__":
    main()
