"""Round 14 (D131): descriptive cohort record for the manuscript (no model is fitted). Reads the Round-13 real-data
loaders and the single-landmark cohorts exactly as run_round13.py builds them, and writes
reports/round14/cohort_summary.json (counts, events, visits per subject in the landmark window, raw-data facts)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kanfae.data.realdata import LOADERS, _rda, single_landmark  # noqa: E402
from run_round13 import SINGLE  # noqa: E402

OUT = ROOT / "reports/round14/cohort_summary.json"


def main():
    out = {}
    for ds, t_L in SINGLE.items():
        ld = LOADERS[ds]()
        lm = single_landmark(ld, t_L)
        nv = np.array([len(t) for t in lm.times])
        out[ds] = dict(n_all=int(len(ld.ids)), n_cohort=int(len(lm.ids)), n_events=int(lm.delta.sum()),
                       landmark=float(t_L), unit=ld.unit, channels=list(ld.channels), baseline=list(ld.z_names),
                       visits_median=float(np.median(nv)), visits_q1=float(np.percentile(nv, 25)),
                       visits_q3=float(np.percentile(nv, 75)), visits_min=int(nv.min()), visits_max=int(nv.max()))
    aids = _rda(ROOT / "data/raw/jm/data/aids.rda", "aids")
    pro = _rda(ROOT / "data/raw/jm/data/prothro.rda", "prothro")
    out["raw_facts"] = dict(aids_rows=int(len(aids)), aids_patients=int(aids.patient.nunique()),
                            aids_cd4_max=float(aids.CD4.max()),
                            aids_cd4_noninteger=bool((aids.CD4.dropna() % 1 != 0).any()),
                            prothro_rows=int(len(pro)), prothro_patients=int(pro.id.nunique()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
