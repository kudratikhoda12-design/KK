"""D71 pre-registered search for Revision 4b (compact frequency-aware front + survival-path generalisation fix).
Seed-21 TRAIN/VALIDATION only. One factor at a time, D41 selection rule. Records: reports/learnability_search_fa4b.jsonl
(resumable); selection: reports/learnability_selected_fa4b.json.

Base = revision-4 selection with the compact summary: front fa, fa_compact, 17 frequencies, r 12, alpha 1, Adam 3e-3,
wd 1e-4, early stopping on the joint objective, 3000 epochs / patience 100, H 32, hidden 64.
G1 early-stopping target {objective, nll}; G2 survival-path dropout {0, 0.1, 0.25}; G3 head weight decay
{global, 1e-2, 1e-1}; G4 global weight decay {1e-4, 1e-3}; G5 latent r {8, 12, 16}; G6 seeds 1, 2;
G7 information only: fae_linear, fae_kan.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import learnability_search_fa as S  # noqa: E402

S.OUT = ROOT / "reports" / "learnability_search_fa4b.jsonl"
BASE = dict(S.BASE, front="fa", fa_compact=True, fa_nfreq=17, r=12)


def steps(run, best):
    c = best("G1-early-stopping", [dict(BASE), dict(BASE, es_target="nll")])
    c = best("G2-head-dropout", [dict(c), dict(c, head_dropout=0.1), dict(c, head_dropout=0.25)])
    c = best("G3-head-wd", [dict(c), dict(c, head_wd=1e-2), dict(c, head_wd=1e-1)])
    c = best("G4-wd", [dict(c), dict(c, wd=1e-3)])
    c = best("G5-latent", [dict(c, r=8), dict(c), dict(c, r=16)])
    for s in (1, 2):
        run("G6-stability", "fae_mlp", c, seed=s)
    for kind in ("fae_linear", "fae_kan"):
        run("G7-encoder-info", kind, c)
    return c


if __name__ == "__main__":
    S.main(steps=steps, selected=ROOT / "reports" / "learnability_selected_fa4b.json")
