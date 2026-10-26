"""Round 16 post-hoc description (exploratory; no verdict changes; written after the formal analysis, D139):
where the S-Sparse loss of b2s_anchor_cox (non-cliff gate) against B2-S Cox comes from, split by the gate decision,
and the formulas rejected by the |a x + b| <= 50 rule. -> reports/round16/posthoc_exploratory.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_round12 import paired  # noqa: E402
from analyse_round15 import per, sim  # noqa: E402
from analyse_round16 import load  # noqa: E402


def main():
    recs = load()
    out = {}
    for kind in ("sparse", "last"):
        R = sim(recs, kind)
        a = per(R, "family_b", "b2s_anchor_cox_noncliff", "curve_error")
        b = per(R, "b2s_cox", "curve_error")
        kept = {s for s, r in R.items() if sum(r["gates"]["b2s_anchor_cox_noncliff"]["residual_on"]) > 0}
        for lab, keep in (("gate kept >= 1 channel", True), ("gate all off", False)):
            sel = {s for s in R if (s in kept) == keep}
            out[f"{kind}: anchor - b2s, {lab}"] = paired({s: a[s] for s in sel}, {s: b[s] for s in sel})
        pat = {}
        for r in R.values():
            k = "".join(str(int(x)) for x in r["gates"]["b2s_anchor_cox_noncliff"]["residual_on"])
            pat[k] = pat.get(k, 0) + 1
        out[f"{kind}: gate pattern counts (channels 1, 2, 3)"] = pat
    out["rejected_formulas"] = [dict(job=r["job_name"], **f) for r in recs if r["status"] == "ok"
                                for f in r.get("symbolic", {}).get("formulas_r15_rejected", [])]
    (ROOT / "reports/round16/posthoc_exploratory.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: (dict(n=v.get("n"), mean_diff=v.get("mean_diff"), ci95=v.get("ci95"), n_pos=v.get("n_pos"))
                          if isinstance(v, dict) and "n" in v else v) for k, v in out.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
