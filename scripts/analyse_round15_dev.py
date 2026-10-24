"""Round 15 (D133): summary of the development-seed dry run (reports/round15/dev/*.jsonl) ->
reports/round15/dev_summary.json and the tables of docs/round15_dryrun.md (rendered by this script)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "reports/round15/dev"


def load():
    recs = []
    for f in sorted(DEV.glob("results_*.jsonl")):
        recs += [dict(json.loads(l), file=f.name) for l in f.read_text().splitlines() if l.strip()]
    return recs


def f4(v):
    return "–" if v is None else f"{v:.4f}"


def f3(v):
    return "–" if v is None else f"{v:.3f}"


def table(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    R = load()
    out = dict(n_jobs=len(R), failed=[r["job_name"] for r in R if r["status"] != "ok"])
    md = []
    rows = []
    for r in R:
        if r.get("job") != "distill":
            continue
        cells = [("after pruning", r["after_pruning"])] + list(r["variants"].items())
        for k, v in cells:
            if v.get("status"):
                rows.append([f"{r['scenario']} {r['seed']}", k, v["status"], "", "", "", "", ""])
                continue
            s = v["shape"]
            rows.append([f"{r['scenario']} {r['seed']}", k, f3(s["mean_id"]), f3(s["phi1"]["rise_id"]), f3(s["phi2"]["rise_id"]),
                         f4(v["curve_error"]), f4(v["ibs"]), f4(v["val_loss"])])
        out[f"distill_{r['scenario']}_{r['seed']}"] = dict(
            chosen_mu=[g["mu"] for g in r["distill_grid"] if g.get("chosen")], n_guarded=r["symbolic"]["n_guarded"],
            n_substituted=r["symbolic"]["n_substituted"], selected=r["selected"], seconds=r["wall_seconds"])
    md.append("**symbolic_distill** (B2-S Cox, seed-0 member after pruning; identifiable RISE; validation NLPL is the "
              "unpenalised loss)\n\n" + table(["Data", "Model", "Mean RISE", "RISE φ₁", "RISE φ₂", "Curve error", "IBS",
                                                "Validation NLPL"], rows))
    rows = []
    for r in R:
        if r.get("job") not in ("res", "res_dev_real"):
            continue
        for k, v in r["models"].items():
            if k in ("b2s_res_cox", "b2s_anchor_cox") or "followup" not in r["file"]:
                rows.append([f"{r['scenario']} {r['seed']}", k, f4(v.get("curve_error")), f4(v["ibs"]), f4(v["harrell_c"]),
                             f4(v["uno_c"]), f3(v["dcal_p"]),
                             str(r["res_on"]) if k in ("b2s_res_cox", "b2s_anchor_cox") else ""])
        key = "anchor" if "followup" in r["file"] else "res"
        out[f"{key}_{r['scenario']}_{r['seed']}"] = dict(gate_on=r["res_on"], models=r["models"], seconds=r["wall_seconds"])
    md.append("**b2s_res_cox and the anchored follow-up** (curve error only on simulated data)\n\n" + table(
        ["Data", "Model", "Curve error", "IBS", "Harrell C", "Uno C", "D-cal p", "Functional channels kept by the gate"], rows))
    rows = []
    for r in R:
        if r.get("job") == "afcm":
            for k, lab in (("v3", "V3 (single fit)"), ("v3_bag3", "afcm_ens: 3-resample bag of V3"),
                           ("v3_pc_t", "V3, per-channel λ_t")):
                s = r[k]["shapes"]
                rows.append([str(r["seed"]), lab, f3(s["phi1"]["rise_id"]), f3(s["phi2"]["rise_id"]),
                             str(r[k].get("lam_t", [q["lam_t"] for q in r[k].get("resamples", [])]))])
                out[f"afcm_{r['seed']}_{k}"] = dict(phi1=s["phi1"]["rise_id"], phi2=s["phi2"]["rise_id"])
        if r.get("job") == "afcmpars":
            s = r["v3_pars"]["shapes"]
            rows.append([str(r["seed"]), "V3, D120 parsimonious λ rule (tol 0.01)", f3(s["phi1"]["rise_id"]),
                         f3(s["phi2"]["rise_id"]), str(r["v3_pars"]["lam_t"])])
            out[f"afcm_{r['seed']}_v3_pars"] = dict(phi1=s["phi1"]["rise_id"], phi2=s["phi2"]["rise_id"])
    rows.sort(key=lambda x: (x[0], x[1]))
    md.append("**afcm_ens and alternatives** (the CK-6c procedure, n 2,000; identifiable RISE; CK-6c criterion "
              "φ₁ < 0.15 and φ₂ < 0.10)\n\n" + table(["Seed", "AFCM variant", "RISE φ₁", "RISE φ₂", "Selected λ_t"], rows))
    (ROOT / "reports/round15/dev_summary.json").write_text(json.dumps(out, indent=1, default=str))
    (ROOT / "reports/round15/dev_tables.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
