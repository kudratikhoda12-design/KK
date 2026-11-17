"""D168 Stage B / D169: the fresh acceptance gate of the frozen mgcv reference AFCM. ONE seed (6001), ONE attempt.

Procedure (unchanged CK procedure): S-Add, DGP v2, n 2,000, Round-11 60/20/20 split on seed 6001, smoothed curves on
the training subjects; the frozen reference AFCM (scripts/afcm_mgcv.R through kanfae.models.afcm_mgcv, files pinned by
SHA-256 in docs/d169_stageB_preregistration.md) fitted on the training subjects; identifiable RISE of phi_1 and phi_2
at the true test curves (tests/test_round12_ck.shapes, the CK-6c metric).
PASS iff phi_1 < 0.15 AND phi_2 < 0.10 (the original CK-6c limits). Anything else, including a failed fit, is FAIL.
Guards: refuses to run if the pre-registration is not committed, if a frozen file's hash differs, or if a result exists.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
SEED, N = 6001, 2000
PHI1_LIMIT, PHI2_LIMIT = 0.15, 0.10
PREREG = ROOT / "docs/d169_stageB_preregistration.md"
OUT = ROOT / "reports/d168/stageB/result.json"
FROZEN = ("scripts/afcm_mgcv.R", "src/kanfae/models/afcm_mgcv.py")


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def guards():
    if OUT.exists():
        raise SystemExit(f"Stage B has already been run ({OUT}); one attempt only (D169).")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PREREG.relative_to(ROOT))], cwd=ROOT,
                             capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PREREG.relative_to(ROOT)), *FROZEN], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit("D169 pre-registration or a frozen file is not committed; Stage B not run.")
    text = PREREG.read_text()
    for f in FROZEN:
        if sha(f) not in text:
            raise SystemExit(f"{f} differs from the hash registered in D169; Stage B not run.")


def main():
    guards()
    import numpy as np
    import test_round11 as R11
    from test_round12_ck import shapes
    from kanfae.data import fsim
    from kanfae.models.afcm_mgcv import AFCMReference
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    dat = fsim.generate("S-Add", n=N, seed=SEED)
    tr, va, te = R11._split(N, SEED)
    xg, lam = R11._smoothed(dat, tr)
    ptr, pte = R11._pack(dat, xg, tr), R11._pack(dat, xg, te)
    y, d = np.asarray(ptr["y"]), np.asarray(ptr["d"], bool)
    ref = AFCMReference().fit(ptr, pte)
    rec = dict(stage="B", decision="D169", seed=SEED, n=N, n_train=len(tr), n_test=len(te), git_head=head,
               frozen_sha256={f: sha(f) for f in FROZEN}, smoothing_lambda=lam,
               tied_event_times_train=int(d.sum() - len(np.unique(y[d]))), reference_info=ref.info,
               limits=dict(phi1=PHI1_LIMIT, phi2=PHI2_LIMIT))
    if ref.ok and ref.finite:
        s = shapes(ref, dat.x_true_grid[te])
        rec["shapes"] = s
        rec["passed"] = bool(s["phi1"]["rise_id"] < PHI1_LIMIT and s["phi2"]["rise_id"] < PHI2_LIMIT)
    else:
        rec["passed"] = False
        rec["failure"] = "reference fit failed or not finite"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["verdict"] = "PASS" if rec["passed"] else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("seed", "verdict", "seconds")} | {"shapes": rec.get("shapes")}, default=float))


if __name__ == "__main__":
    main()
