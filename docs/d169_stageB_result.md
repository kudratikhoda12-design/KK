# D169 Stage B result: PASS (2026-11-17)

Run once, as pre-registered (`docs/d169_stageB_preregistration.md`, commit 4e3f475), by `scripts/d168_stageB.py` at
commit 4e3f475 with the frozen reference files (hashes checked by the runner). Record: `reports/d168/stageB/result.json`.

| Item | Value |
|---|---|
| Seed | 6001 (one attempt) |
| Data | S-Add, DGP v2, n 2,000; 1,200 training subjects; no tied event times |
| Reference fit | mgcv 1.9-5, REML "full convergence" in 8 outer iterations, max \|REML gradient\| 8.6 × 10⁻⁴, no warnings, finite |
| Identifiable RISE φ₁ | **0.1278** (limit < 0.15) |
| Identifiable RISE φ₂ | **0.0204** (limit < 0.10) |
| Raw RISE φ₁ / φ₂ | 0.1286 / 0.0203 |
| Run time | 38.8 s |
| **Verdict** | **PASS**: both limits met |

**How to read it.** This is an implementation acceptance check on one data set: the reference comparator met the
limits that the Python V3 comparator failed (CK-6c, seed 22). It is not evidence about B2-S. On the three development
seeds of Stage A the reference's φ₁ was 0.2255, 0.1382 and 0.1864, so passing on one fresh seed does not show that it
meets the φ₁ limit reliably; Stage C reports its shape error over 110 further data sets. Stage C follows only after
the D170 pre-registration is committed.
