# D169 pre-registration: Stage B, the fresh acceptance gate of the reference AFCM

Owner approval of 2026-11-17 ("Execute Stage B — Fresh Acceptance Gate (only if Stage A is GO)"). Written and committed
**before seed 6001 is used** by any computation. Protocol: `docs/d168_afcm_reference_protocol.md`, Section 5; Stage A
outcome GO (`docs/d168_stageA_development.md`, commit 5d72b37).

| Item | Rule (fixed) |
|---|---|
| Seed | **6001**, one seed, one attempt. Verified unused on 2026-11-17: the string occurs in the repository's records only inside decimal numbers; no record has `seed` or `data_seed` 6001 |
| Procedure | the CK procedure unchanged: `fsim.generate("S-Add", n = 2000, seed = 6001)` (DGP v2), Round-11 60/20/20 split with seed 6001 (`tests/test_round11._split`), smoothed curves on the training subjects (`tests/test_round11._smoothed`) |
| Model | the frozen reference AFCM: `scripts/afcm_mgcv.R` SHA-256 `cacacfe61670851e36962cdd4f35122044b74c49eac8cf807fd17c4bc657f6c9`; `src/kanfae/models/afcm_mgcv.py` SHA-256 `5bb26016cc3edaf2032aba84526c23df4c19e55e9fae4d15e6b1a181707b45a7`; fitted on the 1,200 training subjects; mgcv 1.9-5 in R 4.4.3 |
| Metric | identifiable RISE of φ₁ and φ₂ at the true test curves (`tests/test_round12_ck.shapes`, the CK-6c metric) |
| **Criterion** | **PASS iff identifiable RISE φ₁ < 0.15 AND φ₂ < 0.10** (the original CK-6c limits, unchanged). A failed, non-finite or errored fit is FAIL |
| If FAIL | recorded as a failure; **Stage C is not run**; no second seed is tried without a new owner decision |
| Meaning | an implementation acceptance check of the reference comparator on one data set. It is not evidence about B2-S and not evidence of general accuracy |
| Known before the run | on the three development seeds (Stage A) the reference had φ₁ 0.2255, 0.1382 and 0.1864 and φ₂ 0.0417, 0.0427 and 0.0234; a FAIL on φ₁ is therefore plausible. The limits are not changed because of this |
| Runner | `scripts/d168_stageB.py`: refuses to run unless this file and the frozen files are committed and their hashes match, and refuses a second attempt; writes `reports/d168/stageB/result.json` |
| Historical records | CK-6c, CK-6d, H-R12 and all earlier results are unchanged |
