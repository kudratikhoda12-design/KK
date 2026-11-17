# D170 pre-registration: Stage C, the paired comparison of B2-S Cox with the reference AFCM (H-R25a)

Owner approval of 2026-11-17 (n = 110 on seeds 6100–6209). Written and committed, with the harness, the analysis
script and their tests, **before any Stage-C seed is used**. Protocol: `docs/d168_afcm_reference_protocol.md`, Section 6;
Stage A GO (commit 5d72b37); Stage B PASS on seed 6001 (commit 5a76c81).

## Design

| Item | Rule (fixed) |
|---|---|
| Replicates | **110**, seeds **6100–6209**, each used once. Verified unused on 2026-11-17 (no record in `reports/` has `seed` or `data_seed` in this range; test C-1) |
| Data | the Round-12 / H-R12 design exactly: `scripts/run_round12.Ctx("S-Add", 0, dev_seed = s)`, i.e. S-Add, DGP v2, n 1,000, **data seed s**, `split_groups` 60/20/20 stratified on the event with split seed s, B-spline front end (smoothing parameter chosen on the training subjects), 50-point grid. *Correction to the protocol (Section 6.2): it said data seed `SeedSequence([s, 0])` "as in Round 12"; Round 12 used `SeedSequence` only for S-Misspec and the seed itself for S-Add, which is what this study does* |
| Identical data | each replicate's arrays (grid curves, z, y, δ, the three index sets, true test curves) are written once to `reports/d168/stageC/data/rep_<s>.npz` (git-ignored, SHA-256 in the record), read back and checked equal to the arrays in use; both methods are fitted from them |
| Method a | **B2-S Cox**, three-member ensemble, `scripts/run_round12.run_ensemble(ctx, "b2s", "cox")`, the H-R12 code path unchanged (training subjects for fitting, validation subjects for tuning and early stopping, as in Round 12) |
| Method b | **reference AFCM**, frozen: `scripts/afcm_mgcv.R` SHA-256 `cacacfe61670851e36962cdd4f35122044b74c49eac8cf807fd17c4bc657f6c9`, `src/kanfae/models/afcm_mgcv.py` SHA-256 `5bb26016cc3edaf2032aba84526c23df4c19e55e9fae4d15e6b1a181707b45a7`; mgcv 1.9-5, R 4.4.3; fitted on the training subjects (REML needs no validation subjects) |
| Not included | the Python V3 / REML AFCM (historical, unchanged) and the A7 repair (not accepted in Stage A) |
| Harness | `scripts/d168_stageC.py`, four single-threaded lanes (`--lane k --lanes 4`, seeds by position mod 4), resumable; refuses to start unless Stage B passed and this file, the frozen files, the harness and the analysis script are committed with matching hashes |

## Outcomes and decision rules

- **Primary outcome:** mean identifiable RISE over φ₁ and φ₂ at the true test curves (`run_round12.shape_metrics`
  "mean_id", the H-R12 outcome). Per replicate d = B2-S Cox − reference AFCM.
- **H-R25a (primary):** B2-S Cox and the reference AFCM are equivalent within ±0.05 in mean identifiable RISE, by the
  paired TOST (two one-sided paired t tests at α = 0.05, i.e. the 90% CI inside ±0.05), computed with
  `scripts/analyse_round12.paired` (the H-R12 function). **Supported** iff the TOST p < 0.05 and the analysis is
  evaluable.
- **Always reported:** n, both means, mean difference, SD, 95% and 90% CIs, paired-t p, exact Wilcoxon signed-rank p,
  counts of d < 0 and d > 0.
- **Direction (secondary):** "B2-S lower (better)" only if the 95% CI lies below 0; "B2-S higher (worse)" only if it lies
  above 0; otherwise no direction is claimed.
- **Failures:** a method's replicate fails if it raises an error, returns non-finite predictions, surfaces or metrics,
  or (reference) mgcv does not report "full convergence". Failed fits are recorded and never refitted. Primary analysis
  on complete pairs. **Not evaluable** if more than 11 of the 110 replicates (> 10%) fail for either method, or if
  fewer than 110 replicate records exist.
- **Sensitivity:** each failed replicate set to the worst (largest) mean identifiable RISE observed for that method.
- **Secondary (paired, no decision rule):** identifiable RISE of φ₁ and φ₂ separately, mean raw RISE, survival-curve
  error against the true survival curves, IBS, Harrell C, Uno C; D-calibration pass counts; run times; reference REML
  convergence messages; tied event times.
- **Interpretation limits (fixed now):** equivalence within ±0.05 is a statement about this simulation design only;
  the reference AFCM is the paper's estimator as specified by the author's vignette, not the authors' own complete code.
  H-R12 (+0.0065, 95% CI [−0.0506, +0.0635], 20 replicates, against the Python V3 comparator, inconclusive) is
  unchanged and stays as the historical result.

## Analysis and reporting

`scripts/analyze_d168_stageC.py` → `reports/d168/stageC/analysis_summary.json` and `docs/d168_stageC_results.md`. Tests
`tests/test_d168_stageC.py` (C-1 fresh seeds; C-2 decision rules on synthetic records; C-3 quantile map; C-4 the runner
needs Stage B PASS; C-5 a small reference fit) pass before the launch. Budget: about 4.6 min per replicate (Stage A,
A8), about 2.1–2.5 h wall on four lanes.
