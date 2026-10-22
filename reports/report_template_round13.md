<p class="small">Research prototype report, revision 15 · generated from saved records by
<code>scripts/build_report_round13.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A is Round 13, the real-data study (D114 as amended by D128 and D129;
outcome D130). <b>Every Round-13 result is exploratory.</b> Part B is revision 14's Part A (Round 12); Parts C–D are the
L39 and L38 parts; Parts E–O are the round-11 to round-1 reports.</p>

## 0. Round-13 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Data | PBC `pbcseq` (730-day landmark; E2 death or transplant primary, E1 death sensitivity; 278 patients), JM `prothro` (1 year; 332) and JM `aids` (6 months; 403), each over 5 outer splits; C-MAPSS FD001 Option B (5 splits by engine) |
| Execution | {{V:n_records}} records; one failed cell (the PACE sensitivity model on PBC split 1: non-finite PACE curves). Everything else ran |
| Prediction (single landmark) | **No functional or KAN model was clearly better than the simple baselines.** Nadeau–Bengio intervals for B2-S Cox minus each baseline include 0 for IBS on every data set. On PBC the static landmark Cox model on the last values had the best IBS |
| Selection and symbolic (B2-S Cox) | On PBC, pruning kept bilirubin, albumin and platelets in every split. Substituting the symbolic formulas **without** refit kept the model (risk correlation 0.999 with the pruned model; same IBS and Uno C); the refit degraded it, as in Round 12 |
| Stability and clinical alignment | The bilirubin and albumin effects were stable and aligned with the stated clinical directions. The prothrombin-index effect (prothro) aligned for B1 but not for B2-S (72% of fits, threshold 80%). Details in R13.5 |
| B3 stacked landmarks | The KAN supermodel was not better than the linear landmark-stratified Cox supermodel (R13.6) |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What this means

1. **Small cohorts give no room for flexible functional effects.**
   - The cohorts are small (278–403 patients, 126–183 events) with a median of 3 visits per subject.
   - On these data the flexible functional models (B2, AFCM) do not predict better than a Cox model on the last
     observed values.
   - This is consistent with the simulation finding that B2-S and AFCM agree with each other but need the shape
     signal to matter.
2. **The interpretability measures are informative even without a predictive gain.**
   - Pruning selects a stable, clinically plausible subset of PBC markers.
   - Symbolic substitution without refit preserves the model exactly.
   - The leading effects (bilirubin, albumin, CD4) are stable across 100 bootstrap fits and point in the clinically
     expected direction.
3. **No novelty or superiority claim follows** (D111).

## R13.1 Owner decisions (D128, approved) and plan (D129)

- **D128 (L41 closed; defaults confirmed by the owner):**
  - H-R12 stays inconclusive.
  - Round 13 is exploratory.
  - AFCM stays as a baseline flagged "not validated".
  - The primary symbolic model is the substituted model without refit.
- **Data facts recorded in D128:**
  - The `aids` CD4 values are already square-root transformed.
  - `aids` has 1,405 rows, not the 1,408 its documentation states.
  - `prothro` has 488 patients (its documentation says both 488 and 467).
- **D129** fixed the operational plan before any real-data fit, after a mechanics dry run on synthetic data shaped
  like the real records.

## R13.2 Execution

{{T:exec_table}}

## R13.3 Single-landmark prediction (5 outer splits; mean over splits)

**PBC, E2 (primary)**

{{T:single_pbc_E2}}

**PBC, E1 (sensitivity)**

{{T:single_pbc_E1}}

**prothro**

{{T:single_prothro}}

**aids**

{{T:single_aids}}

**B2-S Cox minus each baseline** (Nadeau–Bengio corrected interval over the 5 splits; exploratory)

{{T:contrast_table}}

## R13.4 Channel selection and symbolic formulas (B2-S Cox, seed 0, per split)

{{T:prune_table}}

**Accepted formulas carrying ≥ 5% of their layer's score, split 0** (x in the original units of the input: log scale
for the log-transformed PBC markers):

{{T:formula_table}}

## R13.5 Stability, clinical alignment and effect plots (100 bootstrap fits per model and data set)

{{T:stab_table}}

{{T:figures}}

## R13.6 B3 stacked-landmark supermodel (mean over 5 splits)

{{T:b3_table}}

## R13.7 C-MAPSS FD001 (Option B; calibration)

{{T:cmapss_table}}

## R13.8 Limitations

- **Small cohorts, few visits.** The real-data cohorts are small with few visits per subject, so all intervals are
  wide. All contrasts are exploratory, with no multiplicity correction.
- **AFCM is not validated** (CK-6c FAIL).
- **PACE failed on one PBC split**, a PBC visit design with large gaps.
- **Clinical alignment is a coarse monotonicity check** (D114). Clinical sign-off is external.

## R13.9 Decisions (D128–D130)

{{T:decisions}}

## R13.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R13.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–O: historical records

Part B is revision 14's Part A (Round 12); Parts C and D are the L39 and L38 parts; Parts E–O are the round-11 to
round-1 reports, each rendered from its own records. Embedded living documents show their current version.
