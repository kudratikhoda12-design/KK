<p class="small">Research prototype report, revision 24 · generated from saved records by
<code>scripts/build_report_round22.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; both manuscripts are
drafts for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 22: per-channel REML AFCM and
CK-6d, the untouched joineRML cohorts renal and heart.valve with C-MAPSS FD003 Option B, and NP-KAN-AFT under three
further error laws (D156–D159). Part A continues with the journal main manuscript, embedded, and the verification of
both manuscripts; the cumulative Round-22 manuscript is its Supplementary Material (`reports/manuscript_round22.pdf`).
Part B is revision 23's Part A (Round 21 and the Round-21 manuscript); Parts C–X are revision 23's Parts B–W.</p>

## 0. Round-22 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Design (D156) | Registered before any Round-22 fit: per-channel REML AFCM with a development gate on seeds 22, 914, 966 before the held-out seed 24 (CK-6d) and the confirmatory re-test H-R22a; the locked Round-21 models on two untouched joineRML cohorts and C-MAPSS FD003 (H-R22b); NP-KAN-AFT under bimodal, t₃ and logistic errors on fresh seeds 5000–5019 (H-R22c) |
| Fact check | R and mgcv not available (Python Fellner–Schall implementation); FD003 was analysed in Round 3, so it is not untouched; the decision-log truncation came from unescaped pipes in table cells, not from `<`; the named distillation options do not exist (the locked Round-21 function is used); the Round-21 manuscript template's Appendix A named Round-20 files |
| AFCM gate and CK-6d | Gate **{{V:gate}}** on the development seeds; CK-6d on seed 24 **{{V:ck6d}}** (φ₁ {{V:ck_phi1}} against 0.15, φ₂ {{V:ck_phi2}} against 0.10); kept as a visible test failure |
| H-R22a | {{V:hA}} |
| Execution | {{V:n_ok}} registered jobs completed, {{V:n_failed}} failed |
| H-R22b (external cohorts) | **{{V:hB}}** (Section R22.4) |
| H-R22c (three error laws) | **{{V:hC}}** (Section R22.5) |
| Journal main manuscript | `docs/manuscript_journal_main.md` / `reports/manuscript_journal_main.pdf` ({{V:j_pages}} pages), embedded below: {{V:j_values}} values, each also a verified value of the Supplementary Material, {{V:j_refs}} references, {{V:j_check}} |
| Supplementary Material | `docs/manuscript_round22.md` / `reports/manuscript_round22.pdf` ({{V:ms_pages}} pages), Rounds 1–22: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c, the 4f gate's original R1 and CK-6d), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

- **AFCM is still unvalidated.** Giving every channel and direction its own smoothing penalty fixed the development
  seeds but not the held-out seed 24, so the confirmatory AFCM re-test (H-R22a) was not run and the original
  comparison of B2-S with AFCM (H-R12) stays inconclusive.
- **On the two new cohorts the functional models were neither better nor worse than the simple baselines.** Every
  interval for the difference in conditional IBS includes 0. On renal, where patients have many measurements before
  the landmark, B2-S Cox had a lower mean than landmark Cox on four of five splits; on heart valve, with about one
  visit per patient, the two were almost equal. Five splits cannot turn this into a finding.
- **H-R22b failed on a technicality that the rule makes binding.** Anchoring and distillation met their safety margins
  on both cohorts (parts 2 and 3), but one baseline, summary Cox, did not converge on one heart-valve split, and part 1
  required every model and baseline to be finite. The verdict is "not supported" as registered; the failed fit was
  not rerun with a different penalty (D158).
- **NP-KAN-AFT is clearly better when no parametric error law fits (H-R22c supported).** Under a bimodal error law it
  had less than half the survival-curve error of the Cox, Weibull and log-normal KAN models in every replicate; under
  heavy-tailed (t₃) and logistic errors it was better than Cox and Weibull and close to log-normal, its advantage
  shrinking as the error law approaches the normal. These are simulated laws only.
- **FD003** (not untouched) ranked NP-KAN-AFT first in mean IBS, but no model was well calibrated (D-calibration
  passed on at most one split of five).
- **Execution.** All 95 jobs finished after four relaunches caused by container restarts; no code, setting, seed or
  rule was changed by a relaunch (pre-registration §5).

# Part A: Round 22

## R22.1 Owner instruction (D156) and fact check

- **Instruction (2026-11-06).** Task 1: per-channel REML AFCM (`fit_afcm(..., selection="reml_per_channel")`, default
  off), optional bagging, development gate on seeds 22, 914, 966, then CK-6d on seed 24 and H-R22a on S-Add 3000–3019
  only if the gate passes. Task 2: the untouched joineRML cohorts renal and heart.valve and C-MAPSS FD003 Option B with
  the locked Round-21 models and five baselines (H-R22b). Task 3: NP-KAN-AFT under bimodal, t₃ and logistic errors
  (H-R22c). Task 4: tests R22-1..R22-4, Appendix A and decision-log fixes, the cumulative manuscript, a journal main
  manuscript of 18–22 pages, report revision 24, the full suite; commit locally, do not push.
- **Fact check** (`docs/round22_preregistration.md` §1): seed 24 and seeds 5000–5019 unused (checked in every record);
  Rscript absent; joineRML 0.4.8 downloaded from CRAN with checksums; renal 407 patients / 126 graft failures, all
  followed beyond one year; heart.valve 256 / 54, one-year landmark 175 / 39 chosen by the pre-stated rule, a median of
  one visit before the landmark; FD003 previously analysed; the decision-log truncation was caused by unescaped `|`
  inside table cells (fixed by escaping, D156).

## R22.2 Methods (as implemented)

- `kanfae.models.afcm.fit_afcm(..., selection="reml_per_channel")` → `fit_afcm_reml`: one λ_x and one λ_t per channel,
  common-λ LAML grid pilot, Fellner–Schall updates with warm refits, convergence on the change in log λ or in the
  training linear predictor, otherwise the best-LAML iterate; `fit_afcm_reml_bagged` (bootstrap, B = 3). Default fits
  unchanged (hash-checked, R22-1).
- `kanfae.data.fsim`: scenarios `S-Misspec-bimodal`, `-studentt3`, `-logistic05` (mean 0, SD 0.5; oracle survival per
  law); every earlier scenario bit-identical (R22-2). `kanfae.data.realdata`: `load_renal`, `load_heart_valve` (R22-3).
- Harness `scripts/run_round22.py` (lanes `scripts/run_round22_lanes.py`); analysis `scripts/analyze_round22.py` →
  `reports/round22/analysis_summary.json`, `docs/round22_results.md`.

## R22.3 Task 1: per-channel REML AFCM, gate and CK-6d

{{T:afcm}}

## R22.4 Task 2: external cohorts and FD003 (H-R22b)

{{T:cohorts}}

**C-MAPSS FD003 Option B** (previously analysed in Round 3; descriptive)

{{T:fd003}}

## R22.5 Task 3: three further error laws (H-R22c)

{{T:sims}}

## R22.6 Hypotheses

{{T:hyp}}

## R22.7 Limitations

- **AFCM.** The per-channel estimator passed the development seeds and failed the held-out check; it is a Python
  implementation (no mgcv), so H-R12 stays inconclusive and AFCM unvalidated.
- **External data.** heart.valve has a median of one visit per patient before the landmark; FD003 is not untouched; five
  splits cannot power superiority claims (D111); every comparison beyond the registered criteria is exploratory.
- **Simulation.** The three error laws share the S-Add location and one censoring mechanism.
- **No novelty claim (D111).**

# Part A (continued): the journal main manuscript (Round 22)

## {{V:j_title}} {#manuscript-title-journal}

{{MANUSCRIPT}}

## R22.8 Manuscript verification

{{T:ms_verification}}

## R22.9 Decisions (D156–D159)

{{T:decisions}}

## R22.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R22.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–X: historical records

Part B is revision 23's Part A (Round 21 and the Round-21 manuscript); Part C is revision 22's Part A (Round 20 and the
Round-20 manuscript); Part D is Round 19 and its manuscript; Part E is Round 18 and its manuscript; Part F is Round 17
and its manuscript; Part G is Round 16 and its manuscript; Part H is Round 15; Part I is Round 14 (the first manuscript
draft); Part J is Round 13; Part K is Round 12; Parts L and M are the L39 and L38 parts; Parts N–X are the round-11 to
round-1 reports, each rendered from its own records. Embedded living documents show their current version.
