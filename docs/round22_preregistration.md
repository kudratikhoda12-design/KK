# Round 22 pre-registration (D156): per-channel REML AFCM, external longitudinal cohorts, multi-law NP-KAN-AFT

Owner instruction of 2026-11-06 ("Round 22: AFCM per-channel REML validation, untouched external longitudinal cohorts,
multi-law NP-KAN-AFT simulations, and journal manuscript"). This plan, the harness (`scripts/run_round22.py`,
`scripts/run_round22_lanes.py`), the analysis (`scripts/analyze_round22.py`) and the tests (`tests/test_round22.py`,
`tests/test_round22_ck.py`) are committed **before** any registered seed (22/914/966 for the gate, 24, 3000–3019,
5000–5019) or external cohort (renal, heart.valve, FD003) is used by a Round-22 model fit. Development mechanics were
checked on non-registered data only (S-Add seed 11, n 400; fsim seed 1 with reduced budgets; the Round-13 synthetic
`dev_aids` and `dev_cmapss` sets).

## 1. Pre-run checks and fact check of the instruction

| Statement in the instruction | Finding (records) |
|---|---|
| Seed 24 and seeds 5000–5019 never used | **Confirmed.** Structural scan of all 278 JSON/JSONL record files (every key containing "seed", and rep/replicate/split): neither value occurs. Code/test/script scan: seed 24 appears only as text (the Round-15 docstring and the Round-14 verifier's registry, both stating it is unused); no integer range in `src/`, `scripts/` or `tests/` covers 5000–5019 as seeds (5000 occurs only as sample sizes and epoch counts). |
| `which Rscript`; `mgcv` reference | **Rscript is not installed** (`which Rscript`, `which R`: none). The `mgcv` reference surface errors are therefore not computed; R is not installed (it would modify the system outside the project). |
| 39 Round 15–21 deterministic tests pass before any change | **Confirmed**: `tests/test_round15.py` .. `test_round20.py` (35) + `tests/test_round21_real_npaft.py` (4) = 39 passed (`reports/logs/r22_prerun_regression.log`, 524 s). |
| V3 and shared-penalty LAML failed CK-6c; per-channel validation selection overfit | V3 on seed 22: identifiable RISE φ₁ 0.247 / φ₂ 0.053 (CK-6c FAIL, D123/D126). LAML (D134 check C2): seed 22 0.158 / 0.094, 914 0.057 / 0.058, 966 0.190 / 0.048 → FAIL (D135). Per-channel λ chosen by validation: D123 V4 on seed 914 0.040 / 0.044 ("4× the fits without improving φ₁"); Round-15 `v3_pc_t` (per-channel λ_t): 22: 0.240 / 0.108, 914: 0.224 / 0.048, 966: 0.065 / 0.056. The records show that per-channel validation selection did not help φ₁; "overfit" is an interpretation, not a tested finding. The φ₁/φ₂ trade-off under one shared λ_t is documented (D126 (b), exploratory). |
| CK-6c procedure n 2,000 | Confirmed (`tests/test_round12b_ck.py`). |
| H-R12 on "20 S-Add replicates, seeds 3000–3019" | Confirmed, **n 1,000** per replicate (D126); quantity B2-S Cox ensemble − AFCM in mean identifiable RISE over φ₁, φ₂; rule D113 (TOST ±0.05); outcome inconclusive (D127). These replicates were analysed in Round 12 with AFCM V3, so H-R22a is **not a fresh-data test**: the comparator is new, the data are not. |
| joineRML `renal`, 407 patients; `gfr`, `haem`; `age`, `weight`, `sex`; `failure`, `fuyears` | joineRML **0.4.8** (current CRAN) downloaded to `data/raw/joinerml_download/` (MD5 203e8511…b06b4c8e equals the CRAN index; SHA-256 5391b059…2fdeb6c3, recorded in `reports/run_manifest.json`); data, DESCRIPTION and man pages extracted to `data/raw/joinerml/` as for JM 1.5-2 (D128). `renal` is a list of four tables (`prot`, `haem`, `gfr`, `surv`); 407 patients, **126 graft failures**; the sex variable is `gender`; the time column of the long tables is `years` (the Rd says `time`). **Every patient has follow-up beyond 1 year** (minimum 1.04 years): the 1-year landmark cohort is all 407 (126 events). Median 97 merged observation times per patient before 1 year (GFR and haematocrit are often measured on different days). Reference Rizopoulos & Ghosh (2011) confirmed in the Rd. |
| `heart.valve` (Lim et al. 2008), landmark 1 year or the rule (≥ 150 at risk, ≥ 35 events) | 256 patients, 988 rows, **54 deaths**. Landmark cohort (y > t_L and ≥ 1 measurement in [0, t_L]): t_L 0.5: 135 / 32; **t_L 1.0: 175 / 39**; 1.5: 200 / 45; 2.0: 202 / 47. t_L = 1.0 meets the rule, so it is used (fixed from these counts only). `hs` is stored as "Homograft" / "Stentless valve" (coded homograft = 1); `sex` 1 = female. `log.grad` is missing in 359 of 988 rows; 38 of the 175 cohort patients have no gradient before 1 year (the Round-13 rule: the training-mean curve). Median **1** measurement per patient before 1 year (maximum 3): little longitudinal information. |
| FD003 "untouched" | **Not correct.** FD003 Options A and B were run in Round 3 (D51; 290 Option-B records in `reports/experiment_results.jsonl`) with the Round-3 models. FD003 is kept as instructed but reported as **previously analysed, not a prospective holdout**; FD002/FD004 are untouched but need operating-condition normalisation (D51), a new preprocessing design not improvised here. |
| `distill_fix_mu=True, nan_safe_refit=True, symbolic_gate_mode="two_way"` | These options do not exist. The locked Round-21 distillation is used: `fkan_surv.distill_and_replace` (stages 1–2, guarded substitution, fixed μ = 10 (D138), NaN-safe refit, validation-loss rollback). The "two-way gate" is the Round-17 anchoring gate (D141), not part of distillation. |
| Baselines landmark_cox, fpca_cox, lme2_cox, sum_cox, sum_rsf on all cohorts | These are the Round-13 single-landmark baselines (clinical cohorts: run as listed). The Round-13 C-MAPSS baselines were sum_aft and fpca_aft; FD003 uses landmark_cox (last values + u), sum_aft and fpca_aft. |
| Round-21 table truncation caused by raw `<` in D153/D154 | **Not the cause.** The cells were split by unescaped `\|` inside cells ("max \|diff\|", "log\|S\|", "E[ε \| …]", "round13\|15"). Rows D42, D50, D134, D144, D147, D153, D154 had extra cells; the pipes are now escaped (`\|`; rendering only, text unchanged). D149 has one cell fewer (no basis column) and is left as recorded. |
| Appendix A of the Round-21 manuscript names Round-20 files | **Confirmed** (an error of Round 21); corrected in the Round-22 template. The Round-21 manuscript is kept as recorded. |
| "log-logistic / heteroscedastic errors" | The detailed specification (standard logistic, standardised to SD 0.5) is implemented; no heteroscedastic law. The Round-12 law "S-Misspec-loglogistic" is 0.5 × standard logistic (SD 0.907); the new one is named `S-Misspec-logistic05`. The other names: `S-Misspec-bimodal`, `S-Misspec-studentt3`. |
| NP-KAN-AFT so far tested under one error law | Confirmed (D151: log-normal only). |

## 2. Locked designs

### 2.1 Task 1: per-channel REML AFCM (CK-6d, H-R22a)

- **Method** (`src/kanfae/models/afcm.py`, default off; `fit_afcm` and `fit_afcm_laml` reproduce their earlier fits bit
  for bit, test R22-1): `fit_afcm(train, val, selection="reml_per_channel", **AFCM_D123)` = `fit_afcm_reml`. AFCM_D123's
  basis (10 × 10 cubic B-spline tensor per channel), penalties (second differences in x, first + second in t,
  normalised by basis size) and ridge 1e-6 (null space, fixed). Pilot: the common (λ_x, λ_t) maximising LAML on the
  AFCM_D123 grid (9 × 5; the D134 C2 choice). Then Fellner–Schall updates (Wood & Fasiolo 2017) of the six per-channel
  parameters λ_{x,d}, λ_{t,d}: λ_j ← λ_j [tr(S⁻¹S_j) − tr((H + S)⁻¹S_j)] / (θ_dᵀS_jθ_d), S in log-likelihood units (2E ×
  the fitted penalty), H the Breslow observed information of the partial likelihood, each update followed by a
  warm-started refit; bounds [1e-9, 1e9]; convergence when max |Δ log λ| < 0.01 or the training linear predictor moves by
  < 1e-3 (sup norm) (a λ that no longer moves the fit, e.g. λ_t of the null channel once its λ_x reached the bound, can
  oscillate in a flat direction of LAML; seen on the development sample, seed 11); at most 100 updates, otherwise the
  iterate with the largest finite LAML. Training data only (no validation split).
- **Variants, in priority order (fixed now):** (1) `afcm_reml` (the single fit above); (2) `afcm_reml_ens`
  (`fit_afcm_reml_bagged`: 3 bootstrap resamples of the training subjects, seed 0, each fitted by the REML procedure on
  one shared design, coefficients averaged, Breslow baseline on the full training set).
- **Development gate** (seeds 22, 914, 966; CK-6c procedure, n 2,000; jobs `afcmdev_<s>`): **GO iff** one variant meets
  identifiable RISE φ₁ < 0.15 AND φ₂ < 0.10 on **all three** seeds; the GO variant is the first in the priority order
  that does. Reproduction check: the REML pilot's (λ_x, λ_t) equals the Round-15 C2 LAML choice on each seed.
- **If GO:** CK-6d (`tests/test_round22_ck.py`) on seed 24 with the GO variant, same procedure and criterion. If CK-6d
  passes: **H-R22a** on the Round-12 S-Add replicates 3000–3019 (n 1,000, `run_round12.Ctx`, B-spline front end; the GO
  variant fitted on the training subjects), against the saved Round-12 B2-S Cox ensemble records: quantity B2-S Cox −
  AFCM_REML in mean identifiable RISE over φ₁, φ₂; two-sided paired t, 95% CI, exact Wilcoxon, sign counts, TOST ±0.05;
  interpretation the H-R12 rule (D113: CI below 0 → B2-S better; above 0 → worse; otherwise TOST equivalent; otherwise
  inconclusive). Harness check first: the Round-12 afcm_v3 refitted on replicate 3000 must equal its record. φ₁ and φ₂
  separately are reported as secondary.
- **If NO-GO** (or CK-6d fails): seed 24 stays unused (or is reported as failed), the development table is reported
  beside Table 47, H-R12 stays inconclusive (D127).

### 2.2 Task 2: external cohorts (H-R22b)

- **Cohorts:** `renal` (t_L 1 year; channels GFR, haematocrit; z age, weight, female; event graft failure) and
  `heart_valve` (t_L 1 year; log LVMI, log gradient; z age, female, homograft; event death), loaders
  `kanfae.data.realdata.load_renal`, `load_heart_valve`; single landmark (history [0, t_L] rescaled to [0, 1]; outcome
  the residual time); outer splits 0–4 (`split_groups` 60/20/20 stratified on the event, split seed k); the Round-13
  harness (`RealCtx`, B-spline front end, conditional/marginal IPCW IBS, Harrell C, Uno C, D-calibration). **FD003
  Option B** (previously analysed; §1): the Round-13/21 engine split, 30-cycle windows every 10 cycles, u = s/100, every
  window an event.
- **Locked models (no tuning beyond the Round-21 protocol):** the Round-21 job (`run_round21_real.job_real`): B2-S Cox,
  Weibull, log-normal and NP-KAN-AFT (rank_km, τ 0.05) D112 3-seed ensembles; cross-confirmed anchors for Cox,
  log-normal and NP-AFT (D150 margins: Cox 0.02 absolute, log-normal 0, NP-AFT 0 relative; halves
  `run_round18.split_val`); fixed-μ distillation of the seed-0 NP-AFT and Cox members. Baselines (clinical):
  landmark_cox, fpca_cox, lme2_cox, sum_cox, sum_rsf (`run_round13.run_single_cell`). FD003: the four B2-S families, the
  NP-AFT cross-confirmed anchor (as FD001 in Round 21), landmark_cox, sum_aft, fpca_aft; one job per (split, family) so
  that the long NP-AFT jobs run in parallel (fits are seeded per network and full batch: splitting does not change
  them).
- **H-R22b supported iff** (1) every external job (10 clinical, 25 FD003) completes and every model and baseline record
  has finite metrics and finite survival predictions; (2) in each clinical cohort, for each of Cox, log-normal and
  NP-AFT, mean conditional IPCW IBS of the cross-confirmed anchor − its unanchored B2-S ≤ +0.005, with exact fallback
  (max |diff| = 0.0 on the test split, survival and risk) on every non-confirmed split; (3) in each clinical cohort, the
  distillation of both members is finite on all 5 splits and changes mean conditional IPCW IBS by at most +0.002 (after
  − before; the spline member is the reference). (4) Descriptive, not part of the verdict: b2s_npaft and b2s_cox against
  landmark_cox, lme2_cox, fpca_cox and sum_rsf (mean, plain and Nadeau–Bengio 95% intervals, wins/losses, D-calibration
  pass counts), and whether the denser renal histories narrow the gap to landmark_cox. Five splits cannot power
  superiority claims (D111).

### 2.3 Task 3: multi-law simulation (H-R22c)

- **Data** (`kanfae.data.fsim`, scenarios added; earlier ones bit-identical, test R22-2): log T = 1 − η/1.5 + ε, η the
  S-Add predictor, DGP v2, dense curves, n 1,000; ε of mean 0 and SD 0.5: `S-Misspec-bimodal` (0.5 N(−1.1, 0.3²) + 0.5
  N(1.1, 0.3²), standardised), `S-Misspec-studentt3` (t₃, standardised), `S-Misspec-logistic05` (standard logistic,
  standardised); censoring U(0, c_max), c_max calibrated per law on the fixed sample (seed 99) to 30%. Replicates s =
  5000–5019 per law; data seed SeedSequence([s, k]) with k = 2, 3, 4 (`run_round12.sub_seed`); split seed s.
- **Models:** D112 3-seed full-validation B2-S ensembles b2s_cox, b2s_waft (Weibull), b2s_lnaft (log-normal), b2s_npaft
  (rank_km, τ 0.05), on the B-spline front end. Metric: oracle survival-curve error (Round-12 `survival_metrics`; IBS, C,
  D-calibration reported).
- **H-R22c supported iff** (1) on S-Misspec-bimodal the upper 95% paired-t bound of curve error is < 0 for b2s_npaft −
  b2s_cox, − b2s_waft and − b2s_lnaft; and (2) on S-Misspec-studentt3 and S-Misspec-logistic05 the upper bound is < 0
  against b2s_cox and b2s_waft and < +0.005 against b2s_lnaft. All 60 jobs must be complete.

## 3. Analysis

`scripts/analyze_round22.py gate` (after the three development jobs) writes `reports/round22/afcm_gate.json`;
`scripts/analyze_round22.py final` writes `reports/round22/analysis_summary.json` and `docs/round22_results.md`. Every
job is recorded (failures with tracebacks); none is dropped; no threshold, margin, model setting or seed is changed after
this registration. Every other comparison is exploratory, with no multiplicity correction and no superiority claim
without a registered criterion (D111).

## 4. Execution

Queue (`scripts/run_round22_lanes.py`, 4 single-threaded lanes, resumable): the five FD003 NP-AFT jobs first, then the
other FD003 parts, the clinical jobs and the simulations (95 jobs); the AFCM jobs run outside the queue (gate first,
then CK-6d, then H-R22a). Expected cost: C-MAPSS dominates (Round 21: about 24 hours of CPU per FD001 split).

## 5. Execution notes

(Appended during the run; mechanics only.)

- 2026-11-06: development gate run (`afcmdev_22/914/966`, 425–485 s each): GO, `afcm_reml` (D157). CK-6d run once on
  seed 24 (`r22_ck6d`): FAIL (φ₁ 0.170); H-R22a not run (D157). Lanes 0–3 launched after the gate commit.
- 2026-11-06 13:18Z: container restart (no records written yet; ensembles saved up to 11:13Z in `ckpt/`). Lanes
  relaunched 13:19Z unchanged (logs `r22_lane<k>_r1`); the first attempt's queue state kept as
  `queue_state_attempt1.json`; saved ensembles are reloaded, unfinished ones refit from their fixed seeds.
