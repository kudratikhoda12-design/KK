# D168: AFCM reference investigation and staged protocol (proposal for approval)

**Status: proposal only.** Written on 2026-11-17 in response to the owner's D168 instruction. Nothing in Stages A–C, nor
in the calibration (Section 7) and joint-model (Section 8) studies, has been run: no package was installed, no model
was trained, no seed was used, no MCMC was run and no data were downloaded. Every earlier record, failure and verdict
(including CK-6c, CK-6d, H-R12 and the Round-25 seed-22 diagnostic) is kept unchanged.

**Environment at the time of writing.** The session's container was replaced between 2026-11-15 and 2026-11-17. The
project-local Python (`.venv/`) and R (`.rlocal/`) environments, both git-ignored, no longer exist, and the unpushed
Round-25 audit commit was lost and has been restored from the session record (audit Addendum A1). Any execution of
this protocol therefore starts by rebuilding both environments (Section 6.1), which itself needs approval because it
installs packages.

---

## 1. What was inspected, and how far

| Category | Item | Status on 2026-11-17 | What it established |
|---|---|---|---|
| **Public author repository** (`ecui1/AFCM`) | `README.md` | **obtained** (raw URL; identical on `master` and `main`; 917 bytes; SHA-256 `8fa7be2f…bc57`) | the repository "contains the code to implement 'Additive Functional Cox Model (Cui et al. 2021)'", lists `vignette_fcm.pdf` and `NHANES.rds` ("temporarily unavailable due to privacy issue"), and says "Other files are available in the online supplements" |
| | `vignette_fcm.pdf` | **obtained and read** (raw URL; 5 pages; "Fitting functional Cox models in R using mgcv package", E. Cui, dated 2/9/2021; SHA-256 `4de54e1d…ff77`) | the exact `mgcv` calls for AFCM on simulated data (Part 1) and on NHANES (Part 2); see Section 2 |
| | `NHANES.rds` | **not requested** | marked private by the author; not needed |
| | full file listing of the repository | **not obtained** | the GitHub web and API routes are outside this session's permitted access and were not used; only the files named in the README or in the owner's instruction were requested by raw URL |
| | `vignette_afcm.Rmd`, `vignette_afcm.html`, `process_data.R`, `supplementary_materials.pdf` at the repository root | **not present** there (raw URL 404 on both `master` and `main`) | consistent with the README: these are publisher supplements, not repository files |
| **Publisher's supplementary files** | `vignette_afcm.Rmd`, `vignette_afcm.html`, `process_data.R`, `supplementary_materials.pdf` (the owner's listing; the PMC page lists one archive, `NIHMS1733544-supplement-Supp_1.zip`, 4 MB, described as "the vignette ... containing code and instructions to reproduce all results shown in the paper") | **not obtained** | 2026-11-15: the PMC download returned a browser-check page, not the archive. 2026-11-17: the PMC and Europe PMC download routes were refused by this session's permission policy and were not retried |
| **Methods in the paper** | author manuscript PMC8664082 | **read on 2026-11-15** (the local copy was lost with the container; the passages used are quoted in the Round-25 audit, Section 2.1) | model form, identifiability constraint (11), quantile-type transformation, `mgcv` call, Laplace-approximate marginal-likelihood smoothing |
| **`mgcv` source** | version 1.9-5 (CRAN mirror): `DESCRIPTION`, `R/coxph.r`, `R/mgcv.r`, `man/coxph.Rd`, `man/te.Rd`, `man/identifiability.Rd` | **read** (re-fetched; hashes of `DESCRIPTION`, `coxph.r`, `coxph.Rd` and `te.Rd` identical to 2026-11-15) | tie handling, the `mc` constraint, the smoothing criterion forced for the Cox family |
| **Runnable `mgcv`** | — | **not available** (not installed; installation not approved) | nothing in this document was executed in R |
| **Our re-implementation** | `src/kanfae/models/afcm.py` (V3 = `AFCM_D123`; REML variant `fit_afcm_reml`) | read | Section 3 |

**What can and cannot be claimed.** The author's **runnable reproduction code** (`vignette_afcm.Rmd`,
`process_data.R`) has **not** been obtained or inspected. What has been obtained is the author's short public vignette,
whose two `gam()` calls are runnable code that specifies the AFCM fit exactly, but not the paper's full analysis
pipeline (simulation design, NHANES processing, plotting and inference). The reference implementation below is
therefore "AFCM as specified by the paper and the author's public `mgcv` vignette", not "the authors' code".

---

## 2. The original statistical implementation, reconstructed

From the author's vignette (verbatim, Part 1, simulated data on a non-periodic domain):

```r
data_analysis$lmat <- I(matrix(1/S, ncol=S, nrow=nrow(data_analysis)))              # quadrature weights
data_analysis$tmat <- I(matrix(seq(0, 1, len=S), ncol=S, nrow=nrow(data_analysis), byrow=TRUE))
fit_afcm <- gam(survtime ~ Z + ti(tmat, X, by=lmat, bs=c("cr","cr"), k=c(10,10), mc=c(FALSE,TRUE)),
                weights=event, data=data_analysis, family=cox.ph())
```

and Part 2 (NHANES, 24-hour activity): the same call with `bs=c("cc","cr")`, `k=c(5,5)`, on the quantile-transformed
covariate `apply(X_smoothed, 2, function(y) ecdf(y)(y))`.

| Element | Original | Evidence |
|---|---|---|
| Functional term | `ti(tmat, X, by = lmat)`: a tensor-product interaction smooth F(s, x) evaluated at every (s_j, X_i(s_j)) and summed with weights lmat, i.e. Σ_j l_j F(s_j, X_i(s_j)) ≈ ∫F(s, X_i(s))ds | vignette; article Section 2.4.2 |
| Margins | `cr` (cubic regression spline, knots at quantiles of the margin's values) in x; `cr` in s for a non-periodic domain (Part 1), `cc` (cyclic) only for the 24-hour NHANES domain (Part 2) | vignette Parts 1 and 2 |
| Basis size | `k = c(10, 10)` in the simulated example | vignette Part 1 |
| Penalties | `mgcv` tensor-product penalties: one integrated-squared-second-derivative penalty per margin, each with its own smoothing parameter | `mgcv` `te`/`ti` construction |
| **Identifiability** | `mc = c(FALSE, TRUE)`: a **marginal centring constraint on the x margin only**, absorbed into the basis **before fitting**, so the fitted surface contains no component that is a function of s alone. The vignette states that this "exactly matches the additional identifiability constraints for AFCM"; the article's constraint (11) is Σ_i F{s, X_i(s)} = 0 for each s | vignette; article eq. (11); `man/te.Rd` |
| Covariate transformation | quantile-type: at each grid point s, x̃ = empirical CDF of X(s) over subjects (`ecdf` per column) | article Section 2.3; vignette Part 2 |
| Smoothing parameters | Laplace approximation to the marginal likelihood (REML). For the Cox family `mgcv` **forces** `method = "REML"` (or NCV) whatever is passed: `R/mgcv.r`, lines 1853–1856 ("if (!(method %in% c("REML","NCV")) ...) method <- "REML"") | `mgcv` 1.9-5 source; article Section 2.4.3 |
| Coefficient estimation | penalised partial likelihood, Newton-type iteration inside `mgcv`'s general-family fitter | article Section 2.4.2; `mgcv` |
| Cox likelihood and ties | `cox.ph()`: "the partial likelihood used (with Peto's approximation for ties)" | `man/coxph.Rd`, lines 41–42 |

**Why the identifiability constraint matters in estimation, not only in evaluation.** For a Cox model, adding any
function c(s) of time alone to F changes every subject's linear predictor by the same constant ∫c(s)ds, which the
partial likelihood cannot see. Those directions are therefore unidentified. `mgcv` removes them from the basis before
fitting (the constraint is imposed during estimation). Our Python V3 keeps them in the basis and holds them in place
only with a ridge of 10⁻⁶ and the t-direction penalty; the "identifiable RISE" then removes them **after** fitting,
for scoring only. The two are **not equivalent**:

1. the quadratic penalties couple the unidentified directions with the identified ones, so the identified part of a
   ridge-held fit can differ from the identified part of a constrained fit at the same smoothing parameters;
2. the ridge leaves the penalised Hessian nearly singular. The Round-25 seed-22 diagnostic measured a condition number
   of about 4 × 10¹⁰ and found that L-BFGS stopped far from the optimum (largest penalised-gradient component 0.106;
   estimated remaining gain about 46 log-partial-likelihood units). A near-null space of this kind is what an
   unimposed constraint would produce. This is an **explanation to be tested in Stage A, not an established cause**.

**A further point to be checked, not assumed.** `mgcv`'s marginal centring constrains the x-margin basis over **all**
data values used to build it (pooled over subjects and grid points), whereas eq. (11) asks for Σ_i F{s, X_i(s)} = 0 **at
every s**. The two coincide when the distribution of the covariate is the same at every s. The quantile transformation
makes x̃ approximately uniform on (0, 1) at every s, so under the transformation the pooled constraint is close to the
per-s constraint. On raw x values (our V3 default) it is not. Stage A measures max_s |n⁻¹ Σ_i F̂(s, x̃_i(s))| for the
`mgcv` fit to quantify this.

**Adaptation of the time basis (non-periodic domain).** Our history window is rescaled to [0, 1] with no periodicity: the
start and end of a patient's history are different times, so a cyclic basis would wrongly force F(0, x) = F(1, x) and
equal derivatives at the ends. The reference implementation therefore uses `bs = "cr"` in s, exactly as the author's own
simulated-data example does on its non-periodic [0, 1] domain (vignette Part 1). Only the NHANES example uses `cc`,
because 0 and 24 hours are the same clock time. With `cr` in both margins and `k = c(10, 10)` the reference matches the
author's non-periodic specification and our existing 10 × 10 tensor size.

**Ties: Peto (mgcv) and Efron (our code).** `mgcv` uses Peto's approximation; for the log partial likelihood it coincides
with the Breslow form. Our Python AFCM uses Efron's form in the fit (and the Breslow form in its information matrix).
All forms give the same partial likelihood when no two event times are equal. In the simulation (S-Add, DGP v2) event
and censoring times are continuous random variables, so ties occur with probability zero; **Stage A step A3 verifies
that every Stage-A, Stage-B and Stage-C data set has no tied event times** (and records the count). The difference
would matter on data with tied times, for example the Round-13 real-data AFCM V3 rows (PBC in days, `aids` and
`prothro` with coarse visit times); those rows are not part of this protocol and stay as recorded.

---

## 3. Our Python re-implementation against the reference

| Element | Reference (Section 2) | Python V3 (`AFCM_D123`, used for H-R12) | Python REML variant (Round 22) |
|---|---|---|---|
| Margins | `cr` × `cr`, knots at quantiles | cubic B-splines, equally spaced knots on the training range | same as V3 |
| Penalty | integrated squared second derivative per margin | second-order differences in x; first- plus second-order differences in t; scaled by basis size | same as V3, per channel |
| Identifiability | x-margin centring absorbed **before** fitting | **none**; ridge 10⁻⁶; identified part extracted after fitting for scoring | same as V3 |
| Covariate | quantile-transformed | raw values (the quantile option exists but is off) | raw |
| Smoothing parameters | REML (Laplace marginal likelihood), Newton | validation partial likelihood on a 9 × 5 grid, common to all channels | per-channel Fellner–Schall updates of a Laplace marginal likelihood |
| Optimiser | Newton with convergence control | L-BFGS, ≤ 500 iterations, zero start, no stationarity check (**did not converge on seed 22**) | as V3, warm-started along the update path |
| Ties | Peto | Efron | Efron |

---

## 4. Stage A: development (no formal seed)

**Purpose.** Establish a correct, converged reference AFCM in R, compare it with the Python variants, and repair the
Python optimiser only if a mathematically justified repair exists. **Seeds:** only historical development data,
namely the CK procedure (S-Add, DGP v2, n 2,000, Round-11 split) on **seeds 22, 914 and 966** and the mechanics set
(S-Add seed 11, n 400). Seed 22 was the CK-6c acceptance seed (Round 12b) and a Round-22 development seed; it is used
here only as development data. No result of Stage A changes CK-6c, CK-6d or H-R12, and the Stage-B acceptance limits
are not tuned on Stage-A results.

| Step | Action | Output |
|---|---|---|
| A0 | Rebuild environments (Section 6.1); record exact versions and an explicit package lock for R | `docs/` environment record |
| A1 | Write `scripts/afcm_mgcv.R`: one `ti(tmat, X_d, by = lmat, bs = c("cr","cr"), k = c(10,10), mc = c(FALSE,TRUE))` per channel d = 1..3, z linear, `family = cox.ph()`, `method = "REML"` stated explicitly; quantile transformation from the **training** ECDF at each grid point (test values mapped through the training ECDF, as `xin = "quantile"` already does); export F̂_d(s, x) on a fixed (s, x) grid and the fit diagnostics (`converged`, outer iterations, REML score, gradient, effective degrees of freedom) | script, unit test on the vignette's own simulated example (structure only; the vignette's data are random noise) |
| A2 | Data export shared by both languages: smoothed curves on the 50-point grid, quadrature weights, z, y, δ, split indices | identical inputs for R and Python |
| A3 | Tie check: number of tied event times in every data set (expected 0) | recorded counts |
| A4 | Fit `mgcv` AFCM on seeds 22, 914, 966; compute identifiable and raw RISE of φ₁, φ₂ with the **existing** metric (`fsim.rise_identifiable`) on the exported surfaces | development table |
| A5 | Identifiability check: max over s of the mean fitted surface over training subjects, for `mgcv` (constraint in fitting) and for Python V3 and REML (no constraint); compare the identified parts of the surfaces directly (sup-norm after removing c(s)) | development table |
| A6 | Convergence and conditioning of every fit: `mgcv` convergence flags; for Python, largest penalised-gradient component, Newton decrement and condition number of the penalised Hessian (as in the seed-22 diagnostic) | development table |
| A7 | **Python repair, only if justified.** Candidate: reparametrise with the x-margin centring constraint absorbed (null-space basis, as `mgcv` does) to remove the near-null directions, and replace L-BFGS by damped Newton on the penalised partial likelihood using the existing observed information (`cox_information`), with a stationarity stopping rule (max \|penalised gradient\| < 10⁻⁶ and Newton decrement < 10⁻⁸). Accept the repair only if, on all three development seeds, it reaches the stopping rule and the cold start and the warm start give the same surface (max difference < 10⁻⁴). Report how the V3 and REML surfaces change when re-converged | development table; the old V3 and REML code paths are kept unchanged for the historical records |
| A8 | Timing of one `mgcv` fit and one B2-S Cox ensemble on the Stage-C design (n 1,000) to fix the Stage-C budget | timing record |

**Go/no-go rule for Stage B (fixed now).** GO if the `mgcv` reference fits converge without error on all three
development seeds, the surfaces are finite and the specification (A1) is frozen and committed. The value of the RISE on
the development seeds is **not** a GO criterion. A NO-GO is reported and Stage B is not run.

**Deliverable.** `docs/d168_stageA_development.md` with tables A3–A8, committed before Stage B.

---

## 5. Stage B: fresh acceptance gate (one new seed)

| Item | Rule |
|---|---|
| Procedure | the CK procedure unchanged (S-Add, DGP v2, n 2,000, Round-11 split, smoothed curves), with the frozen `mgcv` reference AFCM |
| Seed | **6001**, one seed, one attempt. Checked unused: no occurrence in code, documentation, decision log or result records (scan of 2026-11-17: the ranges 6000–6199 and 7000–7199 are clear). Re-verified and recorded at registration |
| Criterion | the **original limits, unchanged**: identifiable RISE of φ₁ **< 0.15** and of φ₂ **< 0.10**; **both** must hold |
| If it fails | recorded as a failure; Stage C is not run; no second seed without a new owner decision |
| Meaning | an implementation acceptance check of the reference comparator on one data set; **not** evidence about B2-S, and not evidence of general accuracy |
| Pre-registration | a short plan (D169) committed before the seed is used |

---

## 6. Stage C: confirmatory paired comparison (only if Stage B passes)

### 6.1 Dependencies (to be approved; none installed now)

| Dependency | Source | Note |
|---|---|---|
| Python environment | `requirements-lock.txt` into `.venv/` | rebuild of the lost environment; versions as recorded |
| R 4.4.3 | conda-forge via micromamba into `.rlocal/` | as Round 24 |
| **`mgcv`** (version to be pinned and recorded; the source read here is 1.9-5) | conda-forge `r-mgcv` or CRAN | **new dependency**; the reason for the approval request |
| `survival`, `splines`, `jsonlite`, `nlme`, `Matrix` | as Round 24 | |
| JMbayes2 0.6-0 and dependencies | as Round 24 | only if Study B (Section 8) is approved |

No R lock file existed for Round 24; this time an explicit lock (`micromamba env export --explicit`) is committed.

### 6.2 Design

- **Data:** S-Add, DGP v2, n 1,000 per replicate (as H-R12), the Round-12 front end. **Every data set is generated once,
  exported (A2 format) and read by both methods**, so B2-S Cox and the reference AFCM see identical curves, covariates,
  outcomes and splits.
- **Methods:** B2-S Cox, three-member ensemble, the Round-12 protocol unchanged; the frozen `mgcv` reference AFCM.
  Secondary: the repaired Python AFCM if Stage A accepted a repair (otherwise none).
- **Seeds:** base seeds **6100 – 6209** for 110 replicates (or 6100 – 6100 + n − 1 for another approved n); data seed
  `SeedSequence([s, 0])` and split seed s, as in Round 12. All checked unused on 2026-11-17 and re-verified at
  registration.
- **Primary outcome:** identifiable RISE averaged over φ₁ and φ₂, as in H-R12. Difference d = B2-S Cox − reference
  AFCM per replicate.
- **Primary decision rule (H-R25a):** equivalence if the 90% confidence interval of the mean of d (paired t, i.e. two
  one-sided tests at α = 0.05) lies inside ±0.05, the H-R12 margin. Reported in every case: the mean, the 95%
  confidence interval, the Wilcoxon signed-rank test and the counts of d < 0 and d > 0. Directional statements
  ("B2-S better" or "AFCM better") are made only if the 95% interval excludes 0; these are secondary.
- **Secondary outcomes:** per-channel identifiable RISE (φ₁, φ₂), raw RISE, survival-curve error against the oracle,
  test IBS, Harrell C; for every fit its convergence diagnostics; run times.
- **Failures:** a replicate in which either method fails or does not converge is counted and reported. Primary analysis
  on complete pairs; a sensitivity analysis assigns the failed method the worst observed RISE. If more than 10% of
  replicates fail for either method, H-R25a is reported as not evaluable.
- **H-R12 is unchanged.** Its result (+0.0065, 95% CI [−0.0506, +0.0635], 20 replicates, inconclusive) stays in every
  record and manuscript as the historical result against the Python V3 comparator.

### 6.3 Replicate count, recomputed from the saved Round-12 variance

From `reports/round12_summary.json` (H_R12): n = 20 pairs, SD of the paired differences **0.1219**, mean +0.0065. The
power of the paired TOST with margin ±0.05 was computed by Monte Carlo (200,000 draws per n, exact t critical values;
script kept in the session scratch folder, method reproducible from this description):

| SD of differences | True mean difference | Pairs for 80% power | Pairs for 90% power |
|---|---|---|---|
| 0.1219 (Round 12) | 0 | 54 | 66 |
| 0.1219 | 0.0065 (Round-12 estimate) | 56 | 74 |
| 0.1219 | 0.02 | 110 | 150 |
| 0.1524 (Round 12 × 1.25) | 0 | 82 | 110 |
| 0.1524 | 0.0065 | 88 | 120 |
| 0.1829 (Round 12 × 1.5) | 0 | 120 | 150 |

Power of the Round-12 design (20 pairs) at a true difference of 0 was **0.14**, which is why H-R12 could not be decisive.
Half-width of the 95% interval at SD 0.1219: 0.057 (20 pairs), 0.032 (60), 0.024 (100), 0.020 (150).

**Proposed n = 110.** Reasons: (a) the Round-12 SD was estimated from only 20 pairs and against a comparator whose
optimiser is now known to be unreliable; a converged, constrained reference AFCM may have a different spread, so a
25% inflation of the SD is prudent; (b) at SD × 1.25 and a true difference of 0, 110 pairs give 90% power, and at the
Round-12 point estimate of the difference they give over 80%; (c) 60 pairs, the earlier starting estimate, would give
about 86% power only if the Round-12 SD were exactly right and the true difference exactly 0. If the owner prefers a
smaller study, **66** pairs (90% power at the unchanged Round-12 SD, true difference 0) is the minimum this protocol
would support. The count is fixed before any Stage-C seed is used; there is no interim look.

### 6.4 Computational budget

| Stage | Work | Estimate |
|---|---|---|
| Environment rebuild | `.venv` from the lock file; `.rlocal` with R and `mgcv` | 20–40 min wall |
| A | 3 development seeds × (`mgcv` fit + Python V3 + Python REML + checks), plus the repair attempt | 1–2 h wall (the `mgcv` fit time is not yet known; measured in A8) |
| B | one data set, one `mgcv` fit | < 15 min |
| C | per replicate: B2-S Cox ensemble 203 s median (528 s maximum) in Round 12; `mgcv` AFCM estimated 1–5 min (to be measured in A8); data, export and evaluation about 1 min | 5–10 min per replicate; **9–18 CPU hours for 110 replicates, about 2.5–4.5 h wall on four single-threaded lanes** |
| Re-checks after any step | manuscript tests and full suite before deliverables | about 2 h (full suite) |

---

## 7. Calibration study design (Study C; not run)

**Principle.** No calibration problem is called resolved without saved individual predictions and the analyses below.

1. **Save individual predictions.** For every test subject or engine window: predicted survival S_i(t) on the
   evaluation grid, the outcome (y_i, δ_i), the IPCW weights used, the identifiers (subject or engine, window index,
   split, seed) and the model. Stored as compressed arrays outside the manuscript files, with checksums committed.
2. **Two different questions on the two kinds of data.**
   - **Engine data (C-MAPSS FD002/FD004, Option B): remaining useful life (RUL), not right-censored survival.** Every
     window ends in failure, so there is no censoring. Calibration is the calibration of a predicted RUL distribution:
     probability-integral-transform histograms, D-calibration without censoring, coverage of central prediction
     intervals, the continuous ranked probability score, and point accuracy of the median RUL (MAE, RMSE). These are
     reported **separately** from the survival calibration of the clinical cohorts and are not called "survival
     calibration".
   - **Clinical cohorts: right-censored survival calibration.** At pre-specified horizons after the landmark (for
     example 1, 2 and 3 years, limited to where at least 10 test events remain), IPCW calibration curves, the
     integrated calibration index and observed-against-expected risk; calibration-in-the-large against the
     Kaplan–Meier estimate. To avoid counting a subject many times across the twenty overlapping partitions, the
     calibration analysis uses **out-of-fold predictions from a single K-fold cross-fitting** (each subject predicted
     once), fixed before the run.
3. **Engine-level clustering.** C-MAPSS test sets have about 50 engines and 450–590 overlapping windows per split (about
   9–12 windows per engine). Uncertainty is computed by a **cluster bootstrap over test engines** within each split
   (combined over splits with the Nadeau–Bengio correction), and D-calibration is also computed on **one
   pre-specified window per engine** (the last), so that its test treats independent units.
4. **Optional hypothesis (H-R25c).** A monotone recalibration of NP-KAN-AFT's predicted RUL distribution, fitted on
   **validation data only**, improves calibration on new engine splits without raising IBS by more than 0.005. Test
   outcomes are never used for recalibration.
5. **Seeds and data.** Descriptive re-analysis: deterministic re-runs of the saved Round-23 configurations with
   predictions written out (new evidence about calibration, not about discrimination). H-R25c: new engine split seeds
   **7000–7009** (checked unused).
6. **Budget.** C-MAPSS re-runs about 3–4 h wall on four lanes (Round 23 cost); clinical cross-fitting about 1–2 h.

---

## 8. Joint-model diagnostic study design (Study B; not run)

**Principle.** No convergence claim is made without saved draws and the diagnostics below.

1. **Same data and pairing as Round 24** (six clinical cohorts, partitions 5–24), so results remain paired with the
   Round-23 records. The Round-24 records are kept as the historical result.
2. **Sampling.** 4 chains, 15,000 iterations, 3,000 burn-in, thinning 3 (4,000 retained draws per chain), one core per
   fit; MCMC seeds **2505–2524** (2400 + k was used in Round 24; the 2500 range is unused).
3. **Saved evidence per fit.** All retained draws of the association parameters, fixed effects, survival coefficients,
   residual SDs, random-effects covariance and baseline-hazard coefficients (compressed, outside git, checksums
   committed); trace and rank plots of the association parameters for every fit and of all parameters for one
   partition per cohort; the JMbayes2 summary.
4. **Diagnostics.** Rank-normalised split R-hat and bulk and tail effective sample sizes (Vehtari et al. 2021) for every
   monitored parameter, Monte-Carlo standard errors of the posterior means. Computed either with the `posterior` R
   package (a further dependency needing approval) or with own code validated against known test chains.
5. **Convergence rule (fixed before the run).** A fit converges if every association parameter and fixed effect has
   R-hat < 1.01 and bulk and tail ESS > 400, and every variance parameter has R-hat < 1.05. A fit that fails is extended
   **once** to twice the length; if it still fails it is classified as **not converged** and treated as a failed fit.
   A cohort with more than two non-converged or failed partitions is not evaluable.
6. **Hypotheses.** H-R25b: as H-R24b (NP-KAN-AFT − joint model, conditional IBS, Nadeau–Bengio upper bound ≤ +0.005 on
   every cohort), evaluated only on converged fits. Secondary: H-R24a repeated; a richer renal trajectory model
   (natural spline with spline random effects or a shared-parameter slope association), pre-specified as a sensitivity
   analysis, not as a re-selection of the comparator.
7. **Budget.** Round 24 took about 2.6 h wall (11:49–14:24 UTC) for 3 chains × 6,500 iterations, both trajectory
   forms, on four lanes; 4 × 15,000 iterations is about 3.1 times the sampling work, giving roughly **8–10 h wall on four
   lanes** (more if fits need the one permitted extension), plus storage of about 0.5–1 GB of draws outside git.

---

## 9. Exact execution steps if approved (in order)

1. Owner approves this protocol (and states the Stage-C n: 110 proposed, 66 minimum) and the installation of `mgcv`.
2. Rebuild `.venv` from `requirements-lock.txt` and `.rlocal` (R 4.4.3, `mgcv` pinned, `survival`, `splines`,
   `jsonlite`); commit the explicit environment lock and versions.
3. Stage A (A1–A8) on development seeds 22, 914, 966 and the mechanics set; write and commit
   `docs/d168_stageA_development.md`; apply the GO/NO-GO rule.
4. If GO: commit the Stage-B plan (D169) with seed 6001 verified unused; run Stage B once; record PASS or FAIL.
5. If PASS: commit the Stage-C plan (D170) with n, seeds 6100–6100 + n − 1, the analysis script and tests, before any
   Stage-C seed is used; run; analyse; record.
6. Only after that: decide with the owner how the manuscripts change (Section 5 of the audit addendum and
   `docs/d168_journal_wording_proposal.md`).
7. Studies B and C (Sections 7 and 8) are independent of Stages A–C and each needs its own approval.

**Stopping point.** Nothing above has been started. Awaiting the owner's approval.
