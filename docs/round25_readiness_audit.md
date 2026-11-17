# Round 25 readiness audit (2026-11-15)

**Question asked:** is the paper ready to submit?
**Answer: not yet.** Three findings of this audit are serious enough that a biostatistics reviewer would very likely
raise them, and one of them (Section 3) is new: the Python AFCM comparator was found not to have converged in the one
diagnostic fit that was allowed. Details and a prioritised list are in Sections 6 to 9.

**Scope and limits of this audit.** It was run under the "Final execution constraints — Round 25 audit". The earlier
Round-25 prompt that those constraints refer to was **not received in this session**; the tasks below were therefore
taken from the constraints themselves (Tasks 1D, 2, 3 and the nine required report sections). No model was trained
apart from the single diagnostic fit of Section 3, no MCMC was run, no formal seed was used, no package was installed,
no threshold, hypothesis or verdict was changed, the Round-24 manuscripts and report were not modified, and nothing
was pushed.

---

## 1. Repository and branch status

| Item | State at the start of the audit |
|---|---|
| Branch | `claude/blissful-faraday-2unozw` (no other branch created or checked out) |
| HEAD | `7e91ef6` "Rewrite journal manuscript for outside readers; correct report test tables (D166)" |
| Working tree | clean (no uncommitted or untracked files) |
| Remote | local branch and `origin/claude/blissful-faraday-2unozw` identical (0 ahead, 0 behind); pushed on 2026-11-14 on the owner's instruction. **Nothing was pushed during this audit.** |
| Rounds 1–24 | all commits and deliverables untouched |

Files added by this audit (new files only): `docs/round25_readiness_audit.md`, `scripts/diag_round25_afcm_mechanics.py`,
`scripts/diag_round25_afcm_check.R`, `reports/round25/diag/` (diagnostic inputs and results), one decision-log row
(D167) and one ledger row (L54).

**Environment (not changed).** Project-local R in `.rlocal/` (git-ignored):

| Package | Version | Note |
|---|---|---|
| R | 4.4.3 (2025-02-28) | |
| JMbayes2 | 0.6-0 | used in Round 24 |
| survival | 3.8-12 | used in the Section 3 check |
| splines | 4.4.3 | used in the Section 3 check |
| nlme | 3.1-170 | |
| GLMMadaptive | 0.9-7 | JMbayes2 dependency |
| Matrix | 1.7-6 | |
| JM | 1.5-2 | installed in Round 24, unused |
| **mgcv** | **not installed** | the package the original AFCM is fitted with |
| **refund** | **not installed** | |

**Network reachability from the container:** PMC, CRAN and `raw.githubusercontent.com` reachable; `github.com` pages
are restricted to this project's repository by the proxy; `doi.org` (publisher page of the AFCM article) returned 403.

---

## 2. AFCM: original source and our implementation

### 2.1 What could be read

| Source | Read? | What it gave |
|---|---|---|
| AFCM article, author manuscript (Cui, Crainiceanu & Leroux 2021, PMC8664082) | **yes** | the model, the identifiability constraint, the exact `mgcv` call, the smoothing-parameter criterion |
| `mgcv` source, version 1.9-5 (CRAN mirror on GitHub: `DESCRIPTION`, `R/smooth.r`, `R/coxph.r`, `R/gam.fit4.r`, `man/coxph.Rd`, `man/te.Rd`) | **yes, read only** (kept in the session scratch folder, not installed, not committed) | how `ti(..., mc=)` builds the constrained tensor product; how `cox.ph` treats ties |
| The authors' own code (`vignette_afcm.Rmd`, inside the article's supplementary archive `NIHMS1733544-supplement-Supp_1.zip` on PMC) | **no** | the archive exists, but the download returned a browser-check page instead of the file; the article page links no code repository |
| A runnable `mgcv` | **no** | not installed; installing is not allowed in this audit |

The original fit, quoted from the article:

`gam(Y ~ z + ti(x, s, by = l, bs = c("cr", "cc"), k = c(Kx, Ks), mc = c(TRUE, FALSE)), weights = delta, family = cox.ph())`

with smoothing parameters chosen by "maximizing the Laplace approximation of the marginal likelihood" (Wood, Pya &
Säfken 2016), and a quantile-type transformation of the functional covariate to fill the rectangular domain.

### 2.2 Comparison

| Aspect | Original AFCM (article + mgcv 1.9-5) | Our `src/kanfae/models/afcm.py` (version "V3" used in the paper) | Same? |
|---|---|---|---|
| Model form | η = Σ ∫F(s, X(s))ds + zγ, Cox partial likelihood | the same form, per channel | yes |
| Basis in x | cubic regression spline (`cr`), knots at quantiles | cubic B-splines, 10 equally spaced knots on the training range | **no** |
| Basis in s / t | cyclic cubic regression spline (`cc`) for their 24-hour data | cubic B-splines on [0, 1], not cyclic (our domain is not periodic, so this difference is appropriate) | no (justified) |
| Penalty | integrated squared second derivative of each margin, one smoothing parameter per margin | second-order **difference** penalties (P-spline), plus a first-order difference penalty in t, each divided by the basis size | **no** |
| Identifiability | marginal centring constraint on the x margin (`mc = c(TRUE, FALSE)`), absorbed into the basis | **no constraint**; a small ridge (1e-6) on all coefficients, and the non-identifiable part is removed afterwards by the "identifiable RISE" metric | **no** |
| Covariate transformation | quantile-type transformation recommended | available (`xin="quantile"`) but **not used** in V3 (raw values) | **no** |
| Smoothing-parameter choice | Laplace approximate marginal likelihood, Newton optimisation of all smoothing parameters jointly | V3: validation partial likelihood on a 9 × 5 grid, one common pair for all channels. Round 22: per-channel Fellner–Schall updates of a Laplace marginal likelihood | **no** (V3); approximately (Round 22) |
| Ties | Peto's approximation (`man/coxph.Rd`) | Efron in the fit, Breslow form in the information matrix | no (irrelevant in simulation: continuous times) |
| Optimiser for the coefficients | Newton / penalised iteratively re-weighted least squares | L-BFGS, at most 500 iterations, from a zero start | **no** — see Section 3 |
| Numerical integration | Riemann weights in the `by` variable | trapezoid weights on a 50-point grid | equivalent |

**Conclusion of this section.** Our AFCM is a **methodology-based re-implementation** of the model form. It is **not**
a reproduction of the authors' code and differs from the `mgcv` fit in the basis, the penalty, the identifiability
constraint, the covariate transformation, the smoothing-parameter criterion (for V3) and the optimiser. The manuscripts
already call it "a Python re-implementation" and "an unvalidated comparator"; they do not list these differences, and
they give "R was not available" as the reason (Supplementary Sections 2.4, 5.10 and 6.3). That reason was true when
written but R has been installed since Round 24, so a reviewer will ask why `mgcv` was not then used.

---

## 3. The one diagnostic run (Task 1D)

| Item | Value |
|---|---|
| Seed | **22 only**, a development seed already used in Rounds 15 and 22; reused as a historical diagnostic reproduction, **not** a fresh or confirmatory seed |
| Replicates | one |
| Design | the saved Round-22 development procedure (S-Add, n 2,000, 1,200 training subjects, 843 events). The Python AFCM was fitted **once**, from the usual zero start, at the per-channel smoothing parameters **already saved** in `reports/round22/dev/results_afcmdev_22.jsonl`. No smoothing parameter was searched |
| R side | `survival::coxph` and `splines::splineDesign` on the identical exported design matrix, times, events, coefficients, penalty and knots (`reports/round25/diag/`) |
| Run time | **22.2 s** in total (data 10.4 s, fit 7.2 s, R check 4.2 s); limit 300 s. A further arithmetic step on the saved outputs (no fit) took under 2 s |
| Scripts | `scripts/diag_round25_afcm_mechanics.py`, `scripts/diag_round25_afcm_check.R`; result `reports/round25/diag/summary.json` (restored on 2026-11-17 from the run's console output, see the addendum; the exported input files were lost with the container) |
| Effect on records | none: CK-6c, CK-6d, their thresholds and their recorded results are unchanged |

**Outcome.**

| Check | Python | R | Agreement |
|---|---|---|---|
| Efron log partial likelihood at the fitted coefficients | −5031.901826 | −5031.901826 (`coxph`, no iteration) | difference 2 × 10⁻¹² — **the likelihood code is correct** |
| Cubic B-spline basis (201 points × 10 functions) | — | `splineDesign` | largest difference 2 × 10⁻¹⁶ — **the basis code is correct** |
| Gradient of the penalised objective at the returned coefficients | largest component 0.1063 | 0.1063 (score written out from the definition) | the two agree, **but the gradient is not zero** |
| Identifiable RISE of φ₁ / φ₂ | 0.534 / 0.158 (this cold-start fit) | saved record: 0.141 / 0.057 (warm-started path) | **not reproduced** |

Read together with the arithmetic on the saved outputs (same coefficients, no new fit):

- the estimated remaining gain in the objective is 0.055 per event, about **46 units of log partial likelihood**;
- the Newton step to the optimum is as large as the coefficients themselves (largest step 3.03, largest coefficient
  2.83), and would change the linear predictor by an SD of 0.28 against its SD of 0.62;
- the penalised Hessian has condition number about 4 × 10¹⁰.

**What this shows.** The building blocks of the Python AFCM (basis, partial likelihood, gradient) agree with R to
machine precision. Its **optimiser does not reach the optimum**: with these smoothing parameters the problem is very
badly conditioned and L-BFGS stops far from the solution, so the fitted shape depends on the starting point (0.53
from a zero start against 0.14 along the warm-started path, at the same smoothing parameters). The problem is convex,
so this is an incomplete optimisation, not a second solution.

**What this does not show (could not be verified within the limits).**

- Which of the two fits is nearer the true optimum, and what the shape error is at the optimum. Finding out needs a
  converged solve, which was not run because the limit allowed one fit and no further optimisation.
- Whether the registered AFCM fits (the CK-6c and CK-6d checks, the 20 replicates behind H-R12, the real-data AFCM
  rows) had converged. Those fits start from zero on a grid that includes equally extreme smoothing parameters, so the
  same weakness is **possible**, but it is not established for them. Their recorded results stand as recorded.
- Any agreement or disagreement with `mgcv`'s AFCM.

**Consequence for the paper.** The paper's one primary confirmatory comparison (H-R12, B2-S Cox against AFCM) uses a
comparator that is not the original implementation and whose optimiser is now shown to be unreliable. "Inconclusive
against an unvalidated comparator" remains an accurate description, but the comparison is weaker than the manuscripts
convey, and AFCM's failed acceptance checks may be partly an artefact of our optimiser rather than of the method.

---

## 4. Calibration audit (saved Round-23 and Round-24 records only)

**What is saved.** Each record holds per-split summary metrics only: IBS (and conditional IBS), Harrell C, Uno C, the
D-calibration p-value, test-set size and, for the clinical cohorts, the number of test events.

| Needed for further calibration diagnostics | Saved? |
|---|---|
| Individual predicted survival curves or probabilities | **no** (`reports/round23/ckpt/` is empty; no per-subject array in any record) |
| Individual event times and indicators aligned with predictions | no (re-derivable from the data, but useless without the predictions) |
| Censoring weights | no |
| Prediction horizons | only implicitly (the IBS grid) |

**Therefore not computable from saved records, and not computed:** calibration curves, calibration slope and
intercept, the integrated calibration index, Brier scores at fixed horizons, observed-versus-expected ratios, and any
cluster-aware calibration test. No model was refitted and no prediction was regenerated to obtain them.

**What the saved D-calibration p-values show** (computed from the records):

| Data | Model | Splits passing (p ≥ 0.05) | Median p | Smallest p |
|---|---|---|---|---|
| FD002 | NP-KAN-AFT | 2 / 10 | 0.011 | 1.9 × 10⁻⁷ |
| FD002 | landmark Cox | 0 / 10 | 2 × 10⁻¹⁵ | 1.3 × 10⁻³⁰ |
| FD004 | NP-KAN-AFT | 5 / 10 | 0.042 | 1.5 × 10⁻¹² |
| FD004 | landmark Cox | 0 / 10 | 5 × 10⁻¹¹ | 7 × 10⁻²² |
| FD002 / FD004 | summary and FPCA Weibull AFT | 0–1 / 10 each | ≤ 1.3 × 10⁻⁶ | ≤ 3.9 × 10⁻¹³ |
| Six clinical cohorts | NP-KAN-AFT, landmark Cox, blend | 19–20 / 20 each | 0.77–0.999 | 0.013 |
| Renal | joint model (chosen form) | 1 / 19 | 5 × 10⁻⁵ | 4.9 × 10⁻⁸ |
| PBC E2 / PBC E1 | joint model, spline form (rarely chosen) | 3 / 20 and 5 / 19 | 0.0016 / 0.025 | 4.8 × 10⁻⁶ |

Findings:

1. **On the engine data every model is miscalibrated, NP-KAN-AFT least badly.** The manuscripts say calibration was
   "weak"; the size of the failure (p-values down to 10⁻¹²) is not stated. The IBS advantage of NP-KAN-AFT is real as
   measured, but it is an advantage among miscalibrated models.
2. **The engine test sets are clustered.** Each test set has about 480–570 windows cut from a much smaller number of
   engines, and the D-calibration test treats windows as independent. Its p-values are therefore too small by an
   unknown amount; the same clustering affects any window-level interval.
3. **On the clinical cohorts "D-calibrated" is weak evidence.** Test sets have 35–81 subjects and 8–36 events. Under a
   well-calibrated model D-calibration p-values should be spread evenly between 0 and 1 (median about 0.5); here the
   medians are 0.77–0.999 (heart valve 0.995–0.999). The test is evidently conservative with this much censoring and
   so few events, so 20 passes out of 20 says little. The sentence "NP-KAN-AFT was finite and D-calibrated" should be
   read, and worded, as "D-calibration did not reject".
4. The renal joint-model result (Section 5.14 of the Supplementary Material) is confirmed by the records.

---

## 5. MCMC diagnostic audit of the Bayesian joint model (saved Round-24 records only)

**What is saved.** For each fitted form: the R-hat of the **association parameters only**, the run time and the
prediction metrics. Settings in all 120 records: 3 chains, 6,500 iterations, 1,500 burn-in.

| Needed | Saved? |
|---|---|
| Posterior draws (any parameter) | **no** — fits ran in a temporary folder that was deleted |
| Effective sample sizes | **no** |
| R-hat of the other parameters (fixed effects, residual and random-effect variances, baseline hazard) | **no** |
| Trace or density plots, divergence or acceptance information | **no** |

**Therefore not reconstructable:** trace plots, effective sample sizes, rank-normalised split R-hat, Monte-Carlo
standard errors, or any independent verification of convergence. The R-hat values are those JMbayes2 reports
(Gelman–Rubin type), taken as recorded.

**What the saved R-hat values show** (largest association R-hat per fit; chosen form unless stated):

| Cohort | Association parameters | Fits | Median | **Largest** | > 1.01 | > 1.05 | > 1.1 | > 1.2 | > 1.5 |
|---|---|---|---|---|---|---|---|---|---|
| PBC E2 | 6 | 20 | 1.111 | **1.53** | 20 | 17 | 11 | 6 | 2 |
| PBC E1 | 6 | 20 | 1.105 | **9.22** | 20 | 17 | 11 | 2 | 1 |
| prothro | 1 | 20 | 1.005 | 1.02 | 3 | 0 | 0 | 0 | 0 |
| aids | 1 | 20 | 1.039 | 1.40 | 18 | 9 | 3 | 1 | 0 |
| renal | 2 | 19 | 1.042 | 1.13 | 16 | 8 | 4 | 0 | 0 |
| heart valve | 2 | 18 | 1.168 | **5.73** | 18 | 17 | 13 | 8 | 4 |
| **All** | | **117** | | | **95** | **68** | **42** | | |

Findings:

1. **Convergence is not established for most fits.** 42 of 117 chosen fits exceed 1.1, 68 exceed 1.05 and 95 exceed
   1.01 on the association parameters alone; the worst values (9.22 and 5.73) mean the chains disagreed completely.
   Only prothro is clean on this single measure.
2. **The manuscripts under-report this.** They give the counts above 1.1 for PBC and heart valve but not the largest
   values, the counts at stricter thresholds, or the fact that nothing but the association parameters was monitored.
3. **Prediction metrics from non-converged fits are of uncertain meaning.** Descriptively, the mean conditional IBS of
   fits above and below 1.1 differs in both directions (PBC E2 0.133 against 0.149; heart valve 0.138 against 0.128;
   renal 0.212 against 0.189), with small groups; no conclusion is drawn.
4. No claim of satisfactory convergence can be made for PBC E2, PBC E1 or heart valve, and for aids and renal only
   with reservations. The statements "the joint model was not better than NP-KAN-AFT" on those cohorts describe a
   joint model that may not have been fitted properly.

---

## 6. Evidence-to-claim review of the journal manuscript

Each claim of the abstract, Section 5 and the conclusions was set against the saved evidence.

| # | Claim in the manuscript | Evidence | Assessment |
|---|---|---|---|
| 1 | Functional KANs give readable nonlinear trajectory effects | PBC effect plots, bootstrap stability, clinical direction check (one exception) | **Supported, descriptive.** Not a registered test; the alignment rule is coarse and has no clinician sign-off |
| 2 | B2-S and AFCM recovered shapes to a similar degree; comparison inconclusive; AFCM unvalidated | H-R12: +0.0065 [−0.0506, +0.0635], 20 replicates; equivalence not shown | **Accurate but weaker than it reads.** Comparator is not the original AFCM (Section 2) and its optimiser is unreliable (Section 3). "To a similar degree" should not be said of a comparator in this state |
| 3 | The distillation refit is safe | H-R16a supported (all refits finite, shapes preserved); finite on held-out cohorts | **Supported.** One simulator; the refit is safe because it barely moves the model, which the text says |
| 4 | Cross-confirmed anchoring keeps the gain and stays within the margin | H-R20b supported for three model families | **Supported in simulation.** Margins were tuned on re-used development seeds (disclosed). On real data the anchor was accepted on at most one split per cohort, so a practical benefit on real data is **not shown** |
| 5 | NP-KAN-AFT is robust to the error law | H-R20a and H-R22c supported | **Supported in simulation**, with the stated boundary (not better than the correct parametric family). It is the classical semiparametric AFT estimator with a new location model, as the text says |
| 6 | On every clinical cohort no functional model was clearly better than landmark Cox | every interval includes 0 | **Supported** |
| 7 | On two held-out engine data sets NP-KAN-AFT had lower IBS than landmark Cox (H-R23a) | 19 of 20 splits, both intervals below 0 | **Supported as stated, over-read in the framing.** (a) C-MAPSS is **simulated** engine data; the manuscript text never says so. (b) The two data sets come from one simulator, so they are not independent confirmations. (c) All models are badly miscalibrated there (Section 4). (d) Comparators are landmark Cox and two-stage AFT models only; none of the sequence models normally used on C-MAPSS. (e) Windows are clustered within engines |
| 8 | Thinning renal histories makes NP-KAN-AFT worse (H-R23b) | 20 partitions, interval below 0 | **Supported.** One cohort, re-analysed data (disclosed) |
| 9 | "On long, dense histories the functional model was better on held-out data" (conclusions) | items 7 and 8 | **Too general.** The evidence is one simulated engine family plus one clinical cohort. Should be limited to those data |
| 10 | NP-KAN-AFT better than the joint model on renal (H-R24a), with the caveat that the joint model fitted poorly | interval below 0; joint model also worse than landmark Cox, D-calibration 1 of 19 | **Supported with the caveat, which the text gives** |
| 11 | "A standard Bayesian joint model was not better than the functional model or landmark Cox on the clinical cohorts" (Section 5.1) | no interval favours the joint model | **Needs qualifying.** One specification; convergence not established on three of six cohorts (Section 5); non-inferiority of NP-KAN-AFT not shown either. The abstract carries no convergence caveat |
| 12 | NP-KAN-AFT was "finite and D-calibrated" on the clinical cohorts | D-calibration passes | **Overstated**: the test has little power there (Section 4, finding 3) |
| 13 | The study is pre-registered | plans committed before each run | **Accurate as now worded** (internal, time-stamped, not a public registry). There is no control for the number of registered hypotheses (about 25), which should be said |
| 14 | Every number is rendered from saved results and verified | verification scripts pass | **Supported** for the numbers; it says nothing about whether the underlying fits were sound |
| 15 | No novelty claim; literature review targeted | stated | **Honest, but a journal will expect more**: a proper search of functional and dynamic deep survival models is missing |

Presentation issues that are not claims: 23 tables in a 22-page main paper is far more than journals accept; there is
no figure of the main results; author details and a public code location are missing; the reason "R was not available"
is out of date.

---

## 7. Remaining scientific and methodological limitations

In order of importance.

1. **The primary confirmatory test has no sound comparator.** H-R12 is inconclusive, the AFCM used is not the
   original implementation, it failed two acceptance checks, and its optimiser did not converge in the one fit examined.
2. **The main positive finding rests on simulated data.** FD002 and FD004 are simulated turbofan data from one
   generator. No real data set with long, dense histories was held out; renal is the only real long-history cohort and
   it was re-analysed.
3. **Calibration is poor where the model wins and untestable where it does not.** Severe miscalibration on the engine
   data; an uninformative test on the small clinical cohorts; no calibration curves anywhere because predictions were
   not saved.
4. **The joint-model comparison is not on firm ground.** One specification, convergence unverified or clearly absent
   on PBC and heart valve, no draws kept, three fits failed.
5. **Small real cohorts.** A few hundred patients, test sets of 35–81 with 8–36 events; no non-inferiority test could
   succeed with the registered margin, which was not justified clinically.
6. **Weak comparator set on the engine data.** No random survival forest or sequence model on the windows.
7. **One simulation design**, 20 replicates for the primary test, about 50–60 would have been needed to show
   equivalence within ±0.05 at 80% power (from the Round-12 spread).
8. **Many registered hypotheses, no multiplicity adjustment**, and the later methods were designed after earlier
   failures (disclosed).
9. **Registration was internal**, not public.
10. **The literature review is targeted**, and one cited comparator is known only from an abstract.
11. **Software and authorship.** The code is not yet in a public, citable location; the work was produced with an AI
    coding assistant under recorded decisions, which the target journal's policy must permit and the human authors
    must be able to vouch for.

---

## 8. Prioritised publication-readiness recommendations

**Must be done before submission.**

| Priority | Recommendation | Needs |
|---|---|---|
| 1 | **Fit the original AFCM with `mgcv`** on the simulation, and report whether it passes the existing acceptance thresholds; then re-test the primary comparison against it (Study A below). Until then, state in the paper the differences of Section 2.2 and drop "to a similar degree" | approval to install `mgcv`; new seeds; a registered plan |
| 2 | **Repair the Python AFCM optimiser** (a Newton or iteratively re-weighted solver with a convergence check), confirm on already-used development seeds that it reaches a zero gradient, and record how many earlier fits change. Report the earlier AFCM results as "optimiser not verified" | development seeds only; a registered plan before any re-test |
| 3 | **Refit the joint model with adequate sampling and keep the draws** (Study B); report R-hat and effective sample size for all parameters and treat non-converged fits as failures. Until then, add the largest R-hat values and the stricter counts to the paper and qualify claim 11 | approval for about 12–15 hours of computing |
| 4 | **Save predictions and report calibration properly** (Study C): calibration curves, Brier scores at fixed horizons, engine-clustered intervals. State in the abstract that C-MAPSS is simulated and that all models were miscalibrated on it | deterministic re-run of saved-seed fits, or new splits |
| 5 | **Narrow the headline claim** to what was tested: "on two simulated engine data sets and one clinical cohort with long histories" | manuscript edit only |
| 6 | **Reword "D-calibrated"** on the clinical cohorts as "D-calibration did not reject (low power)" | manuscript edit only |

**Should be done.**

| Priority | Recommendation |
|---|---|
| 7 | Add at least one **real** held-out data set with long, dense histories (Study D) |
| 8 | Add a random survival forest and one sequence model as comparators on the engine data |
| 9 | Cut the main paper to about six to eight tables and figures; add one figure of the main contrasts; move the rest to the Supplementary Material |
| 10 | Replace "R was not available" by the true present position |
| 11 | Do and document a structured literature search; say that about 25 hypotheses were registered without multiplicity adjustment |
| 12 | Deposit code and records in a public, citable archive; complete authors, funding and competing interests; check the target journal's policy on AI-assisted work |

Items 5, 6, 10 and 11 are wording changes that could be made at once on approval; they were **not** made in this audit.

---

## 9. Proposed new confirmatory studies (for approval; none started)

No seed has been assigned or consumed. Each study would be registered, with its seeds, before any run.

**Study A — the original AFCM as comparator (highest value).**
- *Step 1, acceptance:* fit AFCM with `mgcv` (`ti(..., bs = "cr", mc = c(TRUE, FALSE))`, `family = cox.ph()`, REML) on
  the existing acceptance procedure with one new seed. Pass rule unchanged: identifiable RISE of φ₁ below 0.15 and of
  φ₂ below 0.10. A failure is reported as a failure.
- *Step 2, only if step 1 passes — hypothesis H-R25a:* B2-S Cox minus `mgcv` AFCM in identifiable RISE on S-Add;
  equivalence within ±0.05 by two one-sided tests, superiority if the interval excludes 0.
- *Requirements:* install `mgcv` in `.rlocal/`; about 60 new replicates (the Round-12 spread implies about 51 for 80%
  power at ±0.05); a common data export so that both fits use identical inputs; half a day of computing.

**Study B — joint model with verified convergence.**
- *Hypothesis H-R25b:* as H-R24b (NP-KAN-AFT minus joint model, upper bound ≤ +0.005 on every cohort), evaluated only
  on fits that meet a pre-set convergence rule; H-R24a repeated as a secondary check.
- *Requirements:* the same partitions; longer chains with thinning; draws saved; rule fixed in advance (for example
  R-hat below 1.05 and effective sample size above 400 for every monitored parameter); a richer trajectory model for
  renal as a pre-specified secondary analysis; about 12–15 hours.

**Study C — calibration.**
- *Descriptive part:* re-run the saved-seed Round-23 fits with predictions written to disk and report calibration
  curves, Brier scores at fixed horizons and engine-clustered intervals. No hypothesis.
- *Hypothesis H-R25c (optional):* a recalibration step fitted on validation data only improves D-calibration of
  NP-KAN-AFT on new engine splits without raising IBS by more than 0.005.
- *Requirements:* new engine splits for H-R25c; about 3–4 hours.

**Study D — a real long-history data set.**
- *Hypothesis H-R25d:* NP-KAN-AFT minus landmark Cox in IBS has an upper bound below 0 on a real, previously unused
  data set with long, dense histories.
- *Requirements:* approval to download a public data set (not done here); selection of the data set before any model
  is fitted.

---

## Could not be verified in this audit

- Agreement of our AFCM with the `mgcv` AFCM, and the authors' own code (`vignette_afcm.Rmd` could not be downloaded).
- Whether the registered AFCM fits (CK-6c, CK-6d, H-R12, real-data rows) had converged.
- Which seed-22 fit is nearer the optimum, and the shape error at the optimum.
- Any calibration measure beyond the saved D-calibration p-values.
- Joint-model convergence beyond the saved association R-hat values (no draws, no effective sample sizes).
- The content of the "previous Round-25 prompt", which was not received.

**Stopping point.** The audit ends here. No further training, MCMC, confirmatory experiment or manuscript change has
been started. The findings are presented for the owner's review and approval.
