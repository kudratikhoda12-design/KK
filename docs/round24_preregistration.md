# Round 24 pre-registration (D163): a Bayesian joint model as the missing standard comparator

Owner instruction (2026-11-10): "yes install R and fit the joint model", in reply to the readiness assessment that the
largest gap a reviewer would raise is that no joint model was compared (journal §5.4: "No joint model or deep dynamic
model was fitted"). This plan was written and committed **before any formal job of `scripts/run_round24.py` was run**.
The rules below are not changed after results are seen; nothing is refitted, re-split or re-selected because of a
result.

## 1. Software (project-local; the system and the Python environment are unchanged)

R 4.4.3 with JMbayes2 0.6.0, nlme 3.1.170, survival 3.8.12 and jsonlite 2.0.0 from conda-forge, and JM 1.5.2 from
CRAN, installed with micromamba into `.rlocal/` (git-ignored; not a project dependency of the Python code). JMbayes2 is
used because it fits several longitudinal markers jointly (JM fits one); JM was installed but is not used.

## 2. Data, partitions and pairing

The six clinical cohorts and the **Round-23 partitions k = 5..24** (`split_groups(ids, k, strata=event)`), with the
Round-23 single-landmark cohorts (`run_round13.RealCtx`; landmarks pbc 730 days, prothro 1 year, aids 6 months, renal
and heart_valve 1 year). These are **re-analyses of previously analysed data**, as in Round 23. The NP-KAN-AFT,
landmark Cox and blend results are **read from the Round-23 records, not refitted**; a partition is paired only if the
cohort size, test-set size and test events agree (test R24-2 checks on real_heart_valve_5 that the landmark Cox test
metrics recomputed now equal the Round-23 record exactly). C-MAPSS is **not** analysed: every Option-B window is an
event by construction and each engine has 14–17 sensors over hundreds of cycles, which is not a joint-model setting
and is beyond JMbayes2's practical scope; this is stated as a limitation.

## 3. The joint model (fixed; `scripts/jm_fit.R`, `scripts/run_round24.py`)

- **Fitted on the training subjects only.** Longitudinal submodel per marker on each training subject's **full
  observed history** (all visits up to the subject's own event or censoring time; standard joint-model practice; no
  validation or test data). Survival submodel: Cox on the same baseline covariates as landmark Cox, with **delayed
  entry at t_L** (every cohort member is event-free at t_L). Association: the current value of every marker (JMbayes2
  default). All markers used by the functional models enter jointly (pbc 6, renal 2, heart_valve 2, prothro 1,
  aids 1). pbc times are converted from days to years inside R (numerical scale only).
- **Two longitudinal forms, chosen per partition on validation:** `lin` = `lme(value ~ t, random = ~ t | id)`;
  `ns2` = `lme(value ~ ns(t, 2), random = ~ t | id)` (natural cubic spline mean trajectory, subject-specific intercept
  and slope). "jm" is the form with the lower **validation conditional IBS** (ties → `lin`; a form whose fit failed
  or gave non-finite validation curves is not eligible). Both forms are recorded. This mirrors the Round-23 rule for
  the two landmark Cox versions on C-MAPSS, so the comparator is not handicapped by one fixed trajectory shape.
- **MCMC:** 3 chains, 6,500 iterations, 1,500 burn-in (the JMbayes2 defaults are 3,500 and 500; lengthened because
  with the defaults one association parameter had Rhat 1.26 on the renal development partition, a convergence
  reason only), seed 2400 + k, one core for fitting and for prediction (results do not depend on the machine; test
  R24-4 checks that a repeat run is identical). The max Rhat of the association parameters is recorded and reported
  per cohort; a fit is not excluded or refitted because of its Rhat.
- **Prediction:** for validation and test subjects, JMbayes2 dynamic prediction from the history up to t_L (the same
  window the landmark models use), event-free at t_L, giving S(t_L + s | T > t_L, history). The test-subject entry
  time is set just before t_L, so the posterior of the random effects uses the history and the landmark cohort's
  random-effects distribution (the model was fitted with delayed entry) without a second survival-to-t_L factor.
  Curves are predicted on 201 equally spaced residual times from 0 to the longest training follow-up (JMbayes2 does
  not predict beyond it), interpolated linearly and held flat after the last grid point. Risk score for concordance:
  minus the mean predicted survival over the test IBS grid (the Round-23 blend convention).
- **Metrics:** exactly the Round-23 clinical metrics (`run_round13.real_metrics`): conditional IPCW IBS (primary),
  marginal IBS, Harrell C, Uno C, D-calibration p.

## 4. Hypotheses and decision rules

Intervals: Nadeau–Bengio corrected 95% t-intervals over the paired partitions (`analyse_round13.nb`, frac = mean
n_test / n_train of the cohort). Primary metric: test conditional IBS.

- **H-R24a (renal; pre-specified as in H-R23c (ii): the only clinical cohort with long histories).** Mean of
  b2s_npaft − jm below 0 with NB upper bound < 0.
- **H-R24b (non-inferiority on every cohort).** On each of the six cohorts the NB upper bound of b2s_npaft − jm is
  ≤ +0.005 (the Round-20/21/23 margin). Supported if all six meet it; per-cohort results are reported either way.
- **Descriptive (no verdict):** landmark_cox − jm and stack − jm intervals; Harrell C contrasts; D-calibration passes;
  the chosen form per partition; run time.
- **Failed fits.** Every failed fit is reported with its error and **never refitted with other settings**. A
  hypothesis is **not evaluable** if jm failed (no eligible form) on more than 2 of the 20 partitions of a cohort it
  needs. Contrasts use the partitions where both models are finite.
- **Not done:** no other joint-model specification, association structure, prior or MCMC length is tried after
  results are seen; no deep dynamic model (e.g. Dynamic-DeepHit) is fitted in this round; no superiority claim
  beyond the rules above (D111).

## 5. Development (mechanics only; partitions k = 0, already used in Rounds 13–22)

Short MCMC (1 chain, 600 iterations, 200 burn-in) unless stated. Problems found and fixed, each before any formal job:
(i) JMbayes2 requires every mixed model to be fitted on the same data frame → one long data frame with `na.exclude`;
(ii) prediction needs an entry time below the prediction start → test entry just before t_L (rationale in §3);
(iii) JMbayes2 does not predict beyond the longest training follow-up (prothro: the last 20 grid points were missing)
→ grid up to the longest training follow-up, held flat after it; (iv) a spline form with spline random effects
(`random = ~ ns(t, 2) | id`) failed to converge in `lme` on aids, renal and pbc_E2 → `ns2` uses the spline only in the
mean trajectory; (v) `predict()` starts parallel workers by default → one core with the job seed.

**Numbers seen in development (k = 0; reported here because they were seen before the rules were fixed):** test
conditional IBS, `lin` form: pbc_E2 0.0957, pbc_E1 0.0986, prothro 0.2228, aids 0.1563, renal 0.1956, heart_valve
0.1123; `ns2` form (spline mean, linear random effects): pbc_E2 0.1342, aids 0.1583, renal 0.2336. The two-form rule
(§3) was added after these were seen, **to avoid handicapping the comparator** with a single straight-line trajectory
on cohorts with long, curved histories (renal); the choice between the forms uses validation data only. Earlier
records on the same k = 0 partitions (Round-23 development, D160 §2): pbc_E2 NP-KAN-AFT 0.0988, landmark Cox 0.1026;
renal NP-KAN-AFT 0.1540, landmark Cox 0.1751. A bug check at the default MCMC length on renal k = 0 compared the mean
predicted test survival with the test Kaplan–Meier estimate at the test-time quartiles (`lin` 0.791 / 0.760 / 0.735
against 0.753 / 0.720 / 0.720; `ns2` 0.872 / 0.849 / 0.828) and gave S = 1 at t_L for every subject; max Rhat of the
association parameters 1.26 (`lin`) and 1.25 (`ns2`), and fit times 222 s and 299 s. The longer MCMC (§3) was then
timed on pbc_E2 and renal k = 0, printing only run times and Rhat values: pbc_E2 `lin` 213 s (max Rhat 1.07), `ns2` 280 s (1.11); renal `lin` 329 s (1.08), `ns2` 461 s (1.15).

## 6. Execution

120 jobs (six cohorts × 20 partitions) on four single-threaded lanes from one shared queue
(`scripts/run_round24_lanes.py`), resumable (a job with a final record, ok or a recorded fit failure, is skipped).
Expected run time about 3.5–5 hours (about 8 minutes per pbc job and 13 per renal job for both forms; the one-marker cohorts are faster). Tests R24-1..R24-4 (`tests/test_round24.py`) pass before the launch. Relaunches after container
restarts are logged here with their time and change nothing. Analysis: `scripts/analyze_round24.py` (written before
the run; it reads the records only).

**Run log (added after the run; nothing above changed).** Lanes launched 2026-11-10 11:49:54 UTC; all 120 jobs had a
final record at 14:24 UTC (117 ok, 3 recorded fit failures: real_heart_valve_8, real_heart_valve_12, real_renal_12,
both forms failing in `lme` (nlminb "false convergence" or "function evaluation limit"); not refitted). No relaunch.
One reading bug in `scripts/analyze_round24.py` was fixed before any result was printed: on the one-marker cohorts R
writes the association Rhat as a scalar, not a list, and the Rhat summary failed on it; the fix
(`np.atleast_1d`) changes how the records are read, not any rule or number. Results: `docs/round24_results.md`, D164.
