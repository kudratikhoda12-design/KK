# Experiment protocol (revision 2)

Status: **revision 2, 2026-10-16.** The owner's corrections are incorporated.
Execution is autonomous and staged under the owner's mandate (D21). This
protocol, including the primary outcome and primary comparison, is fixed
**before** any experiment runs. Any later change is logged in `decisions.md`
with a date and a reason. No result in this document is an experimental
finding. The landmark counts in §4.3 are descriptive counts from the raw data.

## 1. Staged execution (D15, D21)

| Stage | Scope | Depends on | Mandatory gate |
|---|---|---|---|
| A | Project-local environment and lock file; AFT likelihoods | – | E-1..E-7, A1..A8 |
| B | Synthetic generator; Observed/Oracle separation | A | G1..G12 |
| C | B-spline basis, smoother, roughness and Gram matrices, functional linear layer, FPCA; in-house KAN layer | A | F1..F5, F-KAN-1..4 |
| D | Models (§2), training strategies, baselines | A, B, C | M1..M5, K1..K4, then R1..R3 (learnability gate) |
| E | Evaluation module; synthetic experiments | D (and Q1..Q5) | Q1..Q5 before any metric is reported |
| F | C-MAPSS: download, inspect, loader, landmark options A and B | A, C, D, E | Provenance recorded; K1..K4 on C-MAPSS |
| G | PBC: provenance, landmark data set, endpoints E1/E2 as sensitivity analyses | A, C, D, E | Provenance recorded; K1..K4 on PBC |

**Failure policy.**
1. When a test fails, diagnose it and attempt at most **two targeted repairs**.
   Rerun the tests after each repair.
2. If the test still fails, the stage is marked **FAILED** or **BLOCKED**.
   Downstream work that needs it is not run. Independent work continues.
3. Thresholds are never changed to obtain a pass (D11).
4. If the functional MLP autoencoder fails the learnability gate (R2), the
   KAN-vs-MLP comparison is not run, because the comparison would be
   uninterpretable. Baselines that do not depend on the gate continue.

**Logging.**
- Every test and experiment command runs through `scripts/run_logged.sh`.
- It records the command, exit code, start and end time and elapsed seconds in
  `reports/logs/commands.jsonl`, and the full output in `reports/logs/<name>.log`.
- `reports/logs/execution_log.md` is the human-readable resumable log.

## 2. Models compared (D8)

| ID | Model | Representation | Head |
|---|---|---|---|
| fpca_aft | FPCA + AFT | FPCA scores (training-fit) | Linear Weibull AFT (L2 1e-3) |
| fpca_cox | FPCA + Cox (secondary reference) | FPCA scores | Cox PH (lifelines, penaliser 0.01) |
| fae_mlp | Functional MLP autoencoder + AFT | §3.4 of `design.md`, MLP blocks | Linear AFT |
| fae_kan | KAN functional autoencoder + AFT | KAN blocks, parameters matched within ±10% | Linear AFT |
| flat_ae | Flattened autoencoder + AFT | MLP autoencoder on a 20-point resampled grid | Linear AFT |
| sum_aft | Summary features + penalised AFT | last value, mean, sd, OLS slope per channel (+ `u`) | Linear AFT |
| sum_cox | Summary features + penalised Cox | same | Cox (lifelines) |
| sum_rsf | Summary features + random survival forest | same | scikit-survival RSF, 200 trees |

**Training controls** (functional autoencoders; `design.md` §4.3):
reconstruction-only (S1), survival-only (`α = 0`), joint (S2), and
pretrain-then-fine-tune (S3).

### 2.1 Tuning rule (D22): equal budget of 4 configurations per model

| Model | 4 configurations |
|---|---|
| fae_mlp, fae_kan, flat_ae | learning rate ∈ {1e-3, 3e-3} × weight decay ∈ {1e-4, 1e-3}. Architecture fixed: r = 4; MLP hidden width 32; KAN grid G = 5, spline order 3, width chosen to match the MLP within ±10%. |
| fpca_aft, fpca_cox | number of components ∈ {2, 4, 8, 12} (capped at the available dimension) |
| sum_aft | L2 ∈ {1e-3, 1e-2, 1e-1, 1} |
| sum_cox | penaliser ∈ {0.01, 0.1, 1, 10} |
| sum_rsf | min_samples_leaf ∈ {5, 10, 20, 40} |

**Selection procedure.**
- Within each replicate or outer split, each configuration is trained with
  seed 0 on the training part.
- The configuration with the lowest **validation IBS** (marginal-KM IPCW,
  censoring fitted on training data) is selected.
- The selected configuration is retrained with seeds 1 and 2.
- The replicate's result is the mean over the 3 seeds. Deterministic models
  run once.
- Neural models use early stopping on the validation value of their training
  objective.
- Test data are never used for selection.
- Parameter counts and wall-clock times are recorded for every run.

## 3. Splits and leakage prevention

- **Real data:** 5 outer splits by individual, 60/20/20 train/validation/test,
  stratified on `δ` (where δ varies).
- **Synthetic data:** 5 **independent replicate data sets** (generator seeds
  1000–1004), each split 60/20/20.
- Fitted on the training part only:
  - channel standardisation;
  - smoother λ;
  - FPCA;
  - sensor selection (C-MAPSS);
  - the censoring model for IPCW;
  - summary-feature scaling;
  - all hyperparameters.
- All landmark windows of one individual are in the same part (K3).
- No input observation is dated after the landmark (K4).
- The synthetic `Oracle` is used only by evaluation code (G5, M5).

## 4. Data sets

### 4.1 Synthetic

**Pre-registered primary setting P0:** V1, dense, 30% censoring, Weibull
event model, N = 1000 per replicate, D = 3, `p_miss = 0`.

**Core secondary settings** (one factor changed from P0):

| Setting | Change from P0 |
|---|---|
| P-c0 | 0% censoring |
| P-c60 | 60% censoring |
| P-sparse | Sparse sampling, `p_miss = 0.1` |
| P-V2 | Risk function V2 |
| P-lognormal | Log-normal event model, fitted with the default Weibull head (misspecification); run only if budget allows |

### 4.2 C-MAPSS (D3a; D3b unresolved)

**Framing.** C-MAPSS is a degradation / remaining-useful-life benchmark, not
naturally right-censored data.
- All training engines fail (`δ = 1`).
- The official test set gives true remaining life. That is known outcome,
  not censoring.

**Procedure.**
1. **Source and provenance.** Download from the PHM Society mirror of the NASA
   repository (S6). Record the URL, SHA-256 and file list. Cite
   Saxena & Goebel (2008).
2. **Inspection before any loader** is written:
   - number of engines;
   - training-life distribution;
   - test truncation lengths;
   - constant sensors.
3. **Subset.** FD001 is primary. FD003 is analysed only if budget allows.
4. **Both landmark options are evaluated, separately and labelled.** Neither
   was approved as the definitive design; D3b stays unresolved.
   - **Option A (single landmark):** `s₀` = minimum training-split life − 1;
     window `[0, s₀]`. Test engines with life ≤ `s₀` are excluded and counted.
     Target: `T − s₀`.
   - **Option B (multiple landmarks):**
     - trailing window `w = 30` cycles;
     - landmarks `s = 30, 40, 50, …` with `s ≤ T − 1`;
     - `s/100` is included in `u`;
     - target: `T − s`;
     - the official test set is evaluated at its last observed cycle (engines
       with fewer than `w` cycles are excluded and counted);
     - evaluation is reported per landmark band and engine-weighted, with
       bootstrap clustered by engine.
5. **Channels.** Sensors whose training-split standard deviation is below
   1e-6 are dropped. The decision is recorded per split.
6. **Metrics.**
   - **RUL** (reported separately from survival metrics): RMSE and MAE of the
     predicted median residual life, uncapped.
   - **Survival framing:** IBS (with no censoring, all IPCW weights are 1) and
     Harrell's/Uno's C.
7. **Artificial censoring** (Option A only, if budget allows): residual times
   of training-part engines are independently censored with
   `C ~ U(0, max residual)`. Results are labelled **ARTIFICIAL CENSORING**.
   They test the code path, not real censoring.
8. **Integer cycles:** continuous approximation (D25).

### 4.3 PBC sequential data (D9 unresolved)

**Descriptive landmark feasibility** (computed 2026-10-16):

| Landmark (days) | At risk | Deaths after | Transplants after | Median visits ≤ landmark (min) |
|---|---|---|---|---|
| 365 | 290 | 118 | 29 | 2 (1) |
| 730 | 278 | 107 | 28 | 3 (1) |
| 1095 | 245 | 81 | 21 | 4 (1) |

**Analysis plan (D26).**
- **Landmark:** `s = 730` days. Inputs are all visits with `day ≤ 730`. The
  window `[0, 730]` is rescaled to `[0, 1]`. The cohort is patients with
  `futime > 730`. Target: `futime − 730`.
- **Channels:** log(bili), albumin, log(alk.phos), log(ast), platelet,
  log(protime). `chol` is excluded because 42% of its visit rows are missing.
- **Baseline covariates `u`:** age and sex, in all models.
- **Endpoints**, run as two separate, explicitly labelled **sensitivity
  analyses**:
  - **E1:** death, with transplant treated as censoring. The estimand is the
    hypothetical survival without transplant, and it relies on CA-1 holding
    for transplant.
  - **E2:** death or transplant (composite). The estimand is transplant-free
    survival.
  - These estimate different quantities. **Neither is declared definitive.**
  - **E3 (competing risks) is not implemented**, so no competing-risk claims
    are made.
- **Visit process:**
  - visit counts per patient in the window are reported;
  - visit counts are **not** used as model features;
  - a descriptive univariate Cox association between visit count and outcome
    is reported, to show whether the visit process is informative.
- **Censoring:** marginal-KM IPCW is the primary metric, **flagged as a
  limitation**. Conditional (Cox-model) IPCW IBS is a sensitivity analysis
  (`design.md` §2.4).
- **Reconstruction:** errors at observed visits of test patients (no held-out
  points, because sampling is too sparse). Labelled accordingly.

## 5. Metrics (D6)

| Setting | Primary | Secondary | Always reported |
|---|---|---|---|
| Synthetic | Survival-curve error (below) | IBS, Uno's C | Calibration, reconstruction, parameters, runtime |
| PBC (right-censored survival) | **IBS** (marginal KM; limitation flagged) | **Uno's C** | Harrell's C, conditional-IPCW IBS, D-calibration, parameters, runtime |
| C-MAPSS, survival framing | IBS | Uno's C | D-calibration |
| C-MAPSS, RUL framing | RMSE of median residual life | MAE | Per landmark band (Option B) |

**Definitions.**
- **Survival-curve error:**
  `mean_i (1/t_max) ∫₀^{t_max} |Ŝ_i(t) − S_true,i(t)| dt`.
  `t_max` is the 0.9 quantile of `T*` in the generator's pilot sample, so it is
  fixed per setting and independent of the models. The integral uses the
  trapezoidal rule on 100 equally spaced points.
- **IBS:** 50 equally spaced times between the 10th and 90th percentiles of
  the observed test times, with IPCW weights.
- **Uno's C:** truncated at `τ` = the 90th percentile of observed test times.
- **D-calibration:** 10 bins. Censored individuals spread their mass uniformly
  over the bins below `Ŝ(Y_i)`. Reported with a χ² p-value.
- **Calibration-in-the-large** at the 25th, 50th and 75th percentile horizons:
  mean predicted `S(t_h)` vs the Kaplan–Meier estimate on the test set.
- **Reconstruction** (synthetic): 15% of observation points of each test
  individual with ≥ 4 points are hidden from the smoother. RMSE at those
  points on the standardised scale is reported **individual-weighted
  (primary)** and **observation-weighted**, alongside the smoother-only error.

## 6. Uncertainty and comparisons

- **Synthetic:** per setting, the mean ± SD over 5 independent replicates.
- **Primary comparison (H1)** on P0:
  - quantity: `fae_kan − fae_mlp` in survival-curve error;
  - test: paired t-interval over the 5 replicates (df = 4), with 95% CI and a
    sign count;
  - **with n = 5 the power is low; a non-significant result is reported as
    inconclusive, not as equivalence.**
- **Real data:** mean ± SD over 5 outer splits. Paired differences use the
  Nadeau–Bengio corrected resampled t-interval, with variance factor
  `1/5 + n_test/n_train`.
- All other comparisons are **secondary or exploratory** and reported without
  significance claims.
- Every run is reported, including failed or diverged runs.

### Ablations (P0, default configuration, 5 replicates × 3 seeds, no retuning)

Run in this priority order; the lowest priorities are dropped first if the
budget runs out:

| Priority | Ablation |
|---|---|
| AB1 | α ∈ {0, 0.1, 1, 10} × encoder {MLP, KAN}. Includes the 2 × 2 encoder × α ∈ {0, 1}. |
| AB2 | Strategy {S1, S2, S3} × {MLP, KAN} |
| AB3 | Encoder {linear, MLP, KAN} |
| AB4 | r ∈ {2, 4, 8, 16} × {MLP, KAN} |
| AB5 | KAN grid G ∈ {3, 5, 10} |
| AB6 | Head {linear, MLP, KAN} on fae_kan |
| AB7 | Fitted AFT family {Weibull, log-normal, log-logistic} × {MLP, KAN} |
| AB8 | Basis size K ∈ {8, 12, 16} (fae_mlp) |

**Compute budget:** CPU only, 4 cores. Synthetic experiments are capped at
about 4 hours of wall-clock time in total. Unfinished items are listed with
the reason.

## 7. Acceptance criteria

**Policy (D11, D18).**
- **DET** tests are deterministic numerical checks; any failure indicates an
  error.
- **STAT** tests use random samples at a **pre-specified fixed seed**. They
  have a small, non-zero probability of failing by chance.
- A STAT failure at its pre-specified seed is recorded as **FAIL**. It is never
  converted into a pass by changing the seed, the sample size or the
  threshold. Diagnostic reruns at other seeds may be run and reported as
  diagnostics only.
- Thresholds change only through an owner-approved entry in `decisions.md`.

### 7.1 Environment (Stage A)

| ID | Type | Criterion |
|---|---|---|
| E-1 | DET | `scripts/setup_env.sh` creates `.venv` using `python3 -m venv`; all installs use `.venv/bin/pip`. The global `pip list` is identical before and after (`diff`). |
| E-2 | DET | `torch.__version__ == "2.14.1+cpu"` and `torch.version.cuda is None`. |
| E-3 | DET | pandas version `< 3`. |
| E-4 | DET | The full planned set (numpy, scipy, pandas, scikit-learn, torch, lifelines, scikit-survival, pyyaml, matplotlib, pytest, hypothesis) imports; versions are printed. |
| E-5 | DET | `requirements-lock.txt` from `pip freeze`. Reinstalling into a throwaway virtualenv outside the repo (with the CPU index) succeeds, and its `pip freeze` is identical to the lock file. |
| E-6 | DET | `pip check` reports no broken requirements. |
| E-7 | DET | `pytest` collects and runs. |

### 7.2 AFT likelihood (Stage A)

| ID | Type | Criterion |
|---|---|---|
| A1 | DET | Event and censored terms equal scipy `logpdf` / `logsf` (`weibull_min`, `lognorm`, `fisk`) on `η ∈ {−3, …, 3}`, `σ ∈ {0.2, 0.5, 1, 3}`, `y` at ε-quantiles from 1e-6 to 1 − 1e-6. float64; `rtol = 1e-10`, `atol = 1e-10`. |
| A2 | DET | Independent numerical consistency. Integrating the density on the log-time scale with `scipy.integrate.quad`: `∫ f` from 0 to `y` equals `1 − S(y)` within 1e-8, and the total integral is 1 within 1e-8. The hazard computed by autograd of `−log S(t)` with respect to `t` equals `exp(log f − log S)` within 1e-8 (relative). |
| A3 | DET | On a mixed sample of events and censored observations, the summed log-likelihood equals the scipy sum within 1e-9. Each censored individual's term equals scipy `logsf` within 1e-10, and is unchanged when its event-density term is perturbed (it does not depend on the density). |
| A4 | DET | `torch.autograd.gradcheck` (float64) with respect to η and σ passes for all families. |
| A5 | DET | For `w ∈ [−50, 50]` (201 points), the log-likelihood and gradients are finite in float32 and float64. |
| A6 | DET | `S ∈ [0, 1]` and non-increasing on 1000 times. `S > 1 − 1e-6` at the 1e-7 quantile. `S < 1e-6` at the 1 − 1e-7 quantile. The median equals `exp(η + σ·median(ε))` within 1e-10 (relative). |
| A7 | DET | Data are simulated with a fixed seed: N = 3000, 2 covariates, about 30% independent censoring. Our float64 L-BFGS maximum log-likelihood is within 1e-4 (absolute) of lifelines `log_likelihood_` for the Weibull, log-normal and log-logistic AFT fitters. Mapped coefficients agree within 1e-3. |
| A8 | STAT | At seed 2026 and N = 5000, every parameter estimate (intercept, 2 slopes, log σ) is within 3 standard errors (inverse Hessian) of the truth, for each family. |

### 7.3 Synthetic generator (Stage B)

| ID | Type | Criterion |
|---|---|---|
| G1 | DET | `y_obs == min(T*, C)` exactly and `y_obs > 0`, for every combination of {V1, V2} × {dense, sparse} × censoring {0, 0.3, 0.6}. |
| G2 | DET | `delta == (T* <= C)` exactly, for the same combinations. |
| G3 | STAT | At seed 11 and N = 10000, the realised censoring fraction is within ±0.03 of the target for 0.3 and 0.6. Target 0 gives exactly 0. |
| G4 | DET | The same seed gives bit-identical outputs; a different seed gives different outputs. |
| G5 | DET | `Observed` has no oracle fields. `to_training_arrays()` raises `TypeError` for an `Oracle`. |
| G6 | DET | For censored individuals, changing `T*` (keeping `T* > C`) leaves the `Observed` arrays and the AFT loss bit-identical. |
| G7 | DET | Times lie in `[0, 1]` and are sorted. `L_i` is within the configured range. Padding has `mask = 0` and `value = 0`. Shapes match `design.md` §1.2. |
| G8 | STAT | At seed 12 and N = 20000, for each family, `(log T* − η)/σ_true` passes a KS test against ε (p > 0.01). |
| G9 | STAT | At seed 13 and N = 20000, `|corr(log E, η)| < 0.03`. |
| G10 | STAT | At seed 14 and **N = 100000** (raised from 20000 before any run, to make chance failure negligible; the threshold is unchanged; D18): the mean over individuals of `S_true(t)` is within 0.01 of the empirical fraction with `T* > t`, at 20 times (5th to 95th percentiles). |
| G11 | DET | For a fixed seed, `T*`, θ, times, values and masks are identical across censoring targets. |
| G12 | STAT + DET | At seed 15, N = 10000, `p_miss = 0.2`: the masked fraction of real (non-padding) entries is within 0.2 ± 0.01 (STAT). Masked entries have `value = 0` (DET). |

### 7.4 Functional representation and KAN layer (Stage C)

| ID | Type | Criterion |
|---|---|---|
| F1 | DET | The B-spline basis matches `scipy.interpolate.BSpline` within 1e-12. The basis sums to 1 on `[0, 1]` within 1e-12. Derivatives match scipy within 1e-9. |
| F1b | DET | `R` and `G` match an independent dense-grid Simpson integration within 1e-6 (relative). |
| F2 | DET | The smoother recovers spline coefficients exactly (1e-6) from noise-free data with λ = 0. With `L_i < K`, or zero observations, the result is finite (zero observations give `c = 0`). |
| F3 | DET | Discretisation consistency. A fixed smooth test function is sampled on nested regular grids `n ∈ {25, 50, 100, 200, 400, 800}`. For the functional features `h` (fixed random β, seed 0) and the FPCA scores, the relative change between consecutive grids is non-increasing, and below 1e-2 between n = 400 and 800. A jittered grid with n = 800 (seed 0) is also within 1e-2 of the regular n = 800 result. |
| F4 | DET | FPCA: training scores have mean 0 (1e-10) and a diagonal covariance (off-diagonal below 1e-8 relative to the diagonal). Eigenvalues are non-increasing. Reconstruction with all components is exact (1e-8). |
| F5 | DET | The functional linear layer (`cᵀ J b`) equals direct dense-grid quadrature of `∫ β x dt` within 1e-8. |
| F6a | DET | (Revision 3, D40.) With only the identity feature map, the NFL equals `FunctionalLinear` with the same β within 1e-10. |
| F6b | DET | (D40.) NFL features with fixed random weights (seed 0): relative change non-increasing for n = 25…800 regular grids, and < 1e-2 between n = 400 and 800. |
| F6c | DET | (D40.) NFL output shapes; float64 gradcheck. |
| F-KAN-1 | DET | The KAN layer's B-spline basis matches scipy `BSpline` within 1e-10 inside the grid. |
| F-KAN-2 | DET | Output shapes are correct; `gradcheck` (float64) passes. |
| F-KAN-3 | DET | For inputs outside the extended grid, the spline contribution is exactly 0 and the output equals the SiLU path. |
| F-KAN-4 | DET | The parameter count equals `n_in·n_out·(G + k + 2) + n_out`. |

### 7.5 Models (Stage D)

| ID | Type | Criterion |
|---|---|---|
| M1 | DET | All models give the correct output shapes for variable `L` and random masks, including an individual with one observation and an empty channel. |
| M2 | DET | Each neural model memorises one fixed batch of 16 (seed 0): the loss falls below 1% of its initial value within 3000 steps, or the reconstruction loss falls below 1% when trained with `α` only. |
| M3 | DET | fae_kan parameter count is within ±10% of fae_mlp, for every configuration used in comparisons. Any configuration that cannot be matched is recorded. |
| M5 | DET | Every `fit` entry point raises `TypeError` for an `Oracle`. |
| K1 | DET | Individual IDs are disjoint across train, validation and test. |
| K2 | DET | Preprocessing statistics (standardisation, FPCA mean, smoother λ) equal those recomputed from the training part only. |
| K3 | DET | All windows of an individual are in a single part (C-MAPSS Option B). |
| K4 | DET | `max(input time) ≤ s` for every model input (C-MAPSS, PBC). |

### 7.6 Learnability gate (end of Stage D; STAT)

| ID | Criterion |
|---|---|
| R1 | At seed 21, N = 2000, setting P0 without censoring: a 2-layer MLP AFT head (width 32) given the true `(a, b, A, ω)` reaches a test Harrell C within 0.02 of the oracle C (computed from `η_true`), and `σ` within 10% of 0.5. |
| R2 | Same data with `σ_y = 0.05`: fae_mlp with S2 and default configuration reaches test C ≥ oracle C − 0.03. A linear probe (ridge, 1e-3) from `z` to each of `a, b, A, ω`, fitted on the training part, has **mean** test R² ≥ 0.8 over the four parameters. |
| R3 | P0 data (30% censoring): fae_mlp's test C is within 0.02 of its R2 value. A naive fae_mlp trained with all `δ = 1` (censored times treated as events) has a larger survival-curve error than the censoring-aware model at **each** of seeds 0, 1 and 2. |

### 7.7 Evaluation module (Stage E)

| ID | Type | Criterion |
|---|---|---|
| Q1 | DET | Our IPCW Brier and IBS equal scikit-survival `brier_score` / `integrated_brier_score` within 1e-8 on fixtures. Our Harrell C equals lifelines `concordance_index` within 1e-12. |
| Q2 | DET + STAT | An oracle score on uncensored, tie-free data gives C = 1 exactly (DET). At seed 31, N = 5000, a random score gives `|C − 0.5| < 0.03` (STAT). |
| Q3 | DET | The conditional-IPCW Brier with `G(t \| x) = Ĝ_KM(t)` for everyone equals the marginal implementation within 1e-12. |
| Q4 | STAT | At seed 32 and N = 5000: true-model predictions give a D-calibration p-value > 0.01, and predictions with σ doubled give p < 0.01. |
| Q5 | DET | The survival-curve error is 0 for oracle predictions. On a constructed case with a known analytic value, it matches within 1e-6. |
