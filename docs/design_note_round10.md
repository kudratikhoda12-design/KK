# Round 10 design note and pre-registration of Rounds 11–13 (2026-10-19)

Owner decisions: D110. Literature and claim policy: D111 (`docs/literature_review_round10.md`). Pre-registrations:
D112 (Round 11), D113 (Round 12), D114 (Round 13). This document is the binding specification. Any change after
the commit that registers it needs an owner-approved decision entry and is reported as a deviation.

**Notation.**
- Subject i has D longitudinal channels X_id(t), observed at times t_ij (with noise) on a history window rescaled to
  [0, 1], scalar covariates z_i, and the outcome (Y_i, δ_i).
- G is a fixed grid of 50 equispaced points on [0, 1]. Integrals ∫·dt are trapezoid quadratures on G.
- In tests, X̂_id(·) denotes the reconstructed curve on G: smoother output (dense data) or PACE conditional
  expectation (sparse data).

## 1. Survival losses and the shared KAN machinery (Round 11)

- **Cox (CoxKAN).**
  - η_i = f(·) (no intercept).
  - Negative log partial likelihood with **exact Breslow** and **Efron** tie handling (both implemented; Efron is the
    default, as in `lifelines`).
  - Breslow estimator of the baseline cumulative hazard Λ̂_0(t). Predicted survival S(t | x) = exp(−Λ̂_0(t) e^η).
  - CoxKAN's original fast approximation is not used; this is a documented deviation.
- **AFT (KAN-AFT).**
  - log T = μ + η + σε.
  - Losses already in the code base: parametric Weibull, IPCW, transform least squares, Buckley–James.
  - New: **non-parametric residual calibration**. Survival is predicted as S(t | x) = Ŝ_ε((log t − μ̂ − η̂)/σ̂), with
    Ŝ_ε the Kaplan–Meier estimate of the standardised residuals on the training data. This is used for KAN-AFT under
    S-Misspec and for C-MAPSS.
- **KAN layers.**
  - Existing B-spline KAN layer (`kanfae.models.kan`): grid 5, order 3, SiLU base.
  - Inputs are standardised with training statistics, not BatchNorm, so that edge functions are readable in original
    units after back-transformation.
  - Grid range: [−3, 3] in standardised units, plus an extension for out-of-range inputs.
- **Regularisation and scale-aware pruning** (replaces the absolute θ = 1e-2 rule; round 8 showed it prunes nothing
  on unit-scale inputs).
  - Loss + λ·(|Φ|₁ + 2·S(Φ) + |C|₁).
  - The edge score s_e = mean over training inputs of |φ_e − mean φ_e| (centred, D54), normalised as
    s̃_e = s_e / max_e' s_e' within the layer.
  - An edge is pruned if s̃_e < θ_rel. A pruned edge's mean is absorbed into the output bias.
  - Grid: λ ∈ {0, 1e-4, 1e-3, 1e-2} × θ_rel ∈ {0, 0.01, 0.02, 0.05, 0.10}.
  - Procedure: train at each λ; prune at each θ_rel; fine-tune 200 epochs with the mask fixed.
  - Selection by validation loss (Cox: validation negative log partial likelihood, Efron; AFT: the training
    loss on the validation split). Ties go to the larger θ_rel, then the larger λ.
  - **Channel selection:** a channel is "selected" if any of its edges survives pruning.
- **Symbolic extraction** (`kanfae.symbolic.fit`):
  - stage 1: linear;
  - stage 2: library;
  - stage 3: PySR.
  - Acceptance: validation R² > 0.99.
  - PySR (60 s per edge) runs only on edges carrying at least 5% of the layer's total centred score, at most 10 per
    model.
  - Metrics are reported before pruning, after pruning and after substitution.

## 2. Model families

**B1: two-stage interpretable coordinates.**
- **B1-FPCA.** FPCA scores per channel, with K_d chosen by 95% fraction of variance explained (max 5).
  - Dense data: eigen-decomposition of the smoothed curves on G.
  - Sparse data: **PACE** (Yao, Müller & Wang 2005), new module `kanfae.functional.pace`. Mean and covariance by local
    linear smoothing with a bandwidth chosen by leave-one-subject-out CV; scores by conditional expectation.
  - The scores, together with z, feed:
    - a 1-layer additive KAN [p → 1] (B1-FPCA-1), or
    - a 2-layer KAN [p → 4 → 1] (B1-FPCA-2);
    - under the Cox loss (CoxKAN) or AFT loss (KAN-AFT).
  - Effect curves in X are recovered as φ_d(·) composed with the eigenfunctions (reported as β-type functions).
- **B1-SUM.** Per channel:
  - the level (last value before the landmark, and the mean);
  - the slope (OLS);
  - the variability (residual SD).
  - These feed the same KAN heads.

**B2: functional KAN layer on X(t)**, new module `kanfae.models.fkan`.
- **B2-H (time-homogeneous):** η = Σ_d ∫₀¹ φ_d(X̂_d(t)) dt + Σ_k ψ_k(z_k). Each φ_d and ψ_k is a univariate KAN
  edge.
- **B2-S (separable bivariate, rank M = 2):** φ_d(x, t) = Σ_{m=1..M} g_dm(x)·h_dm(t). Each g and h is a univariate
  KAN edge (h on t ∈ [0, 1]). The learned surface and its symbolic form are read in x and t separately.
- **B2-deep (for interactions):**
  - Functional summaries s_dm = ∫ g_dm(X̂_d(t)) h_dm(t) dt feed a second KAN layer [D·M + p → 4 → 1].
- **Input scale.**
  - Primary: X̂ standardised by the training mean and SD pooled over G, then back-transformed for plots and
    formulas.
  - Sensitivity: the pointwise quantile transform x̃(t) = F̂_t(X̂(t)) of Cui et al.
- **Identifiability.**
  - Each φ_d is centred: E over training (i, t) of φ_d = 0, with the constant absorbed in the baseline.
  - For B2-S, g and h are rescaled after training so that ||h_dm||₂ = 1.
- **Losses.** Cox (Efron) or AFT (parametric Weibull primary; residual-calibrated variant).
- **Training.** Adam, lr {1e-3, 3e-3} × weight decay {1e-4, 1e-3} tuned on validation loss; early stopping with
  patience 100; maximum 3,000 epochs; the 3-seed ensemble (seeds 0–2) is the primary prediction.
- **Ensemble shapes.** Shape estimates are the mean of the members' centred φ̂, which is valid because η is additive
  in φ.

**B3: stacked landmark supermodel (real data, Round 13).**
- Landmarks t_L on a pre-registered grid per data set (§5).
- For each subject at risk at t_L: features from history up to t_L only (B1-SUM, B1-FPCA/PACE, or B2 on the window
  [t_L − W, t_L]). The outcome is time since the landmark, administratively censored at horizon w.
- The stacked data set has one row per (subject, landmark).
- Model: CoxKAN with a landmark-stratified baseline, and t_L also entering as a KAN edge input, so that effects can
  vary with the landmark (supermodel).
- Splits are by subject. Evaluation uses time-dependent AUC and Brier score at t_L + w, and C within the horizon,
  per landmark.

## 3. Baselines (Round 11 implementation; all under the same splits and metrics)

| ID | Model | Implementation |
|---|---|---|
| `fpca_cox` | Linear functional Cox (FPCA scores) | existing (round 1) |
| `sum_cox`, `sum_aft` | Linear Cox / Weibull AFT on summaries | existing |
| `afcm` | **Additive functional Cox model** (Cui et al. 2021): ∫F(X(t), t)dt, F a tensor-product cubic B-spline (10 × 10 basis) with second-order difference penalties in x and t; penalised partial likelihood (Efron). Smoothing parameters chosen by validation partial likelihood on a 5 × 5 grid, **instead of REML in `mgcv`**, because R is not available | **Python re-implementation** (`kanfae.models.afcm`); a deviation from the reference software, disclosed |
| `lme2_cox` | Two-stage mixed-effects + Cox: per-channel linear mixed model (random intercept and slope, Gaussian), BLUPs of intercept and slope at the landmark into a Cox model (joint-model-style two-stage) | Python re-implementation (EM for the LME), since R/JM is not available |
| `landmark_cox` | Cox on last observed values at the landmark (static) | lifelines |
| `sum_rsf` | RSF on summaries | existing |
| `static_coxkan`, `static_kanaft` | CoxKAN / KAN-AFT on last values + z | new Cox head; existing KAN-AFT |
| `fae_mlp_s2`, `fae_mlp_s3`, `fae_kan_s2`, `fae_kan_s3`, `fae_kan_unb_s3` | Locked autoencoders (C4F; S2/S3; Mode A unbottlenecked) | existing (rounds 8–9) |

## 4. Simulation study (Round 12): data-generating processes

**Common to all scenarios.**
- n = 1,000 subjects (train/validation/test 60/20/20, the existing `split_groups`), D = 3 channels on t ∈ [0, 1].
- Curves: X_d(t) = μ_d(t) + Σ_{k=1..4} ξ_dk ψ_k(t), with ψ_k orthonormal Fourier functions,
  ξ_dk ~ N(0, λ_k), λ = (1, 0.5, 0.25, 0.125), and μ_d(t) = 0.5·d·t.
- Observations X_d(t_ij) + N(0, 0.2²).
  - Dense: 40 equispaced times.
  - Sparse (S-Sparse only): 3–8 visits (uniform count) at uniform random times.
- Scalar covariates: z_1 ~ N(0, 1) (linear effect 0.3), z_2 ~ Bernoulli(0.5) (effect −0.3).
- Censoring: independent uniform, calibrated per scenario to about 30%.
- Truth is stored in the oracle for every subject.

| Scenario | True linear predictor (channel 3 is a null channel in every scenario) | Event model |
|---|---|---|
| **S-Add** | η = ∫ φ₁(X₁(t)) dt + ∫ φ₂(X₂(t), t) dt + 0.3 z₁ − 0.3 z₂; φ₁(x) = sin(πx/2) (homogeneous); φ₂(x, t) = tanh(x)·(1 + t) (separable, rank 1) | Weibull PH, shape 1.5 (log-hazard η; Cox is correct) |
| **S-Int** | η = S-Add's η + 0.8·[∫φ₁(X₁)dt]·[∫ tanh(X₂(t)) dt] (cross-channel multiplicative interaction) | Weibull PH, shape 1.5 |
| **S-Int/P0** (continuity) | Round-1 simulator setting P0 (V1 risk, dense, 30% censoring); KAN-FAE S3 vs MLP-FAE S3 (C4F with strategy S3, tuned) on fresh seeds | as round 1 |
| **S-Sparse** | S-Add's η; sparse observation scheme (3–8 visits) | Weibull PH, shape 1.5 |
| **S-Misspec** | S-Add's η used as an AFT location: log T = 1 + η' + 0.5ε with η' = −η/1.5, ε log-normal (Gaussian ε) or log-logistic (two sub-scenarios). Non-PH, so Cox is misspecified | log-normal / log-logistic AFT |

**Replicates and seeds** (verified unused on 2026-10-19):
- N = 20 per scenario.
- S-Add 3000–3019; S-Int 3100–3119; S-Sparse 3200–3219; S-Misspec 3300–3319 (each seed generates both sub-scenarios
  from independent streams); S-Int/P0 continuity 3400–3419.

**Models per scenario.**
- S-Add, S-Sparse: B2-H, B2-S (Cox and AFT); B1-FPCA-1 (Cox, AFT); B1-SUM; afcm; fpca_cox; sum_cox; sum_rsf;
  static_coxkan.
  - S-Sparse adds the PACE front end for B1/B2 against the fixed B-spline front end, and flat_ae.
- S-Int: the above, plus B1-FPCA-2 and B2-deep.
- S-Int/P0: fae_kan_s3 vs fae_mlp_s3 only.
- S-Misspec: CoxKAN vs KAN-AFT (parametric Weibull; residual-calibrated) for B2-S, plus afcm.

**Metrics.**
- Survival: survival-curve error against the oracle, IBS (marginal IPCW), Harrell C, Uno C (τ = 90% quantile),
  D-calibration p.
- Shape (S-Add, S-Sparse, S-Misspec): **relative ISE**:
  - RISE_d = ∫∫ (φ̂_d^c − φ_d^c)² w_d dx dt / ∫∫ (φ_d^c)² w_d dx dt, with the superscript c denoting centring and w_d
    the empirical density of (X_d(t), t) in the test data.
  - Homogeneous effects integrate over x only.
  - For Cox, φ is identified up to a constant (centring). For AFT, the comparison uses −φ̂/σ̂-scaled shapes against
    the PH-equivalent truth only in S-Add, and the AFT truth in S-Misspec.
  - Primary shape summary: the mean RISE over the active channels (1, 2).
- Selection: recall of the active channels {1, 2} and the false discovery rate (null channel 3 selected), from
  pruning.
- Symbolic: acceptance rate and validation R² of edges; formula RISE after substitution.

**Statistical analysis.**
- Paired over replicates; mean difference, SD, 95% t-CI (df 19), paired-t p, exact Wilcoxon p, sign counts.
- **Single confirmatory test of Round 12 (H-R12):**
  - **Quantity:** in **S-Add**, B2-S under the Cox loss (functional CoxKAN) minus `afcm` in mean RISE over the
    active channels.
  - **Test:** two-sided paired t-test, α = 0.05, N = 20.
  - **Interpretation, fixed now:**
    - If the 95% CI lies entirely below 0, B2-S recovers shapes better than AFCM.
    - If it lies entirely above 0, worse.
    - If it lies inside ±0.05 (absolute RISE), the two are declared **equivalent within the margin** (TOST at α =
      0.05 each side).
    - Otherwise, inconclusive.
- **Pre-registered secondary tests (exploratory, no multiplicity correction):**
  - S-Add survival-curve error, B2-S vs afcm.
  - B2-S vs B1-FPCA-1 (RISE).
  - Selection recall/FDR.
  - S-Int: B2-deep vs B2-S, and B1-FPCA-2 vs B1-FPCA-1 (curve error).
  - **S-Int/P0 continuity:** TOST equivalence of fae_kan_s3 vs fae_mlp_s3 in survival-curve error, margin ±0.005
    (about 5% of the round-9 MLP S3 error, 0.0915). This re-confirms the round-9 F1 finding with fixed criteria on
    fresh seeds.
  - S-Sparse: PACE vs fixed B-spline front end (RISE, curve error); flat_ae vs B2.
  - S-Misspec: KAN-AFT (residual-calibrated) vs CoxKAN (curve error, D-calibration).
- **Every run is reported.** Failed fits are recorded and counted.

## 5. Real data (Round 13)

| Data set | Cohort / endpoint | Channels | Single landmark | B3 landmarks (horizon w) |
|---|---|---|---|---|
| PBC (`pbcseq`) | E2 death or transplant (primary), E1 death (sensitivity) | The 6 round-1 channels, as in `kanfae.data.pbc.CHANNELS`: log bilirubin, albumin, log alkaline phosphatase, log AST, platelets, log prothrombin time; age, sex | 730 days (aligned with rounds 1–9) | 1, 2, 3, 4 years; w = 5 years; window W = 2 years |
| `prothro` (JM) | Death | Prothrombin index; treatment | 1 year | 0.5, 1, 2, 3 years; w = 5 years; W = 2 years |
| `aids` (JM) | Death | CD4 (sqrt scale, as conventional); drug, gender, prevOI, AZT | 6 months | 2, 6, 12 months; w = 12 months; W = full history |
| C-MAPSS FD001 (secondary) | End of life (Option B windows) | Informative sensors | Option B as round 1 | – |

**Real-data protocol.**
- 5 outer splits by subject (seeds 0–4) and the same models as S-Add, with lme2_cox and landmark_cox added.
- Metrics: IBS (marginal and conditional IPCW), Harrell and Uno C, D-calibration; for B3, time-dependent AUC and Brier
  per landmark.
- KAN vs reference comparisons use the Nadeau–Bengio corrected t-interval.
- C-MAPSS: KAN-AFT with residual calibration (the calibration fix) against round 1's parametric AFT; D-calibration
  reported.

**Interpretability deliverables (pre-registered measures).**
- **Stability:** for each selected effect, the centred φ̂_d across the 5 outer splits × 20 bootstrap resamples of the
  training split. Report the mean pairwise R² and mean pairwise cosine similarity on a common x-grid (the 5th–95th
  percentile of pooled X).
- **Effect plots** with bootstrap 90% bands, in original clinical units.
- **Clinical alignment.** Directions fixed now, from established clinical knowledge:
  - PBC: higher bilirubin increases risk; higher albumin decreases risk; longer prothrombin time increases risk.
  - prothro: a lower prothrombin index increases risk.
  - aids: a lower CD4 count increases risk.
  - An effect "aligns" if its centred φ̂ is monotone in the stated direction over the 5th–95th percentile range in at
    least 80% of the split × bootstrap fits.
- **Symbolic formulas** in original units for accepted edges, reported with validation R².

## 6. Round 11 acceptance tests (unit tests written before any Round-12/13 run)

Thresholds are fixed now. A STAT test failure at its pre-specified seed is recorded as FAIL and is never converted
by changing the seed, sample size or threshold.

| ID | Type | Criterion |
|---|---|---|
| CK-1 | DET | Breslow and Efron negative log partial likelihoods equal `lifelines` (Efron) and a direct Breslow formula, for fixed linear η with ties, within 1e-8 |
| CK-2 | DET | The Breslow baseline cumulative hazard equals `lifelines`' baseline for the same linear fit within 1e-6 |
| CK-3 | STAT | CoxKAN with linear edges recovers lifelines' Cox coefficients within 0.05 (n = 2,000, seed 11) |
| CK-4 | DET | The quadrature of the functional layer with φ(x) = x reproduces ∫X(t)dt for analytic curves within 1e-3 |
| CK-5 | STAT | B2-S (Cox) recovers S-Add's φ₁ and φ₂ with RISE < 0.10 each (n = 2,000, seed 12) |
| CK-6 | STAT | The `afcm` re-implementation recovers a known F(x, t) with RISE < 0.10 (n = 2,000, seed 13) |
| CK-7 | STAT | PACE on sparse curves (3–8 visits) recovers the first eigenfunction with \|cosine\| > 0.95 and reconstructs curves with lower mean ISE than per-subject B-spline smoothing (n = 1,000, seed 14) |
| CK-8 | STAT | Scale-aware pruning with validation-selected (λ, θ_rel) removes the null channel and keeps both active channels in S-Add (n = 2,000, seed 15) |
| CK-9 | DET | The stacked landmark builder uses no observation after t_L, includes only subjects at risk at t_L, and censors at t_L + w |
| CK-10 | DET | Symbolic stages 1–2 recover sin, tanh and quadratic edges with validation R² > 0.99 |
| CK-11 | STAT | The residual-calibrated KAN-AFT has D-calibration p > 0.05 on log-normal AFT data where the Weibull-loss AFT has p < 0.05 (n = 2,000, seed 16) |
| CK-12 | DET | The two-stage LME's BLUPs equal closed-form BLUPs for known variance components within 1e-8 |

Unit-test seeds 11–16 are training/data seeds for tests only (never used for Round 12/13 data).

## 7. Compute budget and order

- Round 12: about 5 scenarios × 20 replicates × about 12 models × 3 training seeds, single-threaded, 4 parallel
  lanes. Planning budget: 12 h wall clock.
- If the budget is exceeded, these cells are dropped first, in this order:
  - B2 quantile sensitivity;
  - S-Misspec log-logistic;
  - flat_ae;
  - sum_rsf.
- H-R12 cells (S-Add B2-S Cox and afcm) are never dropped.
- Round 13: 4 data sets × 5 splits; the bootstrap stability analysis (20 resamples) only for the selected B2 and B1
  models.

## 8. Known limitations accepted at registration

- AFCM and the joint-model baseline are Python re-implementations (no R), so differences from `mgcv`/`JM` are
  possible.
- Dynamic-DeepHit and JM-KAN are not implemented.
- n = 1,000 synthetic subjects; three channels only.
- The clinical-alignment directions are coarse (monotonicity).
- Clinical sign-off for PBC endpoints is external.
