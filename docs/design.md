# KAN-FAE: design document (revision 2)

Status: **revision 2, 2026-10-16.** This revision incorporates the owner's
corrections (`decisions.md` D15–D28). The owner asked for the document to be
corrected and saved, and authorised autonomous staged execution.

Nothing in this document is an experimental result. Results are reported only
in `reports/`. This project does **not** claim that the architecture is novel;
no literature review has been carried out (§8.3).

Every statement is labelled as one of:

- **[FACT]**: verified in this project, with a source ID from §8.5;
- **[ASSUMPTION]**: a modelling or design assumption we choose to make;
- **[HYPOTHESIS]**: a claim that the experiments test;
- **[OPEN]**: an unresolved question.

Stage names follow D15: Stage A (environment and likelihood), B (synthetic
data), C (functional representation and KAN layer), D (models and training),
E (evaluation and synthetic experiments), F (C-MAPSS), G (PBC).

---

## 1. Prediction task and notation

### 1.1 Landmark-based prediction (D3)

For individual *i* we observe a multivariate trajectory on a window that ends
at a landmark time `s`. We predict the **residual** time to event after `s`,
for individuals still at risk at `s`.

```
observation window      [s − w, s]       (w = s means "full history from 0")
observations            t_i1 < … < t_iL_i in the window;  L_i varies
values                  y_i ∈ ℝ^{L_i × D},  mask m_i ∈ {0,1}^{L_i × D}
true residual time      T*_i  (measured from s)   hidden whenever censored
censoring time          C_i   (measured from s)
observed                Y_i = min(T*_i, C_i),   δ_i = 1{T*_i ≤ C_i}
```

- Window times are rescaled affinely to `[0, 1]` before any functional
  processing.
- **[ASSUMPTION]** Predictions are conditional on being at risk at `s`.
- **[FACT, by construction]** Model inputs contain no observation dated after
  `s` (test K4). If whole trajectories up to the event or censoring time were
  used, trajectory length would encode the outcome.

### 1.2 Tensor conventions

B = batch, L = padded number of observations, D = channels, K = basis functions
per channel, K_β = basis functions per weight function, H = number of
functional features, r = latent size.

| Tensor | Shape | Notes |
|---|---|---|
| `times` | `(B, L)` | In `[0, 1]`; padded positions are masked |
| `values` | `(B, L, D)` | Standardised per channel with training statistics |
| `mask` | `(B, L, D)` | 1 = observed; padding is 0 |
| `y_obs`, `delta` | `(B,)` | `y_obs > 0` |
| `u` | `(B, p)` | Optional baseline covariates (PBC: age, sex; D26) |

## 2. Survival component: censoring-aware AFT likelihood

### 2.1 Model

```
log T = η + σ ε,     ε ~ F_ε (standardised, fixed family),   σ > 0
w = (log y − η) / σ
S(t | η, σ) = S_ε((log t − η) / σ)
```

A larger `η` means a longer expected survival, so the risk score is `−η`.
With a shared `σ`, survival curves of different individuals never cross.

### 2.2 Log-likelihood of one individual

```
ℓ_i = δ_i [ log f_ε(w_i) − log σ − log y_i ]  +  (1 − δ_i) log S_ε(w_i)
```

- The bracketed term is the log-density of `T` at `y_i`, including the Jacobian
  `−log σ − log y_i`.
- `−log y_i` does not depend on the parameters. It is kept so that values are
  comparable with independent references.
- The training loss is the mean of `−ℓ_i` over the batch.

### 2.3 Families

| Family of T | ε | log f_ε(w) | log S_ε(w) | Independent reference (scipy) |
|---|---|---|---|---|
| Weibull | standard Gumbel (minimum) | `w − exp(w)` | `−exp(w)` | `weibull_min(c=1/σ, scale=exp(η))` |
| Log-normal | N(0,1) | `−w²/2 − ½ log 2π` | `log Φ(−w)` (`torch.special.log_ndtr`) | `lognorm(s=σ, scale=exp(η))` |
| Log-logistic | standard logistic | `w − 2 softplus(w)` | `−softplus(w)` | `fisk(c=1/σ, scale=exp(η))` |

- Median of ε: `log(log 2)` (Gumbel-min), `0` (normal), `0` (logistic).
- The scipy mappings follow from `S(t) = S_ε((log t − η)/σ)`. They are an
  **[ASSUMPTION]** until test A1 passes.

### 2.4 Censoring assumptions, aligned across likelihood and metrics (D17)

Two different independence assumptions are in play, and they must not be
confused:

- **CA-1, conditional independent censoring** (used by the AFT likelihood):
  `T* ⊥ C | X`, where `X` is everything the model conditions on (the window
  trajectory and `u`). The censoring distribution has no parameters in common
  with `(η, σ)`. This is the weakest assumption under which maximising the
  likelihood in §2.2 is valid. **[ASSUMPTION]**
- **CA-2, completely independent ("random") censoring:** `C ⊥ (T*, X)`. This is
  strictly stronger than CA-1. It is what IPCW metrics assume when they use the
  ordinary (marginal) Kaplan–Meier estimate `Ĝ(t)` of the censoring survival
  function. This applies to the IBS and to Uno's C as implemented in
  scikit-survival. **[ASSUMPTION]**

**When the marginal Kaplan–Meier estimate of censoring is appropriate:** only
when CA-2 is plausible. That is, censoring does not depend on covariates.

| Setting | CA-2 plausible? | Consequence |
|---|---|---|
| Synthetic | Yes, by construction (§5.3) | Marginal-KM IPCW is used |
| C-MAPSS, artificial censoring | Yes, by construction | Marginal-KM IPCW is used |
| C-MAPSS, no censoring | Not applicable (all events) | All weights equal 1 |
| PBC, endpoint E2 | Doubtful. Censoring is mostly administrative (alive at the July 1986 analysis) and depends on calendar time of entry, which the data do not contain; it may correlate with covariates. | Primary: marginal KM, **marked as a limitation**. Sensitivity: conditional IPCW (below). |
| PBC, endpoint E1 | Implausible. Transplant is treated as censoring and plausibly depends on disease severity. | Same as above, with a stronger caveat. The CA-1 assumption of the likelihood is itself doubtful here. |

**Conditional IPCW (sensitivity analysis, feasible):**
- `G(t | X)` is estimated by a Cox model for the censoring time. The censoring
  indicator is `1 − δ`. Covariates are the baseline covariates and the last
  observed values of the channels at the landmark.
- It is fitted on the training split only.
- The Brier score at `t` then uses weights `1/G(Y_i⁻ | X_i)` (events) and
  `1/G(t | X_i)` (still at risk). Weights are truncated at the 95th percentile,
  and the truncation is reported.

This relaxes CA-2 to "CA-1 plus a correctly specified censoring model". It is
not assumption-free.

### 2.5 Further assumptions

- **A2 [ASSUMPTION]** The family of ε is correctly specified. The claim that
  misspecification hurts calibration more than ranking is **[OPEN]**.
- **A3 [ASSUMPTION]** AFT structure, with a shared shape when σ is constant.
  Weibull AFT with constant σ is also proportional-hazards.
- **A4 [ASSUMPTION]** The representation uses only pre-landmark data.
- **A5 [ASSUMPTION]** Individuals are independent. This is violated when one
  individual contributes several landmarks (C-MAPSS Option B), so analyses
  there are clustered by individual.
- **A6 [ASSUMPTION]** Continuous time with `Y > 0`. For C-MAPSS cycle counts we
  use the continuous approximation (D25). Residual life at a landmark `s < T` is
  always ≥ 1 cycle.
- **A7 [ASSUMPTION]** Predictions are conditional on being at risk at the
  landmark.

### 2.6 Numerical policy

- Weibull: no clamping of `exp(w)`. Behaviour is tested for `w ∈ [−50, 50]`
  only.
- Log-normal: `log_ndtr` is used for `log Φ(−w)`.
- Models parameterise `σ = softplus(ρ) + 1e-3`. The likelihood takes σ directly.

## 3. Functional representation (Stage C)

### 3.1 Basis and per-individual smoother (D20)

- Cubic B-splines `φ_1..φ_K` on `[0, 1]` with clamped, equally spaced knots.
- Roughness matrix `R_{k,k'} = ∫ φ_k'' φ_{k'}''`.
- Gram matrix `G_{k,k'} = ∫ φ_k φ_{k'}`.
- Both integrals use 64-point Gauss–Legendre quadrature per knot interval,
  which is exact for these polynomial integrands.

For each individual *i* and channel *d*, let `t_1 < … < t_n` be the observed
times (mask = 1). The quadrature weights `q_l` are the lengths of the cells
around each point (midpoints between neighbours; the first and last cells
extend to 0 and 1, so the weights sum to 1). The smoother is:

```
c_id = argmin_c  Σ_l q_l (y_l − φ(t_l)ᵀ c)²  +  λ cᵀ R c  +  λ₀ ‖c‖²
     = (Φᵀ Q Φ + λ R + λ₀ I)⁻¹ Φᵀ Q y,       Q = diag(q)
```

- The first term approximates `∫ (y − x)² dt` regardless of how many points
  there are or where they fall. This is what gives discretisation consistency
  (test F3). A plain sum over points would make the effective penalty shrink as
  the number of observations grows.
- `λ` is selected on training individuals only, by the error on held-out
  observation points.
- `λ₀ = 1e-6` keeps the system solvable when a channel has fewer points than
  basis functions.
- A channel with no observations gets `c = 0`, which is the training mean
  after standardisation. This is reported as an imputation.
- Shapes: `c: (B, D, K)`.

### 3.2 Functional linear layer

```
h_ij = b_j + Σ_d ∫ β_jd(t) x_id(t) dt = b_j + Σ_d c_idᵀ J b_jd
β_jd(t) = Σ_l b_jdl ψ_l(t),     J_{kl} = ∫ φ_k ψ_l dt
```

- `ψ` is a cubic B-spline basis with `K_β` functions.
- Shapes: `b: (H, D, K_β)`, `J: (K, K_β)`, `h: (B, H)`.
- Roughness penalty: `λ_β Σ_{j,d} b_jdᵀ R_β b_jd`.

**[FACT, mathematical]** With a fixed basis this is a linear map on `c`. The
"functional" properties come from the smoother, the penalties and the
continuous decoder. KAN vs MLP changes only the network after `h`.

### 3.3 Functional PCA baseline (D12)

```
coefficient vector  c_i = (c_i1, …, c_iD) ∈ ℝ^{DK}
inner product       ⟨x, x'⟩ = Σ_d c_dᵀ G c'_d
```

- With `M = I_D ⊗ G^{1/2}`, eigen-decompose the training covariance of `M c_i`.
- Scores: `score_i = V_rᵀ M (c_i − c̄)`.
- The mean, `M` and `V` are fitted on training individuals only.

### 3.4 Autoencoders and head

```
z  = g_enc(h)                          (B, r)
ĉ  = reshape(g_dec(z), (B, D, K))
x̂_id(t) = Σ_k ĉ_idk φ_k(t)             evaluable at any t
η  = f_η(z, u)                         (B,)    linear by default
σ  = softplus(ρ) + 1e-3                scalar
```

- **Functional MLP autoencoder:** `g_enc` and `g_dec` are MLPs with SiLU
  activations.
- **KAN functional autoencoder:** `g_enc` and `g_dec` are stacks of `KANLayer`,
  with parameter counts matched to the MLP within ±10% (D7). The head stays
  linear. A KAN head is an ablation.
- **Flattened autoencoder:**
  - each channel is resampled to a fixed grid of 20 points on `[0, 1]` by
    linear interpolation of its observed points (constant extrapolation; an
    empty channel becomes 0);
  - an MLP autoencoder is trained on the `D·20` vector;
  - the decoded grid is linearly interpolated back to the observed times for
    the reconstruction loss, so the loss is comparable with the functional
    models.

### 3.5 KAN layer (D1; implemented in Stage C)

```
KANLayer(n_in, n_out, G, k):
  out_q = Σ_p [ w^b_pq · SiLU(x_p) + w^s_pq · Σ_g a_pqg B_g(x_p) ]
```

- `B_g` are degree-`k` B-splines on a uniform grid of `G` intervals on
  `[−2, 2]`, extended by `k` knots on each side.
- Each KAN block starts with a `LayerNorm` without affine parameters, so inputs
  mostly fall inside the grid.
- Outside the extended grid the spline term is exactly 0 and only the SiLU path
  remains (test M4).
- Parameters per layer: `n_in·n_out·(G + k + 2)` (spline coefficients plus
  `w^b` and `w^s`) plus `n_out` biases.

The layer counts as **tested only after its own tests pass** (F-KAN-1..4 in
the protocol).

### 3.6 Genuine functional encoder vs. autoencoder on flattened trajectories

| | Functional encoder | Flattened autoencoder |
|---|---|---|
| Domain | Functions on `[0, 1]` | A vector of `D·20` resampled values |
| Irregular or missing sampling | Quadrature-weighted smoother | Linear interpolation onto the grid |
| Weights | Smooth `β(t)` with a roughness penalty | One free weight per grid index |
| Output | A function evaluable at any `t` | Grid values, interpolated |
| Testable property | Discretisation consistency (F3) | None |

## 4. Losses, weighting and training strategies

### 4.1 Reconstruction loss weights every individual equally (D19)

```
L_rec = (1/B) Σ_i  (1/|D_i|) Σ_{d ∈ D_i}  (1/n_id) Σ_l m_ild (y_ild − x̂_id(t_il))²
```

`D_i` is the set of channels with at least one observation for individual *i*.

- Every individual has total weight 1, and within an individual every observed
  channel has equal weight.
- The alternative, weighting every observation equally (pooling all observed
  points), was rejected as the default. Under that weighting, densely sampled
  individuals dominate. In PBC, visit frequency is documented to rise with
  worsening condition (§8.1), so observation weighting would over-represent
  sicker patients in the representation objective. That is a form of
  informative weighting.
- The cost of individual weighting is variance. An individual with 1–2 noisy
  points has as much influence as one with 40. For very sparse trajectories
  this can push the decoder toward noise. The smoother's penalties partly
  mitigate this.
- Reconstruction metrics report **both** weightings (§5 of the protocol), so
  the choice is visible.

### 4.2 Total loss

```
L = (1/B) Σ_i −ℓ_i  +  α L_rec  +  λ_β Σ bᵀ R_β b  +  weight decay
```

- Defaults: `α = 1`, `λ_β = 1e-3`, Adam with learning rate `3e-3`.
- Early stopping on the validation value of the stage's objective, with
  patience 30 and at most 400 epochs.

### 4.3 Training strategies and controls

| Name | What is trained | Purpose |
|---|---|---|
| S1, two-stage (reconstruction-only control) | Autoencoder on `L_rec` only, then a frozen `z` with the AFT head trained on the NLL | Unsupervised representation |
| S2, joint (default) | `NLL + α·L_rec` | Main model |
| S3, pretrain and fine-tune | S1, then S2 starting from the S1 weights | Strategy ablation |
| Survival-only control | S2 with `α = 0` | Is reconstruction needed? |

## 5. Synthetic generator (Stage B)

### 5.1 Latent parameters and trajectories

- Window `[0, 1]`, landmark `s = 1`. Residual times are measured from `s`.
- Latent parameters:
  - `a ~ N(0,1)`, `b ~ N(0,1)`
  - `A ~ U(0.5, 2)`, `ω ~ U(0.5, 2)`
  - `φ ~ U(0, 2π)`, a nuisance with no effect on risk
- Latent curves, defined on a 201-point grid and linearly interpolated:
  - `x₁(t) = a + b·t + g(t)`, with `g` a Gaussian process (squared-exponential
    kernel, length-scale 0.3, sd 0.2)
  - `x₂(t) = A·sin(2π ω t + φ)`
  - `x₃(t) = tanh(x₁(t)) + 0.5·x₂(t)`
  - D = 3 channels by default
- Observation process:
  - dense: `L_i ~ U{8,…,40}`; sparse: `L_i ~ U{3,…,8}`
  - times i.i.d. `U(0, 1)`, sorted, shared across channels
  - `y = x + N(0, 0.1²)`
  - channel-wise missingness probability `p_miss` (default 0)

### 5.2 Event times

- V1: `η = 2 + 1.5·tanh(2b) − 0.8·A² + 0.5·a·ω`
- V2: `η = 2 − 0.7·log(1 + ∫₀¹ x₁(t)² dt)`, integrated with the trapezoidal rule
  on the 201-point grid of the noise-free `x₁`.
- `T* = exp(η + σ_true·ε)`, with ε from the chosen family and `σ_true = 0.5`.
- Oracle survival curve: `S_true(t | θ) = S_ε((log t − η)/σ_true)`.

### 5.3 Censoring (independent of everything, so CA-2 holds)

- Target 0: `C = +∞`, so `δ = 1` for everyone.
- Target `p > 0`:
  - `τ_admin` is the 0.95 quantile of `T*` in a **pilot sample** of 20,000
    drawn from independent random streams;
  - `C = min(E, τ_admin)` with `E ~ Exp(rate μ)`;
  - `μ` is solved by bisection so that the pilot's censored fraction equals
    `p`;
  - `μ` and `τ_admin` are then held fixed.

### 5.4 Random streams and separation

- `numpy.random.SeedSequence(seed).spawn` gives independent streams for: latent
  parameters, GP, observation process, event noise, censoring and pilot.
- **Common random numbers:** for a fixed seed, `T*`, θ and trajectories do not
  depend on the censoring target.
- `generate(...)` returns `(Observed, Oracle)`. Training-facing entry points
  accept only `Observed` and raise `TypeError` for an `Oracle`.

## 6. Comparators (D8)

1. FPCA + AFT; FPCA + Cox (secondary reference).
2. Functional MLP autoencoder + AFT.
3. KAN functional autoencoder + AFT.
4. Flattened autoencoder + AFT.
5. Summary features (last value, mean, sd and OLS slope per channel, plus `u`)
   with: penalised AFT; penalised Cox; random survival forest.

## 7. Risks

### Numerical
- Weibull overflow beyond the tested range.
- `σ` collapsing toward 0.
- KAN inputs outside the grid.
- Ill-conditioned smoothing for very sparse channels.

### Statistical
- Small samples (PBC: 278 at risk at 730 days; C-MAPSS FD001: 100 engines).
- Tuning advantage across models.
- Leakage through trajectory length, preprocessing or overlapping windows.
- Censoring assumptions (§2.4).
- Selection by landmark.
- Informative visit process.
- Misspecification of the AFT family.
- **Latent non-identifiability**: KAN spline shapes on latent coordinates are
  not interpreted.
- Multiple comparisons.
- Chance failures of statistical tests (protocol §7).

## 8. Facts, assumptions, hypotheses and open questions

### 8.1 Verified facts (2026-10-16)

"Observed" means checked by us; "reported" means stated by a third-party
source.

**Environment** (observed; S1)
- Python 3.13; no GPU; 4 CPUs; 15 GB RAM; about 29 GB free disk.
- The global environment has `numpy 2.5.3` and `pandas 3.0.5`; no torch.

**Packages** (observed; S2, S3)
- PyPI versions and dates: `torch 2.14.1` (2026-09-30), `lifelines 0.30.3`
  (2026-03-05; declares `pandas<3.0,>=2.1`), `scikit-survival 0.28.0`
  (2026-07-05).
- `pykan 0.2.8` (2024-11-14; declares no dependencies) and `torchkan 0.1.1`
  are not used (D1).
- A dry run resolved the stack on Python 3.13 with `pandas 2.3.3`. The actual
  installation is verified in Stage A.

**PBC sequential data** (observed from CRAN `survival` 3.8-12, `data/pbc.rda`,
object `pbcseq`; licence LGPL (≥ 2); S4, S5)
- 312 patients, 1945 visit rows; visits per patient: min 1, median 5, max 16.
- Status: 143 censored, 29 transplant, 140 death. Follow-up: 41 to 5225 days.
- The column is `futime`, although the help page says `time`. The help page
  lists `trig` and `copper`, which are absent.
- Missing visit rows: `chol` 42.2%, `platelet` 3.8%, `alk.phos` 3.1%; 0% for
  `bili`, `albumin`, `ast` and `protime`.
- The help page says extra visits were often prompted by worsening condition,
  and missingness at the last visit before death or transplant is "potentially
  informative".

**C-MAPSS** (reported; S6–S8; verified in Stage F if the download succeeds)
- Subsets FD001–FD004. Training trajectories run to failure. Test trajectories
  are truncated, with the true remaining useful life supplied.

### 8.2 Assumptions

CA-1, CA-2 (§2.4), A2–A7, landmark-based prediction, B-spline representations,
individual-weighted reconstruction.

### 8.3 Hypotheses and open questions

- **H1 [HYPOTHESIS, primary]:** On the pre-registered synthetic setting, the
  KAN functional autoencoder + AFT has a lower integrated absolute
  survival-curve error than the parameter-matched functional MLP autoencoder +
  AFT. The null hypothesis of no difference is the default; no direction is
  assumed in interpretation.
- **H2 [HYPOTHESIS, secondary]:** The reconstruction term (`α > 0`) reduces
  survival-curve error compared with survival-only training (`α = 0`).
- **H3 [HYPOTHESIS, secondary]:** Functional encoders outperform the flattened
  autoencoder under sparse sampling.
- **[OPEN]** Sensitivity to misspecification of the AFT family.
- **[OPEN]** Whether PBC supports functional modelling.
- **[OPEN]** Which C-MAPSS landmark design is more appropriate (D3b).
- **[OPEN]** Which PBC endpoint is scientifically appropriate (D9).
- **[OPEN]** What prior work exists. **No novelty claim is made.**

### 8.4 References

Bibliographic details as stated by the sources (S5, S6–S8); not yet checked
against the publications themselves:
- Therneau T, Grambsch P. *Modeling Survival Data: Extending the Cox Model*.
  Springer, 2000.
- Murtaugh PA et al. *Hepatology* 20(1.1):126–34, 1994.
- Fleming T, Harrington D. *Counting Processes and Survival Analysis*.
  Wiley, 1991.
- Saxena A, Goebel K. "Turbofan Engine Degradation Simulation Data Set."
  NASA Ames Prognostics Data Repository, 2008.

Methods named but not bibliographically verified: Uno's C-index; Graf et al.'s
IPCW Brier score; D-calibration (Haider et al.); landmarking (van
Houwelingen); Kolmogorov–Arnold networks (Liu et al.).

### 8.5 Sources (accessed 2026-10-16)

| ID | Source |
|---|---|
| S1 | Commands run in this container (logged in `reports/logs/`) |
| S2 | PyPI JSON API, `https://pypi.org/pypi/<package>/json` |
| S3 | `pip install --dry-run --report`, with the CPU index `https://download.pytorch.org/whl/cpu` |
| S4 | `https://cran.r-project.org/src/contrib/survival_3.8-12.tar.gz` (`data/pbc.rda`) |
| S5 | `survival/man/pbcseq.Rd` in the same package |
| S6 | PHM Society mirror, `https://data.phmsociety.org/nasa/` |
| S7 | `https://ieee-dataport.org/documents/nasa-turbofan-jet-engine-data-set` |
| S8 | `https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data` |
