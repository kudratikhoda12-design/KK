# Source-code and mathematical audit (2026-10-17)

Audit of the repository at commit `a918a56` (the previous final report), plus the revisions made since. Each
equation is mapped to its implementation and to the test that checks it **against an independent computation**.
Test IDs refer to `reports/test_results.csv`.

## 1. Equation → code → test map

| Component | Equation | File / function | Verifying tests (independent reference) | Status |
|---|---|---|---|---|
| AFT model | `log T = η + σε`, `w = (log y − η)/σ` | `src/kanfae/survival/aft.py`: `aft_log_likelihood` | A1 (scipy `weibull_min` / `lognorm` / `fisk`) | verified |
| Weibull | `log f = w − eʷ`, `log S = −eʷ` | `aft.py`: `Weibull` | A1, A2 (quadrature, autograd hazard), A6 | verified |
| Log-normal | `log f = −w²/2 − ½log 2π`, `log S = log Φ(−w)` | `aft.py`: `LogNormal` (`log_ndtr`) | A1, A2, A5 (stability, \|w\| ≤ 50) | verified |
| Log-logistic | `log f = w − 2 softplus(w)`, `log S = −softplus(w)` | `aft.py`: `LogLogistic` | A1, A2, A5 | verified |
| Event / censored contributions | `ℓ = δ[log f − log σ − log y] + (1 − δ) log S` | `aft_log_likelihood` (`torch.where`) | A3 (censored terms do not depend on the density), G6 | verified |
| Interval (integer-cycle) likelihood (L14) | event: `log[S(y − 1) − S(y)]`, censored: `log S(y)`, `S(t ≤ 0) = 1` | `aft.py`: `aft_interval_log_likelihood` (stable `log S(lo) + log(−expm1(Δ))`) | INT-1 (200-digit mpmath), INT-2 (width → 0 gives the density), INT-3 (gradcheck, extremes) | verified (new; not yet used in any experiment) |
| Gradients | – | autograd | A4 (`gradcheck`) | verified |
| MLE parameterisation | – | `linear_aft.fit_linear_aft` | A7 (lifelines maximised log-likelihood within 1e-4; parameter mapping), A8 | verified |
| Smoother | `c = (ΦᵀQΦ + λR + λ₀I)⁻¹ΦᵀQy` | `functional/smoothing.py`: `smooth`, `quadrature_weights` | F2, F3 | verified |
| Basis, R, G | B-splines; `R = ∫φ''φ''ᵀ`; `G = ∫φφᵀ` | `functional/basis.py` | F1 (scipy BSpline), F1b (Simpson) | verified |
| Linear functional encoder (A0; Wu et al. projection layer) | `h = b + Σ_d c_dᵀ J b_d` | `functional/layers.py`: `FunctionalLinear` | F5 (dense quadrature) | verified |
| Nonlinear functional layer (A1, D38) | `h = b + ΣΣ ∫β_dm(t) g_dm(x_d(t)) dt` | `layers.py`: `NonlinearFunctional` | F6a (identity map equals A0 within 4.4e-16), F6b, F6c | verified |
| Energy front (A2, D41) | `+ Σ ∫γ_d(t) x_d(t)² dt = Σ_l q_l cᵀT_l c` | `layers.py`: `EnergyFunctional` | Exactness against dense quadrature: max error 1.6e-14 (session check; see §4) | verified (ad hoc) |
| Tensor-spline front (A3, D41) | `+ Σ ∫F_d(x_d(t), t) dt`, F a tensor-product spline | `layers.py`: `TensorSplineFunctional` | Shapes/gradients through the M1-style search runs | partially verified |
| Decoder | `ĉ = g_dec(z)`; `x̂(t) = Σ ĉ_k φ_k(t)` | `models/fae.py`: `FunctionalAE.forward`, `reconstruct_at` | M1, M2 | verified |
| Reconstruction loss (D19) | Individual-weighted masked MSE | `models/pipelines.py`: `_ind_weighted_rec` | AUD-2 (NumPy reference) | verified |
| Joint objective | `NLL + α·L_rec + λ_β·Σ bᵀR_βb` | `pipelines.NeuralAFT._objective` | AUD-1 (independent recomputation, 4 model/front combinations) | verified |
| Training controls | S1 / S2 / S3 / α = 0 | `NeuralAFT.fit`, `_run` | AUD-3 (S1 freezes the autoencoder), AUD-4 (α = 0 gives no decoder gradient) | verified |
| Time scaling | Train on `y/τ`; `S(t) = S_ε((log(t/τ) − η)/σ)` | `NeuralAFT.predict_survival` | AUD-5 (S = 0.5 at the median) | verified |
| KAN layer | `φ(x) = w_b SiLU(x) + w_s Σ c_g B_g(x)` | `models/kan.py`: `KANLayer` | F-KAN-1..4, KREG-2 | verified |
| KAN regularisation (CoxKAN Eqs. 5–8; KAN-AFT §3.2.2) | `‖Φ‖₁ + λ_ent S(Φ) + λ_coef ‖C‖₁` | `KANLayer.regularization` | KREG-1 (definition and gradients) | verified (new) |
| Pruning (KAN-AFT θ = 1e-2; CoxKAN) | Edge L1 < θ; node `min(I, O) < θ` | `KANLayer.prune_edges`, `KANBlock.prune`, `KANAFT.prune` | KREG-3 | verified (new) |
| Symbolic substitution | Edge → formula | `KANLayer.symbolic`, `kan_aft._torch_formula` | KREG-4 (exact substitution leaves the output unchanged) | verified (new) |
| Symbolic fitting | Linear → library `c·f(a·x + b) + d` → PySR; accept if held-out R² > 0.99 | `symbolic/fit.py`: `fit_symbolic` | SYM-1..4 | verified (new) |
| BJ / IPCW / transform (D42) | KAN-AFT Appendix A (with documented deviations) | `survival/censored_ls.py` | LS-1 (reduce to OLS without censoring), LS-2 (KM vs lifelines; BJ conditional mean), LS-3, LS-4, LS-5 | verified (new) |
| IPCW Brier / IBS, Uno C, D-calibration | protocol §5 | `evaluation/metrics.py` | Q1–Q5 | verified |

## 2. Findings

| ID | Finding | Severity | Action |
|---|---|---|---|
| A-KAN-REG | No KAN-specific regularisation existed; only Adam's coupled L2 weight decay applied to the KAN coefficients. | Missing feature | Implemented (CoxKAN/KAN-AFT form); KREG-1 |
| A-PRUNE | No pruning existed. | Missing feature | Implemented; KREG-3 |
| A-SYM | **No symbolic extraction existed.** Nothing in the earlier report claimed any. | Missing feature | Implemented; SYM-1..4; validated on known truth (`reports/kanaft_symbolic.jsonl`) |
| A-WD | `torch.optim.Adam(weight_decay=…)` is coupled L2, not decoupled decay. | Design note | AdamW option added (`optimizer="adamw"`) and tested in the D41 search |
| A-ES | S2 early-stops on the validation *joint* objective, which reconstruction dominates when α = 1. | Possible cause of under-training of the survival part | `es_target="nll"` option added and tested in the D41 search |
| A-BN | In S1's head-only phase, KAN BatchNorm running statistics are refreshed (train mode) on the same training data. | Negligible | Documented |
| A-PROBE | R2's probe criterion requires `(a, b, A, ω)` to be **linearly** decodable from z (r = 4). With α > 0 the same 4 dimensions must also support reconstruction (phase, frequency, GP wiggles), and b enters the risk only through tanh(2b). The criterion is strict for structural reasons, not only because of learnability. | Interpretation | Kept unchanged (D11/D39); latent size r ∈ {4, 8} searched under D41 |
| A-TRANSFORM | KAN-AFT's transform estimator regresses `log T*`. T* is unbiased for E[T\|z] (LS-5: mean T* = 1.993 vs mean T = 1.991), but not for E[log T\|z]. | Mathematical limitation | Confirmed empirically: slopes 0.652 / −0.417 vs truth 0.8 / −0.5 (LS-4 diagnostic), against 0.803 / −0.50 for BJ and IPCW |
| A-IPCW | IPCW and the transform need CA-2 and positivity. The maximum IPCW weight was 74 in LS-4. | Variance risk | Documented; weights reported |
| A-UPLOAD | The uploaded reference PDFs were absent from the session. | External limitation | Public arXiv versions used, with checksums |
| A-CACHE | **Defect (found 2026-10-17, D53).** The D44 tensor-spline feature cache keyed on `(data_ptr, shape, version)` without holding the input, so a same-shape test tensor could reuse a freed validation tensor's address and receive its features. Found through replication seed 110 (test C 0.49 vs train/val 0.86/0.84). Training was unaffected; evaluation was not. | **Correctness defect** | Fixed; regression test F8b (fails on the old code, passes on the fix); every tensor-spline evaluation since D44 recomputed; superseded records kept in `reports/superseded_D53/` |
| A-PRUNE-OFFSET | The reference pruning score (mean \|φ\| per edge, KAN-AFT/CoxKAN) includes each edge's constant offset, which is not identifiable in an additive model. Unregularised, the true null edge z₄ scored 0.48–0.51, **above** the true sin edge (0.33). The centred score mean \|φ − mean φ\| separates the null edge (0.024–0.031 vs ≥ 0.31 for every true edge), but is still above θ = 1e-2 unless λ ≥ 0.1 (D54, `reports/pruning_threshold_diag.jsonl`). | Interpretation / scale dependence | θ = 1e-2 kept as the reported setting (references); the limitation is reported. Only 1 of 22 reported fits pruned an edge (synthetic seed 203, IPCW, λ = 1e-2: the true null z₄) |
| A-INT-REF | INT-1 originally failed because its float64 scipy reference `log(sf(y−1) − sf(y))` cancels (relative error up to 1.6e-7); the implementation matched 200-digit mpmath to ≤ 8e-14. | Test-reference defect | Reference replaced by mpmath (D48); tolerance unchanged |
| A-TEST-STALE | After D45 changed the model defaults, M1/M3 still assumed the old latent width (process gap: the unit suite was not re-run after D45). E-5 compared the lock with stale reproduction evidence after D43. | Test maintenance | Fixed in D56; full suite re-run |

## 3. Survival loss: parametric likelihood vs censoring-adjusted least squares

| Estimator | Target | Censoring assumption | Error-law assumption | Survival curves | Compatible with KAN-FAE joint training? |
|---|---|---|---|---|---|
| Parametric AFT likelihood (reference) | `η`, `σ` (location of log T for the chosen family) | CA-1 (`C ⊥ T \| X`), non-informative | Parametric family (Weibull, log-normal or log-logistic) | Closed form | **Yes:** a differentiable per-individual loss, summed with α·L_rec |
| Buckley–James | `E[log T \| z]` (up to a constant) | CA-1 | iid errors independent of z (homoscedastic); law unspecified | KM of residuals | Only as an outer iterative loop (impute, then refit). The imputation depends on the current η, so it is not a single fixed loss; costly with joint training |
| IPCW least squares | `E[log T \| z]` | **CA-2** (marginal KM of G) and positivity | None | KM of residuals | Yes (weighted MSE), but uncensored rows only; high variance under heavy censoring |
| Transform least squares | `E[log T* \| z]`, which is not `E[log T \| z]` | **CA-2** | None | KM of residuals | Yes (MSE on fixed targets), but biased on the log scale (finding A-TRANSFORM) |
| Cox partial likelihood (CoxKAN) | Log-partial hazard (proportional hazards) | CA-1 | Proportional hazards | Breslow baseline | **A different model.** Not inserted into the AFT model (owner instruction). It remains only as the FPCA + Cox and summary + Cox baselines |

Conclusion: the parametric likelihood remains the reference for the functional autoencoders, because its
assumptions hold by construction for the synthetic data and it yields full survival curves. The three least-squares
estimators are separate, validated variants in `KANAFT` and are compared in `reports/kanaft_symbolic.jsonl`.

## 4. Evidence locations

- `reports/logs/audit_tests_attempt1.log`
- `reports/logs/censored_ls_tests_attempt1.log`
- `reports/logs/kreg_sym_tests_attempt{1,2}.log`
- `reports/logs/rev3_unit_tests_attempt1.log`
- The energy-front exactness check (1.6e-14) was run interactively in this session. It is recorded here but has no
  test ID; it is repeated as test F7 in the final suite.
