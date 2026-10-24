# Round 15 dry run (development seeds only) and the D133 registration proposal

Status: **dry run complete; awaiting the owner's approval before any formal seed is used.** Records:
`reports/round15/dev/*.jsonl`; summary `reports/round15/dev_summary.json`; code committed locally. Development seeds
961–966 (verified unused), 914 (the D123 AFCM development seed) and 22 (the CK-6c seed, a known failure case, used only
to see whether a variant changes it). No formal seed (24; 3500–3519; 3600–3619; 3700–3719) has been used. Every
dry-run number below is a single development replicate: it shows mechanics and direction, not evidence.

## 1. What the dry run found

1. **symbolic_distill works as intended.**
   - The unconstrained refit again destroyed the learned shapes (identifiable RISE 0.197 → 1.781 on S-Add, 0.503 → 5.184
     on S-Sparse), even though no restricted-domain formula was involved on these seeds.
   - The distillation refit kept them (0.200 and 0.502) and slightly improved the validation partial likelihood on
     S-Add (4.2151 vs 4.2205 after pruning; S-Sparse 4.2479 vs 4.2477), with IBS unchanged.
   - Validation chose μ = 1, the largest value in your grid, on both seeds; smaller μ let the shapes drift (μ = 0.01:
     0.593 and 0.675).
   - The domain guard was never triggered on these two seeds (no log, sqrt, 1/x or exp formula was accepted), so its
     safety property is shown only by the unit tests (R15-1: finite for any parameters; no change on the training range).
2. **b2s_res_cox as specified (residual curve X(t) − X_last) is misspecified when the whole trajectory matters.**
   - On S-Sparse (truth ∫sin(πX/2)dt) it was worse than plain B2-S Cox (curve error 0.0924 vs 0.0828): a nonlinear
     effect of X(t) cannot be written as an effect of X_last plus an effect of X(t) − X_last.
   - On S-Last (truth depends on X(1) only) the gate correctly switched every functional channel off, and the model
     beat B2-S Cox (0.1727 vs 0.1800), close to the static last-value CoxKAN (0.1679).
3. **The anchored variant fixes this.** Same last-value edges and the same validation gate, but the gated B2-S term acts
   on the full curve X(t) instead of the residual, so the model nests both B2-S Cox and the pure landmark model.
   - S-Sparse: 0.0810 (better than B2-S Cox 0.0828 and the residual design 0.0924).
   - S-Last: 0.1736 (gate off; like the residual design).
   - On the small, aids-shaped synthetic set the gate switched the functional part off and both gated variants lost to
     B2-S Cox (IBS 0.1985/0.1993 vs 0.1946): with about 80 validation subjects the gate is conservative.
4. **afcm_ens does not resolve the CK-6c sensitivity.**
   - On seed 22 (CK-6c's failure) every variant still had φ₁ ≈ 0.24: the 3-resample bag 0.239, per-channel λ_t 0.240
     (and φ₂ rose to 0.108), the D120 parsimonious λ rule 0.247. All bootstrap resamples chose λ_t = 0.01: validation
     genuinely prefers the time-varying fit on these data, so averaging over resamples cannot change it.
   - On seed 914 the variants were worse than the single V3 fit (V3 0.023; bag 0.056; parsimonious 0.092; per-channel
     λ_t 0.224). On seed 966 V3 0.055; bag 0.097.
   - V3 itself passed the CK-6c criterion on 2 of these 3 seeds, so a single fresh seed (24) is close to a coin toss
     whichever variant is used.

## 2. Dry-run tables
**symbolic_distill** (B2-S Cox, seed-0 member after pruning; identifiable RISE; validation NLPL is the unpenalised loss)

| Data | Model | Mean RISE | RISE φ₁ | RISE φ₂ | Curve error | IBS | Validation NLPL |
|---|---|---|---|---|---|---|---|
| S-Add 961 | after pruning | 0.197 | 0.292 | 0.103 | 0.0531 | 0.1923 | 4.2205 |
| S-Add 961 | sub_noguard | 0.197 | 0.292 | 0.103 | 0.0518 | 0.1924 | 4.2205 |
| S-Add 961 | refit_old | 1.781 | 2.360 | 1.201 | 0.0906 | 0.2081 | 4.2925 |
| S-Add 961 | sub_guard | 0.197 | 0.292 | 0.103 | 0.0518 | 0.1924 | 4.2205 |
| S-Add 961 | refit_guard | 1.781 | 2.360 | 1.201 | 0.0906 | 0.2081 | 4.2925 |
| S-Add 961 | distill_mu0.01 | 0.593 | 0.573 | 0.613 | 0.0717 | 0.1996 | 4.2463 |
| S-Add 961 | distill_mu0.1 | 0.239 | 0.310 | 0.168 | 0.0573 | 0.1934 | 4.2179 |
| S-Add 961 | distill_mu1 | 0.200 | 0.293 | 0.107 | 0.0524 | 0.1924 | 4.2151 |
| S-Add 961 | distill | 0.200 | 0.293 | 0.107 | 0.0524 | 0.1924 | 4.2151 |
| S-Sparse 962 | after pruning | 0.503 | 0.828 | 0.178 | 0.0862 | 0.1757 | 4.2477 |
| S-Sparse 962 | sub_noguard | 0.503 | 0.830 | 0.176 | 0.0861 | 0.1750 | 4.2477 |
| S-Sparse 962 | refit_old | 5.184 | 6.514 | 3.854 | 0.1126 | 0.1893 | 4.3814 |
| S-Sparse 962 | sub_guard | 0.503 | 0.830 | 0.176 | 0.0861 | 0.1750 | 4.2477 |
| S-Sparse 962 | refit_guard | 5.184 | 6.514 | 3.854 | 0.1126 | 0.1893 | 4.3814 |
| S-Sparse 962 | distill_mu0.01 | 0.675 | 1.007 | 0.343 | 0.0926 | 0.1805 | 4.2875 |
| S-Sparse 962 | distill_mu0.1 | 0.507 | 0.843 | 0.172 | 0.0868 | 0.1770 | 4.2520 |
| S-Sparse 962 | distill_mu1 | 0.502 | 0.828 | 0.177 | 0.0860 | 0.1755 | 4.2479 |
| S-Sparse 962 | distill | 0.502 | 0.828 | 0.177 | 0.0860 | 0.1755 | 4.2479 |

**b2s_res_cox and the anchored follow-up** (curve error only on simulated data)

| Data | Model | Curve error | IBS | Harrell C | Uno C | D-cal p | Functional channels kept by the gate |
|---|---|---|---|---|---|---|---|
| S-Last 963 | b2s_anchor_cox | 0.1736 | 0.1708 | 0.6377 | 0.6438 | 0.746 | [0, 0, 0] |
| S-Sparse 964 | b2s_anchor_cox | 0.0810 | 0.1679 | 0.6614 | 0.6652 | 0.426 | [1, 1, 1] |
| dev_aids 965 | b2s_anchor_cox | – | 0.1993 | 0.5950 | 0.5928 | 0.790 | [0, 0, 0] |
| S-Last 963 | b2s_cox | 0.1800 | 0.1731 | 0.6252 | 0.6359 | 0.263 |  |
| S-Last 963 | static_coxkan | 0.1679 | 0.1699 | 0.6394 | 0.6455 | 0.785 |  |
| S-Last 963 | landmark_cox | 0.1807 | 0.1728 | 0.6357 | 0.6395 | 0.719 |  |
| S-Last 963 | b2s_res_cox | 0.1727 | 0.1702 | 0.6403 | 0.6462 | 0.733 | [0, 0, 0] |
| S-Sparse 964 | b2s_cox | 0.0828 | 0.1684 | 0.6614 | 0.6645 | 0.128 |  |
| S-Sparse 964 | static_coxkan | 0.1131 | 0.1862 | 0.5773 | 0.5792 | 0.995 |  |
| S-Sparse 964 | landmark_cox | 0.1161 | 0.1895 | 0.5601 | 0.5641 | 0.918 |  |
| S-Sparse 964 | b2s_res_cox | 0.0924 | 0.1722 | 0.6352 | 0.6366 | 0.775 | [1, 1, 0] |
| dev_aids 965 | b2s_cox | – | 0.1946 | 0.6266 | 0.6353 | 0.638 |  |
| dev_aids 965 | static_coxkan | – | 0.1987 | 0.6022 | 0.5953 | 0.668 |  |
| dev_aids 965 | landmark_cox | – | 0.2055 | 0.5826 | 0.5883 | 0.375 |  |
| dev_aids 965 | b2s_res_cox | – | 0.1985 | 0.6055 | 0.5997 | 0.686 | [0, 0, 0] |

**afcm_ens and alternatives** (the CK-6c procedure, n 2,000; identifiable RISE; CK-6c criterion φ₁ < 0.15 and φ₂ < 0.10)

| Seed | AFCM variant | RISE φ₁ | RISE φ₂ | Selected λ_t |
|---|---|---|---|---|
| 22 | V3 (single fit) | 0.247 | 0.053 | 0.01 |
| 22 | V3, D120 parsimonious λ rule (tol 0.01) | 0.247 | 0.053 | 0.01 |
| 22 | V3, per-channel λ_t | 0.240 | 0.108 | [0.01, 0.0001, 1.0] |
| 22 | afcm_ens: 3-resample bag of V3 | 0.239 | 0.050 | [0.01, 0.01, 0.01] |
| 914 | V3 (single fit) | 0.023 | 0.053 | 1.0 |
| 914 | V3, D120 parsimonious λ rule (tol 0.01) | 0.092 | 0.056 | 10000.0 |
| 914 | V3, per-channel λ_t | 0.224 | 0.048 | [0.0001, 1.0, 0.01] |
| 914 | afcm_ens: 3-resample bag of V3 | 0.056 | 0.064 | [10000.0, 100.0, 10000.0] |
| 966 | V3 (single fit) | 0.055 | 0.056 | 10000.0 |
| 966 | V3, D120 parsimonious λ rule (tol 0.01) | 0.063 | 0.056 | 10000.0 |
| 966 | V3, per-channel λ_t | 0.065 | 0.056 | [10000.0, 10000.0, 0.0001] |
| 966 | afcm_ens: 3-resample bag of V3 | 0.097 | 0.055 | [10000.0, 0.01, 10000.0] |

## 3. D133: the registration I propose (needs your approval)

**Fresh seeds** (verified unused in every code file, test and record on 2026-10-24): CK-6d **24**; S-Add **3500–3519**;
S-Sparse **3600–3619** (3–8 visits); S-Last **3700–3719** (3–8 visits). Seeds 25 and 26 are also unused and are
not assigned. DGP v2, n 1,000 (CK-6d: n 2,000), split_groups 60/20/20 on δ, B-spline front end, the D112 ensembles,
as in Round 12. Every run is reported, failures included.

**Family A: symbolic_distill (as you specified).** Per S-Add and S-Sparse replicate: the Round-12 pruning procedure on
the B2-S Cox seed-0 model, stages 1–2 (PySR is not needed: its formulas are never substituted), then the
domain-guarded substitution and the distillation refit with μ ∈ {1e-2, 1e-1, 1} chosen by unpenalised validation NLPL.
Also recorded: the no-refit model and the old unconstrained refit.
- **H-R15a (confirmatory; both parts must hold):**
  1. all 40 distillation refits finite;
  2. on S-Add, distilled RISE minus RISE after pruning (mean identifiable RISE, paired over 20 replicates): the upper
     95% t-bound below +0.05 (non-inferiority; the H-R12 margin).
- Secondary (exploratory): the same on S-Sparse; distilled minus old refit; distilled minus no-refit in IBS, curve
  error and validation NLPL; how often μ = 1 (the grid edge) is chosen; how often the guard is active.

**Family B: landmark-anchored functional CoxKAN.** On S-Last and S-Sparse (the same replicates as family A): B2-S Cox,
static CoxKAN on last values + z, landmark Cox, the residual design (your specification) and the anchored variant.
- I recommend the **anchored variant as the primary** model of H-R15b, with the residual design as a registered
  secondary. If you prefer to keep the residual design as primary, the dry run predicts that H-R15b part 2 below
  fails.
- **H-R15b (confirmatory; both parts must hold):**
  1. on S-Last, primary minus B2-S Cox in survival-curve error: the 95% CI entirely below 0;
  2. on S-Sparse, the same difference: the upper 95% bound below +0.005 (non-inferiority; the P0 margin).
- Secondary: the same contrasts against static CoxKAN and landmark Cox; the residual design; the gate's decisions,
  including whether it removes the null channel.
- **Real data (exploratory only, not prospective).** PBC E2/E1, prothro and aids on the Round-13 splits 0–4 (so the
  saved Round-13 records of B2-S Cox, static CoxKAN and landmark Cox are the paired comparators, after a
  deterministic reproduction check on one split); Nadeau–Bengio intervals; the last-value edges checked against the
  pre-stated clinical directions. The design was motivated by Round 13 on these same data, so nothing here can be
  confirmatory.

**Family C: AFCM.** The dry run predicts that afcm_ens does not fix the single-fit sensitivity. I need your choice:
- **C1 (as you specified):** CK-6d on seed 24 with afcm_ens: the CK-6c procedure and criterion (identifiable RISE
  φ₁ < 0.15 and φ₂ < 0.10). If it passes, **H-R15c** (conditional confirmatory) re-runs the H-R12 comparison against
  afcm_ens on S-Add 3500–3519 with the H-R12 rule; if it fails, AFCM stays "not validated".
- **C2 (my recommendation):** do not spend seed 24 now. Implement REML/Laplace-type smoothing selection (the method
  of `mgcv`, which the original AFCM uses) on development seeds first. The dry run shows that validation-NLPL
  selection itself, not variance, causes the failure.
- **C3:** replace the single-seed acceptance check by a multi-seed one (for example, at least 4 of 5 fresh seeds
  pass), because a single fit passes or fails almost by chance (V3 passed 2 of 3 development seeds).

**Budget (estimated from the dry-run timings).** About 25 lane-hours (4 parallel lanes, about 7 h wall clock in this
container).

## 4. Further ways to improve the model (my suggestions, not implemented)

1. **Shrink B2-S towards a time-constant effect.**
   - In Round 12 the time-homogeneous B2-H had the lowest shape error (0.123 vs 0.179 for B2-S).
   - A roughness penalty on the time profiles h(t), chosen by validation, would let B2-S fall back to B2-H when the
     data do not support time variation.
   - This is the most promising change for shape recovery.
2. **Monotone edges in the pre-stated clinical directions.** These would remove sign flips in small cohorts (the
   prothrombin-index misalignment) at little cost; the directions are already registered.
3. **Bootstrap bagging of the KAN ensembles** instead of seed-only ensembles. CK-5b passed only through ensembling, so
   resample variance matters.
4. **A gate that is less conservative on small data:** a cross-validated gate instead of one validation split
   (56–80 subjects on the real data).
5. **REML/Laplace smoothing selection for AFCM** (option C2).

These are not part of D133; each would need its own development run and registration.
