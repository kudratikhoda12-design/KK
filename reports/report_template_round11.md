<p class="small">Research prototype report, revision 11 · generated from saved records by
<code>scripts/build_report_round11.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A is round 11 (implementation and acceptance tests CK-1..CK-12, D112):
code was written and tested; no Round-12 or Round-13 seed was used and no real data were acquired. Parts B–K are the
round-10 to round-1 reports, each regenerated from its own round's records.</p>

## 0. Round-11 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Implementation (D116) | Done: exact Efron/Breslow Cox loss and Breslow baseline; Round-12 simulator; PACE; functional KAN (B2-H, B2-S, B2-deep) and static CoxKAN/KAN-AFT heads for B1; scale-aware pruning; symbolic extraction and substitution; residual-calibrated KAN-AFT; AFCM, two-stage LME + Cox, landmark Cox and B1 feature builders; B3 stacked-landmark builder |
| Acceptance tests (D117) | **7 of 12 criteria pass** (CK-1, 2, 4, 7, 9, 10, 12). **5 fail and are recorded as FAIL, not converted:** CK-3, CK-5, CK-6, CK-8, CK-11 |
| Main finding (D118) | **The registered S-Add design leaves part of each effect unidentified.** ∫X_d(t)dt is the same for every subject, so any part of φ that is linear in x does not change any subject's risk. The registered RISE still counts that part. This affects CK-5, CK-6 and the confirmatory test H-R12 |
| Round 12 | **Not started.** Per D112 the failures go to the owner first. Decisions needed: §0.3 |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (the five CK failures and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 The acceptance tests in one table

{{T:ck_table}}

### 0.3 Decisions needed from the owner before Round 12

1. **S-Add identifiability (affects H-R12).** Options, none taken:
   - (a) keep the registration and report H-R12 with the caveat;
   - (b) re-register RISE as error modulo the unidentified subspace;
   - (c) re-register the DGP so that ∫X dt varies across subjects (for example, a random level per channel), and
     re-check identifiability before any Round-12 seed is used.
2. **CK-3.** The early-stopped linear-edge CoxKAN shrinks its slopes (max error 0.084 > 0.05). Decide whether the
   Cox-machinery check should compare at convergence (no early stopping), or whether this shrinkage is accepted as
   the training protocol's property.
3. **CK-8.** With θ_rel ≤ 0.10 nothing was pruned from the null channel. Decide whether the θ_rel grid, the
   functional edge score, or the expectation of the test should change.
4. **CK-11.** The residual-calibrated model is calibrated (p 0.471). The Weibull model was only borderline
   miscalibrated (p 0.0515), so the contrast the test needs did not occur at this seed and n.

## R11.1 Owner decision and scope

- **Owner instruction:** "start round 11." Scope fixed by D112: the design note §§1–3 (models and baselines) and §6
  (acceptance tests with registered thresholds and seeds 11–16).
- **Rules followed:**
  - thresholds unchanged;
  - a STAT failure at its registered seed is recorded as FAIL;
  - Round-12 seeds 3000–3419 unused;
  - development checks on seeds 901–907 and 99 only, with the exceptions disclosed in D116.

## R11.2 What was implemented

| Module | Content |
|---|---|
| `kanfae.survival.cox` | Exact Breslow and Efron negative log partial likelihood (vectorised, differentiable, optional strata); Breslow baseline; S(t \| x) |
| `kanfae.data.fsim` | Round-12 simulator: S-Add, S-Int, S-Misspec (log-normal, log-logistic); dense or sparse (3–8 visits) curves; censoring calibrated to about 30% |
| `kanfae.functional.pace` | PACE (local linear mean and covariance, leave-one-subject-out bandwidths, σ² by a local quadratic perpendicular to the diagonal, conditional-expectation scores) |
| `kanfae.models.fkan` | B2-H, B2-S (rank 2), B2-deep; static heads (B1, static CoxKAN / KAN-AFT); Cox or AFT loss; residual calibration; pruning grid; 3-seed ensemble with lr × weight-decay tuning; symbolic stages 1–3 per edge in original units, substitution and refit |
| `kanfae.models.afcm` | Additive functional Cox model (10 × 10 tensor cubic B-splines, second-order difference penalties, 5 × 5 validation grid) |
| `kanfae.models.lme` | Random-intercept-and-slope LME by EM, BLUPs, two-stage LME + Cox |
| `kanfae.models.fbaselines` | B1-SUM summaries, last values, dense grid FPCA, linear Cox for fpca_cox / sum_cox / landmark_cox |
| `kanfae.data.landmark` | B3 stacked landmark data set (history window, at-risk set, administrative censoring at the horizon) |

The deviations from the reference software and the operational choices are listed in D116 (R11.6).

## R11.3 Acceptance-test details

The registered criteria are in the design note §6 (Part B, R10.4). The operational details (splits, curve smoothing,
where RISE is evaluated) were fixed in `tests/test_round11.py` before the formal runs. Formal run: `round11_acceptance`
(logs in `reports/logs/round11/`); the final suite reproduced every outcome.

{{T:ck_details}}

## R11.4 Why the failing tests failed

**CK-5 and CK-6 (shape recovery): an identifiability defect in the design, plus estimation error.**
- In S-Add, X_d(t) = 0.5·d·t + Σ_k ξ_dk ψ_k(t), and every Fourier ψ_k integrates to 0 over [0, 1]. So ∫X_d(t)dt is
  the same number for every subject.
- Consequence: adding c·x to φ_d (homogeneous case) adds the same constant to every subject's η. The Cox likelihood
  cannot see that. In the bivariate case the same holds for x·c(t) whenever c(t) lies outside the span of the curves'
  variation.
- The registered RISE removes per-t constants but not these linear-in-x parts, so it penalises a direction that the
  data cannot determine.

{{T:ident_table}}

**Exploratory check (not a registered test; determinism check: the registered RISE values were reproduced exactly).**
"RISE-aff" removes the best per-t affine function of x from the error. That removes more than the unidentified part,
so it is optimistic.

{{T:explore_table}}

- For φ₂, almost all of the registered RISE is the linear-in-x part.
- For φ₁, it is not: B2-S still has RISE-aff 0.139 and AFCM 0.538.
- The two rows come from different seeds (12 and 13), so they are **not** a comparison between the methods.

**CK-3.**
- The slopes stopped at epoch 294, when validation patience ran out, short of the partial-likelihood maximum.
- Early stopping acts as shrinkage: all three slopes are smaller in absolute value than lifelines'.
- On development seeds 905 and 907 the same training loop ran until patience as well and came within 0.0003 of
  lifelines. So the outcome depends on the data seed.

**CK-8.**
- At λ ≤ 1e-3 no edge scored below θ_rel 0.10, the largest value in the registered grid, so nothing was pruned.
- At λ 1e-2, three edges were pruned, but every channel kept at least one edge.
- Selection picked λ 1e-4 with nothing pruned (validation losses tied across θ; the tie rule takes the largest θ).

**CK-11.** The test needs both "residual-calibrated p > 0.05" and "Weibull p < 0.05". The first held (0.471); the
second did not (0.0515).

## R11.5 Before Round 12

- **No Round-12 seed was used and nothing was run on real data.**
- The decisions listed in §0.3 are the owner's.
- Until then, the code is ready for Round 12 except for these open items, and the identifiability finding (D118)
  should be resolved before H-R12 is computed.

## R11.6 Decisions in round 11 (D116–D119)

{{T:decisions_r11}}

## R11.7 Tests

{{T:suite_table}}

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record; the
development-session records, including a test-construction error and the accidental early CK-11 run, are in
`reports/logs/round11/dev_session_test_records.jsonl`)

{{T:test_runs}}

## R11.8 Task ledger

{{F:docs/task_ledger.md}}

## R11.9 Reproducibility

{{F:docs/reproducibility.md}}

# Parts B–K: historical records

Part B (round 10), Part C (round 9), Part D (round 8), Part E (round 7), Part F (round 6), Part G (round 5), Part H
(round 4), Part I (round 3), Part J (round 2) and Part K (round 1) are regenerated from the same records. Each part
renders from the test records up to its own round's final suite run. Their numbers are unchanged history. Embedded
living documents (audit, ledger, decision rows, reproducibility) show their current version.
