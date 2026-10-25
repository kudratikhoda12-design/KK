<p class="small">Research prototype report, revision 17 · generated from saved records by
<code>scripts/build_report_round15.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; not publication-ready;
no novelty claim</b>. Part A is Round 15, the pre-registered Phase-2 methods round (D133–D136). Part B is revision 16's
Part A (Round 14: the manuscript draft); Part C is Round 13; Part D is Round 12; Parts E–F are the L39 and L38 parts;
Parts G–Q are the round-11 to round-1 reports.</p>

## 0. Round-15 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Methods (D133) | **symbolic_distill**: domain-guarded symbolic substitution with a functional-distillation refit. **b2s_anchor_cox** (primary) / **b2s_res_cox** (secondary): last-value edges plus a validation-gated B2-S term on the full curve (anchored) or on the residual curve. **afcm_ens** and Laplace (LAML) smoothing selection for AFCM. All new options are off by default; Rounds 1–14 are unchanged |
| Development (D133–D135) | Dry run on development seeds only, then two owner-requested checks: Laplace AFCM met the CK-6c criterion on only 1 of 3 seeds and the cross-validated gate did not improve dev_aids, so **fresh seed 24 was not used** and AFCM stays "not validated" |
| Formal run | {{V:n_records}} jobs, {{V:n_failed}} failed: S-Add 3500–3519, S-Sparse 3600–3619, S-Last 3700–3719, real data on the Round-13 splits |
| **H-R15a** (symbolic_distill) | **Not supported.** {{V:a_finite}} distillation refits finite (rule: all). S-Add distilled − pruned shape error {{V:a_diff}}, 95% CI {{V:a_ci}} (rule: upper bound below +0.05). Against the old unconstrained refit the distilled model was far better ({{V:a_old}}) |
| **H-R15b** (b2s_anchor_cox) | **Not supported.** Part 1 holds: S-Last anchored − B2-S Cox curve error {{V:b1_diff}}, CI {{V:b1_ci}}. Part 2 fails: S-Sparse {{V:b2_diff}}, CI {{V:b2_ci}} (rule: upper bound below +0.005). The residual design was worse on S-Sparse ({{V:res_sparse}}) |
| Real data | Exploratory; no Nadeau–Bengio interval excludes 0 |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say (all statements after the two verdicts are exploratory)

1. **Distillation solves the refit problem, but choosing μ by validation undermines it.**
   - At a fixed μ of 1 or 10 the distilled shapes equal the pruned shapes (R15.8).
   - The validation partial likelihood sometimes preferred a loose anchor, which cost shape accuracy.
2. **The one non-finite refit was numerical, not a domain failure.** A nearly degenerate formula made the first
   L-BFGS step overflow. A NaN-safe refit with a fallback to the substituted model would prevent it; the domain guard
   cannot.
3. **The anchored model is better than B2-S Cox when the landmark value carries the signal (S-Last)**, and matches it
   on S-Sparse whenever its gate keeps the trajectory channels. It loses only when the gate switches every channel
   off (4 of 20 S-Sparse replicates).
4. **Laplace selection for AFCM is not a fix either**: it improved the known failure seed (22) but broke a seed that
   validation selection handled well (966).

## R15.1 Owner decisions (D133, D134) and fact check

- **D133 (owner).** Open a prospective, pre-registered Phase-2 round for three methods aimed at the failure modes of
  Rounds 12–13; dry run on development seeds first; register exact criteria and fresh seeds; keep Rounds 1–14 intact;
  commit locally.
- **D134 (owner).** Families A and B approved, with μ ∈ {0.01, 0.1, 1, 10} (10 added because the dry run chose the grid
  edge), b2s_anchor_cox primary and b2s_res_cox secondary; real data exploratory on the Round-13 splits. AFCM option
  C2: develop Laplace/REML selection before spending seed 24. The cross-validated gate is checked on dev_aids first.
  The rules for both checks were fixed in D134 before they ran.
- **Fact check.** Seeds 24–26, 961–966 and the blocks 3500–3819 appeared in no code, test or record before use.
  GCV/Laplace selection did not exist in the code base before this round.

## R15.2 Methods (as implemented)

- **symbolic_distill.**
  - Restricted-domain library formulas (log, sqrt, 1/x) get a smooth floor on their argument; exp gets a smooth cap.
    On the training range this changes the formula by less than 5e-6 of the floor.
  - The refit adds μ × the variance-normalised squared distance between each refitted B2 surface (and static edge) and
    the pruned model's, on all training (subject, grid time) points.
  - μ is chosen by the unpenalised validation partial likelihood.
- **b2s_anchor_cox / b2s_res_cox.**
  - The existing B2-S network on transformed inputs: z → [last observed values, z]; the curves stay as they are
    (anchored) or become X(t) − X_last (residual).
  - Validation gate: the residual layer's channels are pruned as wholes over θ_rel ∈ D120's grid plus "all off". The
    D120 parsimonious rule (tolerance 0.01) picks the gate, and the chosen mask is applied to all three ensemble
    members.
- **AFCM variants.**
  - Bootstrap-averaged V3: 3 resamples, coefficients averaged.
  - Per-channel λ_t.
  - The D120 parsimonious λ rule.
  - Laplace approximate marginal likelihood (LAML) selection, with the Breslow-form observed information verified
    against an autograd Hessian (difference 2e-14).
- **S-Last** (new simulator scenario): the hazard depends on the curves' values at the landmark only.
- **Tests:** R15-1 to R15-8 (deterministic).

## R15.3 Dry run (development seeds 961–966, 914 and the known failure seed 22)

{{T:dev_tables}}

## R15.4 The two development checks of D134

{{T:dev_checks}}

- **C2 failed** (rule: the CK-6c criterion on all three seeds). Seed 24 was not used.
- **G failed** (rule: a lower dev_aids test IBS than the single-split gate's 0.1993). The CV gate was not added to the
  real-data lane.

## R15.5 Execution

{{T:exec_table}}

Four single-threaded lanes, one queue, 10:51–16:32 UTC on 2026-10-25. The registration was locked (D135) and the
harness smoke-tested on development data before launch.

## R15.6 H-R15a: symbolic_distill (S-Add and S-Sparse, 20 replicates each)

{{T:a_table}}

## R15.7 H-R15b: landmark-anchored functional CoxKAN (S-Last and S-Sparse, 20 replicates each)

{{T:b_table}}

## R15.8 Post-hoc diagnosis (exploratory; no verdict changes)

{{T:posthoc}}

## R15.9 Real data (exploratory; Round-13 splits 0–4)

{{T:real_table}}

When the gate keeps a channel, that channel's effect is shared between its last-value edge and its trajectory term,
so the edge-level monotonicity check above is not comparable with Round 13's total-effect alignment.

## R15.10 Limitations

- **The real-data analyses are not prospective.** The anchored design was motivated by Round 13 on the same data.
- **The dry run informed the primary model** (anchored rather than residual) before the owner approved it. This is
  disclosed; the formal seeds were fresh.
- **AFCM is still not validated**, and none of the four selection variants tried fixes its single-fit sensitivity.
- **Twenty replicates per scenario on one simulator.**

## R15.11 Decisions (D133–D137)

{{T:decisions}}

## R15.12 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R15.13 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–Q: historical records

Part B is revision 16's Part A (Round 14, the manuscript draft); Part C is Round 13; Part D is Round 12; Parts E and F
are the L39 and L38 parts; Parts G–Q are the round-11 to round-1 reports, each rendered from its own records. Embedded
living documents show their current version.
