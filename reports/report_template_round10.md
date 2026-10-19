<p class="small">Research prototype report, revision 10 · generated from saved records by
<code>scripts/build_report_round10.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A is round 10 (owner decisions D110): the literature review, the design
note and the pre-registration of Rounds 11–13. No model was fitted and no data were acquired in this round. Parts B–J
are the round-9 to round-1 reports, each regenerated from its own round's records.</p>

## 0. Round-10 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Literature review (targeted, documented) | Done: 14 search strings; 26 table entries, each verified individually (R10.2). Not a systematic review |
| Gap analysis | **Single-layer functional CoxKAN (B2) has the same model form as the additive functional Cox model of Cui, Crainiceanu & Leroux (2021)**; B1 composes published parts; no KAN survival model with functional covariates was found, but a KAN joint model with landmarks exists as a conference abstract (R10.3) |
| Claim policy | **No novelty claim.** The paper contributes a pre-registered comparison with AFCM and other functional methods, an interpretability evaluation, the KAN-FAE negative result and a reproducible benchmark (D111) |
| Design note | Written: losses, KAN machinery, B1/B2/B3, baselines, simulation DGPs, metrics, analysis plan, real-data plan, acceptance tests (R10.4) |
| Pre-registration | Round 11 (D112: implementation and 12 acceptance tests), Round 12 (D113: 5 scenarios × 20 fresh-seed replicates; single confirmatory test H-R12), Round 13 (D114: PBC, prothro, aids, C-MAPSS; interpretability measures fixed). Committed (`a7ff68b`) before any Round-11 code |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (the 4f gate's original R1, not gating), {{V:xfailed}} expected failures, {{V:skipped}} skipped. No model code changed in this round |

### 0.2 What the review changed in the plan

1. **The direct competitor is AFCM, not a strawman.**
   - Cui et al.'s model already uses ∫F(X(s), s)ds with penalised tensor-product splines under the Cox loss.
   - A single-layer functional CoxKAN is that model with a different parametrisation and training procedure.
   - The primary test therefore compares shape recovery against AFCM (H-R12), not against a linear model.
2. **Every published KAN survival model uses static covariates.** CoxKAN, KAN-AFT, SurvKAN and KAPLAN all do. Their
   "time-varying" features concern effects that change with time, not covariates measured over time.
3. **Two pieces of closely related work could only be checked at abstract level.**
   - JM-KAN (a KAN joint model with yearly landmarks; conference abstract).
   - F-KAN (FPC scores → KAN for a scalar response).
   - Their full texts are needed before any "first" wording.
4. **No R in this environment.** AFCM and the joint-model-style baseline will be Python re-implementations with their
   own acceptance tests (CK-6, CK-12), disclosed as deviations from `mgcv`/`JM`.

### 0.3 Roadmap (Rounds 11–14)

| Round | Content | Gate to the next round |
|---|---|---|
| 11 | Implement CoxKAN (exact ties), PACE, B1, B2, B3 builder, AFCM, two-stage LME, residual-calibrated KAN-AFT; run CK-1..CK-12 | All acceptance tests run; any failure reported to the owner before Round 12 |
| 12 | Simulation: S-Add, S-Int (+ P0 continuity), S-Sparse, S-Misspec; N = 20; H-R12 | Pre-registered analysis only |
| 13 | Real data: PBC (single landmark + B3), prothro, aids, C-MAPSS; stability, effect plots, clinical alignment, symbolic formulas | – |
| 14 | Paper draft: framing (a) + (b); no novelty claim | Owner review |

## R10.1 Owner decisions and fact check (D110)

- **Framing.** Functional CoxKAN / KAN-AFT (B1, B2, B3) is the main study; KAN-FAE (Rounds 1–9) is the motivating
  architectural section.
- **Scope.** Full scope as listed in D110: baselines, the four simulation scenarios with N = 20, the real data sets,
  and measured interpretability.
- **Fact check.** Every cited work exists as named. The AFCM model form is as stated. aids and prothro are in JM 1.5-2
  (prothro's documentation is internally inconsistent: 488 vs 467 patients). CoxKAN's own loss approximates ties, so
  our exact Efron/Breslow implementation is a disclosed deviation. R is unavailable, so AFCM and the joint-model-style
  baseline will be re-implemented.

## R10.2 Literature review

{{F:docs/literature_review_round10.md}}

## R10.3 Gap analysis and claim policy

The gap table (review §5) and the claim policy (D111) are binding for Rounds 11–14.
- **Contributions** will be stated as empirical, pre-registered findings:
  - shape recovery, selection and symbolic interpretability against AFCM and the other baselines;
  - the KAN-FAE negative result;
  - a reproducible benchmark.
- **Not claimed:** a new model class.

## R10.4 Design note and pre-registration of Rounds 11–13

{{F:docs/design_note_round10.md}}

## R10.5 Decisions in round 10 (D110–D115)

{{T:decisions_r10}}

## R10.6 Tests

{{T:suite_table}}

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record)

{{T:test_runs}}

## R10.7 Task ledger

{{F:docs/task_ledger.md}}

## R10.8 Reproducibility

{{F:docs/reproducibility.md}}

# Parts B–J: historical records

Part B (round 9), Part C (round 8), Part D (round 7), Part E (round 6), Part F (round 5), Part G (round 4), Part H
(round 3), Part I (round 2) and Part J (round 1) are regenerated from the same records. Each part renders from the test
records up to its own round's final suite run. Their numbers are unchanged history. Embedded living documents (audit,
ledger, decision rows, reproducibility) show their current version.
