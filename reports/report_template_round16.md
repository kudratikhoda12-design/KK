<p class="small">Research prototype report, revision 18 · generated from saved records by
<code>scripts/build_report_round16.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 16: the pre-registered
confirmatory round for the fixed-μ NaN-safe distillation and the non-cliff gate (D138, D139), followed by the updated
standalone manuscript (Rounds 1–16) and its verification. Part B is revision 17's Part A (Round 15); Part C is Round 14
(the first manuscript draft); Part D is Round 13; Part E is Round 12; Parts F–G are the L39 and L38 parts; Parts H–R are
the round-11 to round-1 reports.</p>

## 0. Round-16 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Changes tested (D138) | **symbolic_distill_fixed**: distillation refit at a fixed μ = 10, with a NaN-safe fallback to the substituted model and rejection of library formulas whose inner argument exceeds 50 in absolute value on the training range. **Non-cliff gate** for b2s_anchor_cox: "all trajectory channels off" only if strictly better on validation than every setting that keeps a channel. All new options are off by default; Rounds 1–15 are unchanged |
| Development checks | S-Add 3515 (the Round-15 failure) is finite under the Round-16 pipeline: the degenerate formula is rejected and no fallback was needed. Applied to the saved Round-15 gate grids, the non-cliff rule removes only {{V:dev_rescued}} of the {{V:dev_old}} S-Sparse "all off" decisions, so a repeat failure of H-R16b part 2 was anticipated before any Round-16 seed was used |
| Formal run | {{V:n_records}} jobs, {{V:n_failed}} failed: S-Add 3800–3819, S-Sparse 3900–3919, S-Last 4000–4019 |
| **H-R16a** (symbolic_distill_fixed) | **{{V:a_verdict}}.** {{V:a_finite}} distillation refits finite (rule: all; fallbacks used: {{V:a_fallback}}). S-Add distilled − pruned shape error {{V:a_diff}}, 95% CI {{V:a_ci}} (rule: upper bound below +0.05). Against the old unconstrained refit: {{V:a_old}} |
| **H-R16b** (b2s_anchor_cox, non-cliff gate) | **{{V:b_verdict}}.** Part 1: S-Last anchored − B2-S Cox curve error {{V:b1_diff}}, CI {{V:b1_ci}} (rule: CI below 0). Part 2: S-Sparse {{V:b2_diff}}, CI {{V:b2_ci}} (rule: upper bound below +0.005). Gate all off on S-Sparse: {{V:b2_alloff}} (D133 rule: {{V:b2_alloff_d133}}) |
| Manuscript | `docs/manuscript_round16.md` / `reports/manuscript_round16.pdf` ({{V:ms_pages}} pages), Rounds 1–16, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

1. **The symbolic refit is now safe, and nearly inert.**
   - The fixed anchor (μ = 10), the bound on formula arguments and the NaN-safe fallback kept every refit finite; the
     fallback was never needed.
   - The distilled model reproduced the pruned shapes on both scenarios (H-R16a supported).
   - It did not improve the survival-curve error, IBS, Uno C or validation loss over substitution without refit
     (R16.5), so substitution alone remains a sufficient symbolic summary.
2. **The anchored model's S-Sparse loss is not the gate cliff.**
   - The non-cliff rule worked as designed: on S-Last it kept channels more often, without changing accuracy.
   - On S-Sparse "all off" occurred in only 1 of 20 replicates under either rule, yet the anchored model was slightly
     but consistently worse than B2-S Cox (17 of 20; post hoc +0.0024 where channels were kept).
   - The registered prediction ("all off" would cause the repeat failure) was wrong about the mechanism.
3. **Anchoring is a trade-off.** In both rounds the last-value edges gained about 0.018 in curve error when risk is
   carried by the last value (S-Last) and cost about 0.003–0.005 when it runs through the trajectory (S-Sparse). Gating
   the last-value edges themselves was not tested.
4. **The manuscript** now covers Rounds 1–16 and reports both Phase-2 verdicts as registered; every number is traced
   to a record (R16.8).

# Part A: Round 16

## R16.1 Owner instruction (D138) and fact check

- **Instruction.** Open Round 16 with a fixed μ and a NaN-safe refit, include the non-cliff gate fix for
  b2s_anchor_cox, register on unused fresh seeds, check on development seeds first (including S-Add 3515), run, record
  D139, rebuild the report (revision 18) and update the standalone manuscript with full provenance verification;
  commit locally, do not push.
- **Fact check.** H-R15a had two registered parts, not three; H-R16a tests exactly those two (D138). The seed blocks
  3800–3819, 3900–3919 and 4000–4019 appeared in no code, test or record before use.

## R16.2 Methods (as implemented)

- **Degenerate-formula rejection.** `fit_symbolic(max_abs_arg=50)`: a library candidate c·f(a·x + b) + d is skipped if
  |a·x + b| exceeds 50 anywhere on the edge's training range; the next candidate (or PySR,
  or no formula) is used.
- **NaN-safe refit.** `refit_symbolic(safe=True)`: the objective (loss + μ × distillation term) is evaluated before the
  refit; if afterwards a parameter or the objective is non-finite, or the objective did not decrease, every parameter is
  restored to its pre-refit value and `refit_fallback` is recorded.
- **Fixed μ = 10**, one refit of 200 L-BFGS iterations per replicate (no validation selection of μ).
- **Non-cliff gate** (`choose_gate(rule="noncliff")`): the D120 parsimonious rule (tolerance 0.01) among the settings
  that keep at least one functional channel; "all off" only if its validation loss is strictly below the best of them.
- **Tests:** R16-1 to R16-3 (deterministic).

## R16.3 Development checks (before any Round-16 seed)

{{T:dev_checks}}

## R16.4 Execution

{{T:exec_table}}

Four single-threaded lanes, one queue. The harness was smoke-tested on development data before launch.

## R16.5 H-R16a: symbolic_distill_fixed (S-Add and S-Sparse, 20 replicates each)

{{T:a_table}}

## R16.6 H-R16b: b2s_anchor_cox with the non-cliff gate (S-Last and S-Sparse, 20 replicates each)

{{T:b_table}}

## R16.7 Limitations

- **One simulator, 20 replicates per scenario**; the Round-16 changes were not run on real data.
- **The revisions were motivated post hoc** by Round-15 failures (on other seeds) and by a development check on the
  saved Round-15 gate grids; the formal seeds were fresh and unused.
- **H-R16a is a safety result, not an accuracy gain**: at μ = 10 the refit barely moves the model.
- **AFCM is still not validated** (Round 15, D135); no AFCM model was run in Round 16.
- **The rounded summary figures in 0.2** (about 0.018; 0.003–0.005; 17 of 20; +0.0024) are read from the tables
  in R16.5, R16.6 and `reports/round16/posthoc_exploratory.json`.

# Part A (continued): the standalone manuscript, Rounds 1–16

## {{V:ms_title}} {#manuscript-title-r16}

{{MANUSCRIPT}}

## R16.8 Manuscript verification

{{T:ms_verification}}

## R16.9 Decisions (D138–D140)

{{T:decisions}}

## R16.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R16.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–R: historical records

Part B is revision 17's Part A (Round 15); Part C is Round 14 (the first manuscript draft); Part D is Round 13; Part E
is Round 12; Parts F and G are the L39 and L38 parts; Parts H–R are the round-11 to round-1 reports, each rendered
from its own records. Embedded living documents show their current version.
