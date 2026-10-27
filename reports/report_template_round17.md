<p class="small">Research prototype report, revision 19 · generated from saved records by
<code>scripts/build_report_round17.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 17: two-way gated landmark
anchoring for Cox and log-normal KAN-AFT models and fixed-μ symbolic distillation for KAN-AFT (D141–D143), followed by
the updated standalone manuscript (Rounds 1–17) and its verification. Part B is revision 18's Part A (Round 16 and the
Round-16 manuscript); Part C is Round 15; Part D is Round 14 (the first manuscript draft); Part E is Round 13; Part F
is Round 12; Parts G–H are the L39 and L38 parts; Parts I–S are the round-11 to round-1 reports.</p>

## 0. Round-17 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Changes tested (D141) | **Two-way gate** (`gate_last_value=True`, off by default): the last-value edges are pruned as well as the trajectory channels, and the plain B2-S comparator is itself a candidate, so the anchored model reduces exactly to `b2s_cox` / `b2s_aft` unless anchoring beats it on validation by more than `last_tol`. **Log-normal KAN-AFT**: comparator and anchored model; the AFT survival prediction now uses the fitted family (it used the Weibull function for every family). **AFT symbolic_distill_fixed**: μ = 10, max_abs_arg 50, NaN-safe |
| Development check | 49 jobs on already-used Round-16 seeds and development seeds 961–969. The b2s_cox re-fit reproduced Round 16 exactly. The pre-fixed rule chose last_tol {{V:lt_cox}} (Cox) and {{V:lt_aft}} (AFT, the grid edge); go/no-go: GO |
| Formal run | {{V:n_records}} jobs, {{V:n_failed}} failed: S-Last 4100–4119, S-Sparse 4200–4219, S-Misspec-lognormal 4300–4319 |
| **H-R17a** (two-way gated b2s_anchor_cox) | **{{V:a_verdict}}.** Part 1: S-Last two-way − B2-S Cox curve error 95% CI {{V:a1_ci}} (rule: upper bound below 0). Part 2: S-Sparse CI {{V:a2_ci}} (rule: upper bound below +0.005) |
| **H-R17b** (log-normal KAN-AFT) | **{{V:b_verdict}}.** Part 1: {{V:b1_finite}} AFT distillation refits finite (rule: all); S-Misspec distilled − pruned RISE upper bound {{V:b1_up}} (rule: below +0.05). Part 2: S-Misspec two-way − B2-S AFT CI {{V:b2m_ci}} (rule: upper below +0.005); S-Last CI {{V:b2l_ci}} (rule: upper below 0) |
| Manuscript | `docs/manuscript_round17.md` / `reports/manuscript_round17.pdf` ({{V:ms_pages}} pages), Rounds 1–17, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

1. **The two-way gate fixed nothing on fresh S-Sparse seeds.** On the Round-16 seeds it was tuned on, it removed most
   of the penalty. On fresh seeds it fell back to B2-S Cox in about half of the S-Sparse replicates but anchored in the
   rest, mostly for the worse. The upper bound missed the margin by the same 0.0003 as in Round 16. The S-Last gain
   was the largest yet.
2. **Why it probably failed (an interpretation, not tested).** The anchored side's validation loss is the best of up to
   16 fine-tuned candidates, the comparator's that of one model, so the decision to anchor is biased toward anchoring.
   An independent confirmation split, or equally selected candidates, would be the next fix to test.
3. **The log-normal KAN-AFT met every registered criterion (H-R17b).** The anchored distillation refit was finite in all
   20 S-Misspec replicates and kept the shapes; the two-way AFT gate did not lose on S-Misspec and gained on S-Last.
   Its development-tuned tolerance (the grid edge) made it cautious: it kept about a third of the possible S-Last gain
   and stayed worse than landmark Cox there.
4. **Implementation fix.** Log-normal (and log-logistic) KAN-AFT survival predictions used the Weibull formula; no
   earlier round was affected because none used those families.
5. **The manuscript** now covers Rounds 1–17 and reports both verdicts as registered (R17.8).

# Part A: Round 17

## R17.1 Owner instruction (D141) and fact check

- **Instruction.** Implement two-way gating of the last-value edges for b2s_anchor_cox and a log-normal KAN-AFT
  counterpart; tune last_tol only on already-used Round-15/16 records and development seeds 961–969; verify the fresh
  seed blocks; pre-register (D141), run (D142), deliver (D143); commit locally, do not push.
- **Fact check** (`docs/round17_devplan.md` §1, D141): the Round-16 figures quoted (+0.0053 against +0.0050; +0.0024
  in the replicates that kept a channel) are correct. The existing B2-S KAN-AFT was Weibull, and the AFT survival
  prediction used the Weibull function for every family (fixed; no earlier record affected). `_rc` means residual
  calibration in this project, not recurrent-event weighting, so no `b2s_anchor_aft_rc` was registered. S-Last is a
  Weibull proportional-hazards scenario, so the log-normal AFT is misspecified there for both compared models. The
  seed blocks 4100–4119, 4200–4219, 4300–4319 and development seeds 967–969 were unused.
- **Execution.** A container restart interrupted 4 formal jobs, which had written nothing and were re-run from the start
  on the same seeds (D142).

## R17.2 Methods (as implemented)

- **Two-way gate** (`ResFKAN(residual=False, gate_last_value=True, last_tol, loss)`; `choose_twoway`). Candidates on the
  tuned seed-0 anchored member, each fine-tuned 200 epochs with its mask (identical masks fitted once): *joint*
  (θ_rel in D120's grid: trajectory channels by integrated score, last-value edges by their normalised score),
  *static* (trajectory off, last-value edges at θ_rel), and *pure trajectory* (the comparator ensemble itself). Rule:
  D120 parsimonious among joint states (tolerance 0.01); a static state only if strictly better (non-cliff); the
  anchored state only if its validation loss is below the comparator's seed-0 validation loss minus last_tol,
  otherwise the comparator's predictions are returned unchanged.
- **Log-normal KAN-AFT**: `FKANSurv(loss="aft", family="lognormal")`; `predict_survival` now uses the family's survival
  function (the Weibull path is bit-identical to before).
- **AFT symbolic_distill_fixed**: the Round-16 pipeline with the log-normal AFT loss; shapes on S-Misspec are compared
  on the η scale of the data-generating model (as Round 12).
- **Tests**: R17-1 to R17-4 (methods), R17-5 to R17-9 (manuscript).

## R17.3 Development check (before any Round-17 seed)

{{T:dev_checks}}

## R17.4 Execution

{{T:exec_table}}

Four single-threaded lanes, one queue. The registration was locked in D141 before the lanes started.

## R17.5 H-R17a and H-R17b part 2: two-way gated anchoring (20 replicates per scenario)

{{T:a_table}}

## R17.6 H-R17b part 1: symbolic distillation for the log-normal KAN-AFT (S-Misspec, 20 replicates)

{{T:b_table}}

## R17.7 Limitations

- **last_tol was tuned on few, already-used replicates** and the Cox value did not transfer to fresh seeds; the AFT
  value lies at the edge of its grid.
- **The validation comparison between the comparator and the anchored states is not like for like** (one model against
  the best of several); this is the most likely reason for H-R17a's failure, but it was not tested.
- **One simulator, 20 replicates per scenario; no real data in Round 17.**
- **AFCM is still not validated** (Round 15, D135).

# Part A (continued): the standalone manuscript, Rounds 1–17

## {{V:ms_title}} {#manuscript-title-r17}

{{MANUSCRIPT}}

## R17.8 Manuscript verification

{{T:ms_verification}}

## R17.9 Decisions (D141–D143)

{{T:decisions}}

## R17.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R17.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–S: historical records

Part B is revision 18's Part A (Round 16 and the Round-16 manuscript); Part C is Round 15; Part D is Round 14 (the
first manuscript draft); Part E is Round 13; Part F is Round 12; Parts G and H are the L39 and L38 parts; Parts I–S are
the round-11 to round-1 reports, each rendered from its own records. Embedded living documents show their current
version.
