<p class="small">Research prototype report, revision 21 · generated from saved records by
<code>scripts/build_report_round19.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 19: the Round-18 confirmation-split
decision with a full-validation estimator, tested against the usual full-validation B2-S (D147–D149), followed by the
updated standalone manuscript (Rounds 1–19) and its verification. Part B is revision 20's Part A (Round 18 and the
Round-18 manuscript); Parts C–U are revision 20's Parts B–T.</p>

## 0. Round-19 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Change tested (D147) | **Full-validation estimator** (`ConfirmRefitAnchor`, off by default): the Round-18 decision on val_confirm is kept; if anchoring is not confirmed the estimator is the usual full-validation B2-S, exactly; if it is confirmed, the chosen masks are refitted on the whole validation set |
| Development check | 60 jobs on the already-used Round-18 seeds; Round 18 reproduced exactly with the new code; the pre-fixed rule chose refit variant **{{V:variant}}** and margins {{V:m_cox}} (Cox), {{V:m_aft}} (AFT); go/no-go: {{V:go}} |
| Formal run | {{V:n_records}} jobs, {{V:n_failed}} failed: S-Last 4700–4719, S-Sparse 4800–4819, S-Misspec-lognormal 4900–4919 |
| **H-R19a** (Cox, against the full-validation B2-S) | **{{V:a_verdict}}.** S-Last 95% CI {{V:a1_ci}} (rule: upper below 0; anchored {{V:ar_cox_last}}). S-Sparse CI {{V:a2_ci}} (rule: upper below +0.005; anchored {{V:ar_cox_sparse}}) |
| **H-R19b** (log-normal AFT, against the full-validation B2-S) | **{{V:b_verdict}}.** S-Misspec CI {{V:b1_ci}} (rule: upper below +0.005; anchored {{V:ar_aft_misspec}}). S-Last CI {{V:b2_ci}} (rule: upper below 0; anchored {{V:ar_aft_last}}) |
| Manuscript | `docs/manuscript_round19.md` / `reports/manuscript_round19.pdf` ({{V:ms_pages}} pages), Rounds 1–19, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

<!-- R19 SUMMARY -->

# Part A: Round 19

## R19.1 Owner instruction (D147) and fact check

<!-- R19 FACTCHECK -->

## R19.2 Methods (as implemented)

- **Decision (unchanged from Round 18).** val_select for tuning, early stopping, the candidate fine-tunes and the
  step-(1) choice (anchored model and a val_select B2-S); val_confirm only for the one-versus-one decision with a margin.
- **Estimator.** Not anchored: the B2-S ensemble tuned and early-stopped on the whole validation set, returned
  unchanged. Anchored: the chosen state's masks applied to an anchored ensemble that uses the whole validation set
  ("retrain": a new ensemble tuned and early-stopped on it; "finetune": the step-(1) ensemble fine-tuned once more on
  it), every member fine-tuned with the mask fixed, early-stopping on the whole validation set.
- **Tests:** R19-1 to R19-4 (methods), R19-5 to R19-9 (manuscript); the Round 15–18 tests pass with default flags.

## R19.3 Development check (before any Round-19 seed)

{{T:dev_checks}}

## R19.4 Execution

{{T:exec_table}}

Four single-threaded lanes, one queue. The registration was locked in D147 before the lanes started.

## R19.5 H-R19a and H-R19b (20 replicates per scenario)

{{T:a_table}}

## R19.6 Limitations

<!-- R19 LIMITATIONS -->

# Part A (continued): the standalone manuscript, Rounds 1–19

## {{V:ms_title}} {#manuscript-title-r19}

{{MANUSCRIPT}}

## R19.7 Manuscript verification

{{T:ms_verification}}

## R19.8 Decisions (D147–D149)

{{T:decisions}}

## R19.9 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R19.10 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–U: historical records

Part B is revision 20's Part A (Round 18 and the Round-18 manuscript); Part C is Round 17 and its manuscript; Part D is
Round 16 and its manuscript; Part E is Round 15; Part F is Round 14 (the first manuscript draft); Part G is Round 13;
Part H is Round 12; Parts I and J are the L39 and L38 parts; Parts K–U are the round-11 to round-1 reports, each
rendered from its own records. Embedded living documents show their current version.
