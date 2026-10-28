<p class="small">Research prototype report, revision 20 · generated from saved records by
<code>scripts/build_report_round18.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 18: landmark anchoring decided
on a confirmation split (D144–D146), followed by the updated standalone manuscript (Rounds 1–18) and its verification.
Part B is revision 19's Part A (Round 17 and the Round-17 manuscript); Part C is Round 16 and its manuscript; Part D is
Round 15; Part E is Round 14 (the first manuscript draft); Part F is Round 13; Part G is Round 12; Parts H–I are the L39
and L38 parts; Parts J–T are the round-11 to round-1 reports.</p>

## 0. Round-18 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Change tested (D144) | **Confirmation-split gate** (`ResFKAN.fit(confirm=...)`, off by default): each replicate's validation set is split, stratified on the event, into val_select (tuning, early stopping, the 16 candidate fine-tunes and the choice of the anchored state, for both models) and val_confirm (only the final one-versus-one decision against B2-S); when anchoring does not win on val_confirm the estimator returns B2-S's predictions exactly |
| Development check | 60 jobs on the already-used Round-17 seeds; Round 17 reproduced exactly with the new code; the pre-fixed rule chose margin {{V:m_cox}} (Cox) and {{V:m_aft}} (AFT); go/no-go: {{V:go}} |
| Formal run | {{V:n_records}} jobs, {{V:n_failed}} failed: S-Last 4400–4419, S-Sparse 4500–4519, S-Misspec-lognormal 4600–4619 |
| **H-R18a** (Cox) | **{{V:a_verdict}}.** S-Last confirm − B2-S Cox curve error 95% CI {{V:a1_ci}} (rule: upper below 0; anchored {{V:ar_cox_last}}). S-Sparse CI {{V:a2_ci}} (rule: upper below +0.005; anchored {{V:ar_cox_sparse}}) |
| **H-R18b** (log-normal AFT) | **{{V:b_verdict}}.** S-Misspec CI {{V:b1_ci}} (rule: upper below +0.005; anchored {{V:ar_aft_misspec}}). S-Last CI {{V:b2_ci}} (rule: upper below 0; anchored {{V:ar_aft_last}}) |
| Manuscript | `docs/manuscript_round18.md` / `reports/manuscript_round18.pdf` ({{V:ms_pages}} pages), Rounds 1–18, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

<!-- R18 SUMMARY -->

# Part A: Round 18

## R18.1 Owner instruction (D144) and fact check

<!-- R18 FACTCHECK -->

## R18.2 Methods (as implemented)

- **Validation split.** Each replicate's validation subjects (20% of n) are split once, stratified on the event
  indicator, into val_select (half) and val_confirm (half), by a generator seeded with (replicate seed, 18).
- **Selection on val_select** for both models: the comparator ensemble (B2-S Cox / log-normal B2-S AFT) and the anchored
  ensemble are tuned and early-stopped on val_select; the Round-17 candidates (joint and static states) are fine-tuned
  and the anchored state chosen on val_select (`choose_twoway` step 1).
- **Decision on val_confirm** (`heldout_loss`): the chosen anchored candidate against the comparator's seed-0 member,
  one model each; anchored iff its held-out loss is lower by more than the margin; otherwise B2-S's predictions are
  returned unchanged.
- **Exploratory comparator:** B2-S trained as in Rounds 12–17 on the whole validation set (`b2s_<loss>_fullval`).
- **Tests:** R18-1 to R18-4 (methods), R18-5 to R18-9 (manuscript); the Round 15–17 tests pass with default flags.

## R18.3 Development check (before any Round-18 seed)

{{T:dev_checks}}

## R18.4 Execution

{{T:exec_table}}

Four single-threaded lanes, one queue. The registration was locked in D144 before the lanes started.

## R18.5 H-R18a and H-R18b (20 replicates per scenario)

{{T:a_table}}

## R18.6 Limitations

<!-- R18 LIMITATIONS -->

# Part A (continued): the standalone manuscript, Rounds 1–18

## {{V:ms_title}} {#manuscript-title-r18}

{{MANUSCRIPT}}

## R18.7 Manuscript verification

{{T:ms_verification}}

## R18.8 Decisions (D144–D146)

{{T:decisions}}

## R18.9 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R18.10 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–T: historical records

Part B is revision 19's Part A (Round 17 and the Round-17 manuscript); Part C is Round 16 and its manuscript; Part D is
Round 15; Part E is Round 14 (the first manuscript draft); Part F is Round 13; Part G is Round 12; Parts H and I are the
L39 and L38 parts; Parts J–T are the round-11 to round-1 reports, each rendered from its own records. Embedded living
documents show their current version.
