<p class="small">Research prototype report, revision 21 · generated from saved records by
<code>scripts/build_report_round19.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 19: the Round-18
confirmation-split decision with a full-validation estimator, tested against the usual full-validation B2-S in a
development check that failed its go/no-go rule (D147); the owner closed the round as a development result (D148), so
no formal test was run. Part A continues with the updated standalone manuscript (Rounds 1–19) and its verification.
Part B is revision 20's Part A (Round 18 and the Round-18 manuscript); Parts C–U are revision 20's Parts B–T.</p>

## 0. Round-19 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Change tested (D147) | **Full-validation estimator** (`ConfirmRefitAnchor`, off by default): the Round-18 decision on val_confirm is kept; if anchoring is not confirmed the estimator is the usual full-validation B2-S, exactly; if it is confirmed, the chosen masks are refitted on the whole validation set ("retrain" or "finetune") |
| Development check | {{V:n_records}} jobs ({{V:n_failed}} failed) on the already-used Round-18 seeds; Round 18 reproduced exactly with the new code; the pre-fixed rule chose variant **{{V:variant}}**, margins {{V:m_cox}} (Cox) and {{V:m_aft}} (AFT) |
| Go/no-go (against the full-validation B2-S) | **{{V:go}}.** Cox S-Sparse upper bound {{V:u_cox_sparse}} (rule: below +0.005) failed; Cox S-Last {{V:u_cox_last}}, AFT S-Misspec {{V:u_aft_misspec}} and AFT S-Last {{V:u_aft_last}} met their rules |
| Formal run | **Not run.** H-R19a and H-R19b are untested; seeds 4700–4719, 4800–4819 and 4900–4919 remain unused (D147, D148) |
| Manuscript | `docs/manuscript_round19.md` / `reports/manuscript_round19.pdf` ({{V:ms_pages}} pages), Rounds 1–19, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

1. **Refitting does not remove the cost of anchoring on S-Sparse for Cox.** Against B2-S trained on the whole
   validation set, the refitted anchored model was worse on average on S-Sparse, and the confirmation decision still
   anchored in a third to a half of those replicates; no margin of the grid kept the Cox S-Sparse bound within +0.005.
2. **The S-Last gains and the AFT results survive the stricter comparator.** Cox and AFT kept clear S-Last gains, and
   the AFT model stayed within its S-Misspec margin, in development.
3. **Round 18's S-Sparse result rests on the half-validation comparator.** Together with Round 18's exploratory
   contrast, the development check indicates that the confirmation-gated Cox model is not shown to be non-inferior to
   the usual full-validation B2-S on S-Sparse.
4. **Protocol.** The go/no-go rule fixed before the development runs failed, so no formal seed was used; the owner
   closed the round as a development result.

# Part A: Round 19

## R19.1 Owner instruction (D147) and fact check

- **Instruction.** Keep the Round-18 split, return the full-validation B2-S when anchoring is not confirmed, refit the
  chosen anchored model on the whole validation set when it is, test against the full-validation B2-S; development
  check on already-used seeds before pre-registration; tests R19-*; commit locally, do not push.
- **Fact check** (`docs/round19_devplan.md` §1): the Round-18 figures quoted are correct; the "0.0058 loss" of the
  val_select B2-S is a mean difference whose interval included 0 ([−0.0131, +0.0014], p 0.11). The fresh seed blocks
  were unused.
- **Owner decision after the NO-GO (D148):** "continue and send me report pdf", taken as the recommended option: close
  Round 19 as a development-only result, write it into the manuscript and report revision 21, run no formal seed.

## R19.2 Methods (as implemented)

- **Decision (unchanged from Round 18).** val_select for tuning, early stopping, the candidate fine-tunes and the
  step-(1) choice (anchored model and a val_select B2-S); val_confirm only for the one-versus-one decision with a margin.
- **Estimator.** Not anchored: the B2-S ensemble tuned and early-stopped on the whole validation set, returned
  unchanged. Anchored: the chosen state's masks applied to an anchored ensemble that uses the whole validation set
  ("retrain": a new ensemble tuned and early-stopped on it; "finetune": the step-(1) ensemble fine-tuned once more on
  it), every member fine-tuned with the mask fixed, early-stopping on the whole validation set.
- **Tests:** R19-1 to R19-4 (val_confirm untouched before the decision; exact full-validation B2-S when not anchored;
  the refit early-stops on the whole validation set and never sees the test split; both variants), R19-5 to R19-9
  (manuscript); the Round 15–18 tests pass with default flags; a Round-18 job reproduced its record exactly.

## R19.3 Development check (before any Round-19 seed)

{{T:dev_checks}}

## R19.4 Limitations

- **No formal test.** Every Round-19 number is a development result on already-used seeds, with the margin and variant
  chosen on the same replicates (in-sample).
- **One simulator, 20 replicates per scenario; no real data. AFCM is still not validated.**

# Part A (continued): the standalone manuscript, Rounds 1–19

## {{V:ms_title}} {#manuscript-title-r19}

{{MANUSCRIPT}}

## R19.5 Manuscript verification

{{T:ms_verification}}

## R19.6 Decisions (D147–D149)

{{T:decisions}}

## R19.7 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R19.8 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–U: historical records

Part B is revision 20's Part A (Round 18 and the Round-18 manuscript); Part C is Round 17 and its manuscript; Part D is
Round 16 and its manuscript; Part E is Round 15; Part F is Round 14 (the first manuscript draft); Part G is Round 13;
Part H is Round 12; Parts I and J are the L39 and L38 parts; Parts K–U are the round-11 to round-1 reports, each
rendered from its own records. Embedded living documents show their current version.
