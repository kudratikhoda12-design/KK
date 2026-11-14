<p class="small">Research prototype report, revision 26 · generated from saved records by
<code>scripts/build_report_round24.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, pushed to the remote on the owner's instruction of 2026-11-14) · status: <b>research prototype; both manuscripts are
drafts for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 24: a Bayesian joint model
(JMbayes2) fitted as the standard dynamic-prediction comparator on the six clinical cohorts and the Round-23 partitions
(D163–D165). Part A continues with the Round-24 journal main manuscript, embedded, and the verification of both Round-24
manuscripts; the cumulative Round-24 manuscript is its Supplementary Material (`reports/manuscript_round24.pdf`). Part B
is revision 25's Part A (Round 23 and the Round-23 journal manuscript); Parts C–Z are revision 25's Parts B–Y.</p>

## 0. Round-24 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Owner instruction | "yes install R and fit the joint model" (2026-11-10); then "yes update the manuscript and send me the PDFs" (2026-11-12); then, on 2026-11-14, push the branch, rewrite the manuscripts for outside readers and correct two bookkeeping items in this report (D166) |
| Design (D163) | Registered before any formal fit: a Bayesian multivariate joint model (per-marker mixed models on the training subjects' full histories, Cox submodel with delayed entry at the landmark, current-value association; straight-line or spline-mean trajectory chosen on validation; 3 chains × 6,500 iterations) on the six clinical cohorts and the Round-23 partitions 5–24, paired with the Round-23 NP-KAN-AFT, landmark Cox and blend records |
| Execution | {{V:n_records}} jobs: {{V:n_ok}} gave a joint model, {{V:n_failed}} recorded fit failures (not refitted); no relaunch |
| H-R24a (renal) | **{{V:hA}}** ({{V:a_diff}}); caveat: on renal the joint model was also worse than landmark Cox ({{V:a_lc}}) and passed D-calibration on {{V:a_dcal}} partitions (Section R24.3) |
| H-R24b (non-inferiority on every cohort) | **{{V:hB}}** (upper bounds {{V:b_uppers}}; margin +0.005) |
| Journal main manuscript | `docs/manuscript_journal_main_round24.md` / `reports/manuscript_journal_main_round24.pdf` ({{V:j_pages}} pages), embedded below: {{V:j_values}} values, each also a verified value of the Supplementary Material, {{V:j_refs}} references, {{V:j_check}} |
| Supplementary Material | `docs/manuscript_round24.md` / `reports/manuscript_round24.pdf` ({{V:ms_pages}} pages), Rounds 1–24: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c, the 4f gate's original R1 and CK-6d), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

- **The missing comparator is now in the study.** A standard Bayesian joint model was fitted on the same patients,
  splits and metrics as the functional model, so the reviewer question "no joint model was compared" is answered.
- **It was not better than the functional model or landmark Cox on any clinical cohort.** On five cohorts the three
  models were close and every interval included 0, but the intervals were too wide to show that NP-KAN-AFT is within
  the registered margin of the joint model.
- **On renal NP-KAN-AFT was clearly better, but for a reason that limits the claim.** The joint model was also worse
  than landmark Cox there and badly calibrated: its straight-line or smooth trajectories do not follow the long kidney
  histories. The result shows that this joint model fits renal poorly, not that NP-KAN-AFT beats joint models in general.
- **The joint model's sampler did not settle on many PBC and heart-valve partitions** (Rhat above 1.1); as registered,
  these fits were kept and the problem is reported as a limitation.
- **Manuscripts.** The Supplementary Material gains Section 5.14 (Tables 66–68); the journal paper gains Section 4.4
  and Table 21, and its limitations now name the single joint-model specification instead of "no joint model".
- **Rewrite for outside readers (D166).** The journal paper was rewritten in plain terms: study-internal vocabulary
  (round numbers, decision identifiers, "owner", "untouched", "locked") was replaced by descriptions, the hypotheses
  keep their registration labels, and the pre-registration is stated to be internal (a time-stamped decision log, not a
  public registry). Every logged value, every table and every verdict statement is unchanged, and both verifications
  pass. The Supplementary Material keeps its round-by-round structure, gains a short reader's note and loses the
  decision identifiers.
- **Correction to this report (D166).** Revision 26 as first issued showed the previous suite's test counts in
  Section 0.1 and an empty table in Section R24.9, because the test records had not been compiled into the tables the
  report reads. They now show `final_full_suite_round24`, and the deliverable check fails if the two ever differ.

# Part A: Round 24

## R24.1 Owner instruction and pre-registration (D163)

- **Instruction (2026-11-10).** Install R and fit the joint model, the largest gap named in the readiness assessment.
- **Pre-registration** (`docs/round24_preregistration.md`, committed before any formal job): software (project-local R
  4.4.3 with JMbayes2 0.6.0 via micromamba in `.rlocal/`, git-ignored; the system and the Python environment
  unchanged), data and pairing, the model, the hypotheses and decision rules, the development problems fixed and the
  development numbers seen on partition 0, and the execution plan. A run log was appended after the run.

## R24.2 Methods (as implemented)

- `scripts/jm_fit.R`: per form, `lme(value ~ t)` or `lme(value ~ ns(t, 2))` with `random = ~ t | id` per marker on one
  shared data frame; `coxph(Surv(entry, time, event) ~ z)` with entry at the landmark; `jm()` with 3 chains, 6,500
  iterations, 1,500 burn-in, one core, seed 2400 + k; dynamic prediction for validation and test subjects from their
  history up to the landmark, on 201 residual times up to the longest training follow-up, held flat after it.
- `scripts/run_round24.py`: inputs, the R call, the Round-23 metrics, the validation choice of form; lanes
  `scripts/run_round24_lanes.py`; analysis `scripts/analyze_round24.py` → `reports/round24/analysis_summary.json`,
  `docs/round24_results.md`. Tests R24-1..R24-4 (inputs, pairing with a Round-23 record, interpolation and form choice,
  an end-to-end fit with a determinism check).

## R24.3 Conditional IBS and contrasts with the joint model

{{T:ibs}}

## R24.4 Concordance, calibration, convergence and failed fits

{{T:conv}}

## R24.5 Hypotheses

{{T:hyp}}

## R24.6 Limitations

- **One joint-model specification.** Straight-line or spline-mean trajectories with a subject-specific intercept and
  slope and current-value association; no other association structure, trajectory model, prior or chain length was
  tried after the results (D163 §4).
- **Convergence.** The largest association Rhat exceeded 1.1 on many PBC and heart-valve partitions; these fits were
  kept as registered.
- **Re-analysed data.** The clinical cohorts and partitions were analysed in Round 23; C-MAPSS was not analysed with a
  joint model; no deep dynamic model was fitted.
- **No novelty or general superiority claim (D111).**

# Part A (continued): the journal main manuscript (Round 24)

## {{V:j_title}} {#manuscript-title-journal-r24}

{{MANUSCRIPT}}

## R24.7 Manuscript verification

{{T:ms_verification}}

## R24.8 Decisions (D163–D166)

{{T:decisions}}

## R24.9 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R24.10 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–Z: historical records

Part B is revision 25's Part A (Round 23 and the Round-23 journal manuscript); Part C is revision 24's Part A (Round 22
and the Round-22 journal manuscript); Part D is revision 23's Part A (Round 21 and the Round-21 manuscript); Part E is
revision 22's Part A (Round 20 and the Round-20 manuscript); Part F is Round 19 and its manuscript; Part G is Round 18
and its manuscript; Part H is Round 17 and its manuscript; Part I is Round 16 and its manuscript; Part J is Round 15;
Part K is Round 14 (the first manuscript draft); Part L is Round 13; Part M is Round 12; Parts N and O are the L39 and
L38 parts; Parts P–Z are the round-11 to round-1 reports, each rendered from its own records. Embedded living documents
show their current version.
