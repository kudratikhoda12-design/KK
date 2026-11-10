<p class="small">Research prototype report, revision 25 · generated from saved records by
<code>scripts/build_report_round23.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; both manuscripts are
drafts for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 23: where the functional model
helps, tested on the untouched C-MAPSS FD002 and FD004 data, by thinning renal histories and by blending NP-KAN-AFT
with landmark Cox on six clinical cohorts with twenty new partitions (D160–D162). Part A continues with the Round-23
journal main manuscript, embedded, and the verification of both Round-23 manuscripts; the cumulative Round-23
manuscript is its Supplementary Material (`reports/manuscript_round23.pdf`). Part B is revision 24's Part A (Round 22
and the Round-22 journal manuscript); Parts C–Y are revision 24's Parts B–X.</p>

## 0. Round-23 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Owner instruction | "try different methods to overcome the issue and get better result" (2026-11-09); then "yes update the manuscript and send me the PDFs" (2026-11-10) |
| Design (D160) | Registered before any formal Round-23 fit: NP-KAN-AFT against landmark Cox on the untouched C-MAPSS FD002 and FD004 (ten engine splits each; H-R23a); renal histories thinned to three visits (H-R23b); a landmark / NP-KAN-AFT blend on six clinical cohorts with twenty new partitions (H-R23c) |
| Execution | {{V:n_ok}} of {{V:n_records}} registered jobs completed, {{V:n_failed}} failed; no relaunch |
| H-R23a (untouched FD002 / FD004) | **{{V:hA}}** (FD002 {{V:a_fd002}}, FD004 {{V:a_fd004}}; Section R23.3) |
| H-R23b (renal thinning) | **{{V:hB}}** ({{V:b_diff}}; Section R23.4) |
| H-R23c (blend) | (i) non-inferiority on every cohort **{{V:hC1}}**; (ii) better on renal **{{V:hC2}}** (Section R23.5) |
| Journal main manuscript | `docs/manuscript_journal_main_round23.md` / `reports/manuscript_journal_main_round23.pdf` ({{V:j_pages}} pages), embedded below: {{V:j_values}} values, each also a verified value of the Supplementary Material, {{V:j_refs}} references, {{V:j_check}} |
| Supplementary Material | `docs/manuscript_round23.md` / `reports/manuscript_round23.pdf` ({{V:ms_pages}} pages), Rounds 1–23: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c, the 4f gate's original R1 and CK-6d), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

- **With long, dense histories the functional model is better.** On two engine data sets never used before, NP-KAN-AFT
  had lower integrated Brier score than landmark Cox on 19 of 20 splits, and both corrected intervals excluded 0. It
  was also better than the summary and FPCA AFT models, and it ranked engines better (Harrell C). Its calibration was
  weak (D-calibration passed on few splits), so its predicted probabilities should not be taken at face value.
- **The history is the reason.** On renal, cutting each patient's history to three visits made NP-KAN-AFT clearly
  worse, and worse than landmark Cox.
- **On short clinical histories there is still no gain.** The blend of NP-KAN-AFT and landmark Cox had a lower mean than
  landmark Cox on five of six cohorts, but every interval was too wide for the registered non-inferiority margin, so
  neither non-inferiority nor a gain on renal was shown. These clinical results reuse data from earlier rounds with new
  partitions.
- **Manuscripts.** The Supplementary Material gains Section 5.13; the journal paper gains Section 4.3 and a rewritten
  Section 5.2, which now states with evidence when a landmark model is enough and when a functional model helps.

# Part A: Round 23

## R23.1 Owner instruction and pre-registration (D160)

- **Instruction (2026-11-09).** Try different methods to overcome the issue (no functional model better than landmark
  Cox on the clinical cohorts) and get better results, without changing thresholds after results or claiming
  superiority without evidence.
- **Pre-registration** (`docs/round23_preregistration.md`, committed before any formal fit): the diagnosis from the
  existing records (functional models help only where histories are long; five splits lack power under the
  Nadeau–Bengio correction), the data and partitions (FD002 / FD004 untouched; clinical partitions 5–24 new), the models,
  the C-MAPSS budget fixed from run time only (windows every 20 cycles, NP-KAN-AFT at lr 3e-3, wd 1e-4, seeds 0–2), the
  hypotheses and decision rules, and the analysis script written before the run ended.

## R23.2 Methods (as implemented)

- `kanfae.data.cmapss.condition_normalise`: k-means with six clusters on the training engines' three operating
  settings, per-condition training mean and SD, constant-in-condition sensors set to 0 (R23-1).
- `scripts/run_round23.py`: NP-KAN-AFT B2-S ensemble; landmark Cox (clinical: as Rounds 13–22; C-MAPSS: last raw or
  last smoothed values, chosen on validation IBS); the blend S = w·S_landmark + (1 − w)·S_NP with w on the grid
  {0, 0.25, 0.5, 0.75, 1} chosen on validation, ties to the larger w (R23-2); three-visit thinning (R23-3); per-seed
  pickled C-MAPSS fits equal to `FKANEnsemble` (R23-5). Lanes `scripts/run_round23_lanes.py`; analysis
  `scripts/analyze_round23.py` → `reports/round23/analysis_summary.json`, `docs/round23_results.md`.

## R23.3 Untouched C-MAPSS FD002 and FD004 (H-R23a)

{{T:cm}}

## R23.4 Renal history thinning (H-R23b)

{{T:renal}}

## R23.5 Landmark / NP-KAN-AFT blend on six clinical cohorts (H-R23c)

{{T:clin}}

## R23.6 Hypotheses

{{T:hyp}}

## R23.7 Limitations

- **C-MAPSS.** NP-KAN-AFT was fitted at one fixed setting with windows every 20 cycles to bound its cost; its
  D-calibration was weak, and no model was well calibrated on FD002.
- **Clinical data.** The twenty partitions are new, but the cohorts were analysed in earlier rounds, so these are
  re-analyses, not untouched tests; the thinning test used one cohort.
- **Blend.** Its weight is chosen on small validation sets and varied widely between partitions.
- **No novelty or general superiority claim (D111).**

# Part A (continued): the journal main manuscript (Round 23)

## {{V:j_title}} {#manuscript-title-journal-r23}

{{MANUSCRIPT}}

## R23.8 Manuscript verification

{{T:ms_verification}}

## R23.9 Decisions (D160–D162)

{{T:decisions}}

## R23.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R23.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–Y: historical records

Part B is revision 24's Part A (Round 22 and the Round-22 journal manuscript); Part C is revision 23's Part A (Round 21
and the Round-21 manuscript); Part D is revision 22's Part A (Round 20 and the Round-20 manuscript); Part E is Round 19
and its manuscript; Part F is Round 18 and its manuscript; Part G is Round 17 and its manuscript; Part H is Round 16
and its manuscript; Part I is Round 15; Part J is Round 14 (the first manuscript draft); Part K is Round 13; Part L is
Round 12; Parts M and N are the L39 and L38 parts; Parts O–Y are the round-11 to round-1 reports, each rendered from its
own records. Embedded living documents show their current version.
