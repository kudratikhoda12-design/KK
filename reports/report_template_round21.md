<p class="small">Research prototype report, revision 23 · generated from saved records by
<code>scripts/build_report_round21.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 21: the locked Round-20 methods
(NP-KAN-AFT, log-normal KAN-AFT, two-half cross-confirmed anchor, fixed-μ symbolic distillation) applied to the real
longitudinal benchmarks (pbc_E2, pbc_E1, prothro, aids) and C-MAPSS FD001 Option B (D153–D155). Part A continues with
the updated standalone manuscript (Rounds 1–21) and its verification. Part B is revision 22's Part A (Round 20 and the
Round-20 manuscript); Parts C–W are revision 22's Parts B–V.</p>

## 0. Round-21 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Design (D153) | Locked models, no tuning on test splits: B2-S Cox, Weibull, log-normal and NP-KAN-AFT (rank_km) D112 ensembles; cross-confirmed anchors for Cox, log-normal and NP-AFT with the Round-20 margins; fixed-μ distillation diagnostic; Round-13 outer splits 0–4 and C-MAPSS Option B; paired and Nadeau–Bengio intervals |
| Fact check and reproduction | Several statements in the instruction corrected before the run (paths, endpoint labels, landmark, C-MAPSS design, quoted C-MAPSS numbers not in any record, option names, margins, test count); pbc_E2 split 0 landmark Cox and B2-S Cox reproduced Round 13 exactly before the run, and every Round-21 B2-S Cox and Weibull fit equals its Round-13 record |
| Execution | {{V:n_records}} jobs ({{V:n_failed}} failed); three container restarts, each logged; the C-MAPSS jobs resumed from family checkpoints and pickled fits (mechanics only) |
| H-R21a (NP-KAN-AFT, clinical) | **{{V:hA}}.** All {{V:a_n}} clinical fits finite; pbc_E2 NP − Cox {{V:a_cox}} (≤ +0.005, met) but NP − Weibull {{V:a_waft}} (> 0, not met); D-cal p ≥ 0.05 on {{V:a_dcal}} of 5 pbc_E2 splits (met) |
| H-R21b (cross-confirmed anchoring, clinical) | **{{V:hB}}.** pbc_E2: {{V:b_pbc_E2}}; pbc_E1: {{V:b_pbc_E1}}; prothro: {{V:b_prothro}}; aids: {{V:b_aids}}; exact fallback on every non-confirmed split |
| H-R21c (C-MAPSS calibration) | **{{V:hC}}** (stated thresholds IBS ≤ 0.055, Harrell C ≥ 0.88 not reached). B2-S Cox: {{V:c_b2s_cox}}; NP-KAN-AFT: {{V:c_b2s_npaft}}; log-normal: {{V:c_b2s_lnaft}}. Registered relative secondary: **{{V:hC_rel}}** |
| Manuscript | `docs/manuscript_round21.md` / `reports/manuscript_round21.pdf` ({{V:ms_pages}} pages), Rounds 1–21, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

1. **On real cohorts the semiparametric KAN-AFT is safe but not better.** Every fit was finite and D-calibrated, and
   on the primary PBC endpoint it beat B2-S Cox on average, but it trailed the Weibull KAN-AFT there, so the
   registered non-degradation criterion failed. No clinical contrast had a Nadeau–Bengio interval excluding 0, and
   landmark Cox kept the lowest mean on both PBC endpoints.
2. **Cross-confirmed anchoring does no harm on real data.** With validation halves of 27–40 subjects it confirmed
   anchoring on some splits only, never cost more than the registered margin, and fell back to the unanchored model
   exactly otherwise. It was not better than the Round-15 anchored model or landmark Cox.
3. **On C-MAPSS the AFT error laws improve calibration a little, not to the stated targets.** Both AFT models lowered
   the D-calibration χ² of B2-S Cox, but no model reached IBS 0.055 or Harrell C 0.88; those thresholds were stated
   with reference to Round-13 numbers that are not in the records.
4. **Symbolic distillation stays safe on real data.** It substituted about half to nine tenths of the edges, rolled back
   where it would have raised the validation loss, and changed the conditional IBS by less than 0.001.

# Part A: Round 21

## R21.1 Owner instruction (D153) and fact check

- **Instruction (2026-11-02).** Evaluate the Round-20 NP-KAN-AFT (rank_km) and the cross-confirmed anchor on pbc_E2,
  pbc_E1, prothro, aids and C-MAPSS FD001 Option B under the Rounds 15–20 protocol: fact check and reproduction first,
  pre-registration (D153), tests R21-1..R21-4, the benchmark, decisions D154–D155, ledger L48, manuscript round 21,
  report revision 23, full suite; commit locally, do not push.
- **Fact check** (`docs/round21_preregistration.md` §1): named paths do not exist (mapped to the repository's);
  pbc_E2/E1 endpoint labels reversed in the instruction (here E2 = death or transplant, primary); landmark 730 days, no
  fixed horizons; landmark cohorts n 278/278/332/403 and validation sets 55–80; C-MAPSS Option B is the Round-13 engine
  split with every window an event; the quoted Round-13 C-MAPSS `b2s_cox` numbers are in no record; option names
  mapped; D150 margins Cox 0.02 absolute, log-normal 0, NP-AFT 0 relative; 35 Round 15–20 deterministic tests.

## R21.2 Methods (as implemented)

- Harness `scripts/run_round21_real.py` (lanes `scripts/run_round21_lanes.py`): per clinical split, the four B2-S
  families on the Round-13 B-spline front end; the three cross-confirmed anchors with halves from
  `run_round18.split_val(va, event, split)` and reference ensembles on each half and the whole validation set; the exact
  fallback checked on the test split; `distill_and_replace` on the seed-0 member of B2-S NP-AFT and Cox. C-MAPSS: the
  four families and the NP-AFT anchor with engine-level halves.
- Tests R21-1..R21-4 (`tests/test_round21_real_npaft.py`): real-split smoke for NP and log-normal AFT, exact fallback on
  a real split, distillation on PBC, and the 35 Round 15–20 tests in a subprocess; R21-5..R21-9 (manuscript).
- Analysis `scripts/analyze_round21.py` -> `reports/round21/analysis_summary.json`, `docs/round21_results.md`.

## R21.3 Clinical results (per data set: means, contrasts, cross-confirmation, distillation, reproduction, timing)

{{T:clinical}}

## R21.4 Accepted symbolic formulas (split 0, first edges)

{{T:formulas}}

## R21.5 C-MAPSS FD001 Option B

{{T:cmapss}}

## R21.6 Limitations

- **Known data.** The clinical cohorts and their splits were used in Rounds 13 and 15; this round applies fixed methods
  to them. Five splits cannot power superiority tests (D111); every comparison beyond the registered criteria is
  exploratory.
- **H-R21c thresholds.** The absolute C-MAPSS thresholds came with Round-13 numbers that are not in the records and
  were kept as stated; C-MAPSS windows of one engine are correlated, so its D-calibration test is anti-conservative.
- **Interrupted execution.** Three container restarts; the affected C-MAPSS jobs resumed from checkpoints and pickled
  fits (deterministic fits; resume tested identical); their recorded seconds include reloaded parts.
- **No novelty claim (D111).** AFCM remains unvalidated.

# Part A (continued): the standalone manuscript, Rounds 1–21

## {{V:ms_title}} {#manuscript-title-r21}

{{MANUSCRIPT}}

## R21.7 Manuscript verification

{{T:ms_verification}}

## R21.8 Decisions (D153–D155)

{{T:decisions}}

## R21.9 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R21.10 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–W: historical records

Part B is revision 22's Part A (Round 20 and the Round-20 manuscript); Part C is Round 19 and its manuscript; Part D is
Round 18 and its manuscript; Part E is Round 17 and its manuscript; Part F is Round 16 and its manuscript; Part G is
Round 15; Part H is Round 14 (the first manuscript draft); Part I is Round 13; Part J is Round 12; Parts K and L are
the L39 and L38 parts; Parts M–W are the round-11 to round-1 reports, each rendered from its own records. Embedded
living documents show their current version.
