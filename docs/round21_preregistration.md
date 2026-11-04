# Round 21 pre-registration: the Round-20 methods on real data (D153)

Written and committed before any full-budget job of `scripts/run_round21_real.py`. Owner instruction (2026-11-02):
evaluate the Round-20 NP-KAN-AFT (rank_km) and the two-half cross-confirmed anchor on the real longitudinal
benchmarks (pbc_E2, pbc_E1, prothro, aids) and on C-MAPSS FD001 Option B; strict pre-registration, verification and
D111 claim hygiene; commit locally, do not push. The real data and their outer splits were used in Rounds 13 and 15,
so this round is a confirmatory **application** of locked methods to known data, not a test on fresh data.

## 1. Fact check of the instruction (each item checked against the repository on 2026-11-02)

| Statement | Check |
|---|---|
| Paths `experiments/run_real_benchmark.py`, `results/real_r13/`, `results/round15/real/`, `results/cmapss_r13/`, `DECISIONS.md`, `audit/ROUND0_LEDGER.md`, `manuscript/*.tex` | **Do not exist.** The real-data harness is `scripts/run_round13.py` (`RealCtx`, `real_metrics`, `run_cmapss`); records `reports/round13/results_lane*.jsonl` (single-landmark, B3, C-MAPSS) and `reports/round15/results_lane*.jsonl` (job `real`); decisions `docs/decisions.md`; ledger `docs/task_ledger.md`; manuscripts are Markdown rendered by pandoc (`docs/manuscript_roundN.md` -> `reports/manuscript_roundN.pdf`). Round 21 uses these (new files: `scripts/run_round21_real.py`, `scripts/analyze_round21.py`, `tests/test_round21_real_npaft.py`, `reports/round21/`, `docs/round21_results.md`, `docs/manuscript_round21.md`, `reports/manuscript_round21.pdf`) |
| pbc_E2 = death alone, 140 events; pbc_E1 = death/transplant, 169 events | **Reversed.** In this repository (`kanfae/data/realdata.py`, D114) **E2 = death or transplant (primary; 169 events in 312 subjects)** and **E1 = death, transplant censored (sensitivity; 140 events)**. The primary stays pbc_E2 by name, i.e. the composite endpoint |
| PBC t0 = 2.5 years, horizon [2.5, 10] y; prothro horizon [1, 6] y; aids [0.5, 1.5] y | The landmarks are PBC **730 days (2 years)**, prothro 1 year, aids 6 months (D114); there is no fixed horizon: the IBS grid is `metrics.ibs_time_grid` of the test and training residual times, as in Round 13 |
| n = 312 / 488 / 467 subjects | Full cohorts, correct; **after landmarking** (alive and at risk at the landmark): pbc_E2/E1 n = 278 (135 / 107 events), prothro n = 332 (183), aids n = 403 (126) |
| validation n ≈ 36–62, halves 18–31 | Validation sets are **55–80 subjects** (pbc 55–56, prothro 67, aids 80; 21–37 events); halves are 27–40 |
| Splits 0–4, stratified 60/20/20 | Correct (`split_groups(ids, split, strata=event)`); identical every time they are built |
| C-MAPSS Option B = 100 train / 100 test engines, right-censored at the last observed test cycle, t0 = 80, horizon [80, 250] | **Not the locked benchmark.** Round 13's Option B (`run_round13.run_cmapss`) uses the **100 training engines only**, split by engine 60/20/20 (5 splits), 30-cycle windows ending every 10 cycles, u = s/100, target T − s, **every window an event** (no censoring); IBS on the data-driven grid |
| Round-13 `b2s_cox` on C-MAPSS: Harrell C 0.903, Uno C 0.869, IPCW IBS 0.044, D-cal p 0.0005 (Limitation 14) | **Not found.** Round 13 fitted no `b2s_cox` on C-MAPSS; its KAN model there was the Weibull `b2s_aft`: mean IBS 0.0807, Harrell C 0.8690, Uno C 0.8731, D-cal p < 0.05 on 5/5 splits (`reports/round13_summary.json`). No record contains these four numbers |
| Seed convention `1000*split + {0,1,2}` "or the exact Round-13/15 convention" | The Round-13/15 convention is used: D112 ensemble, member seeds 0, 1, 2, lr × wd {1e-3, 3e-3} × {1e-4, 1e-3} tuned on each split's validation set |
| `FKANSurv(..., np_smooth_tau=0.05)`, `rel_margin`, `refit_mode`, `exact_b2s_fallback`, `distill_fix_mu`, `nan_safe_refit`, `symbolic_gate_mode="two_way"` | No such arguments. Equivalents: `np_tau=0.05`; `CrossConfirmRefitAnchor(margin, relative_margin)`; the refit is always the Round-19 finetune and the not-confirmed estimator is always the full-validation B2-S exactly (R20-5); fixed-μ NaN-safe distillation is `fkan_surv.distill_and_replace` (μ = 10, `safe=True`, validation rollback). The Round-17 "two-way gate" is the anchoring gate, not a symbolic gate; distillation's own accept/reject rule is the validation-loss rollback |
| Round-20 locked margins: npaft `rel_margin=0.0`, lnaft `rel_margin=0.0`, Cox `rel_margin=0.02` | D150: Cox **0.02 absolute** (NLPL per event), log-normal AFT **0 absolute**, NP-AFT **0 relative** |
| `landmark_cox` anchor; learned α̂ weights | The anchor is the ResFKAN last-value channels (static KAN edges on each channel's last observed value), not the `landmark_cox` model; there are no α weights. Recorded instead: confirmation, both directions' held-out losses, and the h1-direction channel masks |
| "repaired Round-17 normal-CDF survival path" for the log-normal AFT | Correct: `S = 1 − Φ((log t − μ − η)/σ)` (D141) |
| 33 existing Round 15–20 deterministic tests | **35** (`tests/test_round15.py` .. `test_round20.py`; R20-5 is parametrised over three families), plus 25 manuscript tests |
| Reproduction check on pbc_E2 split 0 | Done in the scratch directory before this plan (script and output in `reports/round21/repro/`): `landmark_cox` and `b2s_cox` reproduce the Round-13 records **exactly** (max absolute difference 0.0 on IBS, conditional IBS, Harrell C, Uno C, D-cal p; `b2s_cox` 97 s) |

## 2. Locked models (no tuning on test splits; settings from D112, D141, D150)

Per clinical (dataset, split): `b2s_cox`, `b2s_waft` (Weibull KAN-AFT, = Round-13 `b2s_aft`), `b2s_lnaft`
(log-normal), `b2s_npaft` (NP-KAN-AFT, `np_objective="rank_km"`, `np_tau=0.05`): D112 3-seed B2-S ensembles on the
Round-13 B-spline front end, lr × wd tuned on the split's validation loss. `b2s_anchor_{cox,lnaft,npaft}_cross`:
`CrossConfirmRefitAnchor` with halves h1 = val_select, h2 = val_confirm from `run_round18.split_val(va, event, split)`,
reference B2-S ensembles on h1, h2 and the whole validation set, margins as D150, finetune refit when confirmed,
otherwise the full-validation B2-S exactly (checked on every test split: maximum absolute difference of the test
survival matrix and risk). Symbolic diagnostic: `distill_and_replace` on the seed-0 member of `b2s_npaft` and
`b2s_cox` (formulas, replacement rate, rollback, test metrics before/after). C-MAPSS (split k = 0..4): `b2s_cox`,
`b2s_waft`, `b2s_lnaft`, `b2s_npaft`, `b2s_anchor_npaft_cross` (halves: the validation engines split in two by a
generator seeded (k, 21), so windows of one engine stay together). Metrics as Round 13 (clinical: marginal and
conditional IPCW IBS, Harrell C, Uno C, D-cal p; C-MAPSS: IBS, Harrell C, Uno C, D-cal p), plus the D-calibration χ².
Wall-clock seconds per model. Existing baselines read from their records on the same splits: Round 13 (`landmark_cox`,
`afcm_v3` (not validated), `fpca_cox`, `lme2_cox`, `sum_cox`, `sum_rsf`, `b2s_cox`, `b2s_aft`) and Round 15
(`b2s_anchor_cox`, unconfirmed parsimonious gate).

## 3. Hypotheses and criteria (fixed now)

All IBS below are conditional IPCW IBS unless stated. Paired over the 5 outer splits; every contrast reported with the
plain paired t interval and the Nadeau–Bengio corrected interval (df 4, variance factor 1/5 + n_test/n_train, as in
Rounds 13 and 15), wins/losses/ties. Five splits cannot power superiority tests (D111); the criteria are
non-degradation and integrity checks, and every comparison beyond them is exploratory.

- **H-R21a (NP-KAN-AFT on clinical data).** Supported iff (1) all 20 clinical fits of `b2s_npaft` give finite
  conditional and marginal IPCW IBS, Harrell C, Uno C and D-cal p, and finite test survival predictions; (2) on pbc_E2
  mean(`b2s_npaft` − `b2s_cox`) ≤ +0.005 **and** mean(`b2s_npaft` − `b2s_waft`) ≤ 0; (3) `b2s_npaft` has D-cal
  p ≥ 0.05 on at least 4 of 5 pbc_E2 splits. Secondary (pbc_E1, prothro, aids): full paired comparisons against
  `b2s_cox`, `b2s_waft`, `b2s_lnaft`, `landmark_cox` (both IBS, both C, D-cal pass counts, wins/losses).
- **H-R21b (cross-confirmed anchoring on clinical data).** Supported iff for each of the 4 clinical data sets
  mean(`b2s_anchor_npaft_cross` − `b2s_npaft`) ≤ +0.005 **and** mean(`b2s_anchor_cox_cross` − `b2s_cox`) ≤ +0.005, and
  on every split where the cross-confirmation does not confirm, the estimator equals the full-validation B2-S exactly
  (test survival and risk max |diff| = 0.0). Reported: confirmation counts (k/5), both directions' losses, channel
  masks, `b2s_anchor_lnaft_cross`, and the contrasts against Round-15 `b2s_anchor_cox` and `landmark_cox`.
- **H-R21c (C-MAPSS calibration).** Owner criterion, as stated: `b2s_npaft` or `b2s_lnaft` improves D-calibration over
  `b2s_cox` — mean D-cal χ² over the 5 splits lower **and** at least as many splits with p ≥ 0.05 — while its mean IBS
  ≤ 0.055 and mean Harrell C ≥ 0.88. Supported iff at least one of the two models meets all three. The absolute
  thresholds were stated with reference to Round-13 numbers that are not in the records (§1); the closest Round-13
  KAN result (Weibull `b2s_aft`: IBS 0.0807, Harrell C 0.8690) would meet neither, so the thresholds are kept as stated
  and this part may well fail. Registered secondary (relative): the same D-calibration improvement with mean IBS ≤
  that of `b2s_cox` + 0.005 and mean Harrell C ≥ that of `b2s_cox` − 0.01.

## 4. Execution

Tests R21-1..R21-4 (`tests/test_round21_real_npaft.py`) pass before the run. 25 jobs (20 clinical, 5 C-MAPSS) on four
single-threaded lanes, file-locked queue, resumable (`scripts/run_round21_lanes.py --lane k`); records
`reports/round21/results_lane*.jsonl`; analysis `scripts/analyze_round21.py` -> `reports/round21/analysis_summary.json`
and `docs/round21_results.md`; decisions D154 (outcome) and D155 (deliverables); ledger L48.

## 5. Execution notes (appended; rules above unchanged)

**Interruption (2026-11-03 ~00:42Z).** The container restarted with the 20 clinical jobs recorded (all ok, committed
in d7445c6) and the four running C-MAPSS jobs (about seven hours each) lost before writing a record. Mechanics changed
for the rerun only, no model or rule: (1) each C-MAPSS job checkpoints its record after every model family
(`reports/round21/ckpt/`) and resumes from it; every network is seeded at construction and trained full-batch, so a
family's fit does not depend on which families ran before (shown by the exact Round-13 reproduction of the Weibull and
Cox ensembles that now run in a different order); (2) a fifth lane so the five C-MAPSS jobs run concurrently. The
queue pointer was reset (logged in `reports/round21/queue_state.json` "interruptions"); completed jobs are skipped.

**Second interruption (2026-11-04 ~01:55Z).** The container restarted again after cmapss_0, cmapss_3 and cmapss_4 had
finished (records written); cmapss_1 and cmapss_2 lost their NP-AFT step (the full ensemble, the two half-validation
references and the cross-confirmation, about 15 hours) but kept their first three families in the checkpoints. For
the rerun every fitted ensemble and the cross-confirmation model are pickled as each finishes (`ENS_CACHE`,
`reports/round21/ckpt/cmapss_<k>_fits/`) and reloaded on resume; a test on pbc_E2 split 0 with small budgets
reproduced identical metrics, gate decisions and alternatives from the reloaded objects. Records mark reloaded parts
(`reloaded`), whose wall-clock seconds are then not fit times. Mechanics only; no model, setting or rule changed.

**Third interruption (2026-11-04 ~17:09Z).** The container restarted while the session was paused; cmapss_1 and
cmapss_2 lost only their cross-confirmation gate step. They resume with the full, h1 and h2 NP-AFT ensembles reloaded
from the cache. Mechanics only.
