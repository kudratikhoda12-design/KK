# Round 23 pre-registration (D160): where do functional KANs help, and can a blend keep the best of both?

Owner instruction (2026-11-09): "try different methods to overcome the issue and get better result". The issue
(D154, D158, journal §5.2): on every clinical cohort no functional model was clearly better than landmark Cox on the
last observed values. This plan was written and committed **before any formal job of `scripts/run_round23.py` was
run**. The rules below are not changed after results are seen; nothing is refitted, re-split or re-selected because
of a result.

## 1. Diagnosis from the existing records (descriptive; known before this plan)

| Data | Visits before the landmark (median) | Functional vs landmark Cox (earlier rounds) |
|---|---|---|
| pbc_E2, pbc_E1, prothro, aids | 3 | landmark Cox lowest or near-lowest mean IBS (D154) |
| heart_valve | 1 | no difference (D158) |
| renal | 97 (GFR) | b2s_npaft − landmark_cox −0.0096 [−0.0329, +0.0137] (D158); every B2-S model's mean below landmark Cox |
| C-MAPSS FD003 Option B | 30-cycle windows | b2s_npaft below landmark Cox on 5 of 5 splits, mean −0.0187 (D158) |

Two readings follow. (i) Functional models may help when the history is long and lose nothing to a landmark model
otherwise; this was stated in the Round-22 discussion but **not tested**. (ii) Power: the Nadeau–Bengio correction
multiplies the paired variance by (1/J + n_test/n_train) ≈ (1/J + 1/3), so more splits help only a little; with J = 5
even the FD003 pattern above would not give an interval excluding 0 (upper bound ≈ +0.001), with J = 10 it would
(≈ −0.004). Round 23 therefore tests (i) directly and uses J = 10 (C-MAPSS) and J = 20 (clinical) splits.

## 2. Data, partitions and what has been seen

- **C-MAPSS FD002 and FD004** (six operating conditions; 260 and 249 training engines) have **never been analysed** in
  this project (ledger L15 "NOT RUN"; no records mention them). Before this plan only their **inputs** were inspected
  (no model was fitted, no outcome looked at): k-means on the training engines' settings of split 0 finds six
  conditions (FD002 training rows per condition 4,883–8,253; FD004 5,535–9,376), 17 sensors pass the
  training-variance rule after per-condition z-scoring, and Option B gives 2,883 / 3,366 training windows on split 0.
  Outer engine splits k = 0..9 (`split_groups(units, k)`, 60/20/20).
- **Clinical cohorts** pbc_E2, pbc_E1, prothro, aids, renal, heart_valve: **all analysed before** (Rounds 13–22, outer
  partitions k = 0..4 only; checked: no real-data record has k > 4). Round 23 uses **new partitions k = 5..24**
  (`split_groups(ids, k, strata=event)`). These are re-analyses of previously analysed data with new partitions,
  **not prospective holdouts**, and are reported as such.
- **Development (mechanics only, shrunk budgets: lr 3e-3, wd 1e-4, one seed)** on partitions already used:
  real_pbc_E2_0, real_renal_0 and cm_FD003_0 (FD003 with one operating condition). Their numbers were seen; no setting
  was changed after them. Seen: pbc_E2 split 0 conditional IBS NP 0.0988, landmark 0.1026, blend w = 1 (0.1026);
  renal split 0 NP 0.1540, NP thinned to three visits 0.1807, landmark 0.1751, blend w = 0.25 (0.1551); FD003
  split 0 (windows every 10 cycles; one setting, one seed) IBS NP 0.0858, landmark (raw chosen) 0.1044, smoothed
  landmark 0.1013, blend w = 0.25 0.0853, sum_aft 0.0888, fpca_aft 0.0886. The FD003 run also measured the cost:
  one NP-KAN-AFT fit took 1,718 s on 1,348 training windows; the rank loss is quadratic in the number of windows,
  which fixed the C-MAPSS budget in §3 (decided from run time only, before any FD002/FD004 fit).

## 3. Models (fixed; `scripts/run_round23.py`)

- **b2s_npaft**: the Round-20 NP-KAN-AFT B2-S ensemble (rank_km, τ 0.05), D112 protocol (lr {1e-3, 3e-3} × wd
  {1e-4, 1e-3} on validation loss, 3 seeds). Chosen in advance as the functional model because it is the Round-20–22
  headline model and does not require choosing an error family; no other KAN family is fitted (no model selection
  across families).
- **landmark_cox**: Cox on the last observed values plus baseline covariates (clinical: as Rounds 13–22). C-MAPSS:
  two versions, last **raw** sensor values + u and last value of the **smoothed** curves + u; per split the one with
  the lower validation IBS is "landmark_cox" (so the comparator is not handicapped by sensor noise).
- **stack (blend)**: S = w·S_landmark + (1 − w)·S_npaft, w ∈ {0, 0.25, 0.5, 0.75, 1} chosen on the validation set
  (clinical: conditional IBS; C-MAPSS: marginal IBS); ties go to the larger w (the simpler model). Risk score for
  concordance: minus the mean blended survival over the test IBS grid.
- **b2s_npaft_sparse3** (renal only): the same NP-KAN-AFT after thinning every history to three observed visits
  (first, middle, last in the window, by visit index); the B-spline front end is refitted on the thinned histories;
  same subjects, partitions and metrics.
- **C-MAPSS budget (cost, fixed before any FD002/FD004 fit):** Option-B windows every **20** cycles (not 10; FD002 /
  FD004 split 0 give 1,480 / 1,714 training windows, close to FD003's 1,348 at step 10, against 2,883 / 3,366 at step
  10), and NP-KAN-AFT at the **fixed setting lr 3e-3, wd 1e-4** with seeds 0, 1, 2 (no lr × wd tuning; this was the
  setting chosen most often for NP-KAN-AFT on C-MAPSS in Rounds 21–22, 5 of 10 splits). Each seed's fit is pickled as
  it finishes (the assembled ensemble equals `FKANEnsemble` at that setting; test R23-5). All C-MAPSS models use the
  same windows.
- C-MAPSS also: **sum_aft**, **fpca_aft** (Round-1 grids tuned on validation IBS; as Rounds 13 and 22).
- C-MAPSS preprocessing (FD002/FD004): per-operating-condition z-scoring (`cmapss.condition_normalise`: k-means with
  six clusters, seed 0, n_init 10, fitted on the training engines' three settings; per-condition training mean and
  SD; constant-in-condition sensors set to 0), then the training-variance sensor rule and the Round-13 Option-B
  pipeline (30-cycle windows every 10 cycles, u = s/100, target T − s, every window an event, B-spline λ 1e-4).

## 4. Hypotheses and decision rules

Intervals: Nadeau–Bengio corrected 95% t-intervals over the splits (`analyse_round13.nb`, frac = mean
n_test/n_train of the cohort). Primary metric: test IBS (C-MAPSS marginal IPCW IBS; clinical conditional IPCW IBS,
as Rounds 13–22).

- **H-R23a (confirmatory, untouched data).** In **each** of FD002 and FD004 (10 splits), the mean of
  b2s_npaft − landmark_cox test IBS is below 0 with NB upper bound < 0. Supported if both data sets meet it.
  Descriptive: b2s_npaft − sum_aft, − fpca_aft; stack − landmark_cox; Harrell C; D-calibration passes; chosen landmark
  version; chosen w.
- **H-R23b (mechanism; renal re-analysis, 20 new partitions).** Mean b2s_npaft − b2s_npaft_sparse3 conditional IBS
  below 0 with NB upper bound < 0 (removing history makes the functional model worse). Descriptive: b2s_npaft −
  landmark_cox and b2s_npaft_sparse3 − landmark_cox intervals.
- **H-R23c (blend; clinical re-analysis, 20 new partitions).**
  (i) non-inferiority: on **every** clinical cohort the NB upper bound of stack − landmark_cox mean conditional IBS
  ≤ +0.005 (the Round-20/21 anchoring margin);
  (ii) superiority on renal (pre-specified: the only clinical cohort with long histories): NB upper bound of
  stack − landmark_cox < 0. Parts (i) and (ii) are reported separately. Descriptive on every cohort: b2s_npaft −
  landmark_cox, w distribution, Harrell C.
- **Failed fits.** Every failed job or model fit is reported with its error. A contrast is computed on the splits
  where both models are finite; a hypothesis is **not evaluable** (not "supported") if any model it needs failed on
  more than 2 of its splits (C-MAPSS: of 10; clinical: of 20). A failed fit is never refitted with other settings.
- **Not done:** no threshold, margin, grid, partition or model is changed after results are seen; no further splits
  are added; no other KAN family is substituted; no superiority claim beyond the rules above (D111).

## 5. Execution

140 jobs (20 C-MAPSS, 120 clinical) on four single-threaded lanes from one shared queue
(`scripts/run_round23_lanes.py`), C-MAPSS jobs first; resumable (completed jobs skipped; each C-MAPSS NP-KAN-AFT seed
fit is pickled when it finishes so a restarted job reloads it; a reloaded fit is the deterministic fit a refit would
produce). Expected run time about 12 hours. Tests R23-1..R23-5 (`tests/test_round23.py`) pass before the launch. Relaunches after container restarts are logged here with their time and change nothing.
Analysis: `scripts/analyze_round23.py` (written before the run ends; it reads the records only).
