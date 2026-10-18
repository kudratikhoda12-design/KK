# Final status: KAN-FAE research prototype, round 5 (2026-10-18)

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (Part A = round 5; Parts B–E = rounds 4, 3, 2, 1, each from its
own snapshot); deliverable verification passes. Status: **research prototype; no successful KAN-FAE model; not
publication-ready; no novelty claim.**

| Step | Result |
|---|---|
| Determinism (D79) | The 0.0053 difference explained (thread-count floating point, amplified by instability; 3 threads non-deterministic). All round-5 fits single-threaded |
| Step 1: Revision 4c (D81) | **PASSED**: spectral balance + clipping (Adam 3e-3) + NLL stopping; zero jumps on seeds 0–2; stopping conflict resolved. Full-budget jumps 12 → 0; thread spread 0.013 → 0.0006 |
| Step 2: amended gate, seed 81 (D82) | R1 original FAIL (reported); R1-A PASS; R2 concordance part PASS (C 0.8421 ≥ 0.8390); **R2 probe part FAIL** (0.772 < 0.78); R3 PASS |
| Confirmations 131–135 | **1/5** → **NOT CONFIRMED** |
| Step 3 | **NOT RUN** (guard) |
| Diagnosis (D84, D85) | NLL early stopping ends training at the survival optimum (epochs 91–430) before the latent learns b and A; the registered tie-break chose it over the identical-risk two-checkpoint rule; the early-epoch latent probe is numerically fragile (±0.01–0.02) |
| Tests | `final_full_suite_round5`: 212 passed, 2 failed (4c gate original R1, R2), 5 expected failures (historical gates), 1 skipped (220). Gate reproduction in the registered environment: exact |
| Reporting fixes | xfail and total columns in every run table; historical parts rendered from their own snapshots; round-1 C-MAPSS tables without FD003/v3 |

## Decision requested

Pre-register two-checkpoint stopping with a burn-in before NLL-based selection, chosen by robustness across
training seeds, and evaluate it on new fresh seeds under the unchanged D82 criteria (L27).

---

# Round 4 status (2026-10-18), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (80 pages; Part A = round 4; Parts B–D = rounds 3, 2, 1);
deliverable verification passes. Status: **research prototype; no successful KAN-FAE model; not publication-ready;
no novelty claim.**

| Item | Result |
|---|---|
| Revision 4b (pre-registered D71) | Compact frequency-aware front + head weight decay 1e-2 (`A5-fae_mlp-8e531fd3`) |
| Revised gate, fresh seed 61 | R1 original FAIL (σ̂ 0.581, reported as is); R1-A PASS (profile σ̂ 0.492); **R2 FAIL** (C 0.8372 < 0.8662; probe 0.755); R3 PASS |
| Confirmations 121–125 | **0/5** → **NOT CONFIRMED**; H1/H2/H3/controls/symbolic extraction for KAN-FAE remain BLOCKED (guard verified) |
| Why (D70, D74, D75, D77) | Survival-path overfitting after ~360 epochs; the frozen latent lacks risk information (an MLP head and the true risk formula do not help); **R2's concordance criterion is at the data's information limit**: near-optimal estimates + true risk formula fall below oracle − 0.03 on 9/10 fresh seeds (b, blurred by GP noise, is limiting); frequency-aware variants train unstably |
| R1 amendment (D68/D72) | Fresh seeds 601–610: original check 0/10, R1-A 10/10 (two different procedures; nothing relabelled) |
| PBC (D69) | E2 provisional primary, E1 sensitivity; justification for the professor: `docs/pbc_endpoint_justification.md`; no competing-risk claims |
| Tests | Final suite `final_full_suite_round4_rerun`: 203 passed, 2 failed (4b gate original R1 and R2), 3 expected failures (historical gates), 1 skipped. A regression found by the first round-4 suite run (D78) was fixed; both runs kept |

## Decisions requested

1. Re-reference R2's concordance criterion to a data-attainable benchmark (or change the simulator noise), prospectively, with fresh seeds.
2. Approve a training-stability study of the frequency-aware front.
3. Confirm the PBC endpoint (E2) with the professor.

---

# Round 3 status (2026-10-17), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction). Report:
`reports/final_research_prototype_report.pdf` (59 pages; Part A = round 3; Parts B/C = rounds 2/1, historical);
all deliverable checks pass (`reports/logs/deliverable_verification.json`).

## Revised learnability gate (owner option (c), D57; pre-registered D61)

| Item | Result |
|---|---|
| New design | Frequency-aware front "fa" (phase-invariant projection spectrum, log gap, derivative energies, soft frequency); FQ-1..FQ-10 PASS |
| Selected (validation only, seed 21) | `A5-fae_mlp-b36715d7` (fa, 17 frequencies, r = 12) |
| Revised gate, fresh seed 41 | R1 FAIL (σ̂ 0.571; C passes); **R2 FAIL** (C 0.8189 < 0.8656; probe mean 0.716; **ω probe 0.865**, was ≈ 0); R3 PASS |
| Fresh confirmations 111–115 | **0/5** |
| Verdict | **NOT CONFIRMED**; H1/H2/H3/controls/ablations/KAN-FAE symbolic extraction remain BLOCKED (guard in code) |
| Diagnosis | D65: the spectral block overfits (train–validation C gap 0.06–0.15); a compact 15-feature summary meets the probe criterion on validation (0.82) but C stays ~0.035 short. D66: R1's σ check fails on 8/12 seeds because σ has not converged at early stopping (profile σ̂ within ±0.05 on 12/12) |

## Owner decisions applied

- PBC (D58): E2 provisional primary, E1 sensitivity; interpretation and censoring assumptions stated; E3 future work.
- C-MAPSS (D59/D62): Option B primary with engine-cluster bootstrap CIs, engine-weighted RMSE and band metrics
  (FD001, FD003; v3 reproduces earlier estimates to ≤ 3e-7); Option A secondary.

## Tests

Final full suite (`final_full_suite_round3`): 198 passed, 2 failed (revised gate R1 and R2: the live, reported
status of the current design), 1 expected failure (historical A3 gate, D64), 1 skipped (documented N/A).

## Decisions requested

1. Revision 4b (compact frequency summary + a fix for the survival-part generalisation gap), with its own
   pre-registration and fresh seeds, or stop.
2. Amend R1's σ check (e.g., profile MLE given the trained head) or keep it as registered.
3. Confirm E2 with the professor.

---

# Round 2 status (2026-10-17), historical

Everything is committed locally on branch `claude/blissful-faraday-2unozw`. **Nothing was pushed** (owner
instruction). The global Python environment is unchanged (E-1, re-verified). The lock file reproduces exactly in a
fresh venv (E-5, re-verified). Full report: `reports/final_research_prototype_report.pdf` (44 pages; all
deliverable checks pass, `reports/logs/deliverable_verification.json`).

## Learnability gate (final)

| Item | Result |
|---|---|
| Selected configuration (validation-only controlled search, D41 + D46; 17 fits) | `A3-fae_mlp-ab243462` (tensor-spline front, r = 8, α = 1, Adam 3e-3) |
| Official gate, seed 21 | R1 PASS (C 0.8910); **R2 FAIL** (C 0.8497 < 0.8636; probe mean 0.663 < 0.8; ω probe 0.014); R3 PASS |
| Fresh replications | D41 seeds 101–105: **0/5**; D46 seeds 106–110: **0/5** |
| Verdict | **R2 FAILED (pre-registered; no further extension).** H1–H3, controls, ablations and neural real-data runs remain BLOCKED |

## Completed in round 2

| Item | Status | Evidence |
|---|---|---|
| Audit (equations → code → tests; findings) | COMPLETED | `docs/audit_report.md` |
| Targeted literature review; reference use | COMPLETED; no novelty claim | `docs/literature_review.md` |
| BJ / IPCW / transform estimators (separate variants) | COMPLETED; LS-1..5 PASSED | `src/kanfae/survival/censored_ls.py` |
| Censoring-assumption / misspecification study (L16) | COMPLETED | `reports/censoring_variants.jsonl` |
| KAN regularisation and pruning | COMPLETED (KAN-AFT); pruning removed an edge in 1/22 fits (the true null); D54 diagnostic | `reports/pruning_threshold_diag.jsonl` |
| Symbolic extraction (linear → library → PySR; R² > 0.99) | COMPLETED for KAN-AFT (synthetic: 34/47 edges accepted, true shapes recovered; PBC: 41/80) | `reports/kanaft_symbolic.jsonl`, `reports/kanaft_pbc.jsonl` |
| KAN-AFT vs linear AFT on PBC | COMPLETED; no consistent difference | `reports/kanaft_pbc.jsonl` |
| Interval likelihood (L14) | COMPLETED; INT-1..3 PASSED (after a test-reference repair, D48) | `tests/test_interval_likelihood.py` |
| C-MAPSS FD003 baselines (L15) | COMPLETED | `reports/experiment_results.jsonl` |
| Feature-cache defect (D53) | FOUND, FIXED, regression test F8b; affected results recomputed | `reports/superseded_D53/` |
| Stale unit tests after D45 (D56) | FIXED | `reports/logs/models_tests_D56.log` |
| Final full test suite | 185 passed, 1 failed (R2), 1 skipped (documented N/A) | `reports/logs/final_full_suite_round2_v3.log` |

## Failed

- Learnability gate R2 (above).

## Blocked (by R2)

- H1 (KAN vs MLP), H2 (α), H3 (functional vs flat), training controls, ablations AB1–AB8, neural models on
  C-MAPSS/PBC, pruning and symbolic extraction for KAN-FAE.

## Not run

- C-MAPSS FD002/FD004: they need operating-condition normalisation (new preprocessing design).
- Interval likelihood in an experiment: implemented and tested only.

## Owner decisions requested

1. How to proceed after the R2 failure (report I.15, options a–d).
2. PBC endpoint (E1 / E2 / competing risks).
3. C-MAPSS landmark design (A / B).

---

# Round 1 status (2026-10-16), historical

Everything is committed locally on branch `claude/blissful-faraday-2unozw`. Nothing was pushed to any remote
(owner instruction, D21). The global Python environment is unchanged (E-1).

## Completed

| Item | Status | Evidence |
|---|---|---|
| Revised design documents (corrections 1–6) | DONE | `docs/design.md`, `docs/experiment_protocol.md`, `docs/decisions.md` (D0–D37) |
| Stage A: environment, lock file, AFT likelihoods | PASSED (attempt 3; 2 repairs, both to the test harness or reference optimiser) | `reports/logs/stageA_*.log` |
| Stage B: synthetic generator | PASSED (40/40) | `reports/logs/stageB_tests_attempt1.log` |
| Stage C: functional representation and in-house KAN layer | PASSED (19/19; re-passed after D30) | `reports/logs/stageC_*.log`, `stageCDE_unit_final.log` |
| Stage D: model unit criteria (M1–M5, K1–K2) | PASSED (1 documented not-applicable skip, D32) | `reports/logs/stageCDE_unit_final.log` |
| Stage E: evaluation module (Q1–Q5) | PASSED (Q1 needed 2 fixture/support repairs) | `reports/logs/stageE_metric_tests_attempt3.log` |
| Synthetic experiments, validated baselines (6 settings × 5 replicates) | COMPLETED | `reports/logs/exp_synthetic_baselines_v2.log` |
| C-MAPSS FD001: Options A, B and A with ARTIFICIAL censoring (baselines) | COMPLETED | `reports/logs/exp_cmapss_baselines.log`, `exp_cmapss_sumcox_rerun_D37.log` |
| PBC: E1 and E2 sensitivity analyses (baselines) | COMPLETED | `reports/logs/exp_pbc_baselines.log` |
| Final consolidated test suite | RUN: 148 passed, 1 failed (R2), 1 skipped | `reports/logs/final_full_test_suite.log` |
| Report, PDF and verification | COMPLETED; all deliverable checks pass | `reports/logs/deliverable_verification.json` |

## Failed

- **Learnability gate R2** (Stage D) failed after the two permitted repairs (training budget; capacity).
  - Required: test C ≥ 0.8636.
  - Achieved: 0.8361.
  - Required: mean linear-probe R² ≥ 0.8.
  - Achieved: 0.471.
  - R1 and R3 passed. No threshold was changed.

## Blocked (pre-registered rule, D33)

- H1 (KAN vs parameter-matched MLP functional autoencoder: the primary question), H2 and H3.
- All ablations AB1–AB8.
- All neural-autoencoder runs on synthetic data, C-MAPSS and PBC.
- **To unblock, the owner needs to choose one of:**
  - (a) revise the design (e.g. the deferred `∫F(x(t),t)dt` variant, or other phase-invariant functional
    features) and re-run the gate;
  - (b) explicitly authorise an exploratory, non-confirmatory neural comparison;
  - (c) revise R2 (owner only).

## Unresolved owner decisions

- **PBC endpoint (D9-R):** E1 and E2 were both analysed and labelled; neither is declared definitive. E3
  (competing risks) is not implemented.
- **C-MAPSS landmark design (D3b-R):** Options A and B were both analysed and labelled.

## Bugs found and fixed during the run (all affected results re-run; superseded records kept)

- **D36:** linear-AFT penalty scaling. The superseded synthetic records are in
  `reports/logs/superseded_synthetic_results_penalty_scaling_bug.jsonl`.
- **D37:** constant feature columns in the Cox baselines. The superseded records are flagged in
  `reports/experiment_results.csv` (`used_in_summary = False`).

## Deliverables

| File | Location |
|---|---|
| PDF report (27 pages, A4) | `reports/final_research_prototype_report.pdf` |
| Markdown report | `reports/final_research_prototype_report.md` (HTML: `reports/final_research_prototype_report.html`) |
| Test results, every attempt | `reports/test_results.csv` |
| Test results, latest per test | `reports/test_results_latest.csv` |
| Experiment results (1668 records) | `reports/experiment_results.csv` (source: `reports/experiment_results.jsonl`) |
| Run manifest (environment, hardware, git, data SHA-256, seeds, configs, 30 logged commands) | `reports/run_manifest.json` |
| Logs | `reports/logs/` (`execution_log.md`, `commands.jsonl`, `test_records.jsonl`, `*.log`, PDF page renders) |
| Figures | `reports/figures/` |
| Summary tables | `reports/tables/` |
| Configurations | `configs/` |
| Reproduction instructions | `docs/reproducibility.md`; report §15 |

The prototype is **not publication-ready**. No literature review was performed, and the key comparison was not
run.
