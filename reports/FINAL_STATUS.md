# Final status: KAN-FAE research prototype, round 2 (2026-10-17)

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
