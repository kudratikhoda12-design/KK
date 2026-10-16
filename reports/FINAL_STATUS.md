# Final status: KAN-FAE research prototype (2026-10-16)

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
