# Execution log (resumable)

Human-readable log of the autonomous staged run (D21). Machine-readable command records are in
`reports/logs/commands.jsonl`; full outputs are in `reports/logs/*.log`.

## 2026-10-16 — Session start
- Repository inspected: branch `claude/blissful-faraday-2unozw`, clean tree, HEAD `f54df73` (Initial commit;
  only `README.md`). README left unchanged (D28).
- Resources: 4 CPUs, 15 GB RAM, ~29 GB free disk, no GPU. Tools: pandoc, pdftotext, pdftoppm, Chromium.
- Global `pip list` snapshot saved before any install (89 lines) for criterion E-1.
- Draft documents copied into `docs/` and revised (revision 2): D14-R..D28 recorded in `docs/decisions.md`.

## Stage A — Environment and survival likelihood: PASSED (attempt 3)
- `scripts/setup_env.sh` built `.venv` (exit 0, 116.6 s): torch 2.14.1+cpu, pandas 2.3.3, lifelines 0.30.3,
  scikit-survival 0.28.0, rdata 1.1.0 (added for Stage G; recorded in D29). `pip check`: no broken requirements.
- `scripts/verify_env.sh` (exit 0): E-1 global pip list unchanged (empty diff); E-5 throwaway venv from the lock
  file reproduces `pip freeze` exactly.
- Tests attempt 1: 41 passed / 4 failed.
  - A2 (x3): NaN from `quad` over (-inf, inf) — `exp(u)` overflow in the test integrand (harness bug).
  - A7 log-logistic: our max log-lik exceeded lifelines' by 7.8e-4 (reference optimiser not converged).
- Repair 1: A2 finite range [-700, 700]; A7 `gtol` for lifelines + diagnostic (our likelihood at lifelines'
  estimates reproduces lifelines' value within 1e-8).
- Attempt 2: 41 passed / 4 failed — A2 quadrature missed narrow peak (sigma=0.2); A7 SLSQP ignores `gtol`,
  BFGS (log-normal) raised ConvergenceError with gtol=1e-10.
- Repair 2: A2 integrate in w = (u - eta)/sigma over [-60, 60] with breakpoints; A7 `ftol=1e-12` for SLSQP only.
- Attempt 3: **45 passed, 0 failed** (log `reports/logs/stageA_tests_attempt3.log`). No threshold changed.
  The likelihood implementation (`src/kanfae/survival/aft.py`) was not modified by either repair.

## Stage B — Synthetic data: PASSED (attempt 1)
- `src/kanfae/data/schema.py` (Observed/Oracle, `to_training_arrays` rejects Oracle), `src/kanfae/data/synthetic.py`.
- `tests/test_synthetic.py`: **40 passed, 0 failed** (log `reports/logs/stageB_tests_attempt1.log`).
  STAT outcomes at pre-specified seeds: G3 realised censoring 0.0000 / 0.3027 / 0.5935 (targets 0/0.3/0.6);
  G8 KS p = 0.583 (Weibull), 0.788 (log-normal), 0.583 (log-logistic); G9 corr = -0.0034; G10 max dev 0.00175;
  G12 masked fraction 0.2002.
- Note: identical KS p-values for Weibull and log-logistic are expected: numpy draws both by inverse CDF from the
  same uniform stream (common random numbers) and the KS statistic is invariant under U -> 1-U reflection.

## Stage C — Functional representation and KAN layer: PASSED (attempt 1)
- `src/kanfae/functional/{basis,smoothing,fpca,layers}.py`, `src/kanfae/models/kan.py`.
- `tests/test_functional.py` + `tests/test_kan.py`: **19 passed, 0 failed** (log `reports/logs/stageC_tests_attempt1.log`).
- F3 relative changes (n = 25→50→…→800): h 6.3e-4, 1.5e-4, 3.9e-5, 9.6e-6, 2.4e-6; FPCA 4.4e-4 … 1.7e-6 (≈O(1/n²)).
- The in-house KAN layer has now passed its own tests F-KAN-1..4 (and only from this point is it described as tested).

## Stage D — Models and learnability gate: FAILED at R2 (after 2 repairs); unit criteria PASSED
- Implemented: `src/kanfae/data/{splits,preprocess}.py`, `src/kanfae/models/{fae,pipelines}.py`,
  `src/kanfae/evaluation/metrics.py` (needed by the gate; validated first by Q1..Q5).
- Design corrections before any training: D30 (KAN BatchNorm input normalisation; single-layer KAN blocks),
  D31 (automatic parameter matching with bottleneck-avoiding tie-breaks), D32 (M2 representable targets).
- Unit tests (`reports/logs/stageCDE_unit_final.log`): **55 passed, 1 skipped** (M2 linear AE — not applicable, D32).
  Q1 needed 2 fixture/support repairs (`stageE_metric_tests_attempt{2,3}.log`): the scikit-survival reference cannot
  evaluate the training censoring KM beyond the largest training time; our IBS now evaluates G(min(Y_i, t_last)),
  which is mathematically exact.
- Learnability gate (STAT, seed 21):
  - attempt 1 (`stageD_gate_attempt1.log`): R1 PASS (C 0.8910 vs oracle 0.8936; sigma 0.4816); R2 FAIL (C 0.7545,
    probe R2 a/b/A/omega 0.861/0.711/0.000/0.022); R3 FAIL (|dC| 0.0313; aware < naive at all seeds).
  - diagnostics: `diag_R2_budget.log` (epoch cap always hit; KAN 0.837 vs MLP 0.805 at 3000 epochs, single seed,
    gate data only — DIAGNOSTIC, not a comparison), `diag_R2_capacity.log` (train C ≈ test C ≈ 0.79–0.84: under-fit;
    capacity plateau ≈ 0.836).
  - repair 1 (training budget 400→3000 epochs, patience 30→100): R2 FAIL (C 0.8050), R3 FAIL (|dC| 0.0247).
  - repair 2 (capacity H 16→32, hidden 32→64; KAN re-matched +1.2%): **R1 PASS, R2 FAIL (C 0.8361 < 0.8636; mean
    probe R2 0.471 < 0.8), R3 PASS** (|dC| 0.0042; curve error aware 0.0958/0.0976/0.0998 < naive 0.1472/0.1451/0.1487).
- Root cause (interpretation): the risk depends on the amplitude A of a random-phase sinusoid, a phase-invariant
  (quadratic) functional; the architecture's first layer computes only linear functionals ∫β x dt, and the networks
  after it under-fit this mapping at N = 1200. No threshold was changed.
- Consequence (pre-registered rule, protocol §1.4; D33): the KAN-vs-MLP comparison (H1) and all neural-autoencoder
  experiments are **BLOCKED**. Independent branches continue with validated baselines.

## Stage E (unblocked part) — synthetic baseline experiments: COMPLETED
- First run stopped after a bug was found (D36, linear-AFT penalty scaling); its 491 records are kept in
  `reports/logs/superseded_synthetic_results_penalty_scaling_bug.jsonl` and not used.
- Re-run `exp_synthetic_baselines_v2` (exit 0, 721 s): 6 settings × 5 replicates × 5 baselines + km_null + oracle.

## Stage F — C-MAPSS: COMPLETED for baselines (neural BLOCKED)
- Source: PHM Society mirror (S3 zip), SHA-256 recorded; inspected before writing the loader
  (`cmapss_inspection.json`); K3/K4 tests passed.
- `exp_cmapss_baselines` (exit 0, 1050 s): Options A, B, A+ARTIFICIAL censoring, 5 engine-level splits.
- D37 bug (constant feature column → lifelines NaN) affected sum_cox on Option A splits 1–4 (16 failed tuning
  records, kept); fixed and re-run (`exp_cmapss_sumcox_rerun_D37`, exit 0).
- Summary+AFT on Option A + artificial censoring: all 4 configs diverged in one split (recorded `diverged`);
  4 of 5 splits contribute.

## Stage G — PBC: COMPLETED for baselines (neural BLOCKED)
- Source: CRAN survival 3.8-12 (SHA-256 recorded); provenance/outcome/K1/K4 tests passed (4).
- `exp_pbc_baselines` (exit 0, 105 s): E1 and E2 as separate sensitivity analyses, 5 patient-level splits,
  marginal and conditional IPCW IBS, visit-process descriptive Cox.

## Final consolidated test suite
- `final_full_test_suite` (exit 1, 1622 s): 148 passed, 1 failed (R2, reproduced exactly: C 0.8361), 1 skipped.

## Report
- Built from saved results by `scripts/{compile_test_results,aggregate_results,make_figures,build_report}.py`,
  PDF by `scripts/build_pdf.sh` (pandoc → Chromium), verified by `scripts/verify_deliverables.py`.
- Final status written to `reports/FINAL_STATUS.md`; deliverable verification ALL_CHECKS_PASS=true.
