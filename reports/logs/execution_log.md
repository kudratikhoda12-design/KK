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

## 2026-10-17: Owner-authorised design revision (D38) and reference-based audit
- Pre-registered D38–D40 (nonlinear functional front end), then F6a–c and the unit regressions (50 passed, 1 skip).
- Gate with revision 3 (`rev3_gate_attempt1`, 2385 s): R1 PASS (C 0.8910), **R2 FAIL** (C 0.8461 < 0.8636;
  probe mean 0.509), R3 PASS (|ΔC| 0.0112; aware 0.089–0.090 < naive 0.136–0.140).
- The diagnostic `diag_rev3_R2` was stopped after 1 of 5 configurations (train C 0.8589, test C 0.8461:
  under-fitting); superseded by the controlled search D41.
- Audit tests AUD-1..6 (9 passed); censoring-adjusted estimators LS-1..5 (5 passed); KREG-1..4 and SYM-1..4
  (SYM-4: 1 repair, wrong test premise); F7–F8 (passed). PySR 2.7.0 installed (D43).
- References: KAN-AFT, CoxKAN and Wu et al. downloaded from arXiv (the uploaded PDFs were absent).
  Literature review in `docs/literature_review.md`.
- D41 search: S1 fronts (validation C): A0 0.8389, A1 0.8445, A2 0.8403, **A3 0.8484 (selected)**. The first
  process was killed at the 2-hour limit; resumed after the D44 efficiency fixes.
- D41 selection A3-fae_mlp-ab243462; D45 set it as DEFAULTS. Gate R2 FAIL (C 0.8497); replications 101–105: 0/5.
- D46 extension (budget, derivative-energy front A4, capacity): selection unchanged (D47). Gate R2 FAIL again;
  replications 106–110: 0/5, with seed 110 at test C 0.4913.
- Seed 110 investigated: feature-cache defect (D53). Fixed; regression test F8b; superseded records moved to
  `reports/superseded_D53/`. The search (S1 pinned to r = 4 as originally run), the gate and seeds 101–110 were
  recomputed: same selection; gate identical; seeds 101–109 identical; seed 110 test C 0.8413 (FAIL). Final:
  R2 FAILED.
- L14 interval likelihood (INT-1 test-reference repair, D48); L16 censoring variants (D50); FD003 baselines (D51);
  KAN-AFT on PBC with symbolic extraction (D52); pruning-threshold diagnostic (D54; correction D55).
- KAN-AFT symbolic validation resumed after the 2-hour limit (D49): 12/12 records.
- Worker restart interrupted a full-suite run (partial record kept). The rerun found stale M1/M3 tests after D45
  and stale E-5 evidence (D56): tests fixed, environment re-verified (E-1 and E-5 YES). Final full suite:
  185 passed, 1 failed (R2), 1 skipped.
- Report regenerated (Part I round 2 + Part II round 1): 44-page PDF; deliverable verification ALL_CHECKS_PASS.
- Round 3 (owner decisions D57–D59). D60 frequency diagnostic (seed-21 training split): ω is recoverable; only the
  quadrature-pair projection spectrum is phase-free. Frequency-aware front implemented; FQ-1..FQ-10 (repairs
  disclosed, D63). Pre-registration D61 committed (396100c) before any search fit.
- Search (11 fits, validation only): fa makes ω decodable but overfits. Revised gate, fresh seed 41: R1 FAIL (σ),
  R2 FAIL (C 0.8189), R3 PASS. Confirmations 111–115: 0/5. Verdict NOT CONFIRMED; neural analyses blocked
  (guard verified).
- D65 overfitting diagnostic (compact summary: probe 0.82 on validation, C still short). D66 R1 diagnostic
  (12 seeds; σ not converged at early stopping; profile σ fine); the diagnostic run was resumed once after a
  background time limit.
- C-MAPSS Option B v3 with engine-cluster bootstrap (FD001, FD003); point estimates reproduce earlier runs.
- Historical A3 gate pinned and marked strict xfail (D64). Final suite: 198 passed, 2 failed (revised gate),
  1 xfailed, 1 skipped. Report revision 3 (59 pages); deliverable verification ALL_CHECKS_PASS.
- Round 4 (owner decisions D67–D69). D70 overfitting curves (survival part peaks at ~360 epochs, objective at
  ~2250; latent generalises). Survival-path options + HR-1..3. Pre-registration D71 committed (63da837) before any
  4b fit. Search (11 + 2 fits); D73 corrected the round-2 wording of the D41 rule (no rule change).
- 4b gate, fresh seed 61: R1 original FAIL, R1-A PASS, R2 FAIL (C 0.8372), R3 PASS; confirmations 121–125 0/5;
  NOT CONFIRMED; neural analyses blocked.
- R1-A on fresh seeds 601–610: original 0/10, amended 10/10 (D72). Diagnoses D74 (latent ceiling), D75 (data
  ceiling; 9/10 fresh seeds below the threshold even for near-optimal estimates), D77 (training instability).
- Revision-4 gate marked historical strict xfail (D76). First round-4 suite found a FlatAE regression (D78), fixed;
  rerun: 203 passed, 2 failed (4b gate R1/R2), 3 xfailed, 1 skipped. Report revision 4 (80 pages); verification passes.
- Round 5 (owner D80). D79 determinism: identical 3-thread runs differ; all round-5 fits single-threaded. Revision
  4c mechanisms + RS tests (RS-5 caught a BatchNorm leak in the refit, fixed); C_ceiling module + CEIL tests.
  Pre-registration D81/D82 committed (4605c50) before any fit.
- Step 1 PASSED (D83; disclosed tie at H3). Step 2 seed 81: R1-A pass, R2 concordance pass / probe fail, R3 pass;
  confirmations 131-135 1/5; NOT CONFIRMED; Step 3 not run. Diagnosis D84 (early NLL stopping; seed 135 stopped at
  epoch 91). D85 gate reproduction exact in the registered environment.
- Full suite round 5: 212 passed, 2 failed, 5 xfailed, 1 skipped. Report revision 5 with per-round snapshots.
- Round 6 (owner D86). Single-threading in the run wrapper and pytest session; R1.9 snapshot; 4c gate strict xfail;
  burn-in `min_epochs` + BI tests (BI-1 exact reproduction). Seed 91 found previously generated (CEIL-2) and replaced
  by 92. Pre-registration D87/D88 committed (29ad9e7) before any 4d fit.
- Step 1 (24 fits, seed-21 train/validation, single-threaded): FAILED (D89); closest candidate short by 0.001 on
  training seed 1's probe mean. Step 2 and Step 3 not run; seeds 92 and 141–145 unused.
- `final_full_suite_round6`: 215 passed, 1 skipped recorded (223; pytest 224). Module-level skip not recorded:
  conftest collection hook added (D90); `final_full_suite_round6_rerun`: 215 passed, 0 failed, 2 skipped, 7 xfailed (224).
- First round-6 verification flagged a near-blank PDF page 2 (contents overflow); contents set in two columns (D90).
  Report revision 6 (106 pages); deliverable verification ALL_CHECKS_PASS.
- Round 7 (owner D91). Owner rationale fact-checked against the records (all cited numbers correct; 0.824 is not an
  upper bound). Pre-registration D92/D93 committed (514d011) before any 4e fit; seeds 92 and 141–145 re-verified unused.
- Step 1 (24 + 2 fits): PASSED via Tier 1, r 16 / patience 100 / α 1 (D94); 4d cand1 reproduced bit for bit;
  selection committed (3387903) right after the gate and confirmation processes were started; neither had produced a
  result yet.
- Gate seed 92: R1-A pass, R2 probe pass, R2 concordance fail (0.0022), R3 fail (0.0254); confirmations 141–145
  4/5; verdict NOT CONFIRMED (D95); guard refuses; Step 3 not run. D96/D97 diagnostic on seed-92 train/validation.
- `final_full_suite_round7`: 216 passed, 3 failed (4e gate), 2 skipped, 7 xfailed (228). Report fixes D98.
  Report revision 7; deliverable verification ALL_CHECKS_PASS.
- Round 8 (owner D99). Fact check: two corrections (no 0.025 ceiling tolerance in the protocol; RSF averaged metrics,
  not predictions); disclosed that the round-7 model passes the round-8 criteria. head_refit, SeedEnsembleAFT, head_wd
  for nonlinear heads; NF-1..4. Pre-registration D100/D101 committed (1ca8092) before any 4f fit.
- Step 1 PASSED (D102; nonlinear joint head, k 3), committed (53646ba) before the gate. Gate seed 93 + confirmations
  151–155: CONFIRMED (D103), also under round-7 criteria. Guard opened.
- Step 3 pre-registered (D104, e6f5572) and run in parallel lanes, 20:03–21:48 UTC. Lane B's wrapper was stopped after
  CONTROLS so that AB1/AB3 and AB4 ran as separate processes (no cell changed). Real data pre-registered (D105, 5a7b218),
  run 20:19–21:22. All runs ok. Outcome D106.
- Analysis fixes before reporting: the C-MAPSS comparator summary restricted to FD001; replicate-1000 symbolic rows
  shown against that replicate's own stages.
- `final_full_suite_round8`: 226 passed, 1 failed (4f original R1), 2 skipped, 10 xfailed (239). Report revision 8
  (144 pages); deliverable verification ALL_CHECKS_PASS.
- Round 9 (owner D107, closing L32). Fact check (parameter counts and controls correct; PBC invariance qualified to
  the conclusion level). Mode A / Mode B decoders, fae_mlp_wide, reconstruction metric; FC-1..FC-5 (one dev-run
  failure from a test construction error, fixed, kept). Pre-registration D108 committed (1b4b740) before any fit.
- Fairness lanes 07:28–08:58 UTC, 400 runs, all ok; S2 repeats bit-identical to H1. Outcome D109. L32 closed.
- `final_full_suite_round9` (run in parallel with the lanes): 231 passed, 1 failed (4f original R1), 2 skipped,
  10 xfailed (244). Report revision 9; deliverable verification below.
- Round 10 (owner D110). Literature review: web searches, landing-page / arXiv / Crossref verification
  (`docs/literature_review_round10.md`); gap analysis and claim policy D111. Design note and pre-registration of Rounds
  11–13 (`docs/design_note_round10.md`; D112–D114), committed (a7ff68b) before any Round-11 code. Seeds 3000–3419
  verified unused. R not installed (AFCM and the joint-model baseline will be re-implemented); JM 1.5-2 reachable on
  CRAN (not downloaded).
- Report rendering fix D115. `final_full_suite_round10`: 231 passed, 1 failed (4f original R1), 2 skipped, 10 xfailed
  (244). Report revision 10.
- Round 11 (owner "start round 11"; D112). Implemented the Cox loss (exact ties), simulator, PACE, functional KAN,
  AFCM, LME2, landmark builder, B1 features, symbolic substitution (D116). Development checks on seeds 901-907/99
  (one PACE check on reserved seed 14, disclosed). PACE σ² fix before any CK run. CK-3 operationalisation revised
  before its formal run (D116). A test filter accidentally ran CK-11 at seed 16 early (disclosed; reproduced exactly).
- Formal acceptance run `round11_acceptance`: CK-1, 2, 4, 7, 9, 10, 12 pass; CK-3, 5, 6, 8, 11 FAIL (D117).
  Identifiability diagnostic and exploratory RISE-aff (D118). Round 12 not started; owner decisions L38.
- Code committed (101ba49) before the final suite. `final_full_suite_round11` run; report revision 11 (D119).
- `final_full_suite_round11`: 241 passed, 6 failed (5 CK + 4f original R1), 2 skipped, 10 xfailed (259); CK records
  identical to the formal run. Report revision 11 (188 pages); deliverable verification ALL_CHECKS_PASS.
- L38 decisions (owner D120, 2026-10-21). DGP v2 (v1 pinned and verified bit-identical), identifiable RISE (tolerance
  placed in the spectrum gap on seed 902), AFCM λ_t grid, converged CK-3b, integrated channel pruning, mixture CK-11b.
  Development dry-runs on seeds 908–915 (results stated in D120 before the formal runs). D120 committed (6632a33).
- Formal repaired tests `round12_ck_acceptance`: CK-3b, CK-5b, CK-8b, CK-11b pass; CK-6b FAIL (D121). Round-12 gate
  closed; seeds 3000–3419 unused. Exploratory AFCM basis floor 0.002. L39 open. D121 committed (abebea6).
- `final_full_suite_round12`: 247 passed, 2 failed (CK-6b, 4f original R1), 2 skipped, 15 xfailed (266); CK records
  reproduced. Report revision 12 (199 pages); deliverable verification ALL_CHECKS_PASS. Round 12 not run (gate).
- L39 decisions (owner D123). AFCM options added (defaults unchanged). Diagnosis on development seed 914 in 4 lanes
  (a first launch writing one shared JSON was stopped after seconds and relaunched): V3 adopted (φ₁ 0.023). Seeds 22/23
  verified unused; seed 21 found used in Rounds 4–6 for the Round-1 simulator (correction to D120). CK-7b mechanics on
  seed 916. D123 committed (08292df).
- Formal run `round12b_ck_acceptance`: CK-7b PASS; CK-6c FAIL (φ₁ 0.247). Gate closed; Round 12 not run (D124). L40.
- `final_full_suite_round12b`: 248 passed, 2 failed (CK-6c, 4f original R1), 2 skipped, 16 xfailed (268); all CK
  records reproduced. Report revision 13 (206 pages; contents type reduced, D125); verification ALL_CHECKS_PASS.
- L40 closed (owner D126); exploratory λ_t refit on the CK-6c data (φ₁ 0.133, φ₂ 0.096). Round-12 harness built,
  dry-run on development seeds 931–935; plan committed (d995c5f) before any Round-12 seed. Launched 19:12 UTC.
- Three pruning/symbolic cells failed (non-finite refits); harness fix committed (0eddcc6). Container restart at about
  04:27 UTC; interrupted jobs resumed; all 160 jobs done 06:49 UTC; the three failed cells rerun with the fix.
- Analysis: H-R12 inconclusive (+0.0065, CI [−0.051, +0.064]); P0 equivalent. D127. L41 opened.
- `final_full_suite_round12c` killed by a container restart; rerun as `final_full_suite_round12c_rerun`: 248 passed, 2 failed
  (CK-6c, 4f R1), 2 skipped, 16 xfailed (268); CK records reproduced. Report revision 14 (236 pages), ALL_CHECKS_PASS.
- Round 13 (owner D128 approved; plan D129). JM 1.5-2 acquired; Round-13 data layer, NaN-aware helpers, harness;
  dry runs on synthetic data only; plan committed (9380ef7) before any real-data fit. 70 jobs on 4 workers; one failed
  cell (PACE sensitivity, PBC E2 split 1). Analysis D130: no predictive gain; interpretability measures informative.

- `final_full_suite_round13`: 251 passed, 2 failed (CK-6c, 4f R1), 2 skipped, 16 xfailed (271; no XPASS). Report
  revision 15 (255 pages); verification ALL_CHECKS_PASS.
- Round 14 (owner D131, 2026-10-23): manuscript draft generated from saved records (`scripts/manuscript_round14.py`;
  ledger of 1361 values; independent verifier ALL_PASS; tests R14-1..6); descriptive cohort record; verified references.
  `final_full_suite_round14` stopped deliberately after 10 tests (manuscript accuracy pass first);
  `final_full_suite_round14_rerun` stopped by the 2-hour background limit after 207 records (container ~44% slower);
  `final_full_suite_round14_rerun2` (detached): 257 passed, 2 failed (CK-6c, 4f R1), 2 skipped, 16 xfailed (277), no XPASS.
  Report revision 16 (283 pages), copied to `final_research_prototype_report_rev16.pdf`; verification ALL_CHECKS_PASS. D132.
