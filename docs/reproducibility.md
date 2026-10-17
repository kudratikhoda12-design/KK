# Reproducibility

Project instructions (the repository README.md is intentionally left unchanged, D28).


```bash
# environment (project-local; never touches the global Python)
bash scripts/setup_env.sh                          # or: KANFAE_FROM_LOCK=1 bash scripts/setup_env.sh
bash scripts/verify_env.sh                         # E-1 / E-5 evidence
# tests (every attempt is appended to reports/logs/test_records.jsonl)
KANFAE_RUN_ID=rerun .venv/bin/python -m pytest -v -rA tests/
# data (public sources; checksums in reports/run_manifest.json)
#   C-MAPSS: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
#            -> extract inner CMAPSSData.zip into data/raw/cmapss/
#   PBC:     https://cran.r-project.org/src/contrib/survival_3.8-12.tar.gz -> data/raw/pbc/{DESCRIPTION,data/pbc.rda}
# experiments (resumable; completed run IDs are skipped)
bash scripts/run_all_experiments.sh                # synthetic (5 replicates x 6 settings) + C-MAPSS (5 splits)
.venv/bin/python scripts/run_cmapss.py --splits 5 --analyses cmapss_A,cmapss_A_artificial --models sum_cox --code-version D37
.venv/bin/python scripts/run_pbc.py --splits 5
# round 2 (2026-10-17): learnability recovery (validation-only search, then gate and fresh-seed confirmations)
.venv/bin/python scripts/learnability_search.py            # D41 steps S1-S7 -> reports/learnability_search.jsonl
.venv/bin/python scripts/learnability_search_ext.py        # D46 steps S8-S11 -> reports/learnability_selected_ext.json
KANFAE_RUN_ID=gate .venv/bin/python -m pytest -v -s tests/test_learnability.py
.venv/bin/python scripts/confirm_learnability.py fae_mlp --seeds 101,102,103,104,105
.venv/bin/python scripts/confirm_learnability.py fae_mlp --selected reports/learnability_selected_ext.json \
    --seeds 106,107,108,109,110 --tag _ext
# round 2: estimators, KAN-AFT, symbolic extraction, censoring variants, pruning diagnostic, FD003
.venv/bin/python scripts/run_kanaft_symbolic.py             # synthetic known-truth KAN-AFT (resumable; --fresh)
.venv/bin/python scripts/run_censoring_variants.py          # L16 / D50
.venv/bin/python scripts/diag_pruning_threshold.py          # D54
.venv/bin/python scripts/run_kanaft_pbc.py                  # D52
.venv/bin/python scripts/run_cmapss.py --subset FD003 --code-version v2   # D51
# round 3 (2026-10-17): revision 4 (frequency-aware front), pre-registered in D61 before these runs
.venv/bin/python scripts/diag_frequency_information.py     # D60 (seed-21 training split only)
.venv/bin/python -m pytest -v tests/test_frequency_front.py # FQ-1..FQ-10
.venv/bin/python scripts/learnability_search_fa.py         # validation-only search -> reports/learnability_selected_fa.json
KANFAE_RUN_ID=revised_gate_fa .venv/bin/python -m pytest -v -s tests/test_learnability_fa.py   # fresh seed 41
.venv/bin/python scripts/confirm_learnability.py fae_mlp --selected reports/learnability_selected_fa.json \
    --seeds 111,112,113,114,115 --tag _fa
.venv/bin/python scripts/revised_gate_verdict.py           # -> reports/revised_gate_verdict.json (D61 rule)
.venv/bin/python scripts/run_neural_synthetic.py           # refuses to run unless the verdict is CONFIRMED (D57)
.venv/bin/python scripts/diag_fa_overfitting.py            # D65 (validation only)
.venv/bin/python scripts/diag_r1_seeds.py                  # D66
.venv/bin/python scripts/run_cmapss.py --subset FD001 --analyses cmapss_B --code-version v3   # D62
.venv/bin/python scripts/run_cmapss.py --subset FD003 --analyses cmapss_B --code-version v3
# report (round 3 part + round 2 part + round 1 part): replace build_report_round2.py by
.venv/bin/python scripts/build_report_round3.py
# report (round 2 part + round 1 part)
.venv/bin/python scripts/compile_test_results.py && .venv/bin/python scripts/aggregate_results.py
.venv/bin/python scripts/make_figures.py && .venv/bin/python scripts/make_figures_round2.py
.venv/bin/python scripts/build_report_round2.py
bash scripts/build_pdf.sh && .venv/bin/python scripts/build_manifest.py
```

Every executed command, with its exit code and timing, is in `reports/logs/commands.jsonl`. The full outputs
are in `reports/logs/*.log`. The resumable narrative log is `reports/logs/execution_log.md`.

