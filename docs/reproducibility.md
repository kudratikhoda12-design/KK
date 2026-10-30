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
# round 4 (2026-10-18): Revision 4b, pre-registered in D71 before these runs
.venv/bin/python scripts/diag_overfit_curves.py            # D70 (seed-21 train/validation only)
.venv/bin/python -m pytest -v tests/test_survival_regularisation.py   # HR-1..HR-3
.venv/bin/python scripts/learnability_search_fa4b.py       # -> reports/learnability_selected_fa4b.json
KANFAE_RUN_ID=revised_gate_fa4b .venv/bin/python -m pytest -v -s tests/test_learnability_fa4b.py   # fresh seed 61
.venv/bin/python scripts/confirm_learnability.py fae_mlp --selected reports/learnability_selected_fa4b.json \
    --seeds 121,122,123,124,125 --tag _fa4b
.venv/bin/python scripts/revised_gate_verdict.py fa4b     # -> reports/revised_gate_verdict_fa4b.json (D71 rule)
.venv/bin/python scripts/diag_r1_seeds.py --seeds 601,602,603,604,605,606,607,608,609,610 --out reports/r1a_fresh_seeds.jsonl
.venv/bin/python scripts/diag_4b_ceiling.py               # D74
.venv/bin/python scripts/diag_data_ceiling.py             # D75 (seed 21 validation)
.venv/bin/python scripts/diag_data_ceiling_seeds.py       # D75 (fresh seeds 701-710)
.venv/bin/python scripts/make_figures_round4.py && .venv/bin/python scripts/build_report_round4.py
# round 5 (2026-10-18): Revision 4c, pre-registered in D81/D82 before these runs; all fits single-threaded (D79)
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/test_training_stability.py tests/test_ceiling.py   # RS, CEIL
OMP_NUM_THREADS=1 .venv/bin/python scripts/learnability_search_fa4c.py        # Step 1 -> reports/learnability_selected_fa4c.json
OMP_NUM_THREADS=1 KANFAE_RUN_ID=revised_gate_fa4c .venv/bin/python -m pytest -v -s tests/test_learnability_fa4c.py  # seed 81
OMP_NUM_THREADS=1 .venv/bin/python scripts/confirm_learnability.py fae_mlp --selected reports/learnability_selected_fa4c.json \
    --seeds 131,132,133,134,135 --tag _fa4c --amended --threads 1
.venv/bin/python scripts/revised_gate_verdict.py fa4c     # -> reports/revised_gate_verdict_fa4c.json (D82 rule)
.venv/bin/python scripts/diag_4c_stability.py             # D83 diagnostics
.venv/bin/python scripts/build_report_round5.py           # report revision 5 (all parts, per-round snapshots)
# round 6 (2026-10-18): every run_logged command and the pytest session are single-threaded (D86)
scripts/run_logged.sh bi_tests .venv/bin/python -m pytest -v tests/test_burn_in.py          # BI-2..BI-4
scripts/run_logged.sh search_fa4d .venv/bin/python scripts/learnability_search_fa4d.py      # Step 1 (D87)
# Step 2 would be: tests/test_learnability_fa4d.py (seed 92) and
#   confirm_learnability.py fae_mlp --selected reports/learnability_selected_fa4d.json --seeds 141,142,143,144,145 \
#   --tag _fa4d --amended --ceiling-margin 0.02 --threads 1 ; revised_gate_verdict.py fa4d   (not run: Step 1 failed)
KANFAE_RUN_ID=final_full_suite_round6_rerun scripts/run_logged.sh final_full_suite_round6_rerun \
    .venv/bin/python -m pytest -v -rA -s tests/                                          # final suite (D90)
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round6.py           # report revision 6
bash scripts/build_pdf.sh && .venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# round 7 (2026-10-18; owner D91)
scripts/run_logged.sh search_fa4e .venv/bin/python scripts/learnability_search_fa4e.py      # Step 1 (D92)
KANFAE_RUN_ID=revised_gate_fa4e scripts/run_logged.sh revised_gate_fa4e \
    .venv/bin/python -m pytest -v -s -rA tests/test_learnability_fa4e.py                       # gate, seed 92 (D93)
scripts/run_logged.sh confirm_fa4e .venv/bin/python scripts/confirm_learnability.py fae_mlp \
    --selected reports/learnability_selected_fa4e.json --seeds 141,142,143,144,145 --tag _fa4e \
    --amended --ceiling-margin 0.02 --probe-rule 4e --threads 1                               # confirmations
scripts/run_logged.sh verdict_fa4e .venv/bin/python scripts/revised_gate_verdict.py fa4e  # -> NOT CONFIRMED (D95)
scripts/run_logged.sh diag_4e_seed92 .venv/bin/python scripts/diag_4e_seed92.py            # D96 (train/validation only)
KANFAE_RUN_ID=final_full_suite_round7 scripts/run_logged.sh final_full_suite_round7 \
    .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round7.py           # report revision 7
# round 8 (2026-10-18; owner D99)
.venv/bin/python -m pytest -v tests/test_revision_4f.py tests/test_step3_tools.py              # NF-1..4, ST-1..3
scripts/run_logged.sh search_fa4f .venv/bin/python scripts/learnability_search_fa4f.py      # Step 1 (D100)
KANFAE_RUN_ID=revised_gate_fa4f scripts/run_logged.sh revised_gate_fa4f \
    .venv/bin/python -m pytest -v -s -rA tests/test_learnability_fa4f.py                       # gate, seed 93 (D101)
scripts/run_logged.sh confirm_fa4f .venv/bin/python scripts/confirm_learnability.py fae_mlp \
    --selected reports/learnability_selected_fa4f.json --seeds 151,152,153,154,155 --tag _fa4f \
    --amended --ceiling-margin 0.025 --probe-rule 4f --ensemble 3 --threads 1                 # confirmations
scripts/run_logged.sh verdict_fa4f .venv/bin/python scripts/revised_gate_verdict.py fa4f  # -> CONFIRMED (D103)
reports/step3/lanes.sh A|B|C|D        # Step 3 (D104): KANFAE_RESULTS=reports/step3/<group>.jsonl scripts/run_neural_synthetic.py
                                      #   --analyses H1 | H2,CONTROLS | H3 | SECONDARY | AB1,AB3 | AB4 | AB5,AB6,AB7,AB8;
                                      #   scripts/kanfae_prune_symbolic.py --variant plain | reg
KANFAE_RESULTS=reports/step3/real_pbc.jsonl .venv/bin/python scripts/run_neural_real.py --data pbc        # D105
KANFAE_RESULTS=reports/step3/real_cmapss.jsonl .venv/bin/python scripts/run_neural_real.py --data cmapss
.venv/bin/python scripts/analyse_step3.py                 # -> reports/step3_summary.json, reports/tables/step3_*.csv
KANFAE_RUN_ID=final_full_suite_round8 scripts/run_logged.sh final_full_suite_round8 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/build_report_round8.py           # report revision 8
# round 9 (2026-10-19; owner D107)
.venv/bin/python -m pytest -v tests/test_kan_fairness.py                                     # FC-1..FC-5
reports/round9/lanes.sh 1|2|3|4       # D108: KANFAE_RESULTS=reports/round9/laneN.jsonl scripts/run_kan_fairness.py --cells
                                      #   mlp:S2,mlp:S3,mlp_wide:S2 | kan:S2,kan:S3 | kan_shared:S2,kan_shared:S3 |
                                      #   kan_unb:S2,kan_unb:S3,mlp_wide:S3
cd scripts && ../.venv/bin/python analyse_round9.py && cd ..   # -> reports/round9_summary.json, reports/tables/r9_*.csv
KANFAE_RUN_ID=final_full_suite_round9 scripts/run_logged.sh final_full_suite_round9 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round9.py           # report revision 9
# round 10 (2026-10-19; owner D110): documents only. The literature review (docs/literature_review_round10.md) and the
# design note / pre-registration of Rounds 11-13 (docs/design_note_round10.md; D111-D114). No model code changed,
# no data acquired.
KANFAE_RUN_ID=final_full_suite_round10 scripts/run_logged.sh final_full_suite_round10 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round10.py          # report revision 10
# round 11 (2026-10-20; owner "start round 11", D112/D116): implementation and acceptance tests CK-1..CK-12.
# Formal acceptance run (one pytest process per STAT test, 4-6 in parallel, each single-threaded via conftest):
KANFAE_RUN_ID=round11_acceptance .venv/bin/python -m pytest tests/test_round11.py -q -k "test_CK3_"   # likewise CK5, CK6,
#   CK7, CK8, and -k "CK1_ or CK2_ or CK4_ or CK9_ or CK10 or CK12_ or CK11_"; logs reports/logs/round11/acceptance_*.log;
#   STAT numbers appended to reports/logs/round11_ck.jsonl
scripts/run_logged.sh round11_diag_identifiability .venv/bin/python scripts/diag_round11_identifiability.py   # D118 (seed 902)
scripts/run_logged.sh round11_explore_rise .venv/bin/python scripts/explore_round11_rise.py                     # D118, exploratory
KANFAE_RUN_ID=final_full_suite_round11 scripts/run_logged.sh final_full_suite_round11 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round11.py          # report revision 11
# L38 decisions / repaired tests (2026-10-21; owner D120): DGP v2, identifiable RISE, repaired CK-3b..CK-11b
scripts/run_logged.sh round12_diag_identifiability_v2 .venv/bin/python scripts/diag_round12_identifiability.py  # seed 902
KANFAE_RUN_ID=round12_ck_acceptance scripts/run_logged.sh round12_ck_acceptance_CK8b .venv/bin/python -m pytest tests/test_round12_ck.py -v -k "test_CK8b_"
#   likewise _CK5b (-k test_CK5b_), _CK6b, and _other (-k "test_CK3b_ or test_CK11b_ or ID1 or ID2"); 4 parallel lanes
scripts/run_logged.sh round12_explore_afcm_floor .venv/bin/python scripts/explore_round12_afcm_floor.py         # D121, exploratory
KANFAE_RUN_ID=final_full_suite_round12 scripts/run_logged.sh final_full_suite_round12 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round12.py          # report revision 12
# L39 decisions (2026-10-21; owner D123): AFCM diagnosis on development seed 914, CK-6c / CK-7b
scripts/run_logged.sh round12_diag_afcm_fix_V0_r12a .venv/bin/python scripts/diag_round12_afcm_fix.py V0_r12a V1_wide_lx
#   likewise _V2_wide_norm (V2_wide_norm V3_norm_d1d2), _V4_d1d2_perchannel, _V5_d1d2_quantile (V5.. V6..); 4 lanes
KANFAE_RUN_ID=round12b_ck_acceptance scripts/run_logged.sh round12b_ck_acceptance_CK6c .venv/bin/python -m pytest tests/test_round12b_ck.py -v -k "test_CK6c_"
KANFAE_RUN_ID=round12b_ck_acceptance scripts/run_logged.sh round12b_ck_acceptance_CK7b .venv/bin/python -m pytest tests/test_round12b_ck.py -v -k "test_CK7b_"
KANFAE_RUN_ID=final_full_suite_round12b scripts/run_logged.sh final_full_suite_round12b .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round12b.py         # report revision 13
# Round 12 (2026-10-21/22; owner D126): the simulation study
for k in 1 2 3 4; do scripts/run_logged.sh r12_lane$k .venv/bin/python scripts/run_round12_lanes.py --lane $k --start-utc 2026-10-21T19:12:34Z & done
#   after the container restart (D127): bash reports/round12/resume_after_restart.sh <slot 1-4> (via run_logged r12_resume_slot<k>)
R12_OUT=reports/round12/results_rerun_SAdd_10.jsonl scripts/run_logged.sh r12_rerun_SAdd_10 .venv/bin/python scripts/run_round12.py --scenario S-Add --reps 10-10 --only prune_symbolic
#   likewise SAdd_18 (S-Add 18) and SSparse_2 (S-Sparse 2)
scripts/run_logged.sh round12_analysis .venv/bin/python scripts/analyse_round12.py   # -> reports/round12_summary.json
KANFAE_RUN_ID=final_full_suite_round12c scripts/run_logged.sh final_full_suite_round12c .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round12c.py         # report revision 14
# Round 13 (2026-10-22; owner D128, plan D129): real data (JM 1.5-2 from CRAN: data/raw/jm/, sha256 in D128)
for k in 1 2 3 4; do scripts/run_logged.sh r13_lane$k .venv/bin/python scripts/run_round13_lanes.py --lane $k & done
scripts/run_logged.sh round13_analysis .venv/bin/python scripts/analyse_round13.py     # -> reports/round13_summary.json, figures
KANFAE_RUN_ID=final_full_suite_round13 scripts/run_logged.sh final_full_suite_round13 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round13.py          # report revision 15
bash scripts/build_pdf.sh && .venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
bash scripts/build_pdf.sh && .venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
bash scripts/build_pdf.sh && .venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# Round 14 (2026-10-23; owner D131): manuscript draft; no model is fitted
scripts/run_logged.sh round14_describe_cohorts .venv/bin/python scripts/describe_cohorts_round14.py   # descriptive counts
.venv/bin/python scripts/make_manuscript_figures_round14.py   # cropped display versions of the Round-13 effect figures
.venv/bin/python scripts/manuscript_round14.py                # docs/manuscript_round14.md + reports/manuscript_values_round14.json
bash scripts/build_manuscript_pdf.sh                          # reports/manuscript_round14.pdf
.venv/bin/python scripts/verify_manuscript_round14.py --pdf   # reports/manuscript_verification_round14.json (ALL_PASS)
# (final_full_suite_round14 was stopped after 10 tests to finalise the manuscript text first; its records are kept)
# (final_full_suite_round14_rerun was stopped by the session's 2-hour background limit after 207 records; kept)
setsid nohup env KANFAE_RUN_ID=final_full_suite_round14_rerun2 scripts/run_logged.sh final_full_suite_round14_rerun2 .venv/bin/python -m pytest -v -rA -s tests/ &
.venv/bin/python scripts/compile_test_results.py
.venv/bin/python scripts/build_report_round14.py              # report revision 16 (Part A = the manuscript)
bash scripts/build_pdf.sh && cp reports/final_research_prototype_report.pdf reports/final_research_prototype_report_rev16.pdf
.venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# Round 15 (2026-10-24/25; owner D133/D134): Phase-2 methods (code default-off), dry run on development seeds only
R15_OUT=reports/round15/dev/results_lane1.jsonl .venv/bin/python scripts/run_round15.py --job distill_S-Add_961   # etc.; see D133
R15_ANCHORED=1 R15_GATE=cv .venv/bin/python scripts/run_round15.py --job res_dev_real_965 --job afcmlaml_22       # D134 checks
.venv/bin/python scripts/analyse_round15_dev.py                     # reports/round15/dev_summary.json, dev_tables.md
for k in 1 2 3 4; do scripts/run_logged.sh r15_lane$k .venv/bin/python scripts/run_round15_lanes.py --lane $k & done   # 80 jobs
scripts/run_logged.sh round15_analysis .venv/bin/python scripts/analyse_round15.py   # reports/round15_summary.json
.venv/bin/python scripts/diag_round15_3515.py                       # post-hoc diagnosis (exploratory)
KANFAE_RUN_ID=final_full_suite_round15 scripts/run_logged.sh final_full_suite_round15 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py && .venv/bin/python scripts/build_report_round15.py   # report revision 17
bash scripts/build_pdf.sh && cp reports/final_research_prototype_report.pdf reports/final_research_prototype_report_rev17.pdf
.venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# Round 16 (2026-10-25/26; D138/D139): fixed-mu NaN-safe distillation and the non-cliff gate; manuscript update
R16_OUT=reports/round16/dev/results_dev3515.jsonl .venv/bin/python scripts/run_round16_formal.py dev_add_3515   # dev check
for k in 1 2 3 4; do scripts/run_logged.sh r16_lane$k .venv/bin/python scripts/run_round16_lanes.py --lane $k & done   # 60 jobs
scripts/run_logged.sh round16_analysis .venv/bin/python scripts/analyse_round16.py   # reports/round16_summary.json
scripts/run_logged.sh round16_posthoc .venv/bin/python scripts/posthoc_round16.py     # exploratory
.venv/bin/python scripts/manuscript_round16.py && bash scripts/build_manuscript16_pdf.sh && .venv/bin/python scripts/verify_manuscript.py --pdf
KANFAE_RUN_ID=final_full_suite_round16 scripts/run_logged.sh final_full_suite_round16 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py && .venv/bin/python scripts/build_report_round16.py   # report revision 18
bash scripts/build_pdf.sh && cp reports/final_research_prototype_report.pdf reports/final_research_prototype_report_rev18.pdf
.venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# Round 17 (2026-10-26/27; D141-D143): two-way gated anchoring (Cox, log-normal KAN-AFT); AFT symbolic distillation
for k in 1 2 3 4; do R17_MODE=dev scripts/run_logged.sh r17dev_lane$k .venv/bin/python scripts/run_round17_lanes.py --lane $k & done   # 49 dev jobs
.venv/bin/python scripts/analyse_round17_dev.py                       # reports/round17/dev_summary.json (last_tol, go/no-go)
for k in 1 2 3 4; do R17_MODE=formal scripts/run_logged.sh r17_lane$k .venv/bin/python scripts/run_round17_lanes.py --lane $k & done   # 60 jobs
scripts/run_logged.sh round17_analysis .venv/bin/python scripts/analyse_round17.py   # reports/round17_summary.json
.venv/bin/python scripts/manuscript_round17.py && bash scripts/build_manuscript17_pdf.sh && .venv/bin/python scripts/verify_manuscript_round17.py --pdf
KANFAE_RUN_ID=final_full_suite_round17 scripts/run_logged.sh final_full_suite_round17 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py && .venv/bin/python scripts/build_report_round17.py   # report revision 19
bash scripts/build_pdf.sh && cp reports/final_research_prototype_report.pdf reports/final_research_prototype_report_rev19.pdf
.venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# Round 18 (2026-10-28; D144-D146): landmark anchoring decided on a confirmation split
for k in 1 2 3 4; do R18_MODE=dev scripts/run_logged.sh r18dev_lane$k .venv/bin/python scripts/run_round18_lanes.py --lane $k & done   # 60 dev jobs
.venv/bin/python scripts/analyse_round18_dev.py                       # reports/round18/dev_summary.json (margins, go/no-go)
for k in 1 2 3 4; do R18_MODE=formal scripts/run_logged.sh r18_lane$k .venv/bin/python scripts/run_round18_lanes.py --lane $k & done   # 60 jobs
scripts/run_logged.sh round18_analysis .venv/bin/python scripts/analyse_round18.py   # reports/round18_summary.json
.venv/bin/python scripts/manuscript_round18.py && bash scripts/build_manuscript18_pdf.sh && .venv/bin/python scripts/verify_manuscript_round18.py --pdf
KANFAE_RUN_ID=final_full_suite_round18 scripts/run_logged.sh final_full_suite_round18 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py && .venv/bin/python scripts/build_report_round18.py   # report revision 20
bash scripts/build_pdf.sh && cp reports/final_research_prototype_report.pdf reports/final_research_prototype_report_rev20.pdf
.venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
# Round 19 (2026-10-29/30; D147-D149): confirmation-split decision with a full-validation refit; development check only (NO-GO)
for k in 1 2 3 4; do R19_MODE=dev scripts/run_logged.sh r19dev_lane$k .venv/bin/python scripts/run_round19_lanes.py --lane $k & done   # 60 dev jobs
.venv/bin/python scripts/analyse_round19_dev.py                       # reports/round19/dev_summary.json (variant, margins, go/no-go)
.venv/bin/python scripts/manuscript_round19.py && bash scripts/build_manuscript19_pdf.sh && .venv/bin/python scripts/verify_manuscript_round19.py --pdf
KANFAE_RUN_ID=final_full_suite_round19 scripts/run_logged.sh final_full_suite_round19 .venv/bin/python -m pytest -v -rA -s tests/
.venv/bin/python scripts/compile_test_results.py && .venv/bin/python scripts/build_report_round19.py   # report revision 21
bash scripts/build_pdf.sh && cp reports/final_research_prototype_report.pdf reports/final_research_prototype_report_rev21.pdf
.venv/bin/python scripts/build_manifest.py && .venv/bin/python scripts/verify_deliverables.py
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

