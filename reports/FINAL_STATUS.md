# Final status: KAN-FAE research prototype, Round 15 (2026-10-25)

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Latest full report: revision 16 (`reports/final_research_prototype_report_rev16.pdf`, Round 14). Round 15 is reported in
`docs/round15_dryrun.md` / `reports/round15_dryrun.pdf` (dry run, D133) and `docs/round15_outcome.md` /
`reports/round15_outcome.pdf` (outcome, D136); it is not yet in a cumulative report revision.

| Item | Result |
|---|---|
| D133–D135 | Phase-2 methods implemented (all options default-off); dry run on development seeds; owner decisions; development checks C2 (Laplace AFCM) and G (CV gate) failed their pre-fixed rules, so seed 24 is unused |
| Round 15 (D136) | 80/80 jobs. **H-R15a not supported** (39/40 refits finite; S-Add upper bound +0.059 > 0.05), although distillation kept the shapes where the old refit destroyed them; **H-R15b not supported** (S-Last part holds, −0.0185; S-Sparse upper bound +0.0094 > 0.005); real data exploratory, no interval excludes 0 |
| Tests | New DET tests R15-1..R15-8 pass; no full-suite run in Round 15 (last full suite: `final_full_suite_round14_rerun2`) |
| Next | Owner decision: next-round candidates in `docs/round15_outcome.md` §3; a cumulative report revision 17 with Round 15 on request |

---

# Round 14 status (2026-10-23), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` = `reports/final_research_prototype_report_rev16.pdf` (revision 16,
283 pages: Part A = the Round-14 manuscript draft with its verification; Part B = Round 13; Part C = Round 12; Parts D–E =
L39/L38; Parts F–P = rounds 11–1). Manuscript: `docs/manuscript_round14.md`, `reports/manuscript_round14.pdf` (24 pages).
Status: **research prototype; manuscript draft for the owner's review, not peer reviewed; no novelty claim.**

| Item | Result |
|---|---|
| D131 (owner) | Round 13 / revision 15 approved; L36, L41 closed; manuscript scope and number-provenance rule |
| Round 14 (D132) | Manuscript under framing (a) + (b), D111 claim policy; every number rendered from saved records (1361 logged values), independently re-verified (ALL_PASS); 46 verified references; L37 closed |
| Tests | `final_full_suite_round14_rerun2`: 257 passed, 2 failed (CK-6c, kept visible by D126; 4f R1), 16 xfailed, 2 skipped (277); two earlier partial runs kept (D132) |
| Verification | `verify_deliverables.py` ALL_CHECKS_PASS (includes the manuscript checks) |
| Next | Owner review of the draft: author block, venue, three statements marked as interpretations |

---

# Round 13 status (2026-10-22), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (revision 15: Part A = Round 13; Part B = Round 12; Parts C–D =
L39/L38; Parts E–O = rounds 11–1). Revision 14 is kept as `reports/final_research_prototype_report_rev14.pdf`.
Status: **research prototype; not publication-ready; no novelty claim.**

| Item | Result |
|---|---|
| D128 (approved by the owner) | L41 closed: H-R12 stays inconclusive; Round 13 exploratory; symbolic primary = substitution without refit |
| Round 13 (D129, D130) | PBC (E2/E1), prothro, aids × 5 splits; B3; C-MAPSS; 100-fit stability. No predictive gain of functional/KAN models over simple baselines; stable, clinically aligned leading effects (except the prothro index under B2-S); symbolic substitution without refit preserves the model; C-MAPSS miscalibrated for every model |
| Tests | `final_full_suite_round13`: 251 passed, 2 failed (CK-6c, kept visible by D126; 4f R1), 16 xfailed, 2 skipped (271); see R13.10 |
| Next | Round 14 (paper draft, L37) on the owner's go-ahead |

---

# Round 12 status (2026-10-22), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (revision 14: Part A = Round 12; Parts B–C = L39/L38; Parts D–N =
rounds 11–1). Status: **research prototype; not publication-ready; no novelty claim.**

| Item | Result |
|---|---|
| Round 12 (D126, D127) | All 160 jobs ran (4 scenarios × 20 + P0 × 20 + phase-2), none dropped; 3 pruning cells failed on non-finite symbolic refits and were rerun after a harness fix; a container restart was resumed |
| **H-R12** | **Inconclusive**: B2-S − AFCM V3 mean identifiable RISE +0.0065, 95% CI [−0.051, +0.064]; TOST ±0.05 p 0.063. **AFCM V3 failed CK-6c (not validated)** |
| Main exploratory findings | B2-S ≈ AFCM V3 in shapes and survival; both ≫ B1-FPCA in shapes; B2-S better than the Round-12a AFCM; residual calibration helps KAN-AFT under log-normal errors; pruning recall 0.95 / FDR 0.13 (S-Add); symbolic substitution keeps shapes, the refit destroys them |
| S-Int/P0 continuity | fae_kan_s3 ≡ fae_mlp_s3 within ±0.005 (TOST p 0.005) |
| Tests | `final_full_suite_round12c_rerun` (the first run was killed by a container restart; partial records kept): 248 passed, 2 failed (CK-6c, visible by D126; 4f original R1), 16 expected failures, 2 skipped (268); every CK record reproduced. Report revision 14 (236 pages; `reports/final_research_prototype_report_rev14.pdf`); verification ALL_CHECKS_PASS |
| Next | Owner decisions L41; Round 13 (L36) |

---

# L39 status (2026-10-21), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (revision 13: Part A = L39 decisions and the Round-12 gate;
Part B = L38; Parts C–M = rounds 11–1). Status: **research prototype; not publication-ready; no novelty claim.**

| Item | Result |
|---|---|
| AFCM diagnosis (D123, development seed 914) | Adopted V3 (λ_x grid 1e-7..1e1, normalised penalties, D1 + D2 in t): φ₁ 0.473 → 0.023 on seed 914 |
| CK-7b (seed 23) | PASS: PACE under DGP v2 (\|cos\| ≥ 0.985; ISE below smoothing on all channels) |
| CK-6c (seed 22) | **FAIL**: repaired AFCM φ₁ identifiable RISE 0.247 (> 0.15); φ₂ 0.053 |
| Round 12 | **Not started** (owner gate); seeds 3000–3419 unused. Owner decision needed: L40 |
| Tests | `final_full_suite_round12b`: 248 passed, 2 failed (CK-6c; 4f original R1, not gating), 16 expected failures, 2 skipped (268). All CK records reproduced. Report revision 13 (206 pages); deliverable verification ALL_CHECKS_PASS |

---

# L38 status (2026-10-21), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (Part A = L38 decisions, repaired tests and the Round-12 gate;
Parts B–L = rounds 11–1). Status: **research prototype; not publication-ready; no novelty claim.**

| Item | Result |
|---|---|
| Owner decisions (D120) | DGP v2 (level + trend), identifiable RISE, AFCM λ_t grid to 1e4, converged CK-3b, integrated channel pruning with parsimonious selection, mixture-error CK-11b; registered before any run on seeds 17–21 |
| Repaired tests (D121) | Pass: CK-3b (2.1e-6), CK-5b (B2-S identifiable RISE 0.085 / 0.031), CK-8b ([1, 1, 0]), CK-11b (p 0.612 vs Weibull 2.2e-22). **FAIL: CK-6b** (AFCM φ₁ 0.556) |
| Round 12 | **Not started** (owner gate); seeds 3000–3419 unused. Owner decision needed: L39 (AFCM comparator) |
| Tests | `final_full_suite_round12`: 247 passed, 2 failed (CK-6b; 4f original R1, not gating), 15 expected failures (incl. the five Round-11 CK records), 2 skipped (266). All CK records reproduced. Report revision 12 (199 pages); deliverable verification ALL_CHECKS_PASS |

---

# Round 11 status (2026-10-20), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (Part A = round 11; Parts B–K = rounds 10–1); deliverable
verification passes. Status: **research prototype; not publication-ready; no novelty claim.**

| Item | Result |
|---|---|
| Implementation (D116) | Exact Efron/Breslow Cox loss and Breslow baseline; Round-12 simulator; PACE; functional KAN (B2-H, B2-S, B2-deep) and static CoxKAN / KAN-AFT heads; pruning grid; symbolic extraction and substitution; residual-calibrated KAN-AFT; AFCM, LME2 + Cox, landmark Cox, B1 features; B3 landmark builder. Deviations disclosed |
| Acceptance tests (D117) | Pass: CK-1, 2, 4, 7, 9, 10, 12. **FAIL (recorded, not converted):** CK-3 (slopes off by 0.084 after early stopping), CK-5 (B2-S RISE 0.146 / 0.686), CK-6 (AFCM RISE 0.615 / 1.095), CK-8 (null channel not pruned), CK-11 (Weibull D-cal p 0.0515, not < 0.05) |
| Finding (D118) | In S-Add, ∫X_d dt is the same for every subject, so the linear-in-x part of each effect is unidentified, but the registered RISE counts it; this affects CK-5/CK-6 and H-R12 |
| Round 12 | **Not started**; owner decisions needed (ledger L38) |
| Tests | `final_full_suite_round11`: 241 passed, 6 failed (CK-3, CK-5, CK-6, CK-8, CK-11; 4f original R1, not gating), 10 expected failures, 2 skipped (259). Every CK record reproduced the formal run bit for bit. Report revision 11 (188 pages); deliverable verification ALL_CHECKS_PASS |

## Next

Owner decisions on L38 (D118 options; CK-3, CK-8, CK-11), then Round 12.

---

# Round 10 status (2026-10-19), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (Part A = round 10; Parts B–J = rounds 9–1); deliverable
verification passes. Status: **research prototype; not publication-ready; no novelty claim.**

| Item | Result |
|---|---|
| Owner decisions (D110) | Rounds 10–14 roadmap: functional CoxKAN / KAN-AFT (B1, B2, B3) as the main study; KAN-FAE as the motivating negative result |
| Literature review (D111) | Targeted, documented (`docs/literature_review_round10.md`). Single-layer functional CoxKAN = AFCM's model form (Cui et al. 2021); published KAN survival models use static covariates; JM-KAN and F-KAN checked only at abstract level. **No novelty claim** |
| Pre-registration (D112–D114) | `docs/design_note_round10.md`: Round 11 implementation + acceptance tests CK-1..CK-12; Round 12 simulation (5 scenarios × N 20, fresh seeds 3000–3419, confirmatory H-R12: B2-S vs AFCM shape recovery); Round 13 real data (PBC, prothro, aids, C-MAPSS) with fixed interpretability measures |
| Rendering fix (D115) | Bounded decision filters for the round-7/8/9 parts |
| Tests | `final_full_suite_round10`: 231 passed, 1 failed (4f original R1; not gating), 10 expected failures, 2 skipped (244). No model code changed |

## Next

Round 11 (implementation and acceptance tests), on the owner's go-ahead.

---

# Round 9 status (2026-10-19), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (Part A = round 9; Parts B–I = rounds 8–1, each from its own
snapshot); deliverable verification passes. Status: **research prototype; not publication-ready; no novelty claim.**
Ledger L32 closed.

| Item | Result |
|---|---|
| Owner decisions (D107) | Fairness study; H1 the only confirmatory test (others exploratory); PBC E2 primary / E1 sensitivity. Fact check: parameter counts and controls correct; PBC invariance holds for conclusions, not point estimates |
| Fairness design (D108) | Mode B (KAN encoder + fae_mlp's MLP decoder), Mode A (single-layer KAN decoder 16→36) vs fae_mlp_wide (hidden 107, +0.3%); S2 and S3, tuned, 3-seed ensembles; FC-1..FC-5 pass |
| Outcome (D109) | 400 runs, all ok; S2 repeats bit-identical to H1. **F1** (S3): KAN ≈ MLP on survival (curve −0.0001 [−0.0030, +0.0028]). **F2:** shared MLP decoder fixes reconstruction, not survival (S3 +0.0061, 5/5 worse). **F3:** unbottlenecked KAN vs wide MLP: S2 +0.0055 (5/5 worse, CI includes 0), S3 −0.0020 (no difference). S3 helps every arm. **KAN never better than the MLP on survival** |
| Tests | `final_full_suite_round9`: 231 passed, 1 failed (4f original R1; not gating), 10 expected failures, 2 skipped (244) |

No open decision is required for the research question as pre-registered. H1 is not supported, the fairness
sensitivity study explains the deficit as a training-strategy effect, and no configuration shows a KAN advantage.

---

# Round 8 status (2026-10-18), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (144 pages; Part A = round 8; Parts B–H = rounds 7–1, each from
its own snapshot); deliverable verification passes. Status: **research prototype; not publication-ready; no novelty
claim.**

| Step | Result |
|---|---|
| Owner decisions (D99) | Revision 4f; R2 margin 0.025; one-sided R3-A; probe 0.78/0.80. Fact check: two statements corrected; disclosed that the round-7 model would have passed the round-8 criteria |
| Step 1 (D100, D102) | **PASSED**: nonlinear joint head + 3-seed ensemble (val C 0.8510); 4e fits reproduced bit for bit; NF-1..4 pass |
| Step 2 (D101, D103) | Gate seed 93: R1-A, R2 (C 0.8405 ≥ 0.8322; probe 0.865 / ω 0.908), R3-A all PASS; confirmations 151–155 **5/5** → **CONFIRMED**. Also confirmed under round-7 criteria |
| Step 3 (D104–D106) | All pre-registered analyses run; no failures. **H1:** fae_kan − fae_mlp curve error +0.0078 [+0.0002, +0.0155], 5/5 KAN worse: no KAN advantage. **H2:** no effect of α. **H3:** reversed (flat_ae better under sparse sampling). **Controls:** S3 best; KAN gap vanishes under S3. **Pruning:** removes ≈ 0 edges at θ 1e-2; **symbolic:** 56% of edges accepted, metrics unchanged, formulas not interpretable at the parameter level. **Real data:** PBC no KAN–MLP difference; C-MAPSS KAN worse; neural arms not better than round-1 comparators |
| Tests | `final_full_suite_round8`: 226 passed, 1 failed (4f original R1; not gating), 10 expected failures, 2 skipped (239). Gate reproduces exactly in the suite |

## Decision requested (L32)

1. Whether to run a pre-registered KAN-favourable fairness check (e.g. S3 training for both arms, a KAN decoder without
   a sub-r bottleneck), given that H1 is conditional on the MLP-selected configuration.
2. Whether to write up H1–H3 as exploratory evidence (no novelty claim).
3. PBC endpoint confirmation with the professor.

---

# Round 7 status (2026-10-18), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (Part A = round 7; Parts B–G = rounds 6–1, each from its own
snapshot); deliverable verification passes. Status: **research prototype; no successful KAN-FAE model; not
publication-ready; no novelty claim.**

| Step | Result |
|---|---|
| Owner decisions (D91) | Revision 4e; two-tier Step-1 rule; R2 probe part re-referenced to 0.75/0.75 (fact-checked; disclosed as set after rounds 4–6) |
| Step 1: Revision 4e (D92, D94) | **PASSED via Tier 1**: r 16, patience 100, α 1 (`A7-fae_mlp-94415d4f`); 6/8 candidates Tier-1 eligible; 4d cand1 reproduced bit for bit; patience 300 identical to 100 |
| Step 2: gate, fresh seed 92 (D93, D95) | R1-A PASS; R2 probe part PASS (0.884, ω 0.903; passes every definition); **R2 concordance part FAIL** (0.8384 < 0.8406, by 0.0022); **R3 FAIL** (\|C − C_R2\| 0.0254; censored-data C higher) |
| Confirmations 141–145 | **4/5** (seed 143 short by 0.0019 on concordance) |
| Verdict | **NOT CONFIRMED**; guard refuses; **Step 3 not run** |
| Diagnostic (D96, D97; seed-92 train/validation) | Training-seed SD of validation C 0.0026 > 0.0022 shortfall; systematic C − C_ceiling ≈ −0.016 (linear head vs nonlinear risk) |
| Tests | `final_full_suite_round7`: 216 passed, 3 failed (4e gate R1, R2, R3: live, real), 7 expected failures, 2 skipped (228). The 4e gate reproduces exactly inside the suite |
| Report fixes (D98) | Pipe-split R3 cells; per-round decision filters; gate-reproduction comparison |

## Decision requested

1. (Recommended) Revision 4f: keep the 4e representation; pre-register a Step-1 search over the risk head (linear vs
   small nonlinear) and averaging over training seeds 0–2; new fresh gate and confirmation seeds.
2. Decide prospectively whether R3 stays two-sided.
3. Or accept the verdict (MLP not gate-qualified; H1 blocked).

---

# Round 6 status (2026-10-18), historical

Everything is committed locally on `claude/blissful-faraday-2unozw`; **nothing pushed** (owner instruction).
Report: `reports/final_research_prototype_report.pdf` (106 pages; Part A = round 6; Parts B–F = rounds 5, 4, 3, 2, 1,
each from its own snapshot); deliverable verification passes. Status: **research prototype; no successful KAN-FAE
model; not publication-ready; no novelty claim.**

| Step | Result |
|---|---|
| Housekeeping (D86, D88) | Single-threaded numerics in `run_logged.sh` and the pytest session; R1.9 reads the round-1 snapshot (×5); 4c gate R1/R2 kept as strict xfail; gate seed 91 replaced by 92 (91 had been generated by CEIL-2 in round 5) |
| Step 1: Revision 4d (D87) | **FAILED** (D89): 8 candidates × seeds 0–2, none eligible. Closest (two_checkpoint, lr 1e-3, burn-in 300/500): mean val C 0.8486, mean probe 0.816, min ω 0.813, zero jumps, but seed 1 probe mean 0.779 < 0.78. Not waived |
| Diagnosis (D89) | Within a learning rate all stopping modes and burn-ins give identical latents; the binding constraint is the representation's encoding of b and A |
| Step 2 (gate seed 92; confirmations 141–145) | **NOT RUN** (no fallback); seeds remain unused |
| Step 3 | **NOT RUN** (guard) |
| Tests | `final_full_suite_round6_rerun`: 215 passed, 0 failed, 7 expected failures (historical gates), 2 skipped (224). The first round-6 run recorded 223 because a module-level skip was not recorded; fixed and re-run (D90), first run kept |
| Report | Contents list set in two columns so it no longer spills onto a near-blank page (D90) |

## Decision requested

1. (Recommended) Pre-register a representation-capacity step (L28): r ∈ {12, 16} × objective patience ∈ {100, 300},
   two-checkpoint stopping, lr 1e-3, the same multi-seed rule, fresh seeds; or
2. an explicit owner override (recorded as a protocol deviation) sending the closest 4d candidate to the fresh gate.

---

# Round 5 status (2026-10-18), historical

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
