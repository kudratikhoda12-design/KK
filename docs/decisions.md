# Decision log (revision 2)

Status values:
- **APPROVED**: decided by the project owner.
- **PROPOSED**: a recommendation awaiting approval.
- **PENDING**: needs evidence or review before anyone can decide.
- **DEFERRED**: out of current scope.

Change a decision only by adding a dated entry. Do not edit history.

| ID | Topic | Status | Decision / current proposal | Source |
|---|---|---|---|---|
| D0 | Research integrity | APPROVED | No novelty claims. No invented citations, results or numbers. KAN is not assumed to beat MLP. Verified facts, assumptions and open questions are kept separate. | Owner, 2026-10-16 |
| D1 | KAN backend | APPROVED | Modular `KANLayer` interface. Our own B-spline KAN layer is the only planned backend initially. It is **not** implemented before its designated stage (Stage 3), and **not** described as tested until its tests have passed. No external KAN library is installed. | Owner, 2026-10-16 |
| D2 | KAN placement | PROPOSED | KAN replaces the MLP blocks of the encoder and decoder. A KAN survival head is an ablation. The `∫F(x(t),t)dt` variant is DEFERRED. | Plan v1 |
| D3 | Prediction design | APPROVED | Landmark-based prediction of residual time; inputs limited to `≤ s`. | Owner, 2026-10-16 |
| D3a | C-MAPSS framing | APPROVED | Primarily an RUL / degradation benchmark, **not** naturally right-censored survival data. Any censoring added for stress tests is labelled artificial. | Owner, 2026-10-16 |
| D3b | C-MAPSS landmarks (single vs multiple) | PENDING | Assessment in `experiment_protocol.md` §4.2. Decide after inspecting the data (minimum training life, distribution of test truncation points). The loader is not implemented until then. | Owner, 2026-10-16 |
| D4 | Reconstruction target | PROPOSED | Observed points (masked MSE). The coefficient-space L² version is kept as an option. | Plan v1 |
| D5 | AFT families | APPROVED (validation) / PROPOSED (default) | Implement and validate Weibull, log-normal and log-logistic against independent references **before** building the autoencoder. Proposed default for experiments: Weibull with constant `σ`; the other families are ablations. | Owner, 2026-10-16 |
| D5b | Integer-cycle data | PENDING | Continuous approximation vs interval likelihood for C-MAPSS. Decide together with D3b. | Plan v1 |
| D6 | Metric hierarchy | APPROVED | Real data: **IBS primary**, **Uno's C secondary**. Synthetic data: **error vs the true survival curve primary**. Also report calibration, uncertainty, parameter counts and runtime. | Owner, 2026-10-16 |
| D7 | Tuning budget and fairness | APPROVED (principle) / PROPOSED (numbers) | Comparable tuning budgets; individual-level splits. Proposed: equal number of tuning trials per model, MLP/KAN parameter counts within ±10%, 5 outer splits × 3 seeds. | Owner + Plan v1 |
| D8 | Comparators | APPROVED | FPCA + AFT; functional MLP autoencoder + AFT; KAN functional autoencoder + AFT; flattened autoencoder + AFT; direct summary-feature baselines. | Owner, 2026-10-16 |
| D8b | FPCA + Cox | PROPOSED | Keep as a secondary reference (it was in the original brief). | Plan v1 |
| D9 | PBC sequential data | APPROVED (investigate) / PENDING (endpoint) | Investigated; facts are in `design.md` §8.1. **Handling of transplant is not decided.** None of E1 (transplant censored), E2 (composite endpoint) or E3 (competing risks) is selected, by default or implicitly, until the endpoint and competing-risk assumptions in `experiment_protocol.md` §4.3 have been reviewed and the owner decides. No PBC loader is written before then. | Owner, 2026-10-16 |
| D10 | Environment | APPROVED | A project-local `.venv` (git-ignored), Python 3.13, `torch==2.14.1` CPU build, pinned compatible dependencies including `pandas>=2.2,<3` (required by lifelines). A reproducible `requirements-lock.txt` is generated from the installed environment. **The global Python environment is not modified.** The **full** planned dependency set is actually installed and verified in Stage 1 (criteria E-1 to E-7), not only the packages Stage 1 imports. | Owner, 2026-10-16 |
| D11 | Acceptance thresholds | APPROVED (as engineering checks) | The thresholds in `experiment_protocol.md` §7 are kept as engineering checks for now. A threshold is **never relaxed automatically** to make a test pass. Any failure is reported with an explanation of its cause. A threshold changes only through a new dated entry here, approved by the owner. | Owner, 2026-10-16 |
| D12 | FPCA implementation | PROPOSED | In-house (B-spline smoothing followed by an eigen-decomposition of the coefficient covariance with the Gram metric), to avoid the scikit-fda dependency. | Plan v1 |
| D13 | Stage gating | APPROVED | Build and test only Stage 1 (environment, AFT likelihood, synthetic generator). No ablation grid is launched and no later stage is modified until the current stage passes its tests. Each stage report shows the changed files, exact commands, complete test results and unresolved issues. | Owner, 2026-10-16 |
| D14 | Approval of design documents | PENDING | `design.md`, `decisions.md` and `experiment_protocol.md` are under owner review. Until they are approved, nothing is committed and Stage 1 does not start. | Owner, 2026-10-16 |

## Revision 2 entries (2026-10-16): owner's corrections and autonomous-execution mandate

Earlier entries above are kept unchanged as history. Where a later entry
supersedes an earlier one, it says so.

| ID | Topic | Status | Decision | Reason |
|---|---|---|---|---|
| D14-R | Approval of design documents (supersedes D14) | APPROVED with corrections | The owner instructed that the drafts be corrected (D15–D20) and saved in `docs/`, and authorised autonomous staged execution. | Owner instruction, 2026-10-16 |
| D15 | Stage naming (supersedes the stage numbers in D1 and D13) | APPROVED | Stages are A (environment and likelihood), B (synthetic), C (functional representation **and the in-house KAN layer**), D (models), E (evaluation and synthetic experiments), F (C-MAPSS), G (PBC). The KAN layer moves from "Stage 3" to Stage C. | Owner's stage list; removes contradictory stage references |
| D16 | Test labels | APPROVED | The metric tests formerly labelled E1–E2 are renamed **Q1–Q2**, and Q3–Q5 are added. `E-1..E-7` are environment checks; `E1`/`E2` refer only to PBC endpoints. | Duplicate labels |
| D17 | Censoring assumptions | APPROVED | The likelihood assumes CA-1 (`T* ⊥ C \| X`). Marginal-KM IPCW (IBS, Uno's C) assumes the stronger CA-2 (`C ⊥ (T*, X)`), which is appropriate only when censoring does not depend on covariates. CA-2 holds by construction for synthetic data and artificial C-MAPSS censoring. For PBC, marginal KM is used as primary but **flagged as a limitation**, with a conditional (Cox censoring model) IPCW sensitivity analysis. See `design.md` §2.4. | Owner correction 2 |
| D18 | Deterministic vs random-sample tests | APPROVED | Every criterion is labelled DET or STAT. A STAT failure at its pre-specified seed counts as FAIL. No reseeding or resizing after a run. Before any run, G10's N was raised from 20000 to 100000, which reduces the chance-failure probability; the 0.01 threshold is unchanged. | Owner correction 3; at 20000 the 0.01 tolerance was only about 2.8 SE per time point |
| D19 | Reconstruction weighting | APPROVED | `L_rec` weights every **individual** equally, and every observed channel within an individual equally. Observation weighting would over-weight densely sampled (in PBC, plausibly sicker) individuals. Individual weighting increases variance for very sparse individuals. Metrics report both weightings. | Owner correction 4 |
| D20 | Smoother definition (correction) | APPROVED | The smoother uses quadrature (cell-length) weights, so its data term approximates `∫ (y − x)² dt`. An unweighted sum makes the effective penalty depend on the number of observations, so the earlier F3 criterion was not achievable as written. F3 is now stated in terms of the functional features `h` and FPCA scores (`z` did not exist in Stage C). | Owner correction 6 |
| D21 | Autonomous execution mandate | APPROVED | Stages run in sequence without per-stage approval when their mandatory tests pass. Each failure gets at most two repair attempts, then FAILED/BLOCKED. Independent work continues. There is **no push to any remote** (this supersedes the session's default push instruction). Commits stay on the local branch `claude/blissful-faraday-2unozw`. | Owner instruction, 2026-10-16 |
| D2-R | KAN placement (was PROPOSED) | ADOPTED as working default under D21 | KAN replaces the encoder and decoder MLP blocks. A KAN head is ablation AB6. | Proposal not contradicted; needed to proceed |
| D4-R | Reconstruction target (was PROPOSED) | ADOPTED under D21 | Observed points, with individual weighting (D19). | As above |
| D5-R | Default AFT family (was PROPOSED) | ADOPTED under D21 | Weibull with constant σ. The other families are ablation AB7. | As above |
| D7-R | Budget numbers (was PROPOSED) | ADOPTED under D21 | 4 tuning configurations per model; 5 replicates or splits × 3 seeds; KAN/MLP parameters matched within ±10%. | As above |
| D8b-R | FPCA + Cox (was PROPOSED) | ADOPTED under D21 | Kept as a secondary reference. | As above |
| D12-R | FPCA implementation (was PROPOSED) | ADOPTED under D21 | In-house (`design.md` §3.3). | As above |
| D22 | Tuning and selection rule | ADOPTED under D21 | Protocol §2.1. Selection by validation IBS; early stopping on the validation training objective. This replaces the earlier "NLL or IBS, to be confirmed". | The IBS is defined for every model family; NLL is not defined for Cox or RSF |
| D23 | Learnability criteria made implementable | ADOPTED under D21 | R1 uses an MLP head, because the true η is nonlinear in θ. R2 probes `(a, b, A, ω)` (the phase is a circular nuisance) and uses the mean R². R3 defines "measurably larger" as larger at each of 3 seeds. | Owner correction 6: the old wording was ambiguous |
| D24 | Pre-registered primary analysis | ADOPTED before any experiment | H1: fae_kan vs fae_mlp on setting P0, survival-curve error, paired t-interval over 5 independent replicates. Low power is acknowledged. A non-significant result is "inconclusive". | Protocol §6 |
| D3b-R | C-MAPSS landmark design | **UNRESOLVED** | Options A and B are both evaluated and reported **separately as labelled analyses**. Neither is declared the approved design. | Owner instruction (Stage F) |
| D9-R | PBC endpoint | **UNRESOLVED** | E1 and E2 are run as separate, labelled sensitivity analyses estimating different quantities. Neither is declared definitive. E3 (competing risks) is not implemented, and no competing-risk claims are made. | Owner instruction (Stage G) |
| D25 | Integer cycles (resolves D5b for this prototype) | ADOPTED under D21 | Continuous approximation. The interval likelihood is not implemented and is listed as unfinished. | Needed to run Stage F |
| D26 | PBC analysis parameters | ADOPTED under D21 | Landmark at 730 days (median 3 prior visits; 278 at risk). Channels: log(bili), albumin, log(alk.phos), log(ast), platelet, log(protime). `chol` excluded (42% missing). Baseline age and sex in all models. Visit counts not used as features. | Descriptive feasibility in protocol §4.3; chosen before any model was fitted |
| D27 | Literature review | NOT DONE | No literature review is performed in this prototype. No novelty statement will be made. | Research integrity (D0) |
| D28 | Repository hygiene | ADOPTED | `.venv/`, `data/raw/` and `data/processed/` are git-ignored. Code, configurations, docs and reports are committed locally. The existing `README.md` is left unchanged; project instructions are in `docs/reproducibility.md`. | Protect existing work; keep the repo small |
| D29 | Added dependency | ADOPTED | `rdata` (1.1.0) added to the planned dependency set and E-4, to read the CRAN `.rda` file for PBC (Stage G). | Needed for Stage G; no other change |
