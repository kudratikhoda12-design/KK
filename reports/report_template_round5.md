<p class="small">Research prototype report, revision 5 · generated from saved records by
<code>scripts/build_report_round5.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no successful KAN-FAE model; no novelty claim</b>. Part A is round 5 (owner decisions D80).
Parts B–E are the round-4, 3, 2 and 1 reports, each regenerated from its own round's records.</p>

## 0. Round-5 summary (plain English)

### 0.1 Results of the three steps

| Step | Result |
|---|---|
| **Step 1: training stability and stopping rule (Revision 4c)**, seed-21 train/validation only | **{{V:step1}}**: zero training-NLL jumps on seeds 0–2 and no stopping conflict. Full-budget runs: 12 jumps → 0. Thread-count spread: 0.013 → 0.0006 |
| **Step 2: amended gate**, fresh seed 81 | R1-A PASS; **R2 concordance part PASS** (C {{V:C}} ≥ {{V:thr}}; C_ceiling {{V:ceil}}); **R2 probe part FAIL** (mean {{V:probe}} < 0.78; ω {{V:omega}}); R3 PASS |
| Fresh confirmations, seeds 131–135 | **{{V:conf}} pass** |
| Verdict (pre-registered) | **{{V:verdict}}** |
| **Step 3: KAN vs MLP and downstream** | **NOT RUN** (owner rule; the code guard refuses) |

### 0.2 What we learned

1. **Instability fixed.**
   - Balancing the spectral path to the A3 path and clipping gradients removed the loss spikes: in full
     3000-epoch runs, jumps fell from 12 (maximum training NLL 12.5) to 0 (maximum 1.14).
   - The run-to-run spread across thread counts fell from 0.013 to 0.0006.
2. **Concordance gap closed.** With stable training and survival-optimal stopping, test C now reaches the amended,
   data-attainable reference on the gate seed (0.8421 ≥ 0.8390). Three of five confirmation seeds come within about
   0.006 of it.
3. **The latent is now the failure.**
   - Stopping at the survival optimum (validation NLL minimum, epochs 91–430) ends training before the latent has
     learned b and A (probe b 0.57–0.67).
   - On one seed (135), the validation NLL dipped at epoch 91 and the whole model was undertrained.
4. **A disclosed tie-break decided this.** At search step H3, NLL early stopping and the two-checkpoint rule gave
   *identical* risk predictions. The registered tie-break kept the simpler single-state rule, whose latent is
   earlier and weaker. The two-checkpoint rule's latent (from the objective minimum) had a higher validation probe
   (0.818 vs 0.787). It was not gate-tested: that would be a post-hoc look.
5. **Determinism.**
   - The 0.0053 difference between Tables R4.3 and R4.5a comes from thread-count-dependent floating point,
     amplified by the instability: 3 threads were even non-deterministic between two identical runs.
   - All round-5 fits are single-threaded and reproducible (R5.1).

### 0.3 Recommended next step (owner decision; nothing has been run)

Pre-register **two-checkpoint stopping with a minimum training period**. The representation trains to the
joint-objective minimum; risk is taken from the validation-NLL minimum, but only over epochs ≥ a burn-in (for
example 300), so that an early transient dip cannot end training.

Choose it by **robustness across training seeds** (seeds 0–2 on validation), not seed 0 alone. Then evaluate on a
new fresh gate seed and new confirmation seeds under the D82 criteria, which stay unchanged.

H1 stays blocked until then.

### 0.4 Evidence, kept separate by kind

| Kind | Status |
|---|---|
| **Software correctness (tests)** | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (the current 4c gate's original R1 and R2: live, real), {{V:xfailed}} expected failures (historical gates, strict), {{V:skipped}} skipped. New tests RS-2..5 and CEIL-1..3 pass. RS-5 caught a real BatchNorm leak in the first refit implementation; it was fixed |
| **Learnability (gate)** | **Not achieved.** Revision 4c meets the concordance part of the amended R2 but not the probe part |
| **Survival prediction (real data)** | Unchanged from round 4: descriptive only. PBC E2 provisional primary, E1 sensitivity |
| **Symbolic approximation** | KAN-AFT only (validated on known truth). **Not performed for KAN-FAE** (gate not passed) |
| **Scientific novelty** | **No claim** |
| **Publication readiness** | **No** |

## R5.1 Determinism and the 0.0053 difference (D79)

The same configuration and training seed on seed-21 validation, run in separate processes:

{{T:det}}

**Explanation.**
- Multithreaded floating-point reductions depend on the thread count; at 3 threads they were not even repeatable
  run to run.
- The training instability then amplified tiny differences into different trajectories.
- Table R4.3 (search, 3 threads) recorded 0.8337. Table R4.5a (diagnostic, 4 threads) recorded 0.8284.

**Consequence.**
- The Round-4 search compared configurations whose differences (about 0.009) are as large as this numerical
  spread, so its selection may partly reflect noise (disclosed; records unchanged).
- From round 5, all fits are single-threaded.
- After the stability fix, the spread is 0.0006 (R5.3).

## R5.2 Revision 4c mechanisms and new tests

- **`fa_balance`:** a fixed spectral-path gain, so that the spectral path's initial RMS on the *training* data
  equals the A3 path's.
- **`clip`:** gradient-norm clipping.
- **Stopping modes (`es_mode`):**
  - `two_checkpoint`: latent and decoder from the validation-objective minimum; risk from the full-model state at
    the validation-NLL minimum;
  - `two_state_refit`: after the joint run, the linear AFT head is re-initialised and refitted on the frozen
    latent with validation-NLL stopping.
- **`kanfae.evaluation.ceiling`:** the per-individual parametric plug-in that defines C_ceiling.
- **RS-1:** with every new option off, the code reproduces the pre-change single-thread fit exactly (validation C
  0.8221679197994988, 1142 epochs; `reports/logs/rs1_regression_check.log`).

{{T:new_tests}}

## R5.3 Step 1: pre-registered search and acceptance (D81)

The search used seed-21 train/validation only, was single-threaded, and changed one factor at a time.

**Amended selection rule:** among candidates with probe ≥ 0.78 (≥ 0.80 under two-state stopping) and, from H2 on,
zero jumps, maximise validation C.

{{T:step1_search}}

**Step-1 acceptance**, the selected configuration on three training seeds:

{{T:step1_accept}}

**Diagnostics** (D83; seed-21 train/validation; informational):

{{T:stab}}

**Disclosed tie at H3.**
- NLL early stopping and `two_checkpoint` restore the same epoch-329 state for risk, so their validation C is
  identical (0.8474).
- The registered tie-break kept the earlier, simpler candidate.
- On the stability seeds, the selected latent's probe is fragile (seed 1: mean 0.674, ω 0.481; seed 2: ω 0.742).

## R5.4 Step 2: amended gate and confirmations (D82)

**Amended criteria** (owner D80; registered before any gate data were generated):
- R1-A gating, with the original R1 reported alongside.
- R2: C ≥ min(C_oracle − 0.03, C_ceiling − 0.015), where C_ceiling is the D75 per-individual parametric plug-in on
  the same test split; **and** mean probe ≥ 0.78 with ω ≥ 0.80.
- R3 unchanged.

The simulator is unchanged.

{{T:gate_4c}}

**Reproduction (D85).**
- Re-running the gate file in the registered fully single-threaded environment reproduces every number exactly.
- Inside the full suite, where numpy's linear algebra used 3 threads, concordance moved by only 0.0001, but the
  latent probe moved by 0.01–0.02 (mean 0.762, ω 0.794).
- So the latent at the early, NLL-optimal stopping epoch is numerically fragile (R5.5). R2 fails in both runs.

**Fresh confirmations (seeds 131–135).**

{{T:confirm_4c}}

**All gate looks so far.** Note that the last row uses the amended criteria.

{{T:gate_overview}}

## R5.5 Why Step 2 failed

NLL early stopping ends training at the survival optimum. The diagnostic below refits on the train/validation splits
of two confirmation seeds; their test splits are not touched.

{{T:seed135}}

**Reading.**
- On a typical seed (131), training stops around epoch 280 with good concordance, but the b and A probes are still
  low.
- On seed 135, an early dip in the validation NLL stopped training at epoch 91, before the model had learned much at
  all.
- The survival optimum and the representation optimum fall at different epochs (D70). The single-state NLL rule
  resolves the conflict entirely in favour of survival.
- The two-state rules keep the representation training. Combined with a burn-in before NLL-based selection, they
  address both problems. That is the recommended next step (§0.3); it needs owner approval and a new
  pre-registration.

## R5.6 Reporting fixes (owner decision 5)

- **Chronological test-run tables, in every part:** they now include an *expected failures (xfail)* column and a
  *total* column, and every row sums to the number of tests in that run. Final suites: 202 in round 3, 209 in
  round 4.
- **Historical parts are rendered from their own snapshots:**
  - **Tests:** records up to that round's final suite run (`build_report.use_snapshot`).
  - **Round 1's C-MAPSS run-status and runtime tables:** read round-1-only aggregates
    (`tables/r1_*.csv`: no FD003 runs and no v3 re-runs), so Option B is no longer double-counted and FD003 no
    longer appears in round 1.
- **The 0.0053 difference** is explained in R5.1.

## R5.7 Decisions in round 5 (D79–D85)

{{T:decisions_r5}}

## R5.8 Tests

{{T:suite_table}}

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record)

{{T:test_runs}}

## R5.9 PBC endpoint (provisional; unchanged)

E2 (death or transplantation) remains the provisional primary endpoint and E1 a sensitivity analysis. The
justification for the professor is in Part B, §R4.7 (`docs/pbc_endpoint_justification.md`).

{{T:pbc_E2}}

{{T:pbc_E1}}

## R5.10 Task ledger

{{F:docs/task_ledger.md}}

## R5.11 Reproducibility

{{F:docs/reproducibility.md}}

# Parts B–E: historical records

Part B (round 4), Part C (round 3), Part D (round 2) and Part E (round 1) are regenerated from the same records. Each
part renders from the test records up to its own round's final suite run. Their numbers are unchanged history;
their conclusions on R2 are superseded by Part A. Embedded living documents (audit, ledger, decision rows,
reproducibility) show their current version.
