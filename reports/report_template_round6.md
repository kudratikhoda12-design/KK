<p class="small">Research prototype report, revision 6 · generated from saved records by
<code>scripts/build_report_round6.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no successful KAN-FAE model; no novelty claim</b>. Part A is round 6 (owner decisions D86).
Parts B–F are the round-5 to round-1 reports, each regenerated from its own round's records.</p>

## 0. Round-6 summary (plain English)

### 0.1 Outcome

| Step | Result |
|---|---|
| **Step 1: Revision 4d** (two-state stopping + burn-in; multi-seed selection; seed-21 train/validation only) | **{{V:step1}}**. No candidate met every pre-registered condition. The closest missed one condition by **0.001** (training seed 1's probe mean 0.779 vs 0.78) |
| **Step 2: gate** (fresh seed 92; confirmations 141–145) | **NOT RUN** (Step 1 failed; no fallback was registered). Seeds 92 and 141–145 remain unused and fresh |
| **Step 3: KAN vs MLP and downstream** | **NOT RUN** (code guard) |
| Housekeeping | Done: global single-threading in the run wrapper and the test session; R1.9 counts now ×5; the 4c gate kept as a strict-xfail record; gate seed 91 replaced by 92 because 91 had been generated in round 5 |

### 0.2 What we learned

1. **The survival side is now solved and stable.** Every candidate on every training seed reaches validation C
   0.84–0.85, the level of the data-attainable reference, with zero loss spikes.
2. **The stopping rule is no longer the issue.**
   - Within a learning rate, all two-state rules and burn-ins give *identical* latents (and so identical probes).
   - The latent always comes from the joint-objective minimum. The stopping rule only chooses which state
     supplies risk.
3. **The binding constraint is the representation's linear encoding of b and A.**
   - The probe for the x₁ slope b is 0.68–0.78, and for the amplitude A 0.73–0.84.
   - Training seed 1 is weakest (b 0.676, A 0.725).
   - The lower learning rate improves the latent (minimum seed probe 0.753 → 0.779; minimum ω 0.666 → 0.813), but
     not enough.
4. **The 0.001 shortfall is not waived.** Relaxing a threshold after seeing the result would be a post-hoc change.
   The rule was applied as registered.

### 0.3 Decision requested (nothing has been run)

The evidence points at the **representation**, not the stopping rule. Two clean options:

1. **(Recommended) A pre-registered representation-capacity step**, with the same multi-seed rule and fresh seeds,
   and lr 1e-3 with two-checkpoint stopping fixed:
   - the latent size r ∈ {12, 16};
   - the objective patience ∈ {100, 300}, letting reconstruction, and with it the b and A encoding, keep improving
     before the objective minimum is taken.
2. **An explicit owner override**, recorded as a protocol deviation: send the closest candidate (two-checkpoint,
   lr 1e-3, burn-in 300) to the fresh gate despite its 0.001 shortfall on one seed.

H1 stays blocked either way until the MLP passes a pre-registered gate.

### 0.4 Evidence, kept separate by kind

| Kind | Status |
|---|---|
| **Software correctness (tests)** | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed, {{V:xfailed}} expected failures (historical gates, strict), {{V:skipped}} skipped (the documented N/A case and the 4d gate, which does not run because Step 1 failed). BI-1..BI-4 pass, BI-1 being an exact reproduction |
| **Learnability (gate)** | **Not achieved.** No design has passed. Revision 4d did not reach the gate (Step 1 failed by 0.001 on one seed) |
| **Survival prediction (real data)** | Unchanged: descriptive only. PBC E2 provisional primary, E1 sensitivity |
| **Symbolic approximation** | KAN-AFT only. **Not performed for KAN-FAE** |
| **Scientific novelty** | **No claim** |
| **Publication readiness** | **No** |

## R6.1 Housekeeping (D86, D88)

- **Single-threading everywhere.**
  - `scripts/run_logged.sh` exports `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`.
  - `tests/conftest.py` sets the same variables before any numeric library is imported, then pins the BLAS pools
    (threadpoolctl) and torch to one thread for the whole session.
  - Suite results now match standalone single-threaded runs (closes D85).
- **R1.9:** the "Selected configurations" table in the round-1 part reads the round-1 snapshot, so the C-MAPSS
  counts are ×5.
- **Contents page:** with six parts the contents list overflowed onto an almost empty second page; it is now set in
  two columns (D90).
- **Historical record:** the 4c gate (seed 81) R1 and R2 are strict-xfail records.
- **Seed check.** The proposed gate seed 91 had been generated in round 5 (test CEIL-2: `SynthConfig(n=60,
  seed=91)`, i.e. the first 60 individuals of a seed-91 data set; no model was trained or evaluated on it).
  Following your instruction to use never-generated seeds, the gate was registered on **seed 92**. Seeds 141–145
  were verified unused. None of these seeds has now been generated.

## R6.2 Burn-in mechanism and tests (D87)

`min_epochs`:
- no early stopping before that epoch;
- validation-NLL-based selection (two-checkpoint risk state, NLL early stopping, head refit) considers only epochs
  ≥ `min_epochs`.

With `min_epochs = 0`, the pre-change single-thread fit is reproduced exactly (BI-1:
`reports/logs/bi1_regression_check.log`).

{{T:new_tests}}

## R6.3 Step 1: pre-registered multi-seed search (D87)

- **Design:** a full factorial of 8 candidates × training seeds 0, 1, 2 on seed-21 validation, single-threaded.
- **Eligibility, all required on all three seeds:**
  - zero jumps;
  - every seed's probe mean ≥ 0.78 and ω ≥ 0.80;
  - mean over seeds of the probe mean ≥ 0.80.
- **Selection:** the highest mean validation C. No fallback.

{{T:step1_candidates}}

**Per seed:**

{{T:step1_seeds}}

## R6.4 Why Step 1 failed

- **Probes are identical across stopping modes.** Within each learning rate, all four stopping modes and burn-ins
  produce exactly the same latent probes. The latent is the joint-objective-minimum state (epochs 442–1076), which
  these settings do not alter. They change only the risk state (validation C 0.839–0.850).
- **b and A are the weak components.** The slope b of x₁ (0.68–0.78) and the amplitude A (0.73–0.84) are what pull
  the probe mean down. Frequency ω meets its threshold at lr 1e-3 (≥ 0.813).
- **The decisive cell.** At lr 1e-3, training seed 1 (b 0.676, A 0.725) gives a probe mean of 0.779, short of 0.78
  by 0.001.
- **Next lever:** the remaining gap is representational, so further progress must come from the representation
  (capacity, or longer reconstruction training before the objective minimum), not from stopping (§0.3).

## R6.5 All gate looks so far

{{T:gate_overview}}

## R6.6 Decisions in round 6 (D86–D90)

{{T:decisions_r6}}

## R6.7 Tests

{{T:suite_table}}

**Two round-6 suite runs (D90).**
- `final_full_suite_round6` recorded 223 tests while pytest counted 224. The module-level skip of the 4d gate file
  (Step 1 failed) produces no test items, so the recording hook missed it.
- The conftest now also records skipped modules. The suite was re-run as `final_full_suite_round6_rerun`, the
  round-6 final run, whose recorded rows equal pytest's counts. The first run is kept unchanged.

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record)

{{T:test_runs}}

## R6.8 PBC endpoint (provisional; unchanged)

E2 (death or transplantation) is the provisional primary endpoint and E1 a sensitivity analysis, pending the
professor's confirmation (`docs/pbc_endpoint_justification.md`).

{{T:pbc_E2}}

{{T:pbc_E1}}

## R6.9 Task ledger

{{F:docs/task_ledger.md}}

## R6.10 Reproducibility

{{F:docs/reproducibility.md}}

# Parts B–F: historical records

Part B (round 5), Part C (round 4), Part D (round 3), Part E (round 2) and Part F (round 1) are regenerated from the
same records. Each part renders from the test records up to its own round's final suite run. Their numbers are
unchanged history; their conclusions on R2 are superseded by Part A. Embedded living documents (audit, ledger,
decision rows, reproducibility) show their current version.
