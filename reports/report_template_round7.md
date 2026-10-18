<p class="small">Research prototype report, revision 7 · generated from saved records by
<code>scripts/build_report_round7.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no successful KAN-FAE model; no novelty claim</b>. Part A is round 7 (owner decisions D91).
Parts B–G are the round-6 to round-1 reports, each regenerated from its own round's records.</p>

## 0. Round-7 summary (plain English)

### 0.1 Outcome

| Step | Result |
|---|---|
| **Step 1: Revision 4e search** (r × patience × α; two-tier rule; seed-21 train/validation only) | **{{V:step1}} via Tier {{V:tier}}**. Selected r 16, patience 100, α 1 (`{{V:sel_id}}`); six of eight candidates met Tier 1, so the fallback was not needed. The 4d candidate was re-fitted and reproduced bit for bit |
| **Step 2: gate, fresh seed 92** | R1-A PASS. **R2 probe part PASS** (mean {{V:probe}}, ω {{V:omega}}; also passes 0.78/0.80 and the original 0.80). **R2 concordance part FAIL**: C {{V:C}} < {{V:thr}}, short by 0.0022. **R3 FAIL**: \|C − C_R2\| 0.0254 > 0.02 (the censored-data C is *higher*) |
| Confirmations, fresh seeds 141–145 | **{{V:conf}} pass** (seed 143 fails the concordance part by 0.0019) |
| Verdict (pre-registered, D93) | **{{V:verdict}}** |
| **Step 3: KAN vs MLP and downstream** | **NOT RUN** (the code guard refuses; `reports/logs/guard_check_fa4e_after.log`) |

### 0.2 What we learned

1. **The representation problem is solved.**
   - With latent size r = 16, the latent encodes all four curve parameters on every fresh seed: test probe means
     0.83–0.88, ω 0.85–0.92.
   - It passes even the strictest probe definition used so far (mean ≥ 0.80) on all six fresh seeds. The
     re-referenced probe threshold (0.75) therefore decided nothing.
2. **Patience was not the lever; capacity was.**
   - Patience 300 gave fits identical to patience 100: the joint objective had no later minimum.
   - Raising r from 12 to 16 lifted the weakest training seed's probe from 0.779 to 0.829.
3. **The remaining gap is concordance relative to C_ceiling.**
   - Across the six fresh test splits, C − C_ceiling lies between −0.008 and −0.022. Two splits (seeds 92, 143)
     fall just beyond the 0.020 margin.
   - On seed-92 validation, five training seeds give C − C_ceiling between −0.014 and −0.021, with a training-seed
     SD of 0.0026.
   - So the 0.0022 shortfall is within training-seed variation. The systematic part (about −0.016) is the gap
     between a linear AFT head on the latent and the nonlinear true risk formula, which C_ceiling applies.
4. **R3 failed in an unexpected direction.** Censoring awareness works: the censoring-aware error is about half the
   naive one on all three training seeds. But R3 also requires \|C_censored − C_R2\| ≤ 0.02, and here C on the
   censored data (0.8638) exceeded R2's C (0.8384) by 0.0254.

### 0.3 Decision requested (nothing has been run)

Seeds 92 and 141–145 are now used; any further gate needs new fresh seeds.

1. **(Recommended) Revision 4f: the risk mapping.**
   - Keep the 4e representation fixed.
   - Pre-register a Step-1 search on seed-21 validation over two factors:
     - the risk head: linear (current) vs a small nonlinear head on the latent;
     - averaging the risk score over training seeds 0–2.
   - Rationale: these target the two measured parts of the gap, the head's approximation error (about −0.016) and
     single-seed variance (SD 0.0026). D74's negative result for an MLP head was obtained on the much weaker 4b
     latent, so it needs re-testing.
2. **R3's form (your call, prospectively only).**
   - R3 is two-sided, so it fails when censored-data concordance is *better* than R2's.
   - A one-sided form (C_censored ≥ C_R2 − 0.02) would match its stated purpose. It would have passed on seed 92,
     which is exactly why it can only be adopted for new seeds.
3. **Or accept the verdict.** The MLP stays not gate-qualified, and H1 stays blocked.

H1 stays blocked under every option until a gate is passed.

### 0.4 Evidence, kept separate by kind

| Kind | Status |
|---|---|
| **Software correctness (tests)** | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (the current 4e gate's original R1, R2 and R3: live, real), {{V:xfailed}} expected failures (historical gates, strict), {{V:skipped}} skipped (the documented N/A case and the 4d gate file, which never ran) |
| **Learnability (gate)** | **Not achieved.** Revision 4e passes R1-A and the probe part, fails R2's concordance part by 0.0022 and fails R3; confirmations 4/5 |
| **Survival prediction (real data)** | Unchanged: descriptive only. PBC E2 provisional primary, E1 sensitivity |
| **Symbolic approximation** | KAN-AFT only. **Not performed for KAN-FAE** (Step 3 not run) |
| **Scientific novelty** | **No claim** |
| **Publication readiness** | **No** |

## R7.1 Owner decisions and fact check (D91)

- **Revision 4e.**
  - Fixed at 4d cand1: fa compact, 17 frequencies, spectral balance, clip 1.0, Adam lr 1e-3, two-checkpoint
    stopping, burn-in 300.
  - Searched: r ∈ {12, 16} × objective patience ∈ {100, 300} × α ∈ {1, 2}.
- **Two-tier selection.**
  - Tier 1 is the D87 multi-seed rule.
  - Tier 2 is an owner fallback with every seed's probe mean ≥ 0.775.
- **Gate criteria.**
  - R1-A.
  - R2 concordance with C_ceiling margin 0.020.
  - **R2 probe part re-referenced to mean ≥ 0.75 and ω ≥ 0.75**, with the 0.78/0.80 and 0.80 definitions reported
    side by side.
  - R3 unchanged.
- **Fact check of the rationale against the records.** All the cited numbers are correct:
  - ω probe 0.824 on the quadrature-projection spectrum;
  - b least-squares R² 0.835;
  - test probe means 0.755 and 0.752 on seeds 61 and 122;
  - ω ≤ 0.20 for the frequency-blind fronts;
  - 4d timing.
  - One nuance: 0.824 is the probe on the raw spectrum, not an upper bound. Log Fourier power gives 0.941, and the
    learned latents reached 0.923.
- **Disclosure.** The probe re-reference and Tier 2 were set after seeing rounds 4–6. They are prospective only with
  respect to seeds 92 and 141–145, which had never been generated. R7.6 shows what they would have changed
  historically: no verdict.

## R7.2 Step 1: pre-registered search (D92) and result (D94)

{{T:step1_candidates}}

**Per seed:**

{{T:step1_seeds}}

**Determinism check:** cand0 is 4d cand1 re-fitted in round 7.

{{T:repro}}

**Reading.**
- **Patience.** Patience 300 restores exactly the same objective-minimum state as patience 100 (epochs 944–1210);
  it only trains 200 epochs longer before restoring it.
- **α.** α = 2 improves the latent at r 12 (minimum seed probe 0.779 → 0.804) at a tiny concordance cost.
- **Latent size.** r 16 improves it more (0.829 at α 1; 0.846 at α 2).
- **Selection.** By the registered rule (highest mean validation C), cand4 was selected; it tied exactly with cand6.
- **Information-only fits (seed 0).** fae_linear: validation C 0.8392, probe 0.901. fae_kan: validation C 0.8298,
  probe 0.714.

## R7.3 Step 2: gate on fresh seed 92 (D93, D95)

{{T:gate_4e}}

**Reproduction inside the final suite** (single-threaded everywhere since D86):

{{T:gate_repro}}

## R7.4 Fresh confirmations, seeds 141–145

{{T:confirm_4e}}

## R7.5 Why the gate failed: diagnostic on seed-92 train/validation (D96, D97)

The diagnostic was registered before running and used no test data.

{{T:diag92}}

**Reading.**
- **The concordance shortfall is within training-seed noise.** The gate fits one training seed (0). Across seeds
  0–4, validation C varies with SD 0.0026, more than the 0.0022 shortfall.
- **The gap to C_ceiling is systematic.** It averages about −0.016 on this split and −0.008 to −0.022 across the six
  fresh test splits. C_ceiling applies the true nonlinear risk formula to per-individual parameter estimates; the
  model uses a linear AFT head on the latent. With the latent now encoding the parameters well, this head gap is
  the most plausible remaining source.
- **R3.** The censored-data model's C (0.8638) exceeds R2's single fit (0.8384), so the symmetric tolerance fails.
  R3's purpose (censoring awareness) is met: the error is about half the naive error.

## R7.6 The probe re-reference, retrospectively (information only)

Every historical gate seed and confirmation, re-scored under the three probe definitions. **No verdict is changed.**

{{T:retro}}

**Reading.**
- Under 0.75/0.75, the probe part would have passed on the 4b and 4c gate seeds, but 4b and 4c would still have
  failed overall (concordance, and 4c's confirmations).
- For Revision 4e, the definition made no difference: it passes all three on every seed.

## R7.7 All gate looks so far

{{T:gate_overview}}

## R7.8 Decisions in round 7 (D91–D98)

{{T:decisions_r7}}

## R7.9 Tests

{{T:suite_table}}

**New tests:** `tests/test_learnability_fa4e.py` (the 4e gate). `tests/test_learnability_fa4d.py` stays as the
skipped 4d record (D93).

{{T:new_tests}}

The module-level SKIPPED row is a collection check run while writing the pre-registration, before the Step-1
selection existed (run `20261018T175917`; "pre-registered selection not produced yet"). No gate code ran in it.

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record)

{{T:test_runs}}

## R7.10 PBC endpoint (provisional; unchanged)

E2 (death or transplantation) is the provisional primary endpoint and E1 a sensitivity analysis, pending the
professor's confirmation (`docs/pbc_endpoint_justification.md`).

{{T:pbc_E2}}

{{T:pbc_E1}}

## R7.11 Task ledger

{{F:docs/task_ledger.md}}

## R7.12 Reproducibility

{{F:docs/reproducibility.md}}

# Parts B–G: historical records

Part B (round 6), Part C (round 5), Part D (round 4), Part E (round 3), Part F (round 2) and Part G (round 1) are
regenerated from the same records. Each part renders from the test records up to its own round's final suite run.
Their numbers are unchanged history; their conclusions on R2 are superseded by Part A. Embedded living documents
(audit, ledger, decision rows, reproducibility) show their current version.
