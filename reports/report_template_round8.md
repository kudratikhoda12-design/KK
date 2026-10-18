<p class="small">Research prototype report, revision 8 · generated from saved records by
<code>scripts/build_report_round8.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A is round 8 (owner decisions D99). Parts B–H are the round-7 to round-1
reports, each regenerated from its own round's records.</p>

## 0. Round-8 summary (plain English)

### 0.1 Outcome

| Step | Result |
|---|---|
| **Step 1: Revision 4f search** (risk head × multi-seed averaging; seed-21 train/validation only) | **{{V:step1}}**: selected the **{{V:head_mode}}** head (a small MLP AFT head trained jointly) with risk averaged over **{{V:k}}** training seeds. The 4e linear fits reproduced bit for bit |
| **Step 2: gate, fresh seed 93** | R1-A PASS; **R2 PASS** (ensemble C {{V:C}} ≥ {{V:thr}}; probe {{V:probe}}, ω {{V:omega}}); **R3-A PASS** |
| Confirmations, fresh seeds 151–155 | **{{V:conf}} pass** |
| Verdict (pre-registered, D101) | **{{V:verdict}}**: the first gate any design has passed. It also holds under round 7's stricter concordance margin and two-sided R3 (R8.5) |
| **Step 3** | **Run** (D104, D105): H1, H2, H3, controls, secondary settings, all ablations AB1–AB8, KAN-FAE pruning and symbolic extraction, PBC and C-MAPSS |
| **H1 (primary): KAN-FAE vs MLP-FAE** | fae_kan − fae_mlp in survival-curve error **{{V:h1_diff}}**, 95% CI {{V:h1_ci}}, p {{V:h1_p}}, replicates KAN worse / better {{V:h1_sign}}. **No evidence that KAN helps; weak evidence it is worse** |

### 0.2 What we learned

1. **The conventional baseline finally qualifies.**
   - A nonlinear risk head on the r = 16 representation, with risk averaged over three training seeds, passed the
     gate on fresh seed 93 and 5/5 fresh confirmations.
   - It would also have passed under round 7's stricter criteria, so the verdict does not rest on the round-8
     relaxations.
2. **H1 (primary): no KAN advantage.**
   - On P0, KAN-FAE has slightly *higher* survival-curve error than the parameter-matched MLP-FAE: +0.0078, 95% CI
     [+0.0002, +0.0155], worse on 5/5 replicates. IBS points the same way.
   - By the protocol this is weak evidence that the KAN encoder is worse under this configuration. It is not
     evidence of an advantage.
3. **The pattern repeats elsewhere.**
   - fae_kan is worse than fae_mlp in every secondary synthetic setting and on C-MAPSS (Harrell C −0.021, Nadeau–Bengio
     CI excludes 0).
   - On PBC the two are indistinguishable.
4. **H2: reconstruction did not help survival.** α = 1 vs α = 0 makes no detectable difference, and the AB1 α sweep
   (0–10) is flat.
5. **H3 is reversed.** Under sparse sampling the *flattened* autoencoder has lower curve error than both functional
   autoencoders (−0.014 vs fae_mlp, 5/5 replicates).
6. **Training strategy matters more than the encoder family.**
   - Reconstruction pre-training followed by joint training (S3) is best for both encoders (C 0.867 / 0.866), and
     under S3 the KAN–MLP gap disappears.
   - A *linear* encoder block (AB3) does as well as either nonlinear one: the frequency-aware front does the
     nonlinear work.
7. **KAN pruning removes almost nothing, and symbolic substitution changes nothing.**
   - At the reference threshold, pruning removes 0–1 of 512 risk-encoder edges; survival metrics are unchanged and
     reconstruction gets worse.
   - Symbolic formulas replace 56% of the risk-encoder edges with no loss. But they map abstract front features to
     latent units, so they are not interpretable risk formulas.

### 0.3 What this does and does not show

- **Statistical strength.** Five replicates, and one pre-registered primary test (p = 0.047). Everything else is
  exploratory, with no correction for multiplicity.
- **H1 is conditional on C4F.** C4F was selected and gated for the MLP. The KAN arm shares it (as the protocol
  requires) and, to stay parameter-matched at r = 16, carries a width-3 decoder bottleneck. The S3 control shows the
  KAN deficit can vanish under another training strategy. H1 therefore says "not better under the MLP-selected
  configuration", not "KAN cannot help".
- **How the gate criteria evolved.** They were relaxed over rounds 5–8, every change disclosed. 4f passes the round-7
  set as well; the strict round-5 margin (0.015) would fail it on the gate seed.
- **Real data.** PBC E2 is still a provisional endpoint. C-MAPSS calibration is poor for both neural arms
  (D-calibration p ≈ 0). The neural arms are not better than the round-1 comparators on either data set.
- **Symbolic formulas** describe single encoder edges on abstract features, so they do not add interpretability at
  the level of the curve parameters or covariates.

### 0.4 Evidence, kept separate by kind

| Kind | Status |
|---|---|
| **Software correctness (tests)** | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (the 4f gate's original R1, which is reported but not gating, as in every round), {{V:xfailed}} expected failures (historical gates, strict), {{V:skipped}} skipped |
| **Learnability (gate)** | **Achieved** by the conventional MLP (Revision 4f), under pre-registered criteria, on fresh seeds |
| **KAN vs MLP (H1)** | Run. Primary contrast favours the MLP (R8.8) |
| **Survival prediction (real data)** | Neural arms run on PBC and C-MAPSS (R8.11). Exploratory; PBC E2 provisional |
| **Symbolic approximation** | KAN-FAE: edge formulas for the risk-state encoder (replicate 1000). These are formulas in abstract front features, **not** clinical or curve-parameter formulas (R8.10) |
| **Scientific novelty** | **No claim** |
| **Publication readiness** | **No** |

## R8.1 Owner decisions and fact check (D99)

- **Revision 4f.**
  - The 4e representation is fixed.
  - Risk head: linear, nonlinear trained jointly, or nonlinear refitted on the frozen latent.
  - Risk and survival averaged over training seeds 0–2, or not.
- **Gate criteria.**
  - R2 probe part restored to 0.78 / 0.80.
  - R2 concordance margin widened to C_ceiling − 0.025.
  - R3 made one-sided (R3-A).
  - Fresh seeds 93 and 151–155.
- **Fact check.** All the cited round-7 numbers are correct. Two statements were corrected:
  - the original protocol had no 0.025 tolerance relative to a data ceiling; R2 was C ≥ C_oracle − 0.03;
  - RSF in round 1 averaged *metrics* over three separately fitted seeds; it did not average predictions.
- **Disclosure.** Under the round-8 criteria, the round-7 model would already have been CONFIRMED. R8.5 shows this,
  and also that the 4f verdict does not depend on it.

## R8.2 Step 1: pre-registered search (D100) and result (D102)

{{T:step1_candidates}}

**Per seed** (and the information-only fae_linear and fae_kan fits under the selected rule):

{{T:step1_seeds}}

**Determinism check:** the 4f linear fits against round 7.

{{T:repro}}

**Reading.**
- **Selection.** The jointly trained nonlinear head gives the best single-seed and ensemble validation C. Its
  latent probes are slightly lower but within the rule.
- **The refit mode was handicapped by the registered burn-in.** It stopped at the epoch-300 floor on every seed:
  its validation NLL was already rising, so the burn-in forced it to over-train.
- **Part of the ensemble gain is the comparison itself.** k = 3 scores an ensemble C, while k = 1 scores a mean of
  single-seed C values, so the rule favours k = 3 by construction.

## R8.3 Step 2: gate on fresh seed 93 (D101, D103)

{{T:gate_4f}}

**Reproduction inside the final suite:**

{{T:gate_repro}}

## R8.4 Fresh confirmations, seeds 151–155

{{T:confirm_4f}}

## R8.5 What decided the verdict: each model under each round's criteria

{{T:criteria_robustness}}

- **4e.** Its round-8 confirmation would have come from the criteria change alone.
- **4f.** It passes both sets, so its confirmation does not rest on the relaxed margin or the one-sided R3. With the
  strict 0.015 margin, the gate would fail (0.8405 < 0.8422).

## R8.6 Parameter matching at r = 16 (D31)

{{T:d31}}

**Caveat.** Every within-tolerance KAN design at r = 16 has a decoder bottleneck narrower than r (width 3 for the
synthetic data). The KAN arm is parameter-matched, not architecture-matched.

## R8.7 Step 3 design and run status (D104, D105)

- **Configuration.** C4F (nonlinear joint head, 3-seed ensemble) for every neural arm.
- **Tuning.** Tuned analyses keep protocol 2.1 (lr × wd at seed 0, by validation IBS); seeds 1–2 are then fitted.
- **What is reported.** The primary result per replicate is the ensemble on the test split; per-seed results are
  secondary.
- **Data.** Synthetic replicates 1000–1004 (n = 1000). Every run is counted below.

{{T:step3_status}}

**Budget.**
- **Synthetic:** 20:03–21:48 UTC, about 1 h 45 min of wall clock on 4–7 parallel single-thread processes, within the
  protocol's 4 h cap.
- **Real data:** 20:19–21:22 UTC.
- **Completeness:** every pre-registered item ran, and no run failed or diverged. The only exclusions are the
  pre-registered budget limits: symbolic fits for decoder edges and for replicates 1001–1004.
- **Lane change:** one lane was re-split mid-run so that AB1/AB3 and AB4 ran in parallel. This changed no cell's
  configuration or seed.

## R8.8 H1 (primary): KAN-FAE vs MLP-FAE on P0

{{T:h1_table}}

**Per replicate (ensembles):**

{{T:h1_reps}}

**Reading (protocol §6).**
- **Primary result.** The 95% t-interval for the curve-error difference excludes 0, on the side favouring fae_mlp.
- **The Wilcoxon test cannot reach 0.05 with five pairs.** Its minimum p is 0.0625, which is what we observe (KAN
  worse in all five).
- **Scale.** The effect is small: about 8% of the MLP's curve error. With n = 5 and p = 0.047, it is weak evidence.

## R8.9 H2, H3 and training controls

{{T:contrasts}}

{{T:cells_main}}

**Reading.**
- **H2.** The reconstruction term does not change survival accuracy in either encoder: all CIs include 0.
- **H3.** The hypothesis that functional autoencoders beat a flattened one under sparse sampling is not supported. The
  difference goes the other way for curve error, and C is also higher for flat_ae (+0.012 to +0.016).
- **Controls.**
  - S1 (reconstruction-only encoder, then a head on the frozen latent) is clearly worst, especially for KAN.
  - S3 is best for both encoders.

## R8.10 KAN-FAE pruning and symbolic extraction

{{T:prune}}

{{T:symbolic}}

**Replicate 1000** (the only replicate with symbolic fits, pre-registered), each stage on the same models:

{{T:prune_rep1000}}

**Reading.**
- **Pruning finds almost nothing to remove.** The KAN encoder's edges are not small at θ = 1e-2, because BatchNorm
  makes every input unit-scale. Sparsity regularisation (λ = 1e-3) did not change that for the risk path. It did
  make the objective-state encoder more fragile: reconstruction after pruning degrades from 0.32 to 0.51–0.58.
- **Substitution is behaviour-preserving but not explanatory.** About 56% of edges have an accurate library formula
  (validation R² > 0.99), and swapping them in leaves every survival metric unchanged. Each edge maps one abstract
  front feature to one latent unit, and the risk then passes through an MLP head. No compact formula for risk in
  terms of the curve parameters results.

## R8.11 Secondary settings, ablations and real data

**Secondary synthetic settings:**

{{T:cells_secondary}}

**Ablations AB1–AB8** (P0, untuned; AB2 = controls; the default cells are H2's):

{{T:cells_ab}}

**Real data** (fae_kan − fae_mlp, Nadeau–Bengio corrected 95% CI over 5 outer splits):

{{T:real}}

**Reading.**
- **PBC.** fae_mlp and fae_kan are indistinguishable. Both sit within the range of the round-1 comparators (E2:
  Harrell C 0.829 vs 0.824–0.846; IBS 0.133–0.138 vs 0.132–0.140).
- **C-MAPSS.** fae_kan is worse than fae_mlp on concordance in 5/5 splits. fae_mlp matches the comparators on C but
  not on RUL RMSE (37.9 vs 34.2–37.2), and both neural arms are poorly calibrated (D-calibration p ≈ 0).
- **Status.** All real-data results are exploratory.

## R8.12 All gate looks, decisions and tests

{{T:gate_overview}}

{{T:decisions_r8}}

{{T:suite_table}}

**New tests:**

{{T:new_tests}}

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record)

{{T:test_runs}}

## R8.13 Task ledger and PBC endpoint

E2 (death or transplantation) is the provisional primary endpoint and E1 a sensitivity analysis, pending the
professor's confirmation (`docs/pbc_endpoint_justification.md`). The round-1 comparators:

{{T:pbc_E2}}

{{T:pbc_E1}}

{{F:docs/task_ledger.md}}

## R8.14 Reproducibility

{{F:docs/reproducibility.md}}

# Parts B–H: historical records

Part B (round 7), Part C (round 6), Part D (round 5), Part E (round 4), Part F (round 3), Part G (round 2) and Part H
(round 1) are regenerated from the same records. Each part renders from the test records up to its own round's final
suite run. Their numbers are unchanged history. Embedded living documents (audit, ledger, decision rows,
reproducibility) show their current version.
