<p class="small">Research prototype report, revision 9 · generated from saved records by
<code>scripts/build_report_round9.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A is round 9 (owner decisions D107, closing L32). Parts B–I are the
round-8 to round-1 reports, each regenerated from its own round's records.</p>

## 0. Round-9 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| **L32 decision 1: KAN-fairness study** (D108; P0, replicates 1000–1004; 10 cells, 400 runs, all completed) | **Run.** Under tuned S3 the KAN–MLP survival gap vanishes (F1: curve error −0.0001, 95% CI [−0.0030, +0.0028]). Removing the decoder bottleneck does **not** close the gap under S2 (F2, F3) and does not help under S3. **In no arm or strategy is KAN-FAE better than the MLP on survival** |
| Determinism | The repeated S2 cells reproduce the round-8 H1 records **bit for bit** |
| **L32 decision 2: reporting scope** | H1 (R8.8) stays the single confirmatory test: **not supported**. Everything else is exploratory. **No novelty claim** |
| **L32 decision 3: PBC endpoint** | E2 primary, E1 sensitivity. The comparisons' conclusions are the same under both endpoints, but the point estimates differ (R9.7). Clinical sign-off is external |
| Ledger L32 | **Closed** (D107–D109) |

### 0.2 What we learned

1. **The round-8 H1 deficit is a training-strategy effect.**
   - With reconstruction pre-training (S3), tuned, the KAN encoder matches the MLP on survival: curve error, IBS,
     Harrell C and Uno C differences are all ≈ 0 with CIs straddling 0.
   - This holds with the D31 decoder and with an unbottlenecked KAN decoder.
2. **The decoder bottleneck is not what held KAN back on survival.**
   - Under S2, removing it narrows the curve-error gap from +0.0078 to about +0.005 but does not close it.
   - Under S3, swapping in the MLP decoder makes KAN slightly *worse* (+0.006, 5/5).
3. **The bottleneck does limit reconstruction, strongly.**
   - KAN's test reconstruction error is 0.31–0.53 with the width-3 decoder, against 0.12–0.16 with either
     unbottlenecked decoder.
   - This is consistent with the owner's hypothesis for the S1 collapse (round 8). The survival head does not depend
     on reconstruction quality here.
4. **S3 helps every architecture.** It lowers curve error by 0.006–0.016 in all five arms, most for KAN.
5. **No configuration shows a KAN advantage.** The best survival cells (C 0.864–0.868; curve error 0.091–0.093) include
   both KAN and MLP arms and are indistinguishable at n = 5.

### 0.3 Scientific claims and their status (L32 decision 2)

| Claim | Status |
|---|---|
| The conventional MLP functional autoencoder learns the synthetic representation and survival structure (gate) | **Confirmed** under pre-registered criteria on fresh seeds (Revision 4f, round 8) |
| **H1 (primary, confirmatory): KAN-FAE improves on a parameter-matched MLP-FAE** | **Not supported.** Under C4F the KAN arm had higher survival-curve error (R8.8, repeated below); this is the protocol's single confirmatory test |
| KAN-fairness contrasts F1–F3 (round 9) | Exploratory, without multiplicity correction (R9.3–R9.6) |
| H2, H3, controls, secondary settings, AB1–AB8, pruning and symbolic extraction, real data | Exploratory (Part B) |
| Novelty | **No claim** |

### 0.4 Evidence, kept separate by kind

| Kind | Status |
|---|---|
| **Software correctness (tests)** | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (the 4f gate's original R1, reported and not gating, as in every round), {{V:xfailed}} expected failures (historical gates, strict), {{V:skipped}} skipped. New: FC-1..FC-5 |
| **Learnability (gate)** | Achieved in round 8 (unchanged) |
| **KAN vs MLP** | H1 not supported (confirmatory). Round-9 fairness evidence: R9.3–R9.6 (exploratory) |
| **Survival prediction (real data)** | Exploratory; PBC E2 primary, E1 sensitivity (R9.7) |
| **Scientific novelty** | **No claim** |
| **Publication readiness** | **No** |

## R9.1 Owner decisions and fact check (D107)

- **Decisions.**
  - (1) A pre-registered KAN-fairness study.
  - (2) H1 under C4F stays the single confirmatory test, and everything else is exploratory, with no novelty claim.
  - (3) PBC: E2 primary, E1 sensitivity; clinical sign-off is external.
  - (4) Same discipline throughout.
- **Fact check.**
  - **Parameter counts: correct.** fae_mlp encoder 3,152 + decoder 3,428 = 6,580; KAN encoder 5,136 + D31 decoder
    1,599 = 6,735.
  - **R8.9 controls: correct.** S3 C 0.8656 / 0.8669; S1 KAN 0.7438.
  - **The bottleneck explanation of the S1 collapse** is a hypothesis; round 9 tests it indirectly (R9.5).
  - **PBC invariance: qualified.** It holds for the conclusions, not for the point estimates (R9.7).

## R9.2 Fairness design (D108)

{{T:arms}}

**Modes.**
- **Mode B** keeps the exact C4F KAN encoder (its initialisation is bit-identical) and gives it fae_mlp's decoder
  architecture. It is decoder-matched; its encoder + decoder total is 30% larger than fae_mlp's (disclosed).
- **Mode A** replaces the D31 decoder with a single KAN layer 16 → 36, so no hidden width is below r. It is
  compared with an MLP whose hidden width (107) matches its encoder + decoder parameters (+0.3%).

**Protocol.** Every arm uses C4F and protocol 2.1 tuning (lr × wd at seed 0, by validation IBS). The primary result
per replicate is the 3-seed ensemble on test. Test reconstruction error is the individual-weighted MSE of the
objective-state model on the observed test points, averaged over members.

**Tests.** FC-1..FC-5 were written and passed before any fairness fit. A first run failed on a construction error in
FC-2's setup; it is kept in the records.

{{T:new_tests}}

## R9.3 Results by arm and strategy

{{T:cells}}

**Determinism check:** the repeated S2 cells against the round-8 H1 records.

{{T:determinism}}

**Tuning choices** (protocol 2.1, per replicate):

{{T:tuning}}

**Run status** (every run counted):

{{T:status}}

## R9.4 F1: KAN vs MLP under tuned S3 (standard D31 matching)

{{T:F1}}

**Reading.**
- **Survival: no difference.** With reconstruction pre-training, the D31-matched KAN arm and the MLP agree on every
  survival metric (all CIs include 0).
- **Reconstruction: much worse for KAN.** Its test error is about five times the MLP's (0.53 vs 0.10). That is what a
  width-3 decoder would cause, and it does not carry over to survival.

For reference, the confirmatory H1 under C4F (S2, tuned; round 8):

{{T:h1_table}}

## R9.5 F2: removing the decoder bottleneck by sharing the MLP decoder (Mode B)

{{T:F2}}

**Reading.**
- **Reconstruction improves strongly.** Giving the KAN encoder the MLP's decoder lowers its reconstruction error
  (−0.16 under S2, −0.40 under S3, 5/5).
- **Survival does not.** The KAN encoder with the shared decoder stays behind the MLP: +0.005 under S2, and
  +0.006 under S3 with the CI excluding 0. Under S3 it is even slightly worse than the bottlenecked KAN.
- **So the encoder comparison does not favour KAN** once the decoder is held identical. Note that this arm has 30%
  more encoder + decoder parameters than fae_mlp.

## R9.6 F3: unbottlenecked KAN decoder vs parameter-matched wide MLP (Mode A); S3 vs S2

{{T:F3}}

**Supplementary: S3 − S2 within each arm:**

{{T:S}}

**Reading.**
- **Parameter-matched and unbottlenecked:**
  - under S2, the KAN arm is again slightly behind (curve error +0.0055, KAN worse in 5/5; CI just includes 0);
  - under S3 the two are indistinguishable on survival (−0.0020, CI [−0.0070, +0.0031]), while KAN has worse
    reconstruction (+0.085) and lower D-calibration p.
- **Strategy.** S3 lowers curve error in every arm. The improvement is largest for the KAN arms (−0.013 to −0.016),
  which is why they catch up.

## R9.7 PBC endpoint (L32 decision 3)

**Endpoints.** E2 (death or liver transplantation; transplant-free survival) remains the **primary** reported PBC endpoint,
and E1 (death, transplantation censored) the **sensitivity** analysis. The supervising professor's clinical sign-off
is outside the codebase.

**What does and does not depend on the choice.** Every PBC model comparison in this project reaches the same
conclusion under both endpoints: no detectable difference. This covers KAN-AFT vs linear AFT (round 2) and fae_kan vs
fae_mlp (round 8). The point estimates are **not** identical, and some change sign. So the invariance is claimed for
the conclusions only.

{{T:pbc_invariance}}

## R9.8 Decisions, tests, ledger and reproducibility

{{T:decisions_r9}}

{{T:suite_table}}

**All test runs, in chronological order** (every failure is kept; `reports/test_results.csv` has every record)

{{T:test_runs}}

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–I: historical records

Part B (round 8), Part C (round 7), Part D (round 6), Part E (round 5), Part F (round 4), Part G (round 3), Part H
(round 2) and Part I (round 1) are regenerated from the same records. Each part renders from the test records up to
its own round's final suite run. Their numbers are unchanged history. Embedded living documents (audit, ledger,
decision rows, reproducibility) show their current version.
