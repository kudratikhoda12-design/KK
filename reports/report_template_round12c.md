<p class="small">Research prototype report, revision 14 · generated from saved records by
<code>scripts/build_report_round12c.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A is Round 12, the pre-registered simulation study (D113 as amended by
D120, D123 and D126; outcome D127). Part B is revision 13's Part A (L39), Part C revision 12's Part A (L38); Parts D–N
are the round-11 to round-1 reports.</p>

## 0. Round-12 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Study | 4 scenarios × 20 replicates on the fsim simulator (DGP v2), plus the S-Int/P0 continuity check on the Round-1 simulator (20 replicates). Every pre-registered cell ran, including the phase-2 cells (the 12-hour budget was not reached). See R12c.2 for the three failed pruning cells, their reruns and the container restart |
| **H-R12 (confirmatory)** | **Inconclusive.** B2-S Cox (3-seed ensemble) minus AFCM (repaired, V3) in mean identifiable RISE: +0.0065, 95% CI [−0.051, +0.064], paired-t p 0.82, exact Wilcoxon p 0.43; B2-S was better in 8 of 20 replicates. Equivalence within ±0.05 was not shown (TOST p 0.063) |
| **AFCM caveat (D126)** | **The AFCM comparator failed its own acceptance check CK-6c** (φ₁ identifiable RISE 0.247 on seed 22). H-R12 compares B2-S against a comparator that was not validated |
| Sensitivity rows (not confirmatory) | Against the Round-12a AFCM (V0), B2-S is clearly better (−0.179, 19 of 20 replicates). The single-seed B2-S is not better than AFCM V3 (+0.035). Per channel: φ₂ equivalent within ±0.05; φ₁ inconclusive |
| S-Int/P0 continuity | **Equivalent:** fae_kan_s3 − fae_mlp_s3 in survival-curve error −0.0005, 90% CI inside ±0.005 (TOST p 0.005). This re-confirms the Round-9 finding on fresh seeds |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c, kept visible by D126, and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 H-R12 and its sensitivity rows

{{T:hr12_table}}

### 0.3 What the results say (all statements other than H-R12 are exploratory)

1. **Functional CoxKAN (B2-S) and the additive functional Cox model (AFCM V3) recover the S-Add shapes about equally
   well.** H-R12 cannot separate them, and the same holds for survival-curve error (difference +0.001).
2. **Both beat the two-stage FPCA models by a wide margin in shape recovery.** B1-FPCA-1's RISE is about 0.74, against
   0.18 for B2-S, because a linear-in-scores (β-type) effect cannot represent sin(πx/2). Their survival metrics are also
   worse.
3. **B2-deep is not better than B2-S under the S-Int interaction** (curve error +0.004).
4. **Residual calibration helps KAN-AFT under the log-normal misspecification** (curve error −0.005, 19 of 20
   replicates), but not under the log-logistic one.
5. **Pruning finds the active channels but does not reliably remove the null one.**
   - S-Add: recall 0.95, FDR 0.13; the null channel was removed in 12 of 20 replicates.
   - S-Sparse: recall 0.80, FDR 0.08.
6. **The symbolic formulas preserve the effects, but the refit afterwards destroys them.**
   - Right after substitution, RISE is unchanged (S-Add 0.212 vs 0.214 after pruning).
   - The unconstrained refit of all parameters then degrades it to 1.60.
   - So the symbolic stage is usable without the refit and harmful with it (R12c.6).
7. **flat_ae is far worse than B2-S on sparse curves** (curve error +0.078, 20 of 20).

## R12c.1 Owner decision (D126) and fact check

- **Owner decision (D126).** Round 12 ran with AFCM explicitly flagged as not validated.
  - CK-6c stays a visible FAIL.
  - AFCM V3 is the primary comparator, and the Round-12a AFCM (V0) runs as a sensitivity row in S-Add.
- **Fact check.**
  - AFCM's basis and solver can recover both surfaces (seed 914).
  - The owner's reading of CK-6c is only **partly** supported. With λ_t restricted to ≥ 1 on seed 22, φ₁ falls from
    0.247 to 0.133, but φ₂ rises from 0.053 to 0.096: one λ_t shared by the two channels trades one against the other.

## R12c.2 Execution, failures and the restart

{{T:exec_table}}

- **Launch.** The study started at 19:12 UTC on 2026-10-21: 4 single-threaded workers on one priority queue, with the
  plan committed beforehand (d995c5f).
- **Three pruning/symbolic cells failed** (S-Add replicates 10 and 18, S-Sparse replicate 2).
  - Cause: a substituted symbolic formula (for example `log(−0.66x + 1.36)`) left its domain during the unconstrained
    refit, and the resulting NaN crashed the whole cell.
  - The harness was fixed to record such a refit as `refit_nonfinite` (D127). The three cells were rerun with the fix,
    deterministically, and the original failed records are kept.
- **The container restarted at about 04:27 UTC.** It killed the workers with 96 of 100 core jobs complete.
  - The four interrupted jobs were resumed: completed cells were skipped and missing cells re-run.
  - The workers then continued with the original budget deadline.
- **Finish.** All 160 jobs finished at 06:49 UTC, before the 07:12 deadline, so no cell was dropped.

## R12c.3 S-Add: all models

{{T:sadd_table}}

## R12c.4 S-Sparse, S-Int and S-Misspec

**S-Sparse (3–8 visits; B-spline and PACE front ends)**

{{T:ssparse_table}}

**S-Int (no shape metric: the interaction is not additive)**

{{T:sint_table}}

**S-Misspec, log-normal errors** (shapes on the DGP's η scale)

{{T:smis_ln_table}}

**S-Misspec, log-logistic errors**

{{T:smis_ll_table}}

## R12c.5 Secondary contrasts (exploratory, no multiplicity correction)

{{T:secondary_table}}

## R12c.6 Channel selection and symbolic extraction (B2-S Cox, seed 0)

{{T:sel_table}}

- **Selection.**
  - Channel 2 was kept in every replicate.
  - The null channel was removed in 12 of 20 S-Add replicates and in 16 of 20 S-Sparse replicates.
  - Channel 1 (the homogeneous sin effect) was dropped in 2 S-Add replicates and 7 S-Sparse replicates.
  - The selection rule chose the largest θ_rel (0.25) in almost every replicate.
- **Symbolic acceptance.** About two thirds of the active edges were accepted, at a mean validation R² of 0.997.
- **Substitution preserves the shapes; the refit destroys them.**
  - Substituting the accepted formulas leaves the RISE unchanged.
  - The CoxKAN-style refit of all parameters by unconstrained L-BFGS then drifts away from the shapes.
  - In three cells a formula left its domain during the refit (`refit_nonfinite`).
  - The pre-refit measurement was added to the registered plan for information after a development seed showed the
    effect (D126).

## R12c.7 S-Int/P0 continuity

{{T:p0_table}}

## R12c.8 Limitations

- **H-R12's comparator is not validated** (CK-6c FAIL).
- **H-R12 rests on 20 replicates of a single DGP.**
- **RISE uses a projection** that removes only exactly-unidentified directions; weakly identified ones remain in the
  error.
- **B1 shapes are β-type linearisations**, so their RISE measures what a linear functional model can represent, not
  the additive KAN on scores itself.
- **No PySR formula is substituted**, because the substitution path covers linear and library formulas only.

## R12c.9 Decisions (D126–D127)

{{T:decisions}}

## R12c.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R12c.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–N: historical records

Part B is revision 13's Part A (L39); Part C is revision 12's Part A (L38); Parts D–N are the round-11 to round-1
reports, each rendered from its own records. Embedded living documents (ledger, decision rows, reproducibility) show
their current version.
