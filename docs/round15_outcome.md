# Round 15 outcome (Phase-2 methods; D133–D136)

Launched 10:51 UTC on 2026-10-25 after the owner's decisions (D134) and the two development checks (D135); 80 of 80
jobs completed without failure (16:32 UTC). Records `reports/round15/results_lane*.jsonl`; analysis
`reports/round15_summary.json` (`scripts/analyse_round15.py`, committed before the run finished). Seed 24 was not used.
Everything is committed locally; nothing is pushed.

## 1. Confirmatory results (registered rules)

| Hypothesis | Part | Result | Verdict |
|---|---|---|---|
| **H-R15a** symbolic_distill | (i) all 40 distillation refits finite | 39 of 40 (S-Add 3515: every mu non-finite) | fails |
| | (ii) S-Add: distilled − pruned RISE, upper 95% bound < +0.05 | +0.026, 95% CI [−0.006, +0.059] | fails |
| | **overall** | | **not supported** |
| **H-R15b** b2s_anchor_cox | (i) S-Last: anchored − B2-S Cox curve error, CI < 0 | −0.0185 [−0.0222, −0.0149], better in 20/20 | **holds** |
| | (ii) S-Sparse: the same, upper 95% bound < +0.005 | +0.0047 [−0.0001, +0.0094] | fails |
| | **overall** | | **not supported** |

## 2. What the results say

1. **Distillation solves the problem it was built for, but validation selection of μ undermines it.**
   - Against the old unconstrained refit, the distilled model kept the shapes: RISE −0.93 on S-Add and −10.5 on
     S-Sparse, with curve error −0.021 and −0.017.
   - Its validation partial likelihood was slightly better than that of the unrefitted substitution; IBS and C were
     unchanged.
   - *Post hoc (exploratory):* at a fixed μ of 1 or 10 the distilled shapes equal the pruned shapes within ±0.002.
   - The registered rule chose μ by validation NLPL, which picked a loose anchor (μ 0.1 or 0.01) in 8 S-Add replicates,
     and those cost shape accuracy (mean +0.033 and +0.277).
2. **The one non-finite refit was a numerical failure, not a domain failure.**
   - A baseline-covariate edge had a nearly degenerate accepted formula: softplus(205.3x + 384.1) with an output scale
     of 0.0018.
   - The first L-BFGS step sent the parameters to about 1e180, and the line search cannot recover from a NaN trial.
   - The domain guard (active in 5 cells) cannot prevent this; a NaN-safe refit with a fallback to the substituted
     model would.
3. **The anchored model is clearly better than B2-S Cox when the landmark value carries the signal (S-Last), and
   matches it when the gate keeps the trajectory channels.**
   - *Post hoc:* on S-Sparse it matched B2-S Cox when the gate kept channels: +0.0012 [−0.0005, +0.0030] in 16
     replicates (the null channel was removed in 6).
   - It lost when the gate switched everything off: +0.018 in 4 replicates. These false "all off" decisions are why
     part (ii) failed.
   - The residual design (your original specification) was worse than B2-S Cox on S-Sparse (+0.0149, 19/20), as the
     dry run predicted.
4. **Real data (exploratory).**
   - The Round-13 records reproduced exactly (difference 0.0).
   - On PBC E2 the anchored model's IBS (0.135) lies between B2-S Cox (0.150) and landmark Cox (0.127). No interval
     excludes 0 on any data set.
   - The last-value edges alone are often not monotone in the clinical direction (albumin 0.2, prothrombin index 0.2,
     CD4 0.4 of splits). This is not comparable with Round 13: when the gate keeps a channel, its effect is shared
     between the last-value edge and the trajectory term.
5. **AFCM** stays "not validated" (D135); no AFCM model was run.

## 3. Candidates for a next round (untested; each would need its own registration)

1. symbolic_distill with a fixed large μ (or the D120 parsimonious rule: the largest μ within a validation tolerance)
   and a NaN-safe refit that falls back to the substituted model.
2. A gate that cannot switch off all channels on weak validation evidence. For example, require the "all off"
   setting to beat the full model by more than the tolerance (the reverse parsimony), or average the gate over
   bootstrap resamples.
3. Laplace selection for AFCM combined with per-channel λ_t, since Laplace helped on seed 22 but hurt on seed 966.
