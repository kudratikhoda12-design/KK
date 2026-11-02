<p class="small">Research prototype report, revision 22 · generated from saved records by
<code>scripts/build_report_round20.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype; the manuscript is a
draft for the owner's review, not peer reviewed; no novelty claim</b>. Part A is Round 20: a semiparametric functional
KAN AFT (NP-KAN-AFT: log T = KAN(Z) + ε with the law of ε left unspecified and estimated by Kaplan–Meier; H-R20a) and a
two-half cross-confirmation of the landmark anchor against the 3-seed B2-S ensembles for Cox, log-normal AFT and NP-AFT
(H-R20b); two development stages on already-used seeds, then a formal run on fresh seeds (D150–D152). Part A continues
with the updated standalone manuscript (Rounds 1–20) and its verification. Part B is revision 21's Part A (Round 19 and
the Round-19 manuscript); Parts C–V are revision 21's Parts B–U.</p>

## 0. Round-20 summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Changes tested (D150) | **NP-KAN-AFT** (`NPKANAFT`, `family="np_aft"`, off by default): B2-S location, error law estimated by Kaplan–Meier of the residuals, objective Buckley–James (bj_km) or smoothed Gehan rank (rank_km); **cross-confirmed anchor** (`CrossConfirmRefitAnchor`, off by default): anchoring is used only if the anchored 3-seed ensemble beats the 3-seed B2-S ensemble on held-out data in both directions of the validation split, and is then fine-tuned on the whole validation set; otherwise the full-validation B2-S, exactly |
| Development stage A | {{V:nA}} jobs ({{V:nA_failed}} failed), already-used seeds: objective rule chose **{{V:objective}}**; GO-a **{{V:go_a}}** |
| Development stage B | {{V:nB}} jobs ({{V:nB_failed}} failed), already-used seeds: margins Cox {{V:m_cox}}, log-normal AFT {{V:m_lnaft}}, NP-AFT {{V:m_npaft}} (relative); GO-b **{{V:go_b}}** |
| Formal run (fresh seeds 4700–4719, 4800–4819, 4900–4919) | {{V:nF}} jobs ({{V:nF_failed}} failed) |
| H-R20a (NP-AFT vs Cox and Weibull AFT on S-Misspec) | **{{V:hA}}.** NP − Cox {{V:a_cox}}; NP − Weibull {{V:a_wei}}; secondary S-Sparse NP − Cox {{V:a_sparse}}; exploratory NP − log-normal AFT {{V:a_ln}} |
| H-R20b (cross-confirmed anchor, all three families) | **{{V:hB}}.** Cox: S-Last {{V:b_cox_last}}, S-Sparse {{V:b_cox_traj}} ({{V:b_cox_ok}}); log-normal AFT: S-Last {{V:b_lnaft_last}}, S-Misspec {{V:b_lnaft_traj}} ({{V:b_lnaft_ok}}); NP-AFT: S-Last {{V:b_npaft_last}}, S-Misspec {{V:b_npaft_traj}} ({{V:b_npaft_ok}}) |
| Manuscript | `docs/manuscript_round20.md` / `reports/manuscript_round20.pdf` ({{V:ms_pages}} pages), Rounds 1–20, embedded below: {{V:ms_values}} values logged with their sources, {{V:ms_refs}} references, {{V:ms_check}} |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 What the results say

1. **A Kaplan–Meier error law works as well as knowing the right error family.** On S-Misspec (log-normal errors) the
   semiparametric NP-KAN-AFT was better than B2-S Cox and the Weibull KAN-AFT in every one of the 20 fresh replicates,
   and indistinguishable from the correctly specified log-normal KAN-AFT ({{V:a_ln}}). Its advantage is that it does not
   need the family to be guessed; it is not better than the right parametric model. On S-Sparse it stayed within the
   +0.005 margin of B2-S Cox ({{V:a_sparse}}).
2. **Confirming in both halves of the validation split fixed the Round-19 gap.** Against B2-S trained on the whole
   validation set, the cross-confirmed anchor kept a clear S-Last gain and stayed within +0.005 on the trajectory
   scenario for Cox, log-normal AFT and NP-AFT alike. Round 19's Cox S-Sparse bound was +0.0063 in development; on fresh
   seeds Round 20's is inside the margin.
3. **It works by deciding less often.** On the trajectory scenarios anchoring was confirmed in one replicate per
   family; with the conservative Cox margin (0.02, the edge of the grid) Cox anchored in only {{V:b_cox_conf}} of 20
   S-Last / S-Sparse replicates and gave up part of its S-Last gain (it did not beat landmark Cox there). Always
   refitting would have cost on every trajectory scenario, so the confirmation step, not the refit, carries the
   non-inferiority.
4. **Protocol.** Both go rules passed on the development seeds before any fresh seed was used, every job finished
   (no failures), and both hypotheses were decided by the rules registered in D150.

# Part A: Round 20

## R20.1 Owner instruction (D150) and fact check

- **Instruction (2026-10-31).** Add a non-parametric-error KAN AFT (log T = KAN(Z) + ε, ε estimated
  non-parametrically; Buckley–James/Kaplan–Meier and smoothed-Gehan objectives compared in development) next to the
  Cox and parametric AFT KANs; fix the Round-19 Cox S-Sparse confirmation gate (+0.0063 against < +0.005) by comparing
  against the 3-seed B2-S ensemble and confirming in both directions of the split; develop on the already-used seeds
  4400–4619 only; run the formal evaluation on 4700–4919 only if the go rules pass; follow D111 and the
  pre-registration rules; commit locally, do not push. Later instruction (2026-11-01): "continue and send me the report
  pdf".
- **Fact check** (`docs/round20_devplan.md` §1): the seed blocks were unused; the Gehan pair order in the instruction
  was reversed and is implemented in the convex Jin–Lin–Wei–Ying form; `fkan_surv.py` and `distill_and_replace` did not
  exist and were created; a Buckley–James KAN-AFT for static covariates is already published (KAN-AFT), so no novelty
  claim is made; the cause of the Round-19 miss (one seed-0 member, one direction) was a hypothesis; the go rule
  "S-Last mean < 0" is weaker than the formal upper-bound rule; the verifier's flag is ALL_CHECKS_PASS.

## R20.2 Methods (as implemented)

- **NP-KAN-AFT.** log T = μ + η(Z) + ε with η the B2-S location (D112 ensemble: lr × wd tuned on the objective's
  validation loss, seeds 0–2). bj_km: before every epoch the Kaplan–Meier law of the training residuals is re-estimated
  (largest residual counted as an event) and the loss is the mean squared distance to the Buckley–James targets.
  rank_km: smoothed Gehan loss (1/n²) Σ_{i: δ=1} Σ_j τ softplus((e_j − e_i)/τ), τ = 0.05, intercept fixed. Prediction:
  S(t | Z) = S_KM(log t − η(Z)), a right-continuous step function, 1 below the smallest residual and S_KM(largest
  residual) above the largest. `distill_and_replace` applies the Round-16 fixed-μ distillation with a validation-loss
  rollback (tested; not part of the hypotheses).
- **Cross-confirmed anchor.** For each direction (select on h1 = val_select, confirm on h2 = val_confirm, and the
  reverse) the D141 two-way gate runs on the selection half; the anchored 3-seed ensemble and the 3-seed B2-S ensemble
  of that half are scored on the other half by their mean member held-out loss. Confirmed iff anchoring wins by more
  than the margin in both directions with an active channel (NP-AFT margin relative to |loss|). Confirmed: the
  h1-direction anchored ensemble fine-tuned once more on the whole validation set (the Round-19 finetune refit).
  Otherwise: the full-validation B2-S ensemble, exactly.
- **Tests:** R20-1 to R20-6 (`tests/test_round20.py`: Kaplan–Meier and Buckley–James building blocks, Gehan gradient
  identity and slope recovery, survival recovery without the error family, prediction properties, distillation and
  rollback, cross-confirmation logic and exact reduction, bit-identity of the default paths with the Round-19 code),
  R20-7 to R20-11 (manuscript); the Round 15–19 tests pass; a Round-19 job reproduced its record exactly (68 values).

## R20.3 Development stage A (objective and GO-a; already-used seeds)

{{T:dev_a}}

## R20.4 Development stage B (margins and GO-b; already-used seeds)

{{T:dev_b}}

## R20.5 H-R20a: NP-KAN-AFT (fresh seeds)

{{T:formal_a}}

## R20.6 H-R20b: cross-confirmed anchoring (fresh seeds)

{{T:formal_b}}

## R20.7 Limitations

- **Settings chosen on reused seeds.** The NP-AFT objective and all three margins were chosen on development seeds used
  for the third time (Round-18 formal, Round-19 development, Round-20 development); the Cox margin lies at the edge of
  its grid.
- **Cause not established.** The Round-19 miss and the Round-20 pass are on different seeds; the fresh-seed test
  supports the cross-confirmed estimator, but does not show that the one-member, one-direction comparison caused the
  Round-19 miss.
- **One error law.** NP-KAN-AFT was tested under log-normal errors only; its gain over Cox and Weibull reflects their
  misspecification there. Its survival curve is not extrapolated beyond the largest training residual.
- **No novelty claim (D111).** The semiparametric AFT model and Buckley–James / rank estimation are classical; a
  Buckley–James KAN-AFT for static covariates is published. New here is only their use with the functional B2-S
  location and the anchored gate.
- **One simulator, 20 replicates per scenario; no real data in Round 20. AFCM is still not validated.**

# Part A (continued): the standalone manuscript, Rounds 1–20

## {{V:ms_title}} {#manuscript-title-r20}

{{MANUSCRIPT}}

## R20.8 Manuscript verification

{{T:ms_verification}}

## R20.9 Decisions (D150–D152)

{{T:decisions}}

## R20.10 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

## R20.11 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–V: historical records

Part B is revision 21's Part A (Round 19 and the Round-19 manuscript); Part C is Round 18 and its manuscript; Part D is
Round 17 and its manuscript; Part E is Round 16 and its manuscript; Part F is Round 15; Part G is Round 14 (the first
manuscript draft); Part H is Round 13; Part I is Round 12; Parts J and K are the L39 and L38 parts; Parts L–V are the
round-11 to round-1 reports, each rendered from its own records. Embedded living documents show their current version.
