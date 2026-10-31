# Round 20 development plan (written and committed before the development runs; D150 is locked after them)

Owner instruction (2026-10-31): open Round 20 adding a non-parametric-error KAN AFT (`log T = KAN(Z) + eps`, eps
unspecified, estimated non-parametrically) next to the Cox and parametric AFT KANs, and fixing the Round-19 Cox S-Sparse
confirmation gate with a two-half cross-confirmation against the 3-seed B2-S ensemble; development check on the
already-used Round-18 seeds 4400–4619 only; formal run on 4700–4919 only if the pre-registered go/no-go rules pass;
deliverables D150–D152; keep Rounds 1–19 untouched; follow D111 (no novelty claim); commit locally, do not push.

## 1. Fact check of the instruction

| Statement | Check |
|---|---|
| Seed blocks 4700–4719, 4800–4819, 4900–4919 unused | Correct (checked 2026-10-31): no record, job name or result file uses them; they appear only as the planned, never-run formal ranges of `scripts/run_round19.py` / `analyse_round19.py` |
| Round-19 Cox S-Sparse "+0.0063 vs < +0.0050" | Correct: +0.0063 is the upper 95% bound (mean +0.0030 [−0.0004, +0.0063]) at the fall-back margin 0.02, variant finetune (D147) |
| Round 19 compared the anchored candidate with a single seed-0 B2-S trained on val_select, in one direction | Correct description of the code (`ResFKAN._fit_twoway`: `heldout_loss(ref.members[0], val_confirm)` against the seed-0 anchored candidate, val_select -> val_confirm only). That this *caused* the miss is a hypothesis, which this round tests; it is not established |
| `src/kanfae/models/fkan_surv.py`, `distill_and_replace` | Neither existed. The model code is `fkan.py` (FKANSurv); the Round-16 distillation is `refit_symbolic(anchor, mu, safe=True)` (fixed-mu, NaN-safe). Created: `fkan_surv.py` (NP-AFT helpers, `NPKANAFT`, `distill_and_replace`) with minimal default-off hooks in `fkan.py` |
| Gehan loss `sum_{i: d_i=1} sum_j max(0, (Y_i - mu_i) - (Y_j - mu_j))` | **Reversed pair.** The convex Gehan objective (Jin, Lin, Wei & Ying 2003) is `sum_{i: d_i=1} sum_j max(0, e_j - e_i)`, whose gradient is the Gehan estimating function `sum_i d_i sum_j (x_i - x_j) 1{e_j >= e_i}` (event i against every j still at risk at e_i). Implemented in the correct form (test R20-1 checks the gradient identity and slope recovery) |
| "True non-parametric" KAN AFT | The model (non-parametric location, non-parametric error law) is the **semiparametric AFT** model of the survival literature; Buckley–James estimation of a KAN AFT location is published for static covariates (KAN-AFT, Jose, Francis & Kattumannil; D42, D111), and this code base already has classical Buckley–James (`survival/censored_ls.py`). New here is only its use with the functional B2-S location and the anchored gate. No novelty claim (D111) |
| Development seeds 4400–4619 | Used three times now: Round-18 formal run, Round-19 development check, Round-20 development check. The cross-confirmation was designed after Round 19 failed on exactly these seeds, so a development pass is weak evidence; the fresh formal seeds are the test |
| Go criterion "S-Last mean contrast < 0" | Weaker than Rounds 17–19 (upper 95% bound < 0). Used as stated for the go/no-go; the formal H-R20b keeps the upper-bound criterion (§6) |
| `verify_deliverables.py -> ALL_CHECKS_PASSED` | The verifier's flag is `ALL_CHECKS_PASS` |

## 2. Design (implemented, default off)

**NP-KAN-AFT** (`FKANSurv(loss="aft", family="np_aft", np_objective=...)`; ensemble `NPKANAFT(objective)` = the D112
protocol: lr x wd tuned on the objective's validation loss, seeds 0–2). `log T = mu + KAN(Z) + eps`, KAN(Z) the B2-S
location (with last-value anchor channels through ResFKAN). Objectives (training and validation loss; the same function
on each split):
- `bj_km`: Buckley–James / Kaplan–Meier residual EM. Before every Adam epoch the KM of the current training residuals is
  re-estimated (largest residual counted as an event); loss = mean (Y* − loc)^2, Y* = log Y for events and
  loc + E_KM[eps | eps > r] = loc + r + ∫_r^∞ S_KM / S_KM(r) for censored subjects (held fixed in the step).
- `rank_km`: smoothed Gehan loss `(1/n^2) sum_{i: d_i=1} sum_j tau softplus((e_j − e_i)/tau)`, tau = 0.05 (log-time
  units; fixed, not tuned); the intercept (not identified) is held at its start value.
- Prediction: `S(t | Z) = S_KM(log t − loc(Z))`, plain KM of the training residuals at the fitted location,
  right-continuous: 1 below the smallest residual, S_KM(largest residual) above the largest.
- `distill_and_replace(model)`: the Round-16/17 fixed-mu pipeline (stages 1–2, max_abs_arg 50, guarded substitution,
  mu = 10, safe=True, 200 L-BFGS iterations); KM re-estimated on the distilled residuals; rollback to the spline model if
  any value is non-finite or the validation loss rises above its pre-distillation value + val_tol (default 0). Tested
  (R20-4); not part of the hypotheses.

**CrossConfirmRefitAnchor** (`fres.py`). The Round-18 split h1 = val_select, h2 = val_confirm. For each direction
(select on h1, confirm on h2) and (select on h2, confirm on h1): step (1) is the D141 two-way gate on the selection half;
the anchored ensemble at the chosen state (3 seeds, mask fixed, fine-tuned on the selection half) and the plain B2-S
ensemble of the same half are scored on the other half by the **mean member held-out loss** (the 3-seed ensemble, not
the seed-0 member). Confirmed iff in **both** directions loss(anchored) < loss(B2-S) − margin and the anchored state keeps
an active channel. Margin units: Cox and log-normal AFT absolute (as Rounds 17–19: NLPL per event, NLL per subject);
NP-AFT **relative** (margin x |loss(B2-S)|), because the BJ and Gehan losses have objective-dependent scales. Not
confirmed -> the full-validation 3-seed B2-S, exactly. Confirmed -> the h1-direction anchored ensemble fine-tuned once
more with its mask, early-stopping on the whole validation set (the Round-19 "finetune" refit, bit-identical; R20-5).

Families per scenario: Cox on S-Last and S-Sparse (as Rounds 17–19); log-normal AFT on S-Last and S-Misspec-lognormal
(as Rounds 17–19); NP-AFT on S-Last and S-Misspec-lognormal (the AFT pairing; NP-AFT on S-Sparse enters through the
full-validation B2-S comparison of H-R20a).

Tests R20-1 to R20-6 (`tests/test_round20.py`): building blocks (KM, BJ conditional mean, Gehan gradient), survival
recovery without the error family (bimodal and log-normal errors), prediction properties, distillation and rollback,
cross-confirmation (exact reduction, both directions, ensemble losses, refit = Round-19 finetune), and default-off
bit-identity with the Round-19 code (commit 495f57e loaded from git). Also: the Round 15–19 tests, and a Round-19 job
(devsparse_4500) re-run with the new code and compared value by value with its record.

## 3. Development stage A (no formal seed): NP objective and H-R20a go check

Jobs: S-Sparse 4500–4519 and S-Misspec-lognormal 4600–4619 (40 jobs, `R20_MODE=devA`), full-validation B2-S ensembles:
Cox, NP-AFT bj_km, NP-AFT rank_km, and on S-Misspec also Weibull KAN-AFT.

**Objective rule (fixed now):** the objective with the smaller average of its S-Misspec and S-Sparse mean survival-curve
errors; ties -> bj_km.

**GO-a (fixed now)**, at the chosen objective, survival-curve error, paired over the 20 development replicates:
S-Misspec upper 95% bound of (NP-AFT − Cox) < 0 AND of (NP-AFT − Weibull AFT) < 0, AND S-Sparse upper 95% bound of
(NP-AFT − Cox) < +0.005.

## 4. Development stage B (no formal seed): cross-confirmation

Jobs: S-Last 4400–4419, S-Sparse 4500–4519, S-Misspec-lognormal 4600–4619 (60 jobs, `R20_MODE=devB`), NP-AFT with the
stage-A objective, margin 0 in the fit; every record holds both directions' ensemble held-out losses and the test
metrics of the full-validation refit whatever the decision, so every margin is evaluated offline: for family f and
margin l the estimator is the refit if cross_confirm_decision(dirs, l) else the full-validation B2-S; contrast =
estimator − full-validation B2-S (survival-curve error).

**Margin rule (fixed now)**, grid {0, 0.0025, 0.005, 0.01, 0.02} (absolute for Cox and log-normal AFT, relative for
NP-AFT), trajectory scenario S-Sparse for Cox and S-Misspec for both AFT families: the smallest l with
U(trajectory) < +0.0025 and mean(S-Last) < 0; fall-back: among the l with mean(S-Last) < 0 the one with the smallest
U(trajectory) (ties: smaller l); else 0. U = upper 95% paired-t bound.

**GO-b (fixed now)**: for each of Cox, log-normal AFT and NP-AFT at its margin: mean(S-Last contrast) < 0 AND
U(trajectory) < +0.005. All three must pass.

## 5. Go / no-go

Formal hypotheses are run only where their go rule passes: GO-a -> H-R20a; GO-b -> H-R20b; neither -> no formal seed is
used and the results are reported to the owner.

## 6. Planned registration (D150)

Fresh seeds S-Last 4700–4719, S-Sparse 4800–4819, S-Misspec-lognormal 4900–4919 (60 jobs); NP objective and margins as
chosen in §3–4; analysis by paired t over 20 replicates, every contrast with mean, 95% interval, two-sided p,
wins / losses / ties.
- **H-R20a** (if GO-a): S-Misspec-lognormal, survival-curve error of b2s_npaft_fullval − b2s_cox_fullval upper 95% bound
  < 0 AND b2s_npaft_fullval − b2s_weibull_fullval upper bound < 0. Registered secondary: S-Sparse NP-AFT − Cox (upper bound
  against +0.005); exploratory: NP-AFT − log-normal AFT on S-Misspec (the correctly specified family).
- **H-R20b** (if GO-b): per family f, b2s_anchor_f_cross − b2s_f_fullval: S-Last upper 95% bound < 0 AND trajectory
  scenario (Cox: S-Sparse; AFT families: S-Misspec) upper bound < +0.005; supported iff all three families meet both.
- Secondary: confirmation rates; the contrasts against the h1 B2-S and landmark Cox; the Round-18-style one-vs-one losses
  recorded in the h1 direction. Everything else exploratory; no multiplicity correction; every run reported.

## 7. Execution

Four single-threaded lanes, file-locked queue, resumable: `R20_MODE=devA scripts/run_round20_lanes.py --lane k`
(k = 0..3), then `scripts/analyse_round20_dev.py A`; after the objective is fixed in `run_round20.SETTINGS["devB"]` (the
only edit between the stages, made by the stage-A rule), `R20_MODE=devB ...`, then `scripts/analyse_round20_dev.py B`.
Cost from the Round-19 timings and a timing run on seed 4600 (one B2-S ensemble: Cox 383 s, NP bj_km 400 s, NP rank_km
232 s): stage A about 5 h, stage B about 20 h on four lanes. Before the stages: the harness was smoke-tested on five
development jobs with 5-epoch budgets (scratch output, not a result).
