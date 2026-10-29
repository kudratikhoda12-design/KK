# Round 19 development plan (written and committed before the development runs; D147 is locked after them)

Owner instruction (2026-10-29): open Round 19, keeping the Round-18 confirmation-split decision but refitting the
chosen model on the full validation set and testing against the usual full-validation B2-S; development check on
already-used Round-17/18 seeds only; pre-register (D147) before any formal seed; formal run (D148); deliverables
(D149); keep Rounds 1–18 untouched; commit locally, do not push.

## 1. Fact check of the instruction

| Statement | Check |
|---|---|
| Against b2s_cox_fullval, Cox S-Sparse +0.0043 [−0.0006, +0.0093] (Round 18) | Correct (D145) |
| b2s_cox on val_select "lost 0.0058 (0.0864 vs 0.0806)" | The means are correct; the paired difference (full validation − val_select) was −0.0058 [−0.0131, +0.0014], p 0.11, so the loss from halving the validation set is a point estimate, not a significant difference |
| Fresh blocks 4700–4719, 4800–4819, 4900–4919 | Not used as a seed, job name or code literal anywhere (checked 2026-10-29) |

## 2. Design (implemented, default off; `kanfae.models.fres.ConfirmRefitAnchor`)

- Steps (1)–(2) are Round 18's, unchanged: val_select for tuning, early stopping, the candidate fine-tunes and the
  step-(1) choice (for the anchored model and for a val_select B2-S that takes part in the decision), val_confirm only
  for the one-vs-one decision with a margin.
- **Not anchored:** the estimator returns the usual **full-validation B2-S** (`b2s_<loss>_fullval`) unchanged, so those
  replicates contribute a difference of exactly 0 to the primary contrast.
- **Anchored:** the chosen state's masks (trajectory channels and last-value edges of the step-(1) candidate) are
  applied to an anchored ensemble that uses the whole validation set; every member is fine-tuned with the mask fixed,
  early-stopping on the whole validation set. Two refit variants are compared in development:
  - **retrain**: a new anchored ensemble, (lr, wd) tuned and early-stopped on the whole validation set (D112), masked
    and fine-tuned;
  - **finetune**: the step-(1) anchored ensemble (trained on val_select) fine-tuned once more with the mask,
    early-stopping on the whole validation set.
- Families: S-Last (Cox and log-normal AFT), S-Sparse (Cox), S-Misspec-lognormal (log-normal AFT). S-Sparse AFT, which
  was in no registered criterion, is dropped to save computation.
- Tests R19-1 (val_confirm untouched before the decision), R19-2 (not anchored -> the full-validation B2-S exactly,
  max |diff| 0.0), R19-3 (refit early-stops on the whole validation set, trains on the training split only, masks on
  every member; the test split is not an input), R19-4 (both variants); Round 15–18 tests pass with default flags;
  reproduction of a Round-18 job (sparse_4500) with the new code.

## 3. Development runs (no formal seed)

Already-used Round-18 seeds, treated as development data: S-Last 4400–4419, S-Sparse 4500–4519, S-Misspec-lognormal
4600–4619 (60 jobs), margin 0, recording both confirmation losses and, for the step-(1) state of every
replicate, the test metrics of both refit variants (and of the Round-18 anchored model), so every (variant, margin) is
evaluated from the same fits.

## 4. Variant and margin rule (fixed now)

Margin grid {0, 0.0025, 0.005, 0.01, 0.02}. For variant v and margin l, the estimator of a replicate is the refitted
anchored model (variant v) if loss_confirm(anchored) < loss_confirm(val_select B2-S) − l, else the full-validation
B2-S; U = upper 95% paired-t bound of (estimator − full-validation B2-S) survival-curve error.
- For each variant: **Cox** margin = the smallest l with U(S-Sparse) < +0.0025 and U(S-Last) < 0; **AFT** margin = the
  smallest l with U(S-Misspec) < +0.0025 and U(S-Last) < 0; fall-backs as Rounds 17–18 (the l with the smallest
  trajectory-scenario U among those with U(S-Last) < 0; else 0).
- **Variant:** the one whose Cox and AFT margins are both obtained by the rule (not a fall-back) with the smaller sum
  of the two margins; ties, or neither variant meeting the rule for both families, -> the one with the smaller
  max(U(S-Sparse, Cox), U(S-Misspec, AFT)) at its margins; remaining ties -> retrain.

## 5. Go / no-go (fixed now)

At the chosen variant and margins, on the development seeds: Cox U(S-Sparse) < +0.005 and U(S-Last) < 0, AND AFT
U(S-Misspec) < +0.005 and U(S-Last) < 0 (all against the full-validation B2-S). Otherwise the results are reported to
the owner and no formal seed is used.

## 6. Planned registration (D147)

Fresh seeds S-Last 4700–4719 (3–8 visits), S-Sparse 4800–4819 (3–8 visits), S-Misspec-lognormal 4900–4919 (dense); 60
jobs.
- **H-R19a (Cox)**, primary contrast b2s_anchor_cox_refit − b2s_cox_fullval (survival-curve error): S-Last upper 95%
  bound < 0 AND S-Sparse upper bound < +0.005.
- **H-R19b (log-normal AFT)**, b2s_anchor_aft_refit − b2s_aft_fullval: S-Misspec upper bound < +0.005 AND S-Last upper
  bound < 0.
- Secondary: the same contrasts against the val_select B2-S and landmark Cox, and the Round-18 estimator from the same
  fit. Every contrast: paired mean, 95% t-interval, two-sided paired-t p, wins / losses / ties; anchor rates.
  Everything else exploratory; no multiplicity correction; every run reported.
