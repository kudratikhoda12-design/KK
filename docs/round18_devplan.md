# Round 18 development plan (written and committed before the development runs; D144 is locked after them)

Owner instruction (2026-10-28): open Round 18 testing landmark anchoring decided on a confirmation split; development
check on development / already-used seeds only; pre-register (D144) before any formal seed; formal run (D145);
deliverables (D146); keep Rounds 1–17 untouched; commit locally, do not push.

## 1. Fact check of the instruction

| Statement | Check |
|---|---|
| Round 17 chose the anchored state among up to 16 candidates on val and compared it with the comparator on the same val | Correct (D141): 8 joint + 8 static states (identical masks fitted once), step (2) on the same validation losses |
| S-Sparse anchored in 9/20, upper bound +0.0053 | Correct (D142) |
| "biased upward by ~+0.0066" | +0.0066 is the **mean observed validation advantage** of the anchored state on S-Sparse (comparator − anchored validation loss, `reports/round17_summary.json`), not a measured bias. On the test data the same anchored alternative was worse (+0.0063 curve error) and, across replicates, the validation advantage was essentially uncorrelated with the test difference (r = −0.05). This is consistent with selection optimism but does not prove it; Round 18 tests it |
| AFT anchored in 8/20 S-Last replicates (−0.0059 vs −0.0177 un-gated) | Correct (D142; "un-gated" = the step-(1) anchored alternative) |
| Fresh blocks 4400–4419, 4500–4519, 4600–4619 | Not used as a seed, job name or code literal anywhere (checked 2026-10-28) |

## 2. Design (implemented, default off; `ResFKAN.fit(..., confirm=...)`)

- Each replicate's validation subjects (20% of n) are split once, stratified on the event indicator, into
  **val_select** (half) and **val_confirm** (half), with a generator seeded by (replicate seed, 18).
- **val_select** is used for everything before the final decision, for both models: (lr, wd) tuning and early stopping
  of the comparator ensemble (b2s_cox / b2s_aft) and of the anchored ensemble, the 16 candidate fine-tunes and the
  step-(1) choice of the anchored state (Round-17 rule: parsimonious among joint states, non-cliff for static ones).
- **val_confirm** is used only for step (2), one model against one model: the step-(1) anchored candidate against the
  comparator's seed-0 member, by their unpenalised held-out losses. Anchored iff
  loss_confirm(anchored) < loss_confirm(comparator) − margin; otherwise the estimator returns the comparator's
  predictions exactly.
- Consequence: the Round-18 comparator is tuned and early-stopped on half the usual validation data. The comparator
  trained as in Rounds 12–17 on the whole validation set (`b2s_<loss>_fullval`) is reported as exploratory, to show the
  price of setting val_confirm aside.
- Tests R18-1 (val_confirm untouched before the decision), R18-2 (exact reduction, max |diff| = 0.0), R18-3 (held-out
  losses and rule), R18-4 (default = Round-17 rule); Round 15–17 tests rerun with default flags.

## 3. Development runs (no formal seed)

Already-used Round-17 seeds, treated as development data: S-Last 4100–4119, S-Sparse 4200–4219, S-Misspec-lognormal
4300–4319 (60 jobs; Cox and log-normal AFT on S-Last and S-Sparse, log-normal AFT on S-Misspec), margin 0, recording
both confirmation losses and the anchored alternative's test metrics so that every margin below is evaluated from the
same fits. Reproduction check: Round-17 job last_4100 re-run with the new code must reproduce its record exactly.

## 4. Margin rule (fixed now)

Grid {0, 0.0025, 0.005, 0.01, 0.02} (Cox: NLPL per event; AFT: NLL per subject). For each margin the confirm model of a
replicate is the anchored alternative if loss_confirm(anchored) < loss_confirm(comparator) − margin, else the
comparator; U = upper 95% paired-t bound of (confirm model − comparator) survival-curve error.
- **Cox:** the smallest margin with U(S-Sparse) < +0.0025 and U(S-Last) < 0.
- **AFT:** the smallest margin with U(S-Misspec) < +0.0025, U(S-Sparse) < +0.0025 and U(S-Last) < 0.
- Fall-backs as Round 17: the margin with the smallest worst trajectory-scenario U among those with U(S-Last) < 0;
  else 0.
Diagnostics reported (not used by the rule): mean (comparator − anchored) loss on val_select and on val_confirm; their
correlation with the test difference.

## 5. Go / no-go (fixed now)

At the chosen margins: Cox U(S-Sparse) < +0.005 and U(S-Last) < 0, AND AFT U(S-Misspec) < +0.005 and U(S-Last) < 0, on
the development seeds. Otherwise the results are reported to the owner and no formal seed is used.

## 6. Planned registration (D144; margins from §4)

Fresh seeds S-Last 4400–4419 (3–8 visits), S-Sparse 4500–4519 (3–8 visits), S-Misspec-lognormal 4600–4619 (dense);
60 jobs; models as §3.
- **H-R18a (Cox):** S-Last upper 95% bound of (b2s_anchor_cox_confirm − b2s_cox) survival-curve error < 0 AND
  S-Sparse upper bound < +0.005.
- **H-R18b (log-normal AFT):** S-Misspec upper 95% bound of (b2s_anchor_aft_confirm − b2s_aft) < +0.005 AND S-Last
  upper bound < 0.
- Reported for every contrast: paired mean, 95% t-interval, two-sided paired-t p, win / loss / tie counts, anchor
  rates. Everything else exploratory; no multiplicity correction; every run reported.
