# Round 17 development plan (written and committed before the development runs; D141 is locked after them)

Owner instruction (2026-10-26): open Round 17 testing gated last-value edges on `b2s_anchor_cox` and extending
two-way gated anchoring and fixed-μ symbolic distillation to KAN-AFT (B2-S AFT); tune `last_tol` only on
already-used Round-15/16 records and development seeds 961–969; verify the fresh seed blocks; pre-register (D141), run
(D142), deliver (D143); keep Rounds 1–16 untouched; commit locally, do not push.

## 1. Fact check of the instruction

| Statement | Check |
|---|---|
| H-R16b part 2 missed by 0.0003 (+0.0053 vs +0.0050) | Correct: upper bound +0.00531 (`reports/round16_summary.json`) |
| +0.0024 cost in the 19/20 replicates where the gate kept a channel | Correct (exploratory, `reports/round16/posthoc_exploratory.json`) |
| "Log-Normal KAN-AFT" | The existing B2-S KAN-AFT (`b2s_aft`, Rounds 12–13) is **Weibull**. `FKANSurv(family="lognormal")` existed for the loss, but `predict_survival` always used the Weibull survival function; no earlier round fitted a non-Weibull FKANSurv, so the fix (D141) changes no earlier record. In Round 17 both the comparator `b2s_aft` and `b2s_anchor_aft` use the **log-normal** family, so the comparison is like for like |
| `b2s_anchor_aft_rc` "if recurrent-event weighting is used" | In this project `_rc` means **residual calibration** (Kaplan–Meier of standardised residuals, Round 12), not recurrent-event weighting; there are no recurrent events. No `_rc` variant is registered |
| "exact reduction to `b2s_cox` / `b2s_aft`" | Implemented literally: the pure-trajectory candidate **is** the comparator ensemble fitted on the same split, so when it is chosen the anchored model's predictions equal the comparator's and the paired difference is exactly 0 (test R17-3) |
| S-Last for AFT | S-Last is a Weibull proportional-hazards scenario; a log-normal AFT is misspecified there, equally for the comparator and the anchored model |
| Fresh blocks 4100–4119, 4200–4219, 4300–4319; development seeds 967–969 | Not used as a seed, job name or code literal anywhere (checked 2026-10-26; the only textual hits are unrelated numbers inside old records) |

## 2. Two-way gate (implemented, default off; `ResFKAN(gate_last_value=True)`)

Candidates on the tuned seed-0 anchored member, each with the 200-epoch masked fine-tune (identical masks fitted once):
- **joint** (θ in THETAS_D120 = 0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25): trajectory channels pruned by the
  integrated score at θ, and each last-value edge ψ_d pruned if its score / the largest last-value score < θ;
- **static** (same θ): every trajectory channel off, last-value edges pruned at θ;
- **pure trajectory**: the comparator ensemble (b2s_cox / b2s_aft) itself, scored by its seed-0 validation loss.

Rule: (1) among joint states, the D120 parsimonious rule (tol 0.01, largest θ); a static state replaces it only if
strictly better (Round-16 non-cliff guard); (2) the anchored state is used only if its validation loss is below the
comparator's minus `last_tol`, otherwise the model is the comparator.

## 3. Development runs (no formal seed)

- `devsparse_3900–3919`, `devlast_4000–4019` (already used in Round 16): b2s_cox, two-way b2s_anchor_cox, log-normal
  b2s_aft, two-way b2s_anchor_aft, landmark Cox. The b2s_cox ensemble re-fit must reproduce the Round-16 record
  (reproduction check).
- `devmisspec_961–969` (development seeds; S-Misspec-lognormal): log-normal b2s_aft, the AFT symbolic pipeline
  (μ = 10, max_abs_arg = 50, safe = True), two-way b2s_anchor_aft.
- All two-way fits with last_tol = 0, recording the comparator's and the anchored state's validation losses and the
  test metrics of both, so every last_tol below is evaluated from the same fits.

## 4. last_tol tuning rule (fixed now)

Grid L = {0, 0.0025, 0.005, 0.01, 0.02, 0.04} (validation-loss units: Cox mean NLPL per event; AFT mean NLL per subject).
For each l the two-way model of a replicate is the anchored alternative if anchored_val_loss < ref_val_loss − l, else
the comparator; U(·) is the upper 95% paired-t bound of (two-way − comparator) survival-curve error.
- **Cox:** the smallest l with U(S-Sparse 3900–3919) < +0.0025 and U(S-Last 4000–4019) < 0. If none, the l with the
  smallest U(S-Sparse) among those with U(S-Last) < 0; if none, l = 0.
- **AFT:** the smallest l with U(S-Misspec 961–969) < +0.0025, U(S-Sparse 3900–3919) < +0.0025 and U(S-Last 4000–4019) < 0;
  fall-backs as for Cox.
(+0.0025 is half the registered margin, a safety allowance for the move from development to fresh seeds.)

## 5. Go / no-go before the formal run (fixed now)

The formal run starts only if, at the chosen values, (i) U(S-Sparse, Cox) < +0.005 and U(S-Last, Cox) < 0 on the
development seeds, and (ii) the AFT distillation refit is finite on all 9 development S-Misspec seeds. Otherwise the
results are reported to the owner and no formal seed is used.

## 6. Planned registration (D141; the numbers in brackets are filled from §4)

- S-Last 4100–4119 (3–8 visits), S-Sparse 4200–4219 (3–8 visits), S-Misspec-lognormal 4300–4319 (dense); 60 jobs.
- **H-R17a** (b2s_anchor_cox_twoway, last_tol [Cox]): part 1 S-Last upper 95% bound of (two-way − b2s_cox) curve
  error < 0; part 2 S-Sparse upper bound < +0.005. Both must hold.
- **H-R17b** (log-normal KAN-AFT): part 1 (AFT symbolic_distill_fixed, S-Misspec 4300–4319): 20/20 distillation refits
  finite (a fallback counts as finite) and upper bound of (distilled − after pruning) identifiable RISE < +0.05;
  part 2 (b2s_anchor_aft_twoway, last_tol [AFT]): S-Misspec upper bound of (two-way − b2s_aft) curve error < +0.005
  AND S-Last 4100–4119 upper bound < 0. All parts must hold.
- Everything else exploratory; no multiplicity correction; every run reported.
