<p class="small">Research prototype report, revision 12 · generated from saved records by
<code>scripts/build_report_round12.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A covers the owner decisions closing L38 (D120), the repaired acceptance
tests and the Round-12 gate (D121). <b>Round 12 was not run</b>: one repaired test failed. Parts B–L are the round-11
to round-1 reports, each regenerated from its own round's records.</p>

## 0. Summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| Owner decisions (D120) | Repaired simulator (DGP v2: random level and trend per curve), identifiable RISE as the primary shape metric, wider AFCM λ_t grid, converged CK-3b, channel-level pruning with a parsimonious selection rule, mixture-error CK-11b. Registered and committed before any run on seeds 17–21 |
| Repaired tests (D121) | **4 of 5 pass:** CK-3b, CK-5b, CK-8b, CK-11b. **CK-6b fails:** AFCM's φ₁ identifiable RISE is 0.556 (threshold 0.10) |
| Round 12 | **Not started** (owner gate: all five must pass). Seeds 3000–3419 unused |
| Historical tests | CK-3, CK-5, CK-6, CK-8, CK-11 (Round 11) are kept unchanged as strict expected failures, on the Round-11 simulator |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6b and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 The repaired acceptance tests

{{T:ck_table}}

### 0.3 Decision needed from the owner (ledger L39)

- **The AFCM re-implementation is not validated.** H-R12 compares B2-S against AFCM.
- Possible routes (none taken):
  - widen the λ_x grid below 1e-4 (the selection sat at its lower boundary);
  - use Cui et al.'s quantile-transformed input;
  - a REML-type smoothing selection;
  - or run Round 12 with AFCM explicitly flagged as not validated.
- B2-S itself passed CK-5b, but only through ensemble averaging: each of the three members alone was above 0.10 for φ₁.

## R12.1 Owner decisions (D120) and fact check

The owner's four decisions are in D120 (R12.7), together with the operational details fixed before the runs. The fact
check made five qualifications:
- **The bivariate class is still not fully identified.** Each channel's curves span a 6-dimensional space, so x·c(t)
  with c outside that span still has no effect on η. This is why the projection in the identifiable RISE stays
  necessary.
- **The new trend basis is not orthogonal to the Fourier sine terms.** So the v2 covariance eigenfunctions are not the
  Fourier functions: CK-7 stays on the Round-11 simulator, and PACE has no acceptance test under v2.
- **Variance and signal changed.** The time-averaged variance is unchanged at 1.875, but pointwise variance now ranges
  1.09–2.59, and the SD of the true η rose from 0.45 to 0.71.
- **A large λ_t removes curvature in t, not linear time variation**, because of the second-order difference penalty.
- **CK-11's figures are as the owner stated them** (n_test 400).

## R12.2 What changed in the code

| Module | Change |
|---|---|
| `kanfae.data.fsim` | DGP "v2" (default) with level and trend terms; "v1" kept and pinned in every Round-11 test and script (bit-identical); test-only scenario "S-Misspec-mixture"; `rise_identifiable`, `unidentified_directions`, `rise_1d` |
| `kanfae.models.afcm` | λ_t grid {1e-4, 1e-2, 1, 1e2, 1e4} (the Round-11 grid is kept for the historical CK-6) |
| `kanfae.models.fkan` | `early_stopping=False` training on the whole training set; L-BFGS refit records the final gradient norm; `channel_scores` (integrated functional effect); `prune(..., channel_score="integrated")`; `select_lambda_theta` with the D120 grid and the parsimonious rule (the D112 behaviour is kept for the historical CK-8) |

## R12.3 Identifiability under the repaired simulator

Development seed 902, n 2,000 (`scripts/diag_round12_identifiability.py`). "Unidentified directions" counts the
directions removed by the identifiable RISE (tolerance 3e-8) on a test-sized sample.

{{T:ident_table}}

- The owner's verification thresholds are met on every channel: SD of ∫X dt > 0.5 and SD of ∫t X dt > 0.1.
- The homogeneous effect is now identified up to a constant.

## R12.4 Repaired-test details

{{T:ck_details}}

## R12.5 Why CK-6b failed (exploratory; not a test)

- **Basis resolution is not the cause.** The best possible fit of φ₁ within AFCM's own x basis has a RISE far below
  0.556:

{{T:floor_table}}

- So the error comes from estimation. Untested candidate explanations:
  - the λ_x selection sits at the lower boundary of its grid;
  - the time-constant φ₁ leaks into weakly identified bivariate directions, which the exact-null projection does not
    remove;
  - AFCM is a single fit, whereas B2-S averages three.

## R12.6 Gate decision

- **Round 12 was not started.** The owner's rule requires all five repaired tests to pass; CK-6b failed.
- No Round-12 seed (3000–3419) was used, and nothing was run on real data.

## R12.7 Decisions (D120–D122)

{{T:decisions_r12}}

## R12.8 Tests

{{T:suite_table}}

**All test runs, in chronological order** (every failure is kept; the development dry-runs of the repaired procedures
are in `reports/logs/round12/dev_ckb_dryruns.jsonl`)

{{T:test_runs}}

## R12.9 Task ledger and reproducibility

{{F:docs/task_ledger.md}}

{{F:docs/reproducibility.md}}

# Parts B–L: historical records

Part B (round 11), Part C (round 10), Part D (round 9), Part E (round 8), Part F (round 7), Part G (round 6), Part H
(round 5), Part I (round 4), Part J (round 3), Part K (round 2) and Part L (round 1) are regenerated from the same
records. Each part renders from the test records up to its own round's final suite run. Their numbers are unchanged
history. Embedded living documents (audit, ledger, decision rows, reproducibility) show their current version.
