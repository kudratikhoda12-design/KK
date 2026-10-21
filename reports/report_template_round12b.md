<p class="small">Research prototype report, revision 13 · generated from saved records by
<code>scripts/build_report_round12b.py</code> · git <code>{{V:git_head}}</code> (local branch
<code>claude/blissful-faraday-2unozw</code>, <b>not pushed</b>) · status: <b>research prototype, not
publication-ready; no novelty claim</b>. Part A covers the owner decisions closing L39 (D123), the AFCM diagnosis, the
repaired tests CK-6c and CK-7b, and the Round-12 gate (D124). <b>Round 12 was not run</b>: CK-6c failed. Part B is
revision 12's Part A (L38); Parts C–M are the round-11 to round-1 reports.</p>

## 0. Summary (plain English)

### 0.1 Outcome

| Item | Result |
|---|---|
| AFCM diagnosis (D123) | On development seed 914, widening the λ_x grid, normalising the penalties and adding a first-order t penalty cut AFCM's φ₁ error from 0.473 to 0.023. Adopted as the repaired AFCM |
| CK-7b (PACE under DGP v2) | **Pass** on seed 23 |
| CK-6c (repaired AFCM) | **Fail** on seed 22: φ₁ identifiable RISE 0.247 (threshold 0.15); φ₂ 0.053 passes |
| Round 12 | **Not started** (owner gate). Seeds 3000–3419 unused |
| Tests | Final suite: {{V:n_tests}} tests: {{V:passed}} passed, {{V:failed}} failed (CK-6c and the 4f gate's original R1), {{V:xfailed}} expected failures, {{V:skipped}} skipped |

### 0.2 Acceptance tests in this part

{{T:ck_table}}

### 0.3 Decision needed from the owner (ledger L40)

- **AFCM, the H-R12 comparator, is still not validated.**
- The repaired specification worked on the development seed but not on the fresh seed. Its validation criterion chose
  a small λ_t (time-varying surface) on seed 22, and φ₁ is then poorly recovered (R12b.4).
- This is a property of single-fit AFCM with validation-chosen penalties, not of one setting. Further changes chosen
  on one development seed risk the same outcome.
- Possible routes (none taken):
  - evaluate AFCM's recovery over several development seeds before registering;
  - fix λ_t large for the comparison (a time-homogeneous prior that matches φ₁ but not φ₂);
  - average AFCM over bootstrap or seed fits, as B2-S averages three;
  - or run Round 12 with AFCM flagged as not validated.

## R12b.1 Owner decisions (D123) and fact check

The decisions, the seed checks and five fact-check items are in D123 (R12b.6):
- **Penalty normalisation only rescales λ:** it shifts the grid by one decade.
- **The Kronecker ordering is equivalent** to the one the code uses.
- **The first-order t penalty puts time-homogeneous surfaces in the λ_t → ∞ limit.**
- **The first launch of the diagnosis lanes was aborted** (shared output file) and relaunched.
- **Correction to D120:** seed 21 had been used in Rounds 4–6 as a seed of the Round-1 simulator (not fsim).

## R12b.2 AFCM diagnosis on development seed 914

{{T:afcm_diag_table}}

- **Adopted: V3.** It is the smallest change that resolved φ₁ on this seed.
- **Rejected options:**
  - per-channel λ cost five times the fits without improving φ₁;
  - the quantile input made φ₁ worse.
- The validation NLPL did not track shape recovery: V6 had the best validation loss and the worst φ₁.

## R12b.3 Repaired tests

CK-6c and CK-7b were registered and committed (08292df) before seeds 22 and 23 were used. Both are in
`tests/test_round12b_ck.py`. The CK-7b mechanics were checked once on development seed 916.

## R12b.4 Why CK-6c failed (from the recorded validation grid; no new fit)

{{T:lamt_table}}

- The validation criterion preferred λ_t 1e-2 on seed 22 by a clear margin.
- With that λ_t the surface may vary in time, and φ₁, which is constant in time, is recovered poorly.
- On seed 914 the same procedure had chosen λ_t 1.

## R12b.5 Gate decision

- **Round 12 was not started.** CK-6c failed at its registered seed.
- No Round-12 seed (3000–3419) was used.

## R12b.6 Decisions (D123–D125)

{{T:decisions}}

## R12b.7 Tests

{{T:suite_table}}

**All test runs, in chronological order**

{{T:test_runs}}

# Parts B–M: historical records

Part B is revision 12's Part A (L38 decisions, 2026-10-21); Parts C–M are the round-11 to round-1 reports, each
rendered from its own round's records. Embedded living documents (ledger, decision rows, reproducibility) show their
current version.
