# D168 Stage A: development results and GO/NO-GO (2026-11-17)

Protocol: `docs/d168_afcm_reference_protocol.md`, Section 4. **Development only:** historical data from seeds 22, 914 and
966 (the CK procedure: S-Add, DGP v2, n 2,000, Round-11 60/20/20 split, smoothed curves; 1,200 training subjects) and
the mechanics set (the same procedure with seed 11, n 400; 240 training subjects). No formal seed was used. Nothing
here changes CK-6c, CK-6d, H-R12 or any earlier record; the Stage-B acceptance limits are fixed and are not tuned on
these values. Records: `reports/d168/stageA/results.jsonl` (one per dataset), `reports/d168/stageA/timing.jsonl`, logs in
`reports/d168/stageA/logs/`. Code: `scripts/afcm_mgcv.R`, `src/kanfae/models/afcm_mgcv.py` (reference),
`src/kanfae/models/afcm_constrained.py` (A7 candidate repair, a new code path), `scripts/d168_stageA.py`,
`scripts/d168_stageA_timing.py`. The historical `src/kanfae/models/afcm.py` is unchanged (last changed in commit 244bbd7).

## A0 Environment

R 4.4.3, **mgcv 1.9-5** (CRAN source, checksum-verified), survival 3.8-12, nlme 3.1-170, jsonlite 2.0.0, Matrix 1.7-6;
Python environment identical to `requirements-lock.txt`. Details and the explicit lock: `env/README.md` (commit b6addb2).

## A1 Reference specification (frozen)

Per channel d = 1, 2, 3: `ti(tmat, X_d, by = lmat, bs = c("cr", "cr"), k = c(10, 10), mc = c(FALSE, TRUE))`; the two
scalar covariates linear; `family = cox.ph()`, `method = "REML"`; fitted on the training subjects only. X_d is the
quantile-transformed smoothed curve (training empirical CDF at each of the 50 grid times, plotting positions
(i − 0.5)/n, linear interpolation, the same map for test curves); `lmat` the trapezoid weights of the 50-point grid
(the quadrature used by every other model in the project). F̂_d is returned on the 50 grid times × 401 points in
[0, 1] and interpolated linearly in x̃. Survival predictions: the R linear predictors with a Breslow baseline on the
training linear predictors. Each fit has 272 coefficients (3 × 9 × 10 + 2).

**Frozen files** (SHA-256 at this commit; any change requires a new decision):

| File | SHA-256 |
|---|---|
| `scripts/afcm_mgcv.R` | `cacacfe61670851e36962cdd4f35122044b74c49eac8cf807fd17c4bc657f6c9` |
| `src/kanfae/models/afcm_mgcv.py` | `5bb26016cc3edaf2032aba84526c23df4c19e55e9fae4d15e6b1a181707b45a7` |

## A2 and A3 Identical inputs; ties

Both languages read the same exported arrays (`AFCMReference.fit` writes them; the Python fits use the same arrays in
memory). Tied event times:

| Dataset | Training subjects | Training events | Tied event times (training) | Tied times (all subjects) |
|---|---|---|---|---|
| ck22 | 1,200 | 843 | 0 | 0 |
| ck914 | 1,200 | 827 | 0 | 0 |
| ck966 | 1,200 | 848 | 0 | 0 |
| mech11 | 240 | 167 | 0 | 0 |

No ties anywhere, so Peto (mgcv), Breslow and Efron (Python) give the same partial likelihood on these data.

## A4 Reference fits and shape recovery (development values; not a GO criterion)

Identifiable RISE at the true test curves (the CK-6c metric). "Historical" are the saved records of the same fits.

| Dataset | Reference φ₁ | Reference φ₂ | Python V3 φ₁ | V3 φ₂ | Python REML φ₁ | REML φ₂ | Historical V3 (CK-6c) | Historical REML (Round 22) |
|---|---|---|---|---|---|---|---|---|
| ck22 | 0.2255 | 0.0417 | 0.2475 | 0.0530 | 0.1412 | 0.0571 | 0.2475 / 0.0530 | 0.1412 / 0.0571 |
| ck914 | 0.1382 | 0.0427 | 0.0232 | 0.0527 | 0.0425 | 0.0431 | — | 0.0425 / 0.0431 |
| ck966 | 0.1864 | 0.0234 | 0.0552 | 0.0561 | 0.1472 | 0.0285 | — | 0.1472 / 0.0285 |
| mech11 | 0.3897 | 0.1678 | 0.0547 | 0.1663 | 0.1462 | 0.2120 | — | — |

- The refitted Python V3 (seed 22) and REML fits (seeds 22, 914, 966) **reproduce their saved historical values
  exactly**, so the historical code paths are deterministic and unchanged.
- **Observation, not a decision input:** the reference's φ₁ error is above the Stage-B limit of 0.15 on two of the
  three development seeds (0.2255 and 0.1864) and below it on one (0.1382); φ₂ is below 0.10 on all three. The Stage-B
  limits are not changed because of this.

Reference fit diagnostics:

| Dataset | REML convergence | Outer iterations | max \|REML gradient\| | Total edf | Warnings | Finite | Fit time (s) |
|---|---|---|---|---|---|---|---|
| ck22 | full convergence | 10 | 5.9 × 10⁻⁴ | 20.5 | none | yes | 43.0 |
| ck914 | full convergence | 10 | 5.8 × 10⁻⁴ | 21.9 | none | yes | 41.7 |
| ck966 | full convergence | 10 | 8.7 × 10⁻⁴ | 18.0 | none | yes | 41.6 |
| mech11 | full convergence | 12 | 1.3 × 10⁻⁴ | 17.2 | none | yes | 13.7 |

## A5 Identifiability

Largest over grid times of |mean over training curves of F̂_d(t, X_i(t))|, relative to the RMS of F̂_d (0 means the
paper's constraint (11) holds exactly):

| Dataset | Reference (ch. 1 / 2 / 3) | Python V3 | Python REML |
|---|---|---|---|
| ck22 | 7.5 × 10⁻⁶ / 6.1 × 10⁻⁶ / 1.2 × 10⁻⁸ | 0.73 / 0.43 / 1.12 | 0.53 / 0.44 / 0.53 |
| ck914 | 9.3 × 10⁻⁶ / 6.2 × 10⁻⁶ / 9.2 × 10⁻⁶ | 0.73 / 0.52 / 1.09 | 0.87 / 0.55 / 0.87 |
| ck966 | 3.7 × 10⁻⁶ / 5.8 × 10⁻⁶ / 4.5 × 10⁻⁸ | 0.15 / 0.52 / 0.79 | 0.38 / 0.79 / 0.80 |
| mech11 | 1.1 × 10⁻⁵ / 5.8 × 10⁻⁶ / 1.8 × 10⁻⁸ | 0.47 / 0.78 / 1.01 | 0.69 / 1.02 / 0.76 |

- The reference satisfies the per-time constraint to interpolation accuracy. This is **exact by construction** here:
  with the training-ECDF transformation, the transformed training values at every grid time are the same set
  {(i − 0.5)/n}, so `mgcv`'s pooled centring of the x margin equals the per-time constraint of the paper.
- The Python fits carry large components that depend on t alone (unidentified; they cancel in the partial
  likelihood). The identifiable RISE removes them for scoring, so they do not affect the reported shape errors, but
  the fitted surfaces themselves are not constrained.

Identified part of each Python surface compared with the reference (identifiable relative difference, reference as
truth): V3 φ₁ 0.08–0.30, φ₂ 0.03–0.33; REML φ₁ 0.11–0.31, φ₂ 0.02–0.05 (per dataset in the record). The Python fits
are different estimators from the reference, not approximations of it.

## A6 Convergence and conditioning of the Python fits (at their own smoothing parameters)

| Dataset | V3: max \|grad\| | V3: Newton decrement | V3: cond(H) | REML: max \|grad\| | REML: decrement | REML: cond(H) |
|---|---|---|---|---|---|---|
| ck22 | 6.4 × 10⁻⁷ | 2.9 × 10⁻⁹ | 5.1 × 10⁵ | 3.5 × 10⁻⁴ | 1.6 × 10⁻⁴ | 4.3 × 10¹⁰ |
| ck914 | 9.4 × 10⁻⁴ | 1.2 × 10⁻³ | 1.9 × 10⁶ | 1.1 × 10⁻⁵ | 5.7 × 10⁻⁶ | 3.5 × 10⁸ |
| ck966 | **1.3 × 10⁻²** | 3.6 × 10⁻³ | 1.9 × 10¹⁰ | 2.0 × 10⁻⁵ | 2.6 × 10⁻⁵ | 2.6 × 10⁹ |
| mech11 | 6.9 × 10⁻⁴ | 2.4 × 10⁻³ | 1.9 × 10⁶ | 1.6 × 10⁻³ | 7.2 × 10⁻⁶ | 1.2 × 10¹⁰ |

Re-converging the historical objective by Newton from the historical coefficients (same parametrisation and ridge;
A7 "unconstrained"): φ₁ / φ₂ change from the historical values to

| Dataset | V3 historical → re-converged | REML historical → re-converged |
|---|---|---|
| ck22 | 0.2475 / 0.0530 → 0.2475 / 0.0530 | 0.1412 / 0.0571 → 0.1402 / 0.0614 |
| ck914 | 0.0232 / 0.0527 → 0.0378 / 0.0517 | 0.0425 / 0.0431 → 0.0425 / 0.0432 |
| ck966 | 0.0552 / 0.0561 → **0.1259** / 0.0603 | 0.1472 / 0.0285 → 0.1478 / 0.0286 |
| mech11 | 0.0547 / 0.1663 → 0.0701 / 0.1669 | 0.1462 / 0.2120 → 0.1456 / 0.2129 |

- The warm-started REML path of Round 22 ends near its optimum (small changes).
- **V3 at its own smoothing parameters is often not at its optimum**: on three of four datasets L-BFGS stopped early,
  and the stopped fits had *lower* shape error than the optimum of the same objective (ck966 φ₁ 0.055 against 0.126 at
  the optimum). The historical V3 numbers, including the H-R12 comparator, therefore partly reflect where the optimiser
  stopped, not only the estimator. This qualifies, but does not change, the recorded H-R12 result.
- The Round-25 seed-22 diagnostic (a cold start at the Round-22 REML smoothing parameters, φ₁ 0.534) remains as
  recorded; the warm-started historical path at the same parameters is near-stationary (φ₁ 0.141, re-converged 0.140).

## A7 Candidate Python repair (constraint absorbed + damped Newton; new code path)

At the V3 and at the REML smoothing parameters of each dataset: the constrained Newton fit from a cold start and warmed
from the historical coefficients. Pre-set acceptance: the stopping rule (max |gradient| < 10⁻⁶, Newton decrement
< 10⁻⁸) reached in both, and the cold and warm surfaces equal (max difference of the per-time-centred surfaces at the
test points < 10⁻⁴) on **all three** development seeds.

| Dataset, λ | Stopping rule (cold / warm) | cond(H) constrained | Cold − warm max difference | φ₁ / φ₂ (cold) | Accepted |
|---|---|---|---|---|---|
| ck22, V3 λ | yes / yes | 2.7 × 10⁵ | 2.1 × 10⁻³ | 0.2424 / 0.0551 | no |
| ck22, REML λ | yes / yes | 4.4 × 10¹⁰ | 5.7 × 10⁻⁴ | 0.1384 / 0.0667 | no |
| ck914, V3 λ | yes / yes | 7.6 × 10⁶ | 2.0 × 10⁻⁴ | 0.0587 / 0.0555 | no |
| ck914, REML λ | yes / yes | 8.9 × 10⁸ | 2.3 × 10⁻⁴ | 0.0631 / 0.0425 | no |
| ck966, V3 λ | yes / yes | 1.9 × 10¹¹ | 3.2 × 10⁻⁴ | 0.1645 / 0.0633 | no |
| ck966, REML λ | yes / yes | 1.9 × 10⁹ | 1.4 × 10⁻⁴ | 0.1699 / 0.0282 | no |
| mech11, V3 λ | yes / yes | 8.1 × 10⁶ | 6.5 × 10⁻⁵ | 0.0989 / 0.2185 | (yes) |
| mech11, REML λ | yes / yes | 1.5 × 10⁹ | 4.2 × 10⁻⁵ | 0.1453 / 0.2029 | (yes) |

**Result: the repair is NOT accepted under the pre-set rule.** Both starts reach the stopping rule everywhere and
their shape errors agree to four decimals, but on every development seed the two converged surfaces still differ by
more than 10⁻⁴ (up to 2 × 10⁻³). Removing the unidentified directions did **not** cure the ill-conditioning: the
condition number stays between 10⁵ and 10¹¹ and in one case rises. So the protocol's suggested explanation (that the
near-singularity came from the unimposed constraint) is **not supported**; the poor conditioning comes mainly from
very small smoothing parameters in some directions. Consequences, as fixed in the protocol: no repaired Python AFCM
is used in Stage C, and the historical V3 and REML code paths stay as they are.

## A8 Run time on the Stage-C design (development seed 11)

S-Add, n 1,000 (600 training subjects), Round-12 front end: B2-S Cox three-member ensemble **245 s**, reference AFCM
**28 s** (full convergence), front end 0.5 s; about **4.6 min per replicate**, i.e. about 8.4 CPU hours for 110
replicates and about **2.1–2.5 h wall on four single-threaded lanes**. (Development values only: B2-S mean identifiable
RISE 0.215, reference 0.170; not used for any decision.)

## GO / NO-GO for Stage B (rule fixed in the protocol before Stage A)

| Condition | Result |
|---|---|
| (a) the reference `mgcv` fits converge without error on all three development seeds | **met**: full convergence, no errors or warnings, on seeds 22, 914 and 966 (and on the mechanics set) |
| (b) the fitted surfaces are finite | **met** on all four datasets |
| (c) the reference specification is frozen and committed | **met** with this commit (file hashes above) |

**Decision: GO.** Development RISE values were not part of the rule and did not influence it. Stage B follows only after
the D169 pre-registration is committed.
