### 10.3 C-MAPSS FD001 (REAL simulated turbofan data; RUL benchmark)

**RUL results are reported separately from survival-framing metrics.**
- With no censoring, the IBS here is an unweighted Brier score, and Harrell's C equals Uno's C up to
  truncation.
- RUL RMSE and MAE use the median of each model's predicted survival curve, uncapped, in cycles.
- "Official" means the NASA test trajectories, which have true RUL.
- Neither landmark option is the approved design (D3b-R). They are separate, labelled analyses.

#### 10.3.1 Option A (single landmark s0 = minimum training-split life − 1; full history)

{{T:cmapss_A}}

#### 10.3.2 Option B (trailing 30-cycle windows every 10 cycles; landmark s as a covariate)

{{T:cmapss_B}}

Per landmark band (internal test; pooled metrics mix landmarks, so per-band metrics are shown):

{{T:cmapss_bands}}

![C-MAPSS RUL](figures/fig_cmapss_rul.png)

![C-MAPSS bands](figures/fig_cmapss_bands.png)

#### 10.3.3 Option A with ARTIFICIAL CENSORING (code-path test only)

Training and validation targets were censored independently with `C ~ U(0, max training residual)`. Test
targets are uncensored. These numbers **test the censoring code path, not real censoring**.

{{T:cmapss_A_art}}

**Observations (descriptive; no significance claims).**
- **Option A, internal test.** RUL RMSE was {{V:S|cmapss_A|fpca_aft|rul_test_rmse}} (FPCA + AFT),
  {{V:S|cmapss_A|fpca_cox|rul_test_rmse}} (FPCA + Cox) and {{V:S|cmapss_A|sum_cox|rul_test_rmse}}
  (Summary + Cox) cycles, against {{V:S|cmapss_A|km_null|rul_test_rmse}} for the null reference.
- **Option A, official test.** Only engines whose observed length reaches `s0` can be evaluated (mean
  {{V:S|cmapss_A|fpca_aft|rul_official_n}} of 100). FPCA + AFT had the lowest RMSE,
  {{V:S|cmapss_A|fpca_aft|rul_official_rmse}}.
- **Option B, official test (all 100 engines at their last cycle).** Summary + RSF had the lowest RMSE,
  {{V:S|cmapss_B|sum_rsf|rul_official_rmse}}. The parametric/Cox baselines were at
  {{V:S|cmapss_B|sum_aft|rul_official_rmse}}–{{V:S|cmapss_B|fpca_aft|rul_official_rmse}}, and the null
  reference at {{V:S|cmapss_B|km_null|rul_official_rmse}}.
- **Option B, internal test pooled over all landmarks.** RMSE was ≈{{V:S|cmapss_B|sum_rsf|rul_test_rmse}}–{{V:S|cmapss_B|fpca_cox|rul_test_rmse}}.
  Per band it falls from ≈{{V:S|cmapss_B|sum_rsf|band_30_100_rmse}} at early landmarks (30–99) to
  ≈{{V:S|cmapss_B|sum_rsf|band_200_999_rmse}} at late landmarks (≥ 200). Discrimination within the early band
  is modest (C ≈ {{V:S|cmapss_B|fpca_aft|band_30_100_harrell_c}}), as expected when degradation is not yet
  visible.
- **The two options are not directly comparable.** They predict at different landmarks and evaluate different
  official-test subsets. This is one reason D3b remains an owner decision.
- **Artificial censoring (Option A).**
  - The parametric AFT predictions stayed close to their uncensored values: FPCA + AFT internal RMSE
    {{V:S|cmapss_A_artificial|fpca_aft|rul_test_rmse}} vs {{V:S|cmapss_A|fpca_aft|rul_test_rmse}}.
  - For the Cox models, some predicted survival curves never fell to 0.5 within the training follow-up, so
    their medians are undefined. On average {{V:S|cmapss_A_artificial|sum_cox|rul_test_median_not_reached}}
    test engines per split were affected for Summary + Cox; these are flagged in the records and set to the
    grid end. This inflates Cox RMSE ({{V:S|cmapss_A_artificial|sum_cox|rul_test_rmse}} ±
    {{V:S|cmapss_A_artificial|sum_cox|rul_test_rmse|std}}).
  - For Summary + AFT, all four configurations diverged in one split, so only
    {{V:S|cmapss_A_artificial|sum_aft|ibs|count}} of 5 splits contribute.
- **The neural autoencoders were not evaluated** (BLOCKED, D33).

### 10.4 PBC sequential data (REAL, right-censored; landmark 730 days)

The two endpoints estimate different quantities, and **neither is declared definitive** (D9-R).

#### 10.4.1 E1: death, with transplant treated as censoring (hypothetical no-transplant survival; relies on CA-1 for transplant)

{{T:pbc_E1}}

#### 10.4.2 E2: death or transplant, composite (transplant-free survival)

{{T:pbc_E2}}

![PBC IBS](figures/fig_pbc_ibs.png)

#### 10.4.3 Visit process (descriptive; full cohort; not used as a model feature)

{{T:visits}}

**Observations (descriptive; no significance claims).**
- **All baselines improved substantially on the covariate-free reference for both endpoints.**
  - E1: IBS {{V:S|pbc_E1|fpca_cox|ibs}}–{{V:S|pbc_E1|sum_rsf|ibs}} vs {{V:S|pbc_E1|km_null|ibs}} for the null.
  - E2: IBS {{V:S|pbc_E2|fpca_aft|ibs}}–{{V:S|pbc_E2|sum_rsf|ibs}} vs {{V:S|pbc_E2|km_null|ibs}}.
- **Differences between baselines were small relative to their split-to-split SDs**, so no ranking is claimed.
  With about 56 test patients per split, the uncertainty is large.
- **The conditional-IPCW sensitivity analysis changed IBS values by less than about 0.01.**
  - E1: FPCA + Cox {{V:S|pbc_E1|fpca_cox|ibs}} (marginal) vs {{V:S|pbc_E1|fpca_cox|ibs_conditional_ipcw}}
    (conditional).
  - E2: {{V:S|pbc_E2|fpca_cox|ibs}} vs {{V:S|pbc_E2|fpca_cox|ibs_conditional_ipcw}}.
  - So, *under the Cox censoring model used*, the metric conclusions are not sensitive to the CA-2 assumption.
    This does not validate CA-1 for transplant under E1.
- **D-calibration p-values were all > 0.05.** With small test sets this test has little power.
- **FPCA-based and summary-feature baselines performed similarly.** This is unsurprising given a median of 3
  visits per patient (§14). The data are too sparse to make the functional representation informative.
- **Visit process (descriptive).** The univariate hazard ratio per additional visit in the window is reported
  in the table above. The absence of a clear association in this crude check does not rule out an informative
  visit process; the dataset documentation itself describes visits triggered by worsening condition.
