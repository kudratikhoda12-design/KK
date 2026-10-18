# PBC endpoint: brief scientific justification (for review with the supervising professor)

Status: **provisional** (owner decisions D58/D69, 2026-10-17/18). Nothing below is a definitive clinical claim.
All counts come from `reports/logs/pbc_inspection.json` and the saved landmark records
(`reports/experiment_results.jsonl`, `phase = descriptive`).

## 1. Data and prediction task

- **Source.** Mayo Clinic PBC sequential data (`pbcseq`, R package `survival` 3.8-12).
  - 312 patients and 1,945 visits.
  - Final status: 143 censored alive, 29 transplanted, 140 died.
- **Task (landmark prediction).**
  - Landmark: s = 730 days.
  - Cohort: patients alive and transplant-free at day 730.
  - Inputs: the longitudinal markers before the landmark (bilirubin, albumin, alkaline phosphatase, AST,
    platelets, prothrombin time), plus age and sex.
  - Target: the residual time from the landmark.
- **Landmark cohort.** n = 278. The median number of visits before the landmark is 3, and 17% of patients have
  2 or fewer.

## 2. Candidate endpoints

| | E2 (proposed provisional primary) | E1 (sensitivity) | E3 (not implemented) |
|---|---|---|---|
| Event | Death **or** liver transplantation, whichever comes first | Death | Death and transplantation as competing events |
| Events in the landmark cohort | 135 (107 deaths + 28 transplants) | 107 | – |
| Transplant handling | Part of the event | Treated as censoring | Its own event type |
| Estimand | Distribution of **transplant-free survival** time | Survival in a hypothetical world without transplantation | Cause-specific hazards / cumulative incidence of each event |
| Censoring | Alive without transplant at last follow-up (administrative / loss to follow-up) | As E2, **plus** transplantation | As E2 |
| Key assumption | Censoring non-informative given the landmark covariates (CA-1) | CA-1 must also hold **for transplantation**, i.e. transplant unrelated to prognosis given the covariates | Correct specification of a competing-risks model |

## 3. Why E2 is proposed as the provisional primary endpoint

1. **Transplantation is not a non-informative censoring event.** Patients are listed and transplanted because
   their liver disease is advanced, so transplant timing depends on prognosis beyond the measured covariates.
   Censoring at transplant (E1) then biases survival upward, and its estimand describes a hypothetical scenario
   without transplantation. E2 needs no assumption about transplant selection.
2. **Clinical meaning.** Both components mark failure of the native liver, so "transplant-free survival" is a
   coherent clinical outcome. It is commonly used in PBC research. *The professor should confirm that this matches
   the intended clinical question and the conventions of the target literature.*
3. **Statistical information.** E2 has 135 events against 107 for E1, so estimates and model comparisons are less
   variable.

## 4. Limitations of E2 (to discuss)

- **Heterogeneous components.** E2 combines death and transplantation. Covariate effects can differ between them,
  and a composite cannot separate them.
- **Context dependence.** Transplantation also depends on factors outside the patient's biology: organ
  availability, listing policies, and practice in the 1974–1984 Mayo era. The composite inherits that
  dependence.
- **Small numbers.** With only 28 transplants after the landmark, transplant-specific inference is not feasible
  in any case.

## 5. Role of E1 and E3 in the analysis

- **E1** is reported as a **sensitivity analysis**, with its stronger assumption stated explicitly.
- **E3** (competing risks) is **future work**. No competing-risk claims are made. A properly specified
  cause-specific-hazards or Fine–Gray analysis, with its own validation tests, would be needed first.

## 6. Results so far (descriptive; no model is claimed superior)

- On both endpoints, all validated baselines beat the no-information reference in integrated Brier score (report
  Part C, §R1.10.4).
- KAN-AFT and linear AFT on the same eight landmark inputs could not be told apart over 5 splits (report §R3.5).
- These results do not depend on the endpoint decision being final.

## 7. Question for the professor

Is **transplant-free survival (E2)** the appropriate primary estimand for this landmark prediction study, with
death-only survival (transplant censored, E1) as a sensitivity analysis? Should a competing-risks analysis (E3) be
planned as a separate, specified study?
