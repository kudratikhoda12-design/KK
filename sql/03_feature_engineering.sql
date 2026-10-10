-- =====================================================================
-- 03_feature_engineering.sql
-- Churn-relevant features built with SQL (ratios, CASE parsing, window
-- functions). Every feature uses information available at origination.
-- =====================================================================
CREATE OR REPLACE TABLE model_table AS
WITH base AS (
    SELECT
        s.*,
        -- --- parsing ---------------------------------------------------
        CASE
            WHEN emp_length IS NULL OR emp_length = 'n/a' THEN NULL
            WHEN emp_length LIKE '<%'  THEN 0
            WHEN emp_length LIKE '10+%' THEN 10
            ELSE CAST(regexp_extract(emp_length, '(\d+)', 1) AS INTEGER)
        END                                                            AS emp_years,
        datediff('month', strptime(earliest_cr_line, '%b-%Y')::DATE, issue_date)
                                                                       AS credit_history_months,
        (CAST(fico_range_low AS DOUBLE) + CAST(fico_range_high AS DOUBLE)) / 2
                                                                       AS fico,
        -- sub-grade A1..G5 -> 1..35 ordinal risk score
        (ascii(substr(sub_grade, 1, 1)) - ascii('A')) * 5
            + CAST(substr(sub_grade, 2, 1) AS INTEGER)                 AS sub_grade_num,
        CAST(int_rate AS DOUBLE)                                       AS int_rate_d,
        CAST(annual_inc AS DOUBLE)                                     AS annual_inc_d,
        CAST(loan_amnt AS DOUBLE)                                      AS loan_amnt_d,
        CAST(installment AS DOUBLE)                                    AS installment_d,
        CASE WHEN home_ownership IN ('MORTGAGE', 'RENT', 'OWN') THEN home_ownership
             ELSE 'OTHER' END                                          AS home_ownership_grp,
        CASE WHEN purpose IN ('debt_consolidation', 'credit_card',
                              'home_improvement') THEN purpose
             ELSE 'other' END                                          AS purpose_grp,
        month(issue_date)                                              AS issue_month
    FROM survival_base s
)
SELECT
    id, issue_date, duration_months, event, exit_reason, loan_status,
    -- contract ------------------------------------------------------------
    loan_amnt_d                                                        AS loan_amnt,
    term_months,
    int_rate_d                                                         AS int_rate,
    sub_grade_num,
    grade,
    purpose_grp                                                        AS purpose,
    CASE WHEN initial_list_status = 'w' THEN 1 ELSE 0 END              AS whole_loan,
    CASE WHEN application_type = 'Joint App' THEN 1 ELSE 0 END         AS joint_app,
    -- borrower ------------------------------------------------------------
    annual_inc_d                                                       AS annual_inc,
    emp_years,
    home_ownership_grp                                                 AS home_ownership,
    verification_status,
    CAST(dti AS DOUBLE)                                                AS dti,
    fico,
    credit_history_months,
    CAST(delinq_2yrs AS DOUBLE)                                        AS delinq_2yrs,
    CAST(inq_last_6mths AS DOUBLE)                                     AS inq_last_6mths,
    CASE WHEN mths_since_last_delinq IS NOT NULL THEN 1 ELSE 0 END     AS ever_delinquent,
    CAST(open_acc AS DOUBLE)                                           AS open_acc,
    CAST(total_acc AS DOUBLE)                                          AS total_acc,
    CAST(pub_rec AS DOUBLE)                                            AS pub_rec,
    CAST(revol_bal AS DOUBLE)                                          AS revol_bal,
    CAST(revol_util AS DOUBLE)                                         AS revol_util,
    CAST(mort_acc AS DOUBLE)                                           AS mort_acc,
    CAST(tot_cur_bal AS DOUBLE)                                        AS tot_cur_bal,
    CAST(acc_open_past_24mths AS DOUBLE)                               AS acc_open_past_24mths,
    CAST(bc_util AS DOUBLE)                                            AS bc_util,
    CAST(percent_bc_gt_75 AS DOUBLE)                                   AS percent_bc_gt_75,
    CAST(num_tl_op_past_12m AS DOUBLE)                                 AS num_tl_op_past_12m,
    -- engineered ratios (affordability / leverage) -------------------------
    loan_amnt_d / nullif(annual_inc_d, 0)                              AS loan_to_income,
    12 * installment_d / nullif(annual_inc_d, 0)                       AS payment_to_income,
    CAST(revol_bal AS DOUBLE) / nullif(annual_inc_d, 0)                AS revol_to_income,
    CAST(open_acc AS DOUBLE) / nullif(CAST(total_acc AS DOUBLE), 0)    AS open_acc_ratio,
    -- engineered window features (customer vs. peers) ----------------------
    -- rate spread: how much more this borrower pays than peers with the same
    -- sub-grade booked in the same month. A high spread is a refinancing
    -- incentive -> prepayment (churn) pressure.
    int_rate_d - avg(int_rate_d) OVER (PARTITION BY sub_grade, issue_month)
                                                                       AS rate_spread,
    -- income relative to the borrower's state median
    annual_inc_d / median(annual_inc_d) OVER (PARTITION BY addr_state) AS income_vs_state,
    -- loan size relative to the average loan of the same purpose
    loan_amnt_d / avg(loan_amnt_d) OVER (PARTITION BY purpose_grp)     AS loan_vs_purpose_avg
FROM base;
