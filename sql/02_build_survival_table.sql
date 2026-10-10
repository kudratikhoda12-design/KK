-- =====================================================================
-- 02_build_survival_table.sql
-- Turns raw loan records into a survival (time-to-event) table.
--
--  * Churn / attrition event = the customer PREPAYS the loan, i.e. it is
--    "Fully Paid" at least 3 months before its contractual maturity. The
--    bank loses the remaining interest income and the relationship ends.
--  * duration_months = months on book from issue date to exit or censoring.
--  * Right-censoring:
--      - still active (Current / Late / Grace)  -> censored at the data
--        snapshot (Mar-2019)
--      - paid on schedule (within 3 months of maturity) -> censored at exit
--        (the customer completed the contract, did not churn early)
--      - charged off / default -> censored at last payment (competing risk,
--        we model the cause-specific hazard of prepayment)
-- =====================================================================
CREATE OR REPLACE TABLE loans_clean AS
SELECT
    *,
    strptime(issue_d, '%b-%Y')::DATE                                   AS issue_date,
    CAST(trim(replace(term, 'months', '')) AS INTEGER)                 AS term_months,
    try_strptime(last_pymnt_d, '%b-%Y')::DATE                          AS last_pymnt_date
FROM raw_sample;

CREATE OR REPLACE TABLE survival_base AS
WITH ends AS (
    SELECT
        *,
        CASE
            WHEN loan_status IN ('Current', 'In Grace Period',
                                 'Late (16-30 days)', 'Late (31-120 days)')
                THEN DATE '2019-03-01'                                 -- data snapshot
            ELSE coalesce(last_pymnt_date, issue_date + INTERVAL 1 MONTH)
        END::DATE                                                      AS end_date
    FROM loans_clean
)
SELECT
    *,
    greatest(datediff('month', issue_date, end_date), 1)               AS duration_months,
    CASE WHEN loan_status = 'Fully Paid'
          AND datediff('month', issue_date, end_date) <= term_months - 3
         THEN 1 ELSE 0 END                                             AS event,
    CASE
        WHEN loan_status = 'Fully Paid'
             AND datediff('month', issue_date, end_date) <= term_months - 3 THEN 'prepaid (churn)'
        WHEN loan_status = 'Fully Paid'                                     THEN 'matured on schedule'
        WHEN loan_status IN ('Charged Off', 'Default')                       THEN 'charged off'
        ELSE 'active at snapshot'
    END                                                                AS exit_reason
FROM ends;
