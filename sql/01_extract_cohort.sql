-- =====================================================================
-- 01_extract_cohort.sql
-- Pull the 2016 origination vintage from the public LendingClub file and
-- keep a deterministic 50,000-loan sample.
--
-- Only ORIGINATION-TIME attributes (known when the loan is booked) plus the
-- two outcome fields needed to build the survival target (loan_status,
-- last_pymnt_d) are kept. Post-origination fields such as total_pymnt,
-- recoveries or last_fico_range_* are deliberately excluded: they leak the
-- outcome into the features.
--
-- {raw_path} and {n_sample} are filled in by src/coxkan_churn/data.py.
-- =====================================================================
WITH vintage_2016 AS (
    SELECT *
    FROM read_csv_auto('{raw_path}', all_varchar = true, sample_size = -1)
    WHERE right(issue_d, 4) = '2016'
      AND loan_status IN ('Fully Paid', 'Charged Off', 'Current', 'Default',
                          'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)')
)
SELECT
    id, issue_d, loan_status, last_pymnt_d,                       -- identifiers / outcome
    loan_amnt, term, int_rate, installment, grade, sub_grade,     -- loan contract
    purpose, initial_list_status, application_type,
    emp_length, home_ownership, annual_inc, verification_status,  -- borrower profile
    addr_state, dti,
    fico_range_low, fico_range_high, earliest_cr_line,            -- credit bureau
    delinq_2yrs, inq_last_6mths, mths_since_last_delinq, open_acc,
    pub_rec, revol_bal, revol_util, total_acc, mort_acc,
    pub_rec_bankruptcies, tot_cur_bal, total_rev_hi_lim,
    acc_open_past_24mths, avg_cur_bal, bc_util, num_actv_bc_tl,
    percent_bc_gt_75, tot_hi_cred_lim, mo_sin_old_rev_tl_op,
    num_tl_op_past_12m
FROM vintage_2016
ORDER BY hash(id || '-coxkan-seed-42')                            -- deterministic pseudo-random sample
LIMIT {n_sample};
