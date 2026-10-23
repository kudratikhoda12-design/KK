"""
Assemble the final report (report/Developer_Attrition_Risk_Analytics_Report.docx)
from the tables/ and figures/ produced by scripts 01-07.

Formatting: Arial 11pt body text, numbered tables and figures, structure per
the project brief (Intro, Data, EDA, MANOVA, PCA, Logistic+Ridge, LDA, Clustering,
Conclusions, Limitations).

Run: python3 scripts/08_build_report.py
"""
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

FIG = "figures/"
TAB = "tables/"

doc = Document()

# ---------------- Global style: Arial 11 -------------------------------
style = doc.styles["Normal"]
style.font.name = "Arial"
style.font.size = Pt(11)
rpr = style.element.get_or_add_rPr()
rFonts = rpr.find(qn("w:rFonts"))
if rFonts is None:
    rFonts = rpr.makeelement(qn("w:rFonts"), {})
    rpr.append(rFonts)
rFonts.set(qn("w:eastAsia"), "Arial")
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.space_before = Pt(0)

for hstyle_name in ["Heading 1", "Heading 2"]:
    hs = doc.styles[hstyle_name]
    hs.paragraph_format.space_before = Pt(8)
    hs.paragraph_format.space_after = Pt(4)

for sec in doc.sections:
    sec.top_margin = Inches(0.7)
    sec.bottom_margin = Inches(0.7)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

tbl_counter = [0]
fig_counter = [0]


def h1(text):
    p = doc.add_heading(text, level=1)
    for run in p.runs:
        run.font.name = "Arial"
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def h2(text):
    p = doc.add_heading(text, level=2)
    for run in p.runs:
        run.font.name = "Arial"
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def para(text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(11)
    return p


def caption(prefix, counter, text):
    counter[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"{prefix} {counter[0]}. {text}")
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return counter[0]


def add_table_from_df(df, index=True, max_rows=None, fmt=None):
    d = df.copy()
    if max_rows:
        d = d.head(max_rows)
    cols = (["" ] if index else []) + list(d.columns.astype(str))
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, c in enumerate(cols):
        hdr[i].text = str(c)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)
                r.bold = True
                r.font.name = "Arial"
    for idx, row in d.iterrows():
        cells = t.add_row().cells
        offset = 0
        if index:
            cells[0].text = str(idx)
            offset = 1
        for j, c in enumerate(d.columns):
            val = row[c]
            if fmt and c in fmt:
                val = fmt[c](val)
            cells[j + offset].text = str(val)
        for cell in cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.name = "Arial"
    return t


def add_figure(path, width=5.8):
    doc.add_picture(FIG + path, width=Inches(width))
    last_p = doc.paragraphs[-1]
    last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# =====================================================================
# TITLE
# =====================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Developer Attrition-Risk Analytics")
r.font.name = "Arial"; r.font.size = Pt(20); r.bold = True

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("A Multivariate Analysis of Job-Seeking Behavior Among Professional "
                 "Developers (2019 Stack Overflow Developer Survey)")
r.font.name = "Arial"; r.font.size = Pt(13); r.italic = True

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = meta.add_run("HR People Analytics Report | October 2026")
r.font.name = "Arial"; r.font.size = Pt(10)
doc.add_paragraph()

# =====================================================================
# 1. INTRODUCTION
# =====================================================================
h1("1. Introduction & Motivation")
para(
    "Developer turnover is one of the costliest and least predictable forms of "
    "attrition in the technology sector: replacing a mid-level engineer typically "
    "costs 50-200% of annual salary once recruiting, onboarding, and lost "
    "productivity are counted. Because resignation decisions build up over months, "
    "HR teams need early, data-driven signals of who is at risk of leaving - not "
    "just exit-interview hindsight. This report uses the Stack Overflow Annual "
    "Developer Survey, a large, standardized, cross-company survey of working "
    "developers, as a proxy for a company-wide engagement/exit-risk survey, and "
    "asks three questions a retention team would ask of its own population:"
)
bullet("Do developer profiles (pay, experience, satisfaction, workload) differ "
       "between those who are job-seeking and those who are not, and does that "
       "difference vary by country?")
bullet("Which measurable factors drive job-seeking behavior, in effect sizes an "
       "HR partner can act on?")
bullet("Can the developer population be segmented into a small number of "
       "actionable talent profiles for targeted retention strategy?")

# =====================================================================
# 2. DATA & VARIABLES
# =====================================================================
h1("2. Data & Variables")
para(
    "Source: Stack Overflow Developer Survey 2019 public results "
    "(https://survey.stackoverflow.co/, archived under the StackExchange/Survey "
    "GitHub repository, ODbL 1.0 licence), 88,883 raw responses. The 2019 wave "
    "was selected for its stable schema and large N."
)
h2("2.1 Filtering and sample construction")
para(
    "The raw file was filtered to MainBranch = \"I am a developer by profession\" "
    "and Employment = \"Employed full-time\" (55,405 respondents), then restricted "
    "to the five largest countries by respondent count - United States, India, "
    "United Kingdom, Germany, Canada (29,851 respondents). After deriving the "
    "analysis variables, rows with any missing value on an analysis variable were "
    "dropped (complete-case analysis: 29,851 -> 17,847, a 40.2% drop driven mainly "
    "by CodeRevHrs (23.4% missing - many respondents are not asked or do not do "
    "code review) and ConvertedComp (19.4% missing - compensation questions are "
    "routinely skipped)). A reproducible random subsample of N = 8,000 was then "
    "drawn (seed = 42) for computational tractability, matching the brief's target "
    "of n ~ 5,000-10,000."
)
h2("2.2 Variable treatment")
para(
    "ConvertedComp (USD/year, adjusted by Stack Overflow) was floored at >0, capped "
    "at the 99th percentile to tame currency-conversion and self-report outliers, "
    "and log-transformed (logComp) to correct its strong right skew. YearsCodePro "
    "and CodeRevHrs were parsed from free-text buckets (\"Less than 1 year\"->0.5, "
    "\"More than 50 years\"->51); CodeRevHrs was capped at its 99th percentile "
    "(20.0 h/week). WorkWeekHrs was capped at 100 h/week per the brief and "
    "non-positive values treated as missing. Age was restricted to a plausible "
    "working-adult range [16, 75]. JobSat (5-point Likert: Very dissatisfied ... "
    "Very satisfied) was coded 1-5, justified because the response categories are "
    "equally spaced verbal anchors, a standard treatment for ordinal survey items "
    "in parametric models."
)
para(
    "Outcome (JobSeek): the source item has three levels. We coded \"I am actively "
    "looking for a job\" and \"I'm not actively looking, but I am open to new "
    "opportunities\" as job-seeking risk = 1 (both represent real flight risk from "
    "an HR standpoint - passive candidates convert easily when approached), and "
    "\"I am not interested in new job opportunities\" as 0. This mapping is the "
    "single most consequential data decision in the report and is stated here "
    "explicitly so results can be re-derived under an alternative (stricter) "
    "mapping if desired."
)
para("Factors used for grouping/profiling: Country, EdLevel, Hobbyist, OrgSize.")

h2("2.3 Missingness and final sample")
miss_df = pd.read_csv(TAB + "table0_cleaning_report.txt", header=None) if False else None
caption("Table", tbl_counter, "Missingness (%) on the filtered, pre-cleaning "
        "sample (N=29,851), analysis variables")
miss_tbl = pd.DataFrame({
    "variable": ["CodeRevHrs", "ConvertedComp", "Age", "WorkWeekHrs", "OrgSize",
                 "EdLevel", "YearsCodePro", "JobSat", "JobSeek", "Hobbyist"],
    "% missing": [23.41, 19.42, 9.81, 8.35, 2.76, 1.75, 0.80, 0.43, 0.00, 0.00],
})
add_table_from_df(miss_tbl, index=False)
para(
    "Final analysis sample: N = 8,000. Class distribution of the outcome: "
    "67.6% job-seeking (actively looking or open), 32.4% not interested. "
    "Country composition: United States 4,250 (53.1%), India 1,122 (14.0%), "
    "United Kingdom 1,053 (13.2%), Germany 979 (12.2%), Canada 596 (7.5%)."
)

# =====================================================================
# 3. EXPLORATORY ANALYSIS
# =====================================================================
h1("3. Exploratory Analysis")
para(
    "Table 1 summarizes the five numeric variables used in the MANOVA/PCA blocks "
    "(plus CodeRevHrs) by job-seeking status. Job-seekers report markedly lower "
    "job satisfaction (3.51 vs 4.48 on the 5-point scale), somewhat lower pay "
    "(median ~$73k vs ~$105k implied by mean logComp) and fewer years of "
    "professional experience, consistent with early/mid-career dissatisfaction "
    "being the dominant visible signal."
)
t1 = pd.read_csv(TAB + "table1_descriptives_by_jobseek.csv")
t1p = t1.pivot(index="variable", columns="JobSeek_lbl", values="mean")
t1p = t1p.reindex(["logComp", "YearsCodePro", "WorkWeekHrs", "Age", "JobSat_num", "CodeRevHrs"])
t1p.columns = [f"Mean ({c})" for c in t1p.columns]
caption("Table", tbl_counter, "Descriptive means of numeric variables by job-seeking status (N=8,000)")
add_table_from_df(t1p.round(2))

para(
    "Table 1b (correlation matrix) motivates the PCA in Section 5: YearsCodePro "
    "and Age are very strongly correlated (r=0.86), as expected (career tenure "
    "tracks age), while logComp correlates moderately with both (r~0.26-0.27). "
    "JobSat and CodeRevHrs are nearly orthogonal to the seniority/earnings block, "
    "suggesting at least two to three independent dimensions in the data."
)
corr = pd.read_csv(TAB + "table1b_corr_matrix.csv", index_col=0)
caption("Table", tbl_counter, "Correlation matrix of the numeric block (N=8,000)")
add_table_from_df(corr)

# =====================================================================
# 4. MANOVA RESULTS
# =====================================================================
h1("4. MANOVA Results")
h2("4.1 One-way MANOVA: DVs ~ JobSeek")
para(
    "DVs = [logComp, YearsCodePro, WorkWeekHrs, Age, JobSat]; factor = JobSeek "
    "(seeking/open vs. not interested)."
)
manova1 = pd.DataFrame({
    "Statistic": ["Wilks' Lambda", "Pillai's Trace", "Hotelling-Lawley Trace", "Roy's Greatest Root"],
    "Value": [0.8562, 0.1438, 0.1679, 0.1679],
    "F": [268.49, 268.49, 268.49, 268.49],
    "Num DF": [5, 5, 5, 5],
    "Den DF": [7994, 7994, 7994, 7994],
    "p": ["<.0001", "<.0001", "<.0001", "<.0001"],
})
caption("Table", tbl_counter, "One-way MANOVA: JobSeek effect on the 5-variable profile")
add_table_from_df(manova1, index=False)
para(
    "The one-way MANOVA is highly significant (Wilks' Lambda=0.856, "
    "F(5,7994)=268.49, p<.0001): job-seekers and non-seekers have reliably "
    "different multivariate profiles across pay, experience, hours, age, and "
    "satisfaction jointly, not just on any single variable. Because the omnibus "
    "test is significant, we proceed to univariate follow-up ANOVAs (Table 2b)."
)
anova = pd.read_csv(TAB + "table2c_univariate_anova.csv")
caption("Table", tbl_counter, "Univariate follow-up ANOVAs (one DV at a time)")
add_table_from_df(anova, index=False)
para(
    "JobSat (F=1235.96, p<.0001) and logComp (F=144.23, p<.0001) carry almost all "
    "of the multivariate signal; YearsCodePro and Age are significant but with "
    "small effect sizes, and WorkWeekHrs does not differ reliably by group "
    "(p=0.225) - job-seekers are not simply the overworked."
)

h2("4.2 Two-way MANOVA: JobSeek x Country")
manova2 = pd.DataFrame({
    "Effect": ["JobSeek (main)", "Country (main)", "JobSeek x Country (interaction)"],
    "Wilks' Lambda": [0.9888, 0.9127, 0.9967],
    "F": [18.12, 37.00, 1.31],
    "Num DF": [5, 20, 20],
    "Den DF": [7986, "26487.5*", "26487.5*"],
    "p": ["<.0001", "<.0001", "0.159"],
})
caption("Table", tbl_counter, "Two-way MANOVA: JobSeek x Country")
add_table_from_df(manova2, index=False)
para(
    "*Approximate denominator df under the Rao F-approximation for Wilks' Lambda. "
    "Both main effects remain significant with Country in the model. The "
    "JobSeek x Country interaction is not significant by Wilks' Lambda (p=0.159), "
    "meaning the seeking vs. non-seeking profile difference is broadly similar "
    "across the five countries - though Roy's Greatest Root (which is sensitive "
    "to a single dominant dimension) does flag a significant interaction "
    "(p<.001), likely attributable to India's distinctly elevated pay-growth "
    "trajectory interacting with job-seeking (see Section 7, Cluster 2). Given "
    "Wilks' Lambda is the standard, more conservative omnibus statistic, we treat "
    "the interaction as modest rather than a first-order driver, and interpret "
    "country mainly as an independent control."
)

h2("4.3 Assumption checks")
para(
    "Box's M (equality of covariance matrices): chi2(15)=729.36, p<.0001 - "
    "formally violated. Box's M is well known to be oversensitive at large N "
    "(here ~8,000), rejecting equal covariances for even trivial differences; we "
    "therefore do not treat this as disqualifying, consistent with the standard "
    "advice to prefer Pillai's Trace (the most robust of the four MANOVA "
    "statistics to covariance heterogeneity) as the primary significance test - "
    "which agrees with Wilks' Lambda here (both p<.0001)."
)
para(
    "Mardia's test of multivariate normality on group-mean-centered residuals "
    "rejects normality in both groups (skewness and kurtosis tests, both "
    "p<.0001) - expected, since the DV block mixes a 5-point ordinal variable "
    "(JobSat) with skewed continuous variables. With N in the thousands per "
    "group, MANOVA's F-approximations are robust to this degree of non-normality "
    "via the central limit theorem, so we retain the parametric MANOVA rather "
    "than a permutation alternative."
)
para(
    "Levene's test per DV found significant heteroscedasticity for logComp, "
    "YearsCodePro, WorkWeekHrs, and JobSat (all p<.01), but not Age (p=0.93). "
    "Combined with the unequal group sizes (67.6%/32.4%), this reinforces the "
    "choice of Pillai's Trace as the headline statistic."
)
caption("Figure", fig_counter, "Standardized mean profiles by job-seeking status "
        "(parallel coordinate plot)")
add_figure("fig1_parallel_coords.png", width=4.3)
para(
    "Figure 1 visualizes the group-mean profile directly: non-seekers sit above "
    "average on pay and (modestly) on experience/age, and far above average on "
    "satisfaction; job-seekers are essentially flat on pay/experience/hours and "
    "markedly below average on satisfaction. The profile difference is almost "
    "entirely a satisfaction story, with a secondary pay/tenure story."
)
para("HR takeaway: a single engagement-survey satisfaction score is, on this "
     "evidence, the most efficient early-warning signal available - cheaper to "
     "collect than a full compensation/tenure audit and carrying most of the "
     "discriminating information.", bold=False, italic=True)

# =====================================================================
# 5. PCA RESULTS
# =====================================================================
h1("5. PCA Results")
para(
    "The 6-variable numeric block (logComp, YearsCodePro, WorkWeekHrs, Age, "
    "JobSat, CodeRevHrs), standardized, was reduced via eigendecomposition of "
    "its correlation matrix (shown in Table 1b above - the Age/YearsCodePro "
    "redundancy motivates this step). Retention rule: eigenvalue > 1 AND "
    "cumulative variance >= 70%."
)
eig = pd.read_csv(TAB + "table3b_pca_eigen_summary.csv")
caption("Table", tbl_counter, "PCA eigenvalues and variance explained")
add_table_from_df(eig, index=False)
caption("Figure", fig_counter, "Scree plot: eigenvalues and cumulative variance explained")
add_figure("fig2_scree.png", width=4.3)
para(
    "Three components clear the eigenvalue>1 threshold and jointly reach exactly "
    "70.0% cumulative variance, so 3 components are retained (PC1-PC3)."
)
load = pd.read_csv(TAB + "table3_pca_loadings.csv", index_col=0)
caption("Table", tbl_counter, "PCA loadings (correlation-scale) for retained components")
add_table_from_df(load)
para(
    "PC1 (33.7% of variance) loads heavily and almost exclusively on "
    "YearsCodePro (0.93) and Age (0.93), with a moderate contribution from "
    "logComp (0.51) - a \"Seniority & Earnings\" axis. PC2 (19.0%) loads "
    "positively on CodeRevHrs (0.76) and WorkWeekHrs (0.52) and negatively on "
    "logComp (-0.35) and JobSat (-0.41) - a \"Workload intensity\" axis "
    "(busier, more code-review-heavy roles that pay/satisfy relatively less). "
    "PC3 (17.3%) loads positively on JobSat (0.74) and WorkWeekHrs (0.54), "
    "and negatively on YearsCodePro/Age - a \"Satisfaction net of seniority\" "
    "axis distinguishing happy vs. unhappy developers of similar tenure."
)
caption("Figure", fig_counter, "PCA biplot (PC1 vs PC2), colored by job-seeking status")
add_figure("fig3_biplot.png", width=4.3)
para(
    "The biplot shows job-seekers (orange) distributed broadly across PC1 "
    "(seniority is not strongly differentiating, consistent with Section 4) but "
    "with a visible density shift toward the lower/negative region of PC2 and "
    "PC3 -lower satisfaction, in the direction opposite the JobSat arrow - "
    "echoing the MANOVA result. PC scores (PC1-PC3) are carried forward as the "
    "feature space for clustering in Section 7."
)

# =====================================================================
# 6. LOGISTIC + RIDGE RESULTS
# =====================================================================
h1("6. Logistic Regression and Ridge Regularization")
para(
    "Model: JobSeek ~ logComp + YearsCodePro + WorkWeekHrs + Age + JobSat + "
    "CodeRevHrs + EdLevel (3 dummies, ref.=Less than Bachelor's) + Hobbyist + "
    "Country (4 dummies, ref.=United States). McFadden's pseudo-R2 = 0.154 "
    "(deviance -2LL = 8,529.6 vs. null -2LL = 10,077.6), a solid fit for a "
    "cross-sectional attitudinal outcome."
)
orl = pd.read_csv(TAB + "table4_odds_ratios.csv", index_col=0).drop(index="const")
orl.columns = ["coef", "OR", "CI low", "CI high", "p"]
caption("Table", tbl_counter, "Odds ratios (95% CI) for job-seeking, natural-unit model")
add_table_from_df(orl.round(3))
para(
    "In plain HR language: each additional point of job satisfaction (5-point "
    "scale) is associated with a 58% lower odds of job-seeking (OR=0.418, "
    "95% CI 0.394-0.444) - by far the largest, most precisely estimated effect "
    "in the model. Each extra year of professional tenure lowers odds by about "
    "2.4% (OR=0.976), holding age constant, while - net of tenure - each extra "
    "year of age raises odds by about 2.3% (OR=1.023); these two coefficients "
    "partially offset because tenure and age are highly collinear (r=0.86, see "
    "Section 6.1) and should be read together, not separately. Hobbyist coders "
    "(who code outside of work) have 30% higher odds of seeking (OR=1.296, "
    "p<.001) - plausibly a proxy for higher general career engagement/options. "
    "Compensation (logComp) and weekly hours are not significant once "
    "satisfaction and tenure are controlled. Country is a strong control: "
    "Indian developers have more than 3x the odds of U.S. developers of "
    "job-seeking (OR=3.195, p<.001), while German (OR=0.683) and UK (OR=0.829) "
    "developers have lower odds than the U.S. Education level is not a "
    "significant driver of job-seeking in this sample."
)
lrt = pd.read_csv(TAB + "table4b_lr_tests.csv")
caption("Table", tbl_counter, "Likelihood-ratio tests for block importance")
add_table_from_df(lrt, index=False)
para(
    "LR tests confirm the numeric block (chi2=1203.4, df=6, p<.0001) and Country "
    "(chi2=148.7, df=4, p<.0001) are the dominant contributors, Hobbyist is a "
    "significant but modest contributor (chi2=15.8, df=1, p<.001), and EdLevel "
    "adds nothing once the other blocks are in the model (chi2=2.7, df=3, "
    "p=0.437)."
)

h2("6.1 Collinearity and ridge regularization")
para(
    "The standardized design matrix (numerics z-scored, categorical dummies, "
    "plus intercept) has condition number 10.3 - moderate, below the common "
    "concern threshold of ~30, but driven almost entirely by the Age/"
    "YearsCodePro redundancy (r=0.86) documented in Section 3/5. To quantify "
    "and correct for this, we fit a ridge-penalized logistic regression across "
    "a grid of 40 penalties (lambda from 10^-3 to 10^3) on the same standardized "
    "design matrix."
)
caption("Figure", fig_counter, "Ridge trace: standardized coefficients vs. log(lambda)")
add_figure("fig4_ridge_trace.png", width=4.6)
para(
    "As expected from the correlation structure, Age (red) and YearsCodePro "
    "(orange) are the two coefficients that move the most under increasing "
    "penalty - their unregularized OLS-logistic estimates are being inflated "
    "and partially offsetting each other, and ridge shrinkage pulls both toward "
    "a smaller, more stable joint effect well before the JobSat or Country "
    "coefficients move appreciably."
)
cv_lambda = 4.924
para(f"A 5-fold cross-validated search (minimizing held-out deviance/log-loss) "
     f"selects lambda* = {cv_lambda:.2f}.")
rc = pd.read_csv(TAB + "table4d_ridge_vs_ols_coefs.csv", index_col=0)
caption("Table", tbl_counter, "Ridge (lambda=4.92) vs. unregularized logistic, standardized coefficients")
add_table_from_df(rc)
para(
    "At the cross-validated lambda, JobSat barely shrinks (0.7%) and remains "
    "the dominant, robust driver; Country and Hobbyist effects are also stable "
    "(<10% shrinkage). YearsCodePro and Age shrink modestly (3.6% and 4.5%) "
    "together, and the sparsely populated Doctoral/Professional education "
    "dummy (n=210 in-sample) shrinks the most (41.5%) as expected for a "
    "high-variance, small-category estimate. The substantive conclusion is "
    "unchanged by regularization: satisfaction dominates, tenure/age act as a "
    "paired (partially collinear) control, and the OLS logistic odds ratios in "
    "Table 4 can be read with confidence."
)

# =====================================================================
# 7. LDA & COMPARISON
# =====================================================================
h1("7. Linear Discriminant Analysis and Comparison with Logistic Regression")
para(
    "To compare classifiers on equal footing, LDA, QDA, and logistic regression "
    "were all fit on the same 6 standardized numeric predictors (no categorical "
    "controls), using 5-fold stratified cross-validation (shuffled, "
    "random_state=42); out-of-fold predictions were pooled into one confusion "
    "matrix per model."
)
para(
    "Box's M on this 6-variable block is again significant (chi2(21)=785.3, "
    "p<.0001), formally favoring QDA's unequal-covariance assumption over LDA's. "
    "Because Box's M is oversensitive at this sample size (Section 4.3), we "
    "instead let cross-validated performance arbitrate between LDA and QDA."
)
cm = pd.read_csv(TAB + "table5_confusion_matrices.csv")
caption("Table", tbl_counter, "Confusion matrices (5-fold CV, pooled out-of-fold predictions)")
add_table_from_df(cm, index=False)
met = pd.read_csv(TAB + "table5b_classifier_metrics.csv")
met = met.rename(columns={
    "accuracy": "accuracy", "error_rate": "error rate",
    "sensitivity_recall_seeking": "sensitivity", "specificity": "specificity",
})
caption("Table", tbl_counter, "Classifier comparison: accuracy, sensitivity, specificity, AUC")
add_table_from_df(met, index=False)
caption("Figure", fig_counter, "ROC curves: LDA, QDA, and logistic regression (5-fold CV)")
add_figure("fig5_roc_curves.png", width=4.0)
para(
    "LDA and logistic regression are statistically indistinguishable on "
    "discrimination (AUC 0.757 vs 0.757) - unsurprising, since both fit "
    "linear decision boundaries, and QDA's added flexibility does not pay off "
    "here (AUC 0.738, lower accuracy), suggesting the extra covariance "
    "parameters mostly fit noise rather than real nonlinearity. The two linear "
    "models trade off differently at the default 0.5 threshold: LDA achieves "
    "much higher sensitivity (88.5% vs 84.6%) at the cost of lower specificity "
    "(37.4% vs 49.5%) than logistic regression, because LDA's threshold is set "
    "by the class priors (67.6% seeking) rather than directly optimizing "
    "log-likelihood for the 0.5 cut; logistic regression gives the better "
    "overall error rate (26.8% vs 28.1%) and, critically, produces directly "
    "interpretable odds ratios (Section 6) that LDA does not. We recommend "
    "logistic regression as the primary model for deployment (better balance, "
    "interpretable coefficients, easy to recalibrate the decision threshold to "
    "an HR-chosen cost ratio), while noting that if the operational goal is to "
    "cast the widest possible net for a proactive retention outreach program "
    "(where missing a flight risk is far costlier than one extra wellness "
    "check-in), LDA's higher recall is a legitimate alternative - and in "
    "practice, either model's predicted probability can simply be "
    "threshold-tuned to the same operating point, since their ROC curves "
    "coincide."
)

# =====================================================================
# 8. CLUSTERING & SEGMENTS
# =====================================================================
h1("8. Clustering & Developer Segments")
para(
    "K-means (k-means++ initialization, 25 restarts) was run on the 3 retained "
    "PC scores (Section 5) across k=1..10. k was selected using both the elbow "
    "in within-cluster sum of squares and the gap statistic (15 uniform "
    "reference datasets per k); both point to k=3."
)
caption("Figure", fig_counter, "Within-cluster SS (elbow) and gap statistic across k=1..10")
add_figure("fig6b_elbow_gap.png", width=4.6)
para(
    "Ward's hierarchical clustering (Euclidean distance) on the same PC scores, "
    "cut at k=3, was compared to the k-means partition via the Adjusted Rand "
    "Index: ARI = 0.648, indicating substantial (though not perfect) agreement "
    "between the two methods - a reassuring robustness check that the 3-segment "
    "structure is not an artifact of one algorithm."
)
caption("Figure", fig_counter, "Ward hierarchical dendrogram (truncated), cut at k=3")
add_figure("fig6_dendrogram.png", width=4.6)
caption("Figure", fig_counter, "K-means clusters (k=3) in PC1-PC2 space")
add_figure("fig7_kmeans_pc1_pc2.png", width=4.2)
para(
    "Clusters 0 and 2 overlap substantially in the PC1-PC2 projection shown "
    "above because their main separation is along PC3 (satisfaction, net of "
    "seniority) rather than PC1/PC2 - visible instead in the cluster profile "
    "table below."
)

prof = pd.read_csv(TAB + "table6_cluster_profiles.csv", index_col=0)
prof["n"] = prof["n"].astype(int)
prof.index = ["Cluster 0: Settled & Satisfied", "Cluster 1: Senior Specialists",
              "Cluster 2: Early-Career Flight Risk"]
caption("Table", tbl_counter, "Cluster profiles: mean numeric values and size")
add_table_from_df(prof)

seek_cross = pd.read_csv(TAB + "table6b_cluster_vs_jobseek_country.csv", header=[0, 1], index_col=0)
para(
    "Cross-tabulated against job-seeking status and country, the three segments "
    "read as distinct, actionable HR profiles:"
)
bullet(
    "Cluster 0 - \"Settled & Satisfied\" (n=4,183, 52.3% of sample): mid tenure "
    "(5.7 yrs), average pay (~$108k), high satisfaction (4.45/5), low risk - "
    "only 3.6% actively looking, 42.3% not interested in a move at all. The "
    "bulk of the stable workforce; retention spend here has the lowest ROI."
)
bullet(
    "Cluster 1 - \"Senior Specialists\" (n=1,652, 20.7%): most experienced "
    "(20.1 yrs), oldest (43.7), highest paid (~$156k), moderately satisfied "
    "(3.78/5); only 7.9% actively looking. Low volume, high replacement cost - "
    "worth a lightweight, low-cost touch (career-growth/mentoring conversations) "
    "given how expensive and slow senior specialists are to replace."
)
bullet(
    "Cluster 2 - \"Early-Career Flight Risk\" (n=2,165, 27.1%): youngest "
    "(28.1), similar tenure to Cluster 0 (5.1 yrs) but lowest pay (~$30k "
    "implied, reflecting a heavy India/lower-cost-market concentration: 40.7% "
    "of this cluster is India vs. 14.0% of the overall sample), lowest "
    "satisfaction (2.66/5) and heaviest code-review load (6.35 h/week). This "
    "segment is 25.2% actively looking for a job - roughly 7x Cluster 0's rate "
    "- and only 11.4% say they are not interested in a move. This is the "
    "highest-priority retention segment."
)
para(
    "HR takeaway: the clustering independently reproduces the MANOVA/logistic "
    "findings (satisfaction and, to a lesser extent, pay/tenure drive "
    "job-seeking) but turns them into a targeting list - Cluster 2 is a "
    "concrete, sizeable (27% of the workforce), identifiable population for a "
    "pay-equity review and workload rebalancing (code-review load), rather than "
    "a generic company-wide engagement initiative.", italic=True
)

# =====================================================================
# 9. CONCLUSIONS & HR RECOMMENDATIONS
# =====================================================================
h1("9. Conclusions & HR Recommendations")
bullet("Monitor job satisfaction as the primary, cheapest leading indicator: it "
       "dominates every model in this report (MANOVA, logistic odds ratios, "
       "ridge-stable coefficient, and the main cluster-separating axis).")
bullet("Treat pay and tenure as a secondary, collinear pair, not independent "
       "levers; a pay review should be paired with a tenure/seniority context "
       "check rather than read in isolation (Section 6.1).")
bullet("Prioritize the Early-Career Flight Risk segment (27% of the workforce, "
       "25% actively job-seeking): a targeted pay-equity audit (this segment is "
       "disproportionately in lower-cost markets, notably India) and code-review "
       "workload rebalancing are the two most defensible, specific interventions "
       "the data support.")
bullet("Give Senior Specialists (21% of the workforce, expensive to replace) a "
       "lighter-touch, non-monetary retention motion - this group is not highly "
       "price-sensitive by this evidence, but has an 8% active-seeking rate "
       "worth tracking.")
bullet("For a predictive flight-risk score, deploy the logistic regression "
       "model (interpretable odds ratios, tunable decision threshold); use "
       "country and hobbyist status as context, not blame - both are strong "
       "statistical controls but not actionable levers on their own.")

# =====================================================================
# 10. LIMITATIONS
# =====================================================================
h1("10. Limitations")
bullet("Self-selection bias: Stack Overflow survey respondents are a "
       "convenience sample of developers engaged enough with the community to "
       "take the survey; they may not represent the full developer population "
       "or any single company's workforce.")
bullet("Cross-sectional design: all variables are measured at one point in "
       "time, so reported associations (e.g., satisfaction -> job-seeking) are "
       "correlational, not causal, and could run in either direction.")
bullet("No true churn outcome: JobSeek is a stated intention/attitude, not "
       "observed turnover; the mapping of \"open to opportunities\" into the "
       "risk group (Section 2.2) is a modeling choice that inflates the "
       "positive class and should be validated against a company's own exit "
       "data before being used operationally.")
bullet("External validity: the 2019 wave predates the 2020-2023 labor-market "
       "disruptions (COVID-19 hiring freezes, 2022-2023 tech layoffs) that "
       "materially changed developer job-seeking behavior; a current "
       "deployment should re-estimate on recent data.")
bullet("Country generalizability: only the five largest-N countries are "
       "included; findings (especially the India effect) should not be "
       "extrapolated to markets not represented in the sample.")

doc.save("report/Developer_Attrition_Risk_Analytics_Report.docx")
print("Saved report/Developer_Attrition_Risk_Analytics_Report.docx")
print(f"Tables numbered: {tbl_counter[0]}, Figures numbered: {fig_counter[0]}")
