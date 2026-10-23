# Developer Attrition-Risk Analytics

A multivariate analysis of job-seeking behavior among professional developers,
using the 2019 Stack Overflow Developer Survey (public results,
https://survey.stackoverflow.co/, ODbL 1.0 licence) as a proxy for a
company-wide HR engagement/exit-risk survey.

See `report/Developer_Attrition_Risk_Analytics_Report.docx` for the full
write-up (introduction, data treatment, MANOVA, PCA, logistic + ridge
regression, LDA vs. logistic comparison, clustering, HR recommendations,
and limitations).

## Reproducing the analysis

Requires Python 3 with `pandas`, `numpy`, `scipy`, `scikit-learn`,
`statsmodels`, `matplotlib`, `python-docx`.

```
pip install pandas numpy scipy scikit-learn statsmodels matplotlib python-docx
```

1. Download the 2019 Stack Overflow Developer Survey results CSV to
   `data/raw/survey_results_public_2019.csv` (not committed to this repo;
   ~190 MB). Source: the `StackExchange/survey` GitHub repository,
   `packages/archive/2019/results.csv` (served via Git LFS; see
   `media.githubusercontent.com/media/StackExchange/survey/main/packages/archive/2019/results.csv`).
2. Run the scripts in order from the repo root:

```
python3 scripts/01_clean_data.py       # filter, clean, subsample (N=8000) -> data/processed/clean_sample.csv
python3 scripts/02_eda.py              # Table 1, correlation matrix
python3 scripts/03_manova.py           # Unit 3: one-way & two-way MANOVA, assumptions, Figure 1
python3 scripts/04_pca.py              # Unit 4: PCA, Figures 2-3, PC scores -> data/processed/pc_scores.csv
python3 scripts/05_logistic.py         # Unit 5: logistic regression, odds ratios, ridge trace, Figure 4
python3 scripts/06_lda.py              # Unit 6: LDA vs QDA vs logistic, Figure 5
python3 scripts/07_clustering.py       # Unit 7: k-means + Ward hierarchical, Figures 6-7
python3 scripts/08_build_report.py     # assembles report/Developer_Attrition_Risk_Analytics_Report.docx
```

All intermediate tables land in `tables/` (CSV/TXT) and all figures in
`figures/` (PNG), each numbered to match the report.

## Repository layout

```
scripts/     01-08, one per analysis unit, commented, each independently runnable
data/raw/    raw survey CSV (gitignored; download per instructions above)
data/processed/  cleaned sample, PCA scores, clustered sample (reproducible from scripts/)
tables/      numbered tables (CSV/TXT), reproducible from scripts/
figures/     numbered figures (PNG), reproducible from scripts/
report/      final Word report
```
