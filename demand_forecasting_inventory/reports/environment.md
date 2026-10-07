# Environment inspection (Phase 0)

* Python: `3.13.16` (CPython) on `Linux-6.18.44-fc-v77-x86_64-with-glibc2.39`
* CPU cores: 4; RAM: 15.7 GB; free disk: 28.7 GB
* Internet access: Hugging Face API HTTP 200; PyPI HTTP 200
* Kaggle credentials present: **False**; unauthenticated Kaggle M5 download endpoint: HTTP 401 (401 = authentication required)

## Installed packages used by the project

| package | version |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| statsmodels | 0.15.0 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| shap | 0.52.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| pyarrow | 25.0.1 |
| joblib | 1.6.0 |
| requests | 2.34.2 |
| tabulate | 0.10.0 |
| pytest | 9.1.1 |
| nbformat | 5.11.1 |
| nbconvert | 7.17.1 |

## Notes
* No GPU is needed or used; deep learning is deliberately *not* used (see README, design decisions).
* Missing packages are installed with `pip install -r requirements.txt` (PyPI was reachable).
* Because the official Kaggle files need credentials, the dataset is obtained from verified public mirrors (see `data_source.md`).
