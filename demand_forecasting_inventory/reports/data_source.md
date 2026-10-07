# Data source and provenance

## Dataset
**M5 Forecasting - Accuracy** (Makridakis Open Forecasting Center, University of Nicosia; Walmart unit sales by item, store and day, 2011-01-29 to 2016-06-19, with calendar/event/SNAP information and weekly selling prices).

## Why a mirror was used (and why it is legitimate)
* The official distribution is the Kaggle competition `m5-forecasting-accuracy`. Downloading it requires Kaggle credentials; an unauthenticated request returns **HTTP 401**, and no `~/.kaggle/kaggle.json` or `KAGGLE_*` variables exist in this environment.
* We therefore used the public Hugging Face dataset repositories `kashif/M5` (primary) and `denephew/M5_Forecasting` (cross-check). Both host the four competition files under their original names.
* **Integrity verification** (see `outputs/tables/raw_data_verification.csv`): every downloaded file has (i) the SHA-256 published by the primary mirror's API, and (ii) a hash identical to the value published by the independent second mirror. Calendar (a plain git blob) is verified through its git blob SHA-1.
* **Limitation of this evidence:** Kaggle's own checksums are not accessible without credentials, so we cannot prove byte-equality with the official download. Our evidence is agreement between two independent uploads plus the structural checks below and in `data_inspection.md` (e.g. 30,490 series, 1,941 days, 6,841,121 price rows, validation file = prefix of the evaluation file).
* Structural check against the documented M5 release: calendar has 1969 days (2011-01-29 to 2016-06-19); the evaluation sales file has 1941 observed day columns (d_1 ... d_1941).

## Verification table

| file                       |   local_bytes | size_matches_primary   | size_matches_crosscheck   | hash_type     | local_hash                                                       | hash_matches_primary   | hash_matches_crosscheck   |
|:---------------------------|--------------:|:-----------------------|:--------------------------|:--------------|:-----------------------------------------------------------------|:-----------------------|:--------------------------|
| calendar.csv               |        103469 | True                   | True                      | git-blob-sha1 | 051cdcb4c955814d3f25c6db8a844b65dbf9d3fa                         | True                   | True                      |
| sell_prices.csv            |     203395785 | True                   | True                      | sha256(lfs)   | 9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef | True                   | True                      |
| sales_train_validation.csv |     120007726 | True                   | True                      | sha256(lfs)   | f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724 | True                   | True                      |
| sales_train_evaluation.csv |     121736518 | True                   | True                      | sha256(lfs)   | 4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0 | True                   | True                      |

## Files used
* `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv` (all 1,941 observed days). `sales_train_validation.csv` is a strict prefix of the evaluation file (d_1 ... d_1913) and is only used for a consistency check.
* The 28 days after d_1941 are *not* in the public data (held out by the competition) - calendar and prices for them exist but sales do not.

## Differences from the competition set-up
* The competition's hidden evaluation period is not available; we run our own chronological train/validation/test split inside the 1,941 observed days (see `validation_strategy.md`).
* Our metrics are MAE/RMSE/sMAPE/WAPE on the sums of daily demand, not the competition's WRMSSE.

## Licence / attribution
M5 data are released by the organisers for research; cite: Makridakis, Spiliotis & Assimakopoulos (2022), *The M5 competition: Background, organization, and implementation*, International Journal of Forecasting 38(4).
