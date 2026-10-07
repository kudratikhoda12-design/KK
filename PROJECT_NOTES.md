# Project notes: licence, data and use

**Author:** Kudrati Khoda, M.Tech Quality, Reliability & Operations Research, Indian Statistical Institute, Kolkata.

## Code and documents

The code, reports and figures in this repository are the author's academic work, shared for review (for example, by a prospective employer). No open-source licence has been chosen; please ask the author before reusing substantial parts.

## Market data

* Market data © Binance, from the Binance Vision public archive (<https://data.binance.vision/>), licensed under **CC BY-NC-SA 4.0** (Binance Vision Dataset Terms). It is used here for non-commercial academic research only.
* **No raw or processed Binance data is included** in this repository or its ZIP package. `data/README.md` and `reproduction/README.md` explain how to download it; `reproduction/checksums/` lists the SHA-256 hash of every file used, so the exact inputs can be verified (`python scripts/verify_checksums.py`).
* Saved result tables (`reports/tables/`) contain derived statistics, not raw market data.

## Credit-risk module

`credit_module/` and `notebooks/01_credit_risk.ipynb` are the author's earlier project, included unchanged. The LendingClub data it uses is distributed through Kaggle and is not included. Its evidence status is described in `credit_module/README.md`.

## Not investment advice

This is a historical research study. Its conclusion is that the tested strategy is **not** profitable after realistic costs. Nothing here is a recommendation to trade.
