# Sample data (for tests only)

These are small, byte-identical subsets of two official Binance public-archive files, so the test
suite and a quick demo run without downloading anything:

| File | Source file | Subset |
|---|---|---|
| `ETHUSDT-aggTrades-2024-06-11_0000-0059.zip` | `futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2024-06-11.zip` (SHA-256 verified against the published `.CHECKSUM` before subsetting) | header + rows with `transact_time` < 2024-06-11 01:00:00 UTC (23,032 aggregated trades) |
| `ETHUSDT-bookDepth-2024-06-11_0000-0059.zip` | `futures/um/daily/bookDepth/ETHUSDT/ETHUSDT-bookDepth-2024-06-11.zip` | header + rows stamped 2024-06-11 00:xx:xx |

Rows were copied unmodified; only the selection and re-zipping are ours, so `SHA256SUMS` lists
checksums of these subset files (not Binance's). Data © Binance, distributed through
https://data.binance.vision under Binance's terms of use; included here only as a minimal test fixture.

The full dataset is **not** in the repository. Rebuild it with `python scripts/run_stage2.py`.
