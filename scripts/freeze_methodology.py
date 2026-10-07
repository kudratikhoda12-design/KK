#!/usr/bin/env python3
"""Freeze the methodology BEFORE the holdout is touched.

Writes reports/tables/frozen_parameters.json (machine-readable, read by run_holdout.py) and
docs/final_methodology.md (human-readable). Every value is copied from config/config.yaml or from the
development-selection outputs; the SHA-256 of the markdown is recorded in the holdout access log on every
holdout access, so any later edit is detectable.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ethof.config import load_config  # noqa: E402
from ethof.features import FEATURE_SETS, GROUPS  # noqa: E402
from ethof.strategy import cost_scenarios  # noqa: E402

TAB = ROOT / "reports" / "tables"


def main() -> None:
    out_md = ROOT / "docs" / "final_methodology.md"
    if (ROOT / "reports" / "holdout_access_log.jsonl").exists():
        sys.exit("holdout already accessed: the methodology may not be re-frozen")
    cfg = load_config()
    rs = cfg["research"]
    sel = json.loads((TAB / "primary_model_selection.json").read_text())
    lgbm = json.loads((TAB / "lgbm_selected_params.json").read_text())
    model = sel["selected"].split("_")[0]
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "feature_set": "E_all", "horizon": rs["primary_horizon"],
        "lgbm_params": lgbm, "logit_C": rs["models"]["logit_C"],
        "rf_params": {k: v for k, v in rs["models"]["rf"].items() if k != "train_stride"},
        "rf_train_stride": rs["models"]["rf"]["train_stride"],
        "delta": sel["delta_selected"], "train_stride": rs["walk_forward"]["train_stride"],
        "embargo_days": rs["walk_forward"]["embargo_days"], "block_months": rs["walk_forward"]["block_months"],
        "flow_rule_z": rs["signal"]["flow_rule_z"], "costs": rs["costs"], "neutral_band": rs["neutral_band"],
        "holdout": rs["holdout"], "selection": {k: v for k, v in sel.items() if k != "delta_table"},
    }
    (TAB / "frozen_parameters.json").write_text(json.dumps(frozen, indent=1, default=float))
    cs = cost_scenarios(cfg)
    reg = json.loads((TAB / "regime_cutoffs.json").read_text())
    feat_lines = "\n".join(f"| {g} | {len(fs)} | {', '.join('`'+f+'`' for f in fs)} |" for g, fs in GROUPS.items())
    md = f"""# Final Methodology (FROZEN)

Frozen at **{frozen['frozen_at_utc']}**, before any holdout row was evaluated.
`scripts/freeze_methodology.py` generated this file from `config/config.yaml` and the
development-period selection outputs. `reports/tables/frozen_parameters.json` is the machine-readable copy
that `scripts/run_holdout.py` reads. Every holdout access is logged in `reports/holdout_access_log.jsonl` together with the
SHA-256 of this file. If the file changes after the holdout is opened, the hashes in the log will no longer match.

Environment: Python {platform.python_version()}, packages pinned in `requirements.txt`. Seeds: model seed 20231
(`ethof.models.SEED`), bootstrap seeds fixed in code.

## 1. Data and periods
- Instrument: Binance USDⓈ-M **ETHUSDT perpetual** (public archive, SHA-256-verified files).
- Full period 2023-03-01 → 2026-09-30 UTC. **Development** 2023-03-01 → 2025-09-30. **Holdout**
  {rs['holdout']['start']} → {rs['holdout']['end_exclusive']} (exclusive).
- Research frequency: complete 1-minute grid. Bar t covers [ts, ts+1 min).
- Data treatment: as in `docs/data_quality_report.md`. Nothing is deleted for being extreme. Book features are missing when
  the snapshot is older than {cfg['quality']['book_stale_seconds']} s, where age is measured from the last snapshot whose values *changed*
  (this covers the frozen feed). Open interest ≤ 0 is set to missing, and ±inf ratios are set to missing.

## 2. Timeline
| Step | Time |
|---|---|
| features of row t | data stamped < τ_t = ts_t + 1 min (trades of bars ≤ t; last book snapshot < τ_t; settled funding ≤ τ_t; OI created ≤ τ_t − 5 min) |
| signal | at τ_t |
| execution | during bar t+1 at its VWAP (taker); never at the signal price |
| holding | {rs['primary_horizon']} bars per signal (staggered: net position = mean of the last {rs['primary_horizon']} signals) |
| exit | VWAP of bar t+1+{rs['primary_horizon']} (implicitly, through the staggered position) |
| target | (P[t+{rs['primary_horizon']}] − P[t]) / P[t], P = last trade price at or before the bar close |

## 3. Features ({len(FEATURE_SETS['E_all'])} in the primary set E)
| Group | n | Features |
|---|---|---|
{feat_lines}

The order-book features are **percentage-band** imbalances (cumulative resting quantity within ±1/2/3/5 % of the price).
They are **not** top-N level features, and no best bid/ask or spread information is used. Trailing windows only. Large-trade threshold
is ≥ 10 ETH (rule: the size bucket closest to the median per-minute p95 trade size in 2023-03..2024-02).

Ablation sets: A = price; B = A + volume; C = B + flow; D = C + book; E = D + funding/OI.

## 4. Target
- Primary: binary `up_15 = 1{{fwd_ret_15 > 0}}`. Secondary (reported, not used for trading): 3-class with neutral band
  ±{rs['neutral_band']*1e4:.0f} bps (= base round-trip cost); regression on fwd_ret_15; horizons 5/30/60.

## 5. Validation and preprocessing
- Expanding walk-forward, {rs['walk_forward']['block_months']}-month blocks, first validation block {rs['walk_forward']['first_validation']}, purge of
  training rows whose label ends after (block start − {rs['walk_forward']['embargo_days']} day embargo), training stride {rs['walk_forward']['train_stride']} minutes.
- Imputation and scaling happen inside sklearn pipelines fitted per training fold. LightGBM uses native missing-value handling.

## 6. Model (frozen)
- Selection rule (pre-declared): the lowest mean validation log loss on set E, h = 15, among logit / RF / LightGBM.
- **Selected: `{sel['selected']}`** (model class `{model}`).
- LightGBM hyper-parameters (8-point grid, chosen by mean validation log loss): `{json.dumps({**lgbm})}`;
  other parameters are fixed in `ethof.models.LGBM_DEFAULT`. Logit C = {rs['models']['logit_C']}. RF: `{json.dumps(frozen['rf_params'])}`, stride {frozen['rf_train_stride']}.
- Holdout procedure: four 3-month blocks. Before each block the frozen model is re-fitted on all data from 2023-03-01 to
  (block start − 1 day) (the same expanding scheme as development). No parameter, feature or threshold is revisited.

## 7. Trading rule (frozen)
- LONG if P(up) > 0.5 + δ, SHORT if P(up) < 0.5 − δ, FLAT otherwise. **δ = {frozen['delta']}**, chosen as the highest
  out-of-fold net Sharpe at base cost among {rs['signal']['prob_deltas']} (development out-of-fold, 2024-03 → 2025-09).
- Position sizing: each signal is a sub-position of 1/{rs['primary_horizon']} of capital held {rs['primary_horizon']} bars; net |position| ≤ 1; no leverage;
  shorting allowed (perpetual).
- Benchmarks: buy-and-hold (long 1×), momentum sign(ret_15), order-flow rule (long if ofi_15_z_1440 > {rs['signal']['flow_rule_z']}, short if < −{rs['signal']['flow_rule_z']}),
  random signals with the ML strategy's activity rate (seeded).

## 8. Costs, slippage, funding (per side, on traded notional)
| Scenario | fee (bps) | slippage (bps) | half spread (bps) | total per side (bps) |
|---|---|---|---|---|
""" + "\n".join(f"| {k} | {c.fee_bps} | {c.slippage_bps} | {c.half_spread_bps} | {c.per_side*1e4:.3f} |" for k, c in cs.items() if k != "gross") + f"""

Sources: Binance USD-M VIP0 fees are 2 bps maker and 5 bps taker, with a 10% discount when paid in BNB. The half spread was measured from `bookTicker` on 3 development
days (`reports/tables/spread_measurement.csv`). Slippage is an assumption, bracketed by the low and high scenarios.
**Base** is the headline scenario. Funding: the realised rate at each settlement is charged to the position held over the settlement
(longs pay a positive rate, shorts receive it).
The break-even cost is (gross P&L + funding) / total turnover, expressed per side.

## 9. Metrics and pre-declared secondary holdout evaluations
- Classification: AUC, log loss vs prior, Brier, accuracy, precision/recall/F1, ECE, Spearman IC, and *tradable AUC* (against the
  VWAP-to-VWAP return actually capturable).
- Trading: total and annualised return, volatility, Sharpe and Sortino (daily, 365-day year), max drawdown, Calmar, win rate,
  profit factor, average trade (bps, net of a full round trip), turnover, exposure, funding, break-even cost.
- Also evaluated on the holdout, with the same frozen procedure: ablation sets A–E × {{logit, LightGBM, RF}} at h = 15; horizons 5/30/60
  (sets A, C, E × logit and LightGBM); a confirmatory HAC test of 1-minute OFI reversal; volatility regimes (tercile cut-offs
  {reg['vol_1440_low_le']:.3e} / {reg['vol_1440_mid_le']:.3e} on vol_1440, estimated on 2023-03..2024-02); monthly stability; SOLUSDT replication
  with identical settings.
"""
    out_md.write_text(md)
    print(md)
    print("sha256", hashlib.sha256(out_md.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
