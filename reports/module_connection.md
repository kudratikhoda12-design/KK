# One framework, two problems: how Module A (credit) maps to Module B (market)

Both modules answer the same three questions in different domains:
1. **Can I predict it?**
2. **Does the prediction hold up out of time?**
3. **What happens in the tail when I am wrong?**

| Step | Module A: credit portfolio (LendingClub) | Module B: market signal (Binance BTCUSDT) | What changes and why |
|---|---|---|---|
| Event predicted | Default over the loan's life (binary) | Next-60-minute return > 0 (binary) | Market labels arrive every hour, not after years, so there are about 80k labelled events instead of ~1.3M loans, but each one carries far less signal |
| Probability model | PD = P(default \| borrower features); logistic regression, 2017 AUC 0.70 | P(up \| market features); test AUC **0.545** | Same model family and interpretation. The difficulty differs: in markets, 0.545 is statistically strong but economically too small |
| Challenger | LightGBM (AUC 0.708 vs 0.700 in the original report; the code is not in the notebook) | LightGBM passed the pre-registered simplicity rule exactly at the threshold (7/10 folds). On test: AUC 0.545 vs 0.540 for LR, but lost **more** money (net Sharpe −5.81 vs −4.35) | Same lesson: a small AUC gain is not enough reason to give up interpretability and stability |
| Out-of-time validation | Train ≤ 2016; validate 2017; test 2018 | Expanding walk-forward, 2019–2023 validation blocks; untouched 2024+ test | Markets drift faster, so the model is refitted every 6 months and labels are purged at boundaries |
| Leakage control | Exclude post-origination fields (recoveries, payments) | Features only from candles closed by *t*; execution 1 min later; perturbation test | Same principle: only information available at decision time |
| Calibration | Predicted PD vs observed default rate by risk decile: **under**-predicted in all deciles | Predicted P(up) vs observed up-rate: **over**-confident (top decile 60.0% predicted vs 56.4% observed); decile mean returns only −1.3 to +1.9 bp | Same tool, opposite error. In trading, the return per decile against the 14 bp cost decides the outcome |
| Aggregate backtest | Expected vs actual defaults (z-test) | Kupiec proportion-of-failures test on VaR exceptions | Both compare predicted frequencies with what later happened |
| Stability | Vintage AUC 0.731 → 0.694; score PSI ≤ 0.027 | Test AUC 0.567 → 0.532 by half-year; volatility-feature PSI up to 3.5 | Credit: little population shift. Market: large input drift, which explains the decay |
| From probability to money | ECL = PD × LGD × EAD | Net P&L = position × return − turnover × cost | The money link is a trading rule with costs instead of an accounting identity |
| Monte Carlo | One-factor Gaussian copula of correlated defaults (ρ assumed) | Stationary block bootstrap of observed daily strategy returns | Default dependence is unobserved, so it has to be modelled; strategy returns are observed, so we resample them and keep fat tails and volatility clustering without assuming a distribution |
| Tail risk | VaR/ES of portfolio credit loss (99%) | 1-day VaR/ES of strategy returns (95/99%): historical, normal and Student-t | The normal-vs-historical comparison quantifies fat tails directly |
| Stress testing | PD odds ×1.5, LGD +5 pp; ρ sensitivity | Costs ×2 and ×3, extra slippage, volatility ×2, signal flips, historical worst days, gap shocks, regimes | The assumption that drives the tail differs: correlation in credit, costs and regime in trading |
| Model risk lesson | Tail risk was very sensitive to ρ (VaR99 $30M → $108M) | Profitability is decided by costs: gross Sharpe +0.16 becomes −5.8 net; break-even is 0.19 bp vs 7 bp | In both, the assumption you are least sure about dominates the answer, so state it and stress it |

**One-sentence summary:** the credit module went from *who will default* to *how much the portfolio could lose under stress*. The market module goes from *can we predict the next hour* to *does that prediction survive costs, regimes and an honest count of how many things we tried*.
