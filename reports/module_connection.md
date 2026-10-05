# One framework, two problems: how Module A (credit) maps to Module B (market)

Both modules answer the same three questions in different domains:
1. **Can I predict it?**
2. **Does the prediction hold up out of time?**
3. **What happens in the tail when I am wrong?**

| Step | Module A: credit portfolio (LendingClub) | Module B: market signal (Binance BTCUSDT) | What changes and why |
|---|---|---|---|
| Event predicted | Default over the loan's life (binary) | Next-60-minute return > 0 (binary) | Market labels arrive every hour, not after years, so there are about 80k labelled events instead of ~1.3M loans, but each one carries far less signal |
| Probability model | PD = P(default \| borrower features); logistic regression | P(up \| market features); logistic regression | Same model family and interpretation. The difficulty differs: credit AUC ≈ 0.70 is useful, while a market AUC of 0.52 may already be a lot |
| Challenger | LightGBM (AUC 0.708 vs 0.700 in the report) | LightGBM under a pre-registered simplicity rule | Same lesson: a small AUC gain is not enough reason to give up interpretability and stability |
| Out-of-time validation | Train ≤ 2016; validate 2017; test 2018 | Expanding walk-forward, 2019–2023 validation blocks; untouched 2024+ test | Markets drift faster, so the model is refitted every 6 months and labels are purged at boundaries |
| Leakage control | Exclude post-origination fields (recoveries, payments) | Features only from candles closed by *t*; execution 1 min later; perturbation test | Same principle: only information available at decision time |
| Calibration | Predicted PD vs observed default rate by risk decile | Predicted P(up) vs observed up-rate by decile, **plus mean forward return per decile** | In trading, what matters is whether deciles differ in *return* by more than the cost |
| Aggregate backtest | Expected vs actual defaults (z-test) | Kupiec proportion-of-failures test on VaR exceptions | Both compare predicted frequencies with what later happened |
| Stability | Vintage AUC, PSI of score vs 2016 | Per-half-year AUC and Sharpe; **PSI of features and score per test block** | Same tool for the same question: did the population shift, and is that why performance changed? |
| From probability to money | ECL = PD × LGD × EAD | Net P&L = position × return − turnover × cost | The money link is a trading rule with costs instead of an accounting identity |
| Monte Carlo | One-factor Gaussian copula of correlated defaults (ρ assumed) | Stationary block bootstrap of observed daily strategy returns | Default dependence is unobserved, so it has to be modelled; strategy returns are observed, so we resample them and keep fat tails and volatility clustering without assuming a distribution |
| Tail risk | VaR/ES of portfolio credit loss (99%) | 1-day VaR/ES of strategy returns (95/99%): historical, normal and Student-t | The normal-vs-historical comparison quantifies fat tails directly |
| Stress testing | PD odds ×1.5, LGD +5 pp; ρ sensitivity | Costs ×2 and ×3, extra slippage, volatility ×2, signal flips, historical worst days, gap shocks, regimes | The assumption that drives the tail differs: correlation in credit, costs and regime in trading |
| Model risk lesson | Tail risk was very sensitive to ρ (VaR99 $30M → $108M) | Profitability is very sensitive to costs and selection; the Deflated Sharpe corrects for the number of trials | In both, the assumption you are least sure about dominates the answer, so state it and stress it |

**One-sentence summary:** the credit module went from *who will default* to *how much the portfolio could lose under stress*. The market module goes from *can we predict the next hour* to *does that prediction survive costs, regimes and an honest count of how many things we tried*.
