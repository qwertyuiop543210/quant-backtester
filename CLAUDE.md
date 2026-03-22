# CLAUDE.md — Quant Backtester Project
## Project Purpose
Systematic backtesting framework to identify durable trading edges across multiple strategy types and asset classes. All strategies must survive realistic transaction costs and demonstrate statistical significance across multi-year datasets before being considered viable.
## Core Principles
### Research Integrity
- NEVER test a strategy on the same data used to develop it. Split data into in-sample and out-of-sample periods.
- A strategy that only works in recent data is a regime artifact, not an edge. Test across the longest available history.
- Minimum 100 trades for any statistical conclusion. Below 50 trades, results are noise.
- Report profit factor, Sharpe ratio, max drawdown, win rate, and total trades for every test.
### Transaction Costs
- Always include realistic commission and slippage.
- Default commissions: $5 round trip per futures contract, $0.01/share for equities, 0.01% for ETFs.
- Default slippage: 1 tick per side on futures, 0.01% on equities/ETFs.
- Never run a backtest with zero costs.
### Avoiding Overfitting
- Fewer parameters is better. Every adjustable parameter is a degree of freedom that can overfit.
- If a strategy requires more than 3-4 parameters to be profitable, it's likely overfit.
- Walk-forward validation: optimize on year 1-2, test on year 3. Roll forward.
- Never optimize parameters to maximize backtest P&L. Optimize for robustness.
### Statistical Significance
- Profit factor between 0.95 and 1.05 is indistinguishable from random.
- Minimum profit factor of 1.2 after costs to consider a strategy viable.
- Calculate bootstrap confidence interval for profit factor.
## Code Standards
### File Structure
```
quant-backtester/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── core/
│   ├── backtester.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── plotting.py
├── strategies/
│   ├── turn_of_month.py
│   ├── vix_reversion.py
│   ├── nq_es_pairs.py
│   ├── overnight_returns.py
│   └── gold_silver_ratio.py
├── results/
└── data/
```
### Backtester Requirements
- Core backtester must be instrument-agnostic. Strategies produce signal series, backtester simulates execution.
- Support both single-instrument and multi-instrument (pairs) strategies.
- Track per-trade P&L, cumulative P&L, max drawdown from peak, and time in market.
### Output Requirements
- Every strategy run must produce: console output with profit factor, Sharpe ratio, win rate, total trades, max drawdown, annualized return, time in market %.
- PNG equity curve saved to results/.
- Trade list CSV saved to results/.
- Print VERDICT: "TRADEABLE" (PF > 1.2, trades > 100) or "NO EDGE" with reason.
## Strategy-Specific Notes
### Turn of Month
- Buy on trading day -2 relative to month end, sell on trading day +3 of next month.
- Trading day means market-open days only.
- Test on SPY daily data, max available history.
### VIX Reversion
- Entry: VIX closes above 30. Buy SPY at next open.
- Exit: VIX closes below 20. Sell SPY at next open.
- Also test thresholds: entry 25/28/30/35, exit 18/20/22.
### NQ/ES Pairs
- Calculate NQ/ES price ratio daily.
- Z-score with 50-day rolling lookback.
- Entry at Z > 2.0 or Z < -2.0. Exit at Z returning to 0.5 / -0.5.
- Track P&L on BOTH legs simultaneously.
- Position sizing: NQ point = $20, ES point = $50. Dollar-neutral exposure.
### Overnight Returns
- Compare: (A) buy SPY at close, sell next open vs (B) buy at open, sell at close.
- Measurement study showing which session captures returns.
### Gold/Silver Ratio
- Use GLD/SLV ratio. Z-score 50-day lookback.
- Same entry/exit logic as NQ/ES pairs. Dollar-neutral exposure.
## What NOT To Do
- Do not add complexity to make a backtest look better.
- Do not discard outlier trades.
- Do not test on data shorter than 5 years.
- Do not use future data in any calculation.
- Do not run parameter optimization without out-of-sample validation.
## Archiving Failed Strategies
When a strategy is backtested and rejected:
1. Add an entry to the "Previously Tested Strategies" section below with: strategy name, what was tested, the key stats, and a one-line VERDICT explaining why it was rejected.
2. Remove all test code for the rejected strategy from the repo. Keep the repo clean — only active and in-development strategy code should exist in the codebase.
3. Never re-implement a rejected strategy unless the rejection reason has fundamentally changed (e.g., new tooling removes a prior limitation, or new data contradicts prior results).
## Previously Tested Strategies (Archived — Do Not Re-Implement)
The following strategies were tested and rejected during development. They are documented here
for context so we don't repeat failed experiments. All test code has been removed from the repo.
### 1. ORB (Opening Range Breakout) on NQ — v4.1 through v9.0
- Tested in Pine Script on 15-min and 10-min NQ1! charts
- v7.0 was best result: +$39,565, PF 1.80, 11.86% max DD over 7 months (day-of-week filter: Mon/Wed/Fri longs, Wed-only shorts, 150-min time exit)
- Extended to 26 years: went negative (PF 0.917). Edge is regime-dependent, not durable.
- v9.0 attempted regime-adaptive gating via rolling trade outcomes — insufficient trade frequency to work
- VERDICT: Rejected. Not robust across regimes.
### 2. VWAP Mean Reversion on NQ
- Tested in Pine Script over 16 years
- Result: +$10,215 total (~$0 annualized), 28% max drawdown
- VERDICT: Rejected. No meaningful edge.
### 3. NQ/ES Pairs Trading (Z-Score Reversion)
- Research indicator confirmed 80% Z-score reversion rate on 434 signals
- Single-leg Pine Script strategy lost money over 26 years — NQ directional drift overwhelms the spread
- TradingView cannot simulate two-legged trades, so proper backtest was impossible
- VERDICT: Rejected due to tooling limitation + directional drift problem.
### 4. TSMOM Short-Only Sleeve (ES)
- 12-month lookback, short when trailing return negative, monthly rebalance. 3 MES contracts ($15/point), 2x ATR stop.
- 68 trades over 25 years. PF 0.602, lost $21,408 total.
- Only 1/3 bear regimes profitable (2008 GFC only). Lost money in dot-com and 2022-23.
- All 6 sensitivity variants (6m/9m/12m lookback, with/without ATR stop) also lost money.
- VERDICT: Rejected. Equity upward drift overwhelms short-side momentum signal.
## Active Strategy
- **Week 1 & Week 4 ES Calendar Strategy with VIX Filter** — validated 2013-2026, PF 2.650, 63.9% Topstep pass rate
- This is the core strategy. All new work should complement it, not replace it.
## In Development
- **Week 2/3 Mean Reversion Filler** — intended to trade during off-weeks when the calendar strategy is flat. Testing in progress.
