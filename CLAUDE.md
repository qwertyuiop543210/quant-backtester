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
