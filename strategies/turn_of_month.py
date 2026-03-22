"""Turn of Month strategy.

Buy on trading day -2 relative to month end, sell on trading day +3 of next month.
Trading day = market-open days only (based on available price data).
Test on SPY daily data, max available history.

Default costs: 0.01% commission per side (ETF), 0.01% slippage per side.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.backtester import run_single
from core.metrics import summary, print_summary
from core.plotting import plot_equity

STRATEGY_NAME = "Turn of Month (SPY)"


def generate_signals(prices: pd.DataFrame) -> pd.Series:
    """Generate turn-of-month signals.

    Buy on trading day -2 before month end, sell on trading day +3 of next month.
    Returns signal series: 1 = long, 0 = flat.
    """
    close = prices["Close"]
    dates = close.index

    # Group trading days by year-month
    signals = pd.Series(0, index=dates, dtype=int)

    # Get trading days per month
    ym = dates.to_period("M")
    months = ym.unique()

    for m in months:
        mask = ym == m
        month_days = dates[mask]
        n = len(month_days)
        if n < 3:
            continue
        # Trading day -2 relative to month end = index n-2 (0-based)
        # Mark day n-2 and n-1 (last two trading days of month) as buy
        buy_start = max(0, n - 2)
        for j in range(buy_start, n):
            signals.loc[month_days[j]] = 1

    # Also need to mark first 3 trading days of each month as long
    for m in months:
        mask = ym == m
        month_days = dates[mask]
        n = len(month_days)
        # Trading day +1, +2, +3 of month (first 3 trading days)
        sell_end = min(3, n)
        for j in range(sell_end):
            signals.loc[month_days[j]] = 1

    return signals


def run():
    """Run the turn-of-month backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Downloading SPY data (max history)...")

    spy = get_data("SPY", start="1993-01-01")
    print(f"Data range: {spy.index[0].date()} to {spy.index[-1].date()} "
          f"({len(spy)} trading days)")

    signals = generate_signals(spy)

    # ETF costs: 0.01% commission, 0.01% slippage per side
    result = run_single(
        prices=spy["Close"],
        signals=signals,
        commission_per_trade=0.0001,  # 0.01%
        slippage_pct=0.0001,          # 0.01%
        initial_capital=100_000.0,
    )

    stats = summary(
        trade_pnls=result["trade_pnls"],
        equity_curve=result["equity_curve"],
        position_series=result["position"],
    )

    print_summary(stats, STRATEGY_NAME)

    # Save equity curve plot
    plot_equity(result["equity_curve"], STRATEGY_NAME)

    # Save trade list
    if len(result["trade_list"]) > 0:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "turn_of_month_trades.csv")
        result["trade_list"].to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats, result


if __name__ == "__main__":
    run()
