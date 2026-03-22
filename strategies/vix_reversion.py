"""VIX Reversion strategy.

Entry: VIX closes above entry_threshold (default 30). Buy SPY at next open.
Exit: VIX closes below exit_threshold (default 20). Sell SPY at next open.
Also tests multiple threshold combinations.

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

STRATEGY_NAME = "VIX Reversion"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def generate_signals(spy: pd.DataFrame, vix: pd.DataFrame,
                     entry_threshold: float = 30.0,
                     exit_threshold: float = 20.0) -> pd.Series:
    """Generate VIX reversion signals.

    Buy SPY when VIX closes above entry_threshold.
    Sell SPY when VIX closes below exit_threshold.
    Signals are shifted by 1 day to execute at next open (no lookahead).
    """
    vix_close = vix["Close"].reindex(spy.index).ffill()
    signals = pd.Series(0, index=spy.index, dtype=int)

    in_trade = False
    for i in range(len(spy)):
        if pd.isna(vix_close.iloc[i]):
            signals.iloc[i] = 1 if in_trade else 0
            continue

        if not in_trade and vix_close.iloc[i] > entry_threshold:
            in_trade = True
        elif in_trade and vix_close.iloc[i] < exit_threshold:
            in_trade = False

        signals.iloc[i] = 1 if in_trade else 0

    # Shift signals by 1 to execute at next open (avoid lookahead)
    signals = signals.shift(1).fillna(0).astype(int)
    return signals


def run_single_threshold(spy, vix, entry_thresh, exit_thresh, verbose=True):
    """Run backtest for a single threshold pair."""
    name = f"{STRATEGY_NAME} (entry={entry_thresh}, exit={exit_thresh})"
    signals = generate_signals(spy, vix, entry_thresh, exit_thresh)

    result = run_single(
        prices=spy["Close"],
        signals=signals,
        commission_per_trade=0.0001,
        slippage_pct=0.0001,
        initial_capital=100_000.0,
    )

    stats = summary(
        trade_pnls=result["trade_pnls"],
        equity_curve=result["equity_curve"],
        position_series=result["position"],
    )

    if verbose:
        print_summary(stats, name)

    return stats, result, name


def run():
    """Run VIX reversion backtest with multiple threshold combinations."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    spy = get_data("SPY", start="1993-01-01")
    vix = get_data("VIX", start="1993-01-01")
    print(f"SPY range: {spy.index[0].date()} to {spy.index[-1].date()} ({len(spy)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    # Primary test: entry 30, exit 20
    stats, result, name = run_single_threshold(spy, vix, 30.0, 20.0)

    plot_equity(result["equity_curve"], name)

    if len(result["trade_list"]) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "vix_reversion_trades.csv")
        result["trade_list"].to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    # Test additional threshold combinations
    print(f"\n{'='*70}")
    print("  Threshold Sensitivity Analysis")
    print(f"{'='*70}")
    print(f"  {'Entry':>6} {'Exit':>6} {'PF':>8} {'Sharpe':>8} {'WinRate':>8} {'Trades':>8} {'MaxDD':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    entry_thresholds = [25, 28, 30, 35]
    exit_thresholds = [18, 20, 22]

    for entry_t in entry_thresholds:
        for exit_t in exit_thresholds:
            if exit_t >= entry_t:
                continue
            s, _, _ = run_single_threshold(spy, vix, entry_t, exit_t, verbose=False)
            print(f"  {entry_t:>6} {exit_t:>6} {s['profit_factor']:>8.3f} "
                  f"{s['sharpe_ratio']:>8.3f} {s['win_rate']:>7.1%} "
                  f"{s['total_trades']:>8} {s['max_drawdown']:>7.1%}")

    print(f"{'='*70}\n")
    return stats, result


if __name__ == "__main__":
    run()
