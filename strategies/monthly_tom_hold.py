"""Monthly Turn-of-Month Hold strategy.

Buy ES at close on trading day -2 of each month, sell at close on trading day +3
of next month. This is a ~5 trading day hold around month boundaries.

Costs: $5 round trip commission per contract, $12.50 slippage per side per contract.
ES point value = $50, starting capital $100K.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity

STRATEGY_NAME = "Monthly TOM Hold (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0


def build_tom_entries_exits(dates: pd.DatetimeIndex) -> list[tuple]:
    """Find (entry_date, exit_date) pairs for TOM holds.

    Entry: trading day -2 of each month (2nd to last trading day).
    Exit: trading day +3 of the next month (3rd trading day).
    """
    ym = dates.to_period("M")
    months = sorted(ym.unique())

    pairs = []
    for k in range(len(months) - 1):
        cur_month = months[k]
        next_month = months[k + 1]

        cur_days = dates[ym == cur_month]
        next_days = dates[ym == next_month]

        if len(cur_days) < 2 or len(next_days) < 3:
            continue

        entry_date = cur_days[-2]  # Trading day -2
        exit_date = next_days[2]   # Trading day +3 (0-indexed: index 2 = 3rd day)
        pairs.append((entry_date, exit_date))

    return pairs


def run():
    """Run monthly TOM hold backtest and compare to overnight version."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    close = es["Close"].astype(float)
    entry_exit_pairs = build_tom_entries_exits(close.index)
    print(f"TOM windows identified: {len(entry_exit_pairs)}")

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)

    pair_idx = 0
    in_trade = False
    entry_price = 0.0
    entry_date = None

    for i in range(len(close)):
        date = close.index[i]

        # Check for entry
        if not in_trade and pair_idx < len(entry_exit_pairs):
            target_entry, target_exit = entry_exit_pairs[pair_idx]
            if date == target_entry:
                entry_price = close.iloc[i]
                entry_date = date
                in_trade = True

        # Check for exit
        if in_trade and pair_idx < len(entry_exit_pairs):
            _, target_exit = entry_exit_pairs[pair_idx]
            if date == target_exit:
                exit_price = close.iloc[i]
                pnl_points = exit_price - entry_price
                gross_pnl = pnl_points * ES_POINT_VALUE
                cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
                net_pnl = gross_pnl - cost
                cash += net_pnl
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "hold_days": (date - entry_date).days,
                    "pnl_points": pnl_points,
                    "gross_pnl": gross_pnl,
                    "costs": cost,
                    "pnl": net_pnl,
                })
                in_trade = False
                pair_idx += 1

        if in_trade:
            position.iloc[i] = 1.0
        equity.iloc[i] = cash + (
            (close.iloc[i] - entry_price) * ES_POINT_VALUE if in_trade else 0.0
        )

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)

    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )

    print_summary(stats, STRATEGY_NAME)

    # Compare with note about overnight version
    print(f"Note: The futures_overnight_tom strategy holds overnight only (close->open).")
    print(f"This strategy holds close-to-close across the TOM window (~5 trading days).")
    if len(trade_list) > 0:
        avg_hold = trade_list["hold_days"].mean()
        print(f"Average hold period: {avg_hold:.1f} calendar days")

    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "monthly_tom_hold_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
