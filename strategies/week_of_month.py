"""Week of Month strategy.

Classify each trading week as week 1-5 of the month.
Buy ES at Monday open, sell at Friday close for each week.
Report results broken down by week number to find consistently profitable weeks.

Costs: $5 round trip commission per contract, $12.50 slippage per side per contract.
ES point value = $50, starting capital $100K.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Week of Month (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0


def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5).

    Week 1 = days 1-7, week 2 = days 8-14, etc.
    """
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find Monday-Friday trading week boundaries.

    Returns list of dicts with: monday_idx, friday_idx, week_of_month.
    Uses the Monday's date to determine week_of_month.
    If Monday is a holiday, uses first available day that week.
    If Friday is a holiday, uses last available day that week.
    """
    weeks = []
    i = 0
    while i < len(dates):
        date = dates[i]
        # Find start of a trading week (Monday = 0)
        if date.dayofweek == 0:
            monday_idx = i
            # Find the Friday (or last day before next Monday)
            friday_idx = i
            j = i + 1
            while j < len(dates) and dates[j].dayofweek > 0 and dates[j].dayofweek <= 4:
                friday_idx = j
                j += 1
            # Only count if we have at least Monday through some weekday
            if friday_idx > monday_idx and dates[friday_idx].dayofweek >= 3:
                wom = get_week_of_month(dates[monday_idx])
                weeks.append({
                    "monday_idx": monday_idx,
                    "friday_idx": friday_idx,
                    "monday_date": dates[monday_idx],
                    "friday_date": dates[friday_idx],
                    "week_of_month": wom,
                })
            i = friday_idx + 1
        else:
            i += 1

    return weeks


def run():
    """Run week-of-month backtest with breakdown."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    weeks = find_trading_weeks(es.index)
    print(f"Trading weeks found: {len(weeks)}\n")

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)

    last_equity_idx = 0

    for w in weeks:
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        wom = w["week_of_month"]

        entry_price = open_.iloc[mi]
        exit_price = close.iloc[fi]

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
        net_pnl = gross_pnl - cost

        # Fill equity for days between trades
        for k in range(last_equity_idx, mi):
            equity.iloc[k] = cash

        # Mark-to-market during hold
        for k in range(mi, fi + 1):
            mtm = (close.iloc[k] - entry_price) * ES_POINT_VALUE
            equity.iloc[k] = cash + mtm
            position.iloc[k] = 1.0

        cash += net_pnl
        last_equity_idx = fi + 1

        trades.append({
            "entry_date": w["monday_date"],
            "exit_date": w["friday_date"],
            "week_of_month": wom,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost,
            "pnl": net_pnl,
        })

    # Fill remaining
    for k in range(last_equity_idx, len(close)):
        equity.iloc[k] = cash

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)

    # Overall stats
    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )
    print_summary(stats, f"{STRATEGY_NAME} — All Weeks")

    # Breakdown by week of month
    print(f"{'='*75}")
    print(f"  Week-of-Month Breakdown")
    print(f"{'='*75}")
    print(f"  {'Week':>5} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")

    for wom in range(1, 6):
        subset = trade_list[trade_list["week_of_month"] == wom]
        if len(subset) == 0:
            continue
        sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
        pf = profit_factor(sub_pnls)
        wr = win_rate(sub_pnls)
        avg = sub_pnls.mean()
        total = sub_pnls.sum()
        n = len(sub_pnls)
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        print(f"  {wom:>5} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}")

    print(f"{'='*75}\n")

    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "week_of_month_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
