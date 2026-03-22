"""OpEx Week strategy.

Identify quarterly options expiration weeks (third Friday of March, June, September,
December). Buy ES at Monday open of OpEx week, sell at Friday close.

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

STRATEGY_NAME = "Quarterly OpEx Week (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0

OPEX_MONTHS = {3, 6, 9, 12}  # Quarterly expiration months


def find_third_friday(year: int, month: int) -> pd.Timestamp:
    """Find the third Friday of a given month."""
    # First day of month
    first = pd.Timestamp(year=year, month=month, day=1)
    # Find first Friday (weekday 4)
    days_until_friday = (4 - first.dayofweek) % 7
    first_friday = first + pd.Timedelta(days=days_until_friday)
    # Third Friday = first Friday + 14 days
    return first_friday + pd.Timedelta(days=14)


def find_opex_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find OpEx week Monday-Friday pairs in the trading calendar.

    Returns list of dicts with monday_idx, friday_idx, opex_date, quarter.
    """
    years = sorted(set(dates.year))
    date_set = set(dates)
    weeks = []

    for year in years:
        for month in OPEX_MONTHS:
            opex_friday = find_third_friday(year, month)

            # Find the Monday of OpEx week
            opex_monday = opex_friday - pd.Timedelta(days=4)

            # Find actual trading days closest to these
            # Monday: find first trading day on or after opex_monday
            monday_candidates = dates[(dates >= opex_monday) &
                                       (dates <= opex_monday + pd.Timedelta(days=2))]
            # Friday: find last trading day on or before opex_friday
            friday_candidates = dates[(dates <= opex_friday) &
                                       (dates >= opex_friday - pd.Timedelta(days=2))]

            if len(monday_candidates) == 0 or len(friday_candidates) == 0:
                continue

            monday = monday_candidates[0]
            friday = friday_candidates[-1]

            monday_idx = dates.get_loc(monday)
            friday_idx = dates.get_loc(friday)

            if friday_idx <= monday_idx:
                continue

            quarter = f"Q{(month - 1) // 3 + 1}"
            weeks.append({
                "monday_idx": monday_idx,
                "friday_idx": friday_idx,
                "monday_date": monday,
                "friday_date": friday,
                "opex_date": opex_friday,
                "quarter": quarter,
                "year": year,
            })

    return weeks


def run():
    """Run quarterly OpEx week backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    opex_weeks = find_opex_weeks(es.index)
    print(f"Quarterly OpEx weeks found: {len(opex_weeks)}\n")

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)

    last_equity_idx = 0

    for w in opex_weeks:
        mi = w["monday_idx"]
        fi = w["friday_idx"]

        entry_price = open_.iloc[mi]
        exit_price = close.iloc[fi]

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
        net_pnl = gross_pnl - cost

        # Fill equity for gap days
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
            "opex_date": w["opex_date"],
            "quarter": w["quarter"],
            "year": w["year"],
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
    print_summary(stats, STRATEGY_NAME)

    # Breakdown by quarter
    if len(trade_list) > 0:
        print(f"{'='*70}")
        print(f"  Quarterly Breakdown")
        print(f"{'='*70}")
        print(f"  {'Quarter':>8} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            subset = trade_list[trade_list["quarter"] == q]
            if len(subset) == 0:
                continue
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            total = sub_pnls.sum()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {q:>8} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}")

        print(f"{'='*70}")

        # By decade
        print(f"\n{'='*70}")
        print(f"  Decade Breakdown")
        print(f"{'='*70}")
        print(f"  {'Period':>12} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

        for decade_start in range(2000, 2030, 10):
            decade_end = decade_start + 9
            subset = trade_list[(trade_list["year"] >= decade_start) &
                                (trade_list["year"] <= decade_end)]
            if len(subset) == 0:
                continue
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            n = len(sub_pnls)
            label = f"{decade_start}-{decade_end}"
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {label:>12} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f}")

        print(f"{'='*70}\n")

    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "opex_week_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
