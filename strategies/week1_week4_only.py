"""Week 1 & Week 4 Only strategy.

Buy ES at Monday open, sell at Friday close, but ONLY during week 1 and week 4
of each month. Skip weeks 2, 3, and 5.

Includes Topstep evaluation simulation: $150K account, $4,500 trailing drawdown
limit, $9,000 profit target.

Costs: $5 round trip commission per contract, $12.50 slippage per side per contract.
ES point value = $50, starting capital $100K, 1 contract.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Week 1 & Week 4 Only (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

# Topstep eval parameters
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0


def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find Monday-Friday trading week boundaries.

    Returns list of dicts with: monday_idx, friday_idx, week_of_month.
    """
    weeks = []
    i = 0
    while i < len(dates):
        date = dates[i]
        if date.dayofweek == 0:
            monday_idx = i
            friday_idx = i
            j = i + 1
            while j < len(dates) and dates[j].dayofweek > 0 and dates[j].dayofweek <= 4:
                friday_idx = j
                j += 1
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


def simulate_topstep(trade_pnls: list[float]) -> list[dict]:
    """Simulate Topstep $150K eval attempts.

    Rules:
    - Start with $150K.
    - Trailing drawdown: $4,500 from equity high-water mark.
    - Profit target: $9,000 cumulative profit.
    - If trailing DD breached, attempt fails -> restart from next trade.
    - If profit target hit, attempt passes.

    Returns list of attempt dicts.
    """
    attempts = []
    i = 0

    while i < len(trade_pnls):
        balance = TOPSTEP_CAPITAL
        high_water = balance
        start_trade = i
        status = "in_progress"

        while i < len(trade_pnls):
            balance += trade_pnls[i]
            high_water = max(high_water, balance)
            trailing_dd = high_water - balance
            profit = balance - TOPSTEP_CAPITAL
            i += 1

            if trailing_dd >= TOPSTEP_TRAILING_DD:
                status = "FAILED"
                break
            if profit >= TOPSTEP_PROFIT_TARGET:
                status = "PASSED"
                break

        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade + 1,
            "end_trade": i,
            "trades_taken": i - start_trade,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - TOPSTEP_CAPITAL,
            "max_trailing_dd": high_water - balance if status == "FAILED" else trailing_dd,
            "status": status,
        })

        if status == "in_progress":
            # Ran out of trades
            attempts[-1]["status"] = "INCOMPLETE"
            break

    return attempts


def run():
    """Run week 1 & week 4 only backtest with Topstep simulation."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    all_weeks = find_trading_weeks(es.index)
    active = [w for w in all_weeks if w["week_of_month"] in ACTIVE_WEEKS]
    print(f"Total trading weeks: {len(all_weeks)}")
    print(f"Active weeks (1 & 4 only): {len(active)}\n")

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)
    last_equity_idx = 0

    for w in active:
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
            "week_of_month": w["week_of_month"],
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

    # Breakdown by week
    if len(trade_list) > 0:
        print(f"{'='*75}")
        print(f"  Breakdown: Week 1 vs Week 4")
        print(f"{'='*75}")
        print(f"  {'Week':>5} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")

        for wom in sorted(ACTIVE_WEEKS):
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

    # --- Topstep Evaluation Simulation ---
    if len(trades) > 0:
        trade_pnl_list = [t["pnl"] for t in trades]
        attempts = simulate_topstep(trade_pnl_list)

        print(f"{'='*80}")
        print(f"  Topstep $150K Eval Simulation")
        print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing drawdown, "
              f"${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
        print(f"{'='*80}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'Peak Bal':>12} {'Final Bal':>12}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

        passed = 0
        failed = 0
        for a in attempts:
            status_str = a["status"]
            print(f"  {a['attempt']:>3} {status_str:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} ${a['final_balance']:>11,.0f}")
            if status_str == "PASSED":
                passed += 1
            elif status_str == "FAILED":
                failed += 1

        total_attempts = passed + failed
        pass_rate = passed / total_attempts * 100 if total_attempts > 0 else 0

        print(f"{'='*80}")
        print(f"  Total attempts: {total_attempts} | "
              f"Passed: {passed} | Failed: {failed} | "
              f"Pass rate: {pass_rate:.1f}%")

        if passed > 0:
            passed_attempts = [a for a in attempts if a["status"] == "PASSED"]
            avg_trades_to_pass = np.mean([a["trades_taken"] for a in passed_attempts])
            print(f"  Avg trades to pass: {avg_trades_to_pass:.0f} "
                  f"(~{avg_trades_to_pass / 2:.0f} weeks)")
        print(f"{'='*80}\n")

    # Save outputs
    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "week1_week4_only_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
