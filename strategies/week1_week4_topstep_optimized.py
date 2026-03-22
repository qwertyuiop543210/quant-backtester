"""Week 1 & Week 4 — Topstep-optimized.

Same as v1 (1 ES contract, week 1 and week 4 only, Monday open to Friday close)
with ONE rule added: if the position is down more than $3,500 at any daily close
during the week, close immediately. This preserves ~$1,000 of the $4,500 trailing
drawdown buffer for the next trade instead of letting a single week breach the
entire limit.

No other changes — no skip-after-loss, no scaling, no MES. 1 ES contract.

Costs: $5 round trip commission, $12.50 slippage per side. ES point value = $50.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Week 1 & Week 4 Topstep-Optimized (ES, $3.5K Stop)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

WEEKLY_STOP_LOSS = -3_500.0  # Close if down $3,500+ at any daily close

# Topstep eval parameters
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_MONTHLY_FEE = 165.0


def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find Monday-Friday trading week boundaries with daily indices."""
    weeks = []
    i = 0
    while i < len(dates):
        date = dates[i]
        if date.dayofweek == 0:
            monday_idx = i
            daily_indices = [i]
            j = i + 1
            while j < len(dates) and dates[j].dayofweek > 0 and dates[j].dayofweek <= 4:
                daily_indices.append(j)
                j += 1
            friday_idx = daily_indices[-1]
            if friday_idx > monday_idx and dates[friday_idx].dayofweek >= 3:
                wom = get_week_of_month(dates[monday_idx])
                weeks.append({
                    "monday_idx": monday_idx,
                    "friday_idx": friday_idx,
                    "daily_indices": daily_indices,
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
            attempts[-1]["status"] = "INCOMPLETE"
            break

    return attempts


def run():
    """Run week 1 & week 4 Topstep-optimized backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    all_weeks = find_trading_weeks(es.index)
    active = [w for w in all_weeks if w["week_of_month"] in ACTIVE_WEEKS]
    print(f"Total trading weeks: {len(all_weeks)}")
    print(f"Active weeks (1 & 4 only): {len(active)}")
    print(f"Position: 1 ES @ ${ES_POINT_VALUE}/pt")
    print(f"Weekly stop: ${abs(WEEKLY_STOP_LOSS):,.0f} max loss at daily close\n")

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)
    last_equity_idx = 0

    for w in active:
        mi = w["monday_idx"]
        fi = w["friday_idx"]

        entry_price = open_.iloc[mi]

        # Fill equity for gap days before this trade
        for k in range(last_equity_idx, mi):
            equity.iloc[k] = cash

        # Simulate day-by-day through the week
        exit_idx = fi
        exit_price = close.iloc[fi]
        exit_reason = "Friday close"

        for day_idx in w["daily_indices"]:
            day_close = close.iloc[day_idx]
            day_pnl_dollar = (day_close - entry_price) * ES_POINT_VALUE

            # Check weekly stop loss at each daily close (except Friday — that's normal exit)
            if day_pnl_dollar <= WEEKLY_STOP_LOSS and day_idx != fi:
                exit_idx = day_idx
                exit_price = day_close
                exit_reason = f"STOP (${day_pnl_dollar:,.0f})"
                break

            # Mark-to-market
            mtm = (day_close - entry_price) * ES_POINT_VALUE
            equity.iloc[day_idx] = cash + mtm
            position.iloc[day_idx] = 1.0

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        # If stopped out early, fill remaining days flat
        if exit_idx < fi:
            for k in range(exit_idx + 1, fi + 1):
                equity.iloc[k] = cash + net_pnl

        cash += net_pnl
        last_equity_idx = fi + 1

        trades.append({
            "entry_date": w["monday_date"],
            "exit_date": es.index[exit_idx],
            "week_of_month": w["week_of_month"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "exit_reason": exit_reason,
        })

    # Fill remaining equity
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

    # Stop-loss summary
    if len(trade_list) > 0:
        stopped_out = trade_list[trade_list["exit_reason"].str.contains("STOP")]

        print(f"{'='*75}")
        print(f"  Stop-Loss Summary")
        print(f"{'='*75}")
        print(f"  Total trades:     {len(trade_list)}")
        print(f"  Stopped out:      {len(stopped_out)} ({len(stopped_out)/len(trade_list)*100:.1f}%)")
        if len(stopped_out) > 0:
            avg_stop_pnl = stopped_out["pnl"].mean()
            worst_stop = stopped_out["pnl"].min()
            print(f"  Avg stop P&L:     ${avg_stop_pnl:,.0f}")
            print(f"  Worst stop P&L:   ${worst_stop:,.0f}")
            # How much worse would these trades have been without the stop?
            # Find the same weeks in v1 (full Friday close)
            stopped_weeks = stopped_out.index.tolist()
            v1_pnl_for_stopped = []
            for idx in stopped_weeks:
                w = active[idx]
                fi = w["friday_idx"]
                full_pnl = (close.iloc[fi] - trade_list.loc[idx, "entry_price"]) * ES_POINT_VALUE - COST_PER_TRADE
                v1_pnl_for_stopped.append(full_pnl)
            avg_v1_pnl = np.mean(v1_pnl_for_stopped)
            saved = np.sum(v1_pnl_for_stopped) - stopped_out["pnl"].sum()
            print(f"  Avg P&L without stop: ${avg_v1_pnl:,.0f}")
            print(f"  Total saved by stop:  ${-saved:,.0f}" if saved < 0 else f"  Total cost of stop:   ${saved:,.0f}")
        print(f"{'='*75}\n")

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

    # --- Topstep Simulation: Optimized vs v1 ---
    if len(trades) > 0:
        opt_pnl_list = trade_list["pnl"].tolist()
        opt_attempts = simulate_topstep(opt_pnl_list)

        # Run v1 (no stop) for comparison
        v1_pnls = []
        for w in active:
            mi = w["monday_idx"]
            fi = w["friday_idx"]
            entry = open_.iloc[mi]
            exit_ = close.iloc[fi]
            gross = (exit_ - entry) * ES_POINT_VALUE
            v1_pnls.append(gross - COST_PER_TRADE)
        v1_attempts = simulate_topstep(v1_pnls)

        # Print optimized results
        print(f"{'='*80}")
        print(f"  Topstep $150K Eval — Optimized ($3.5K Weekly Stop)")
        print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing DD, "
              f"${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
        print(f"{'='*80}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'Peak Bal':>12} {'Final Bal':>12}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

        opt_passed = 0
        opt_failed = 0
        for a in opt_attempts:
            status_str = a["status"]
            print(f"  {a['attempt']:>3} {status_str:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} ${a['final_balance']:>11,.0f}")
            if status_str == "PASSED":
                opt_passed += 1
            elif status_str == "FAILED":
                opt_failed += 1

        opt_total = opt_passed + opt_failed
        opt_rate = opt_passed / opt_total * 100 if opt_total > 0 else 0

        print(f"{'='*80}")
        print(f"  Optimized: {opt_total} attempts | "
              f"Passed: {opt_passed} | Failed: {opt_failed} | "
              f"Pass rate: {opt_rate:.1f}%")

        # v1 summary
        v1_passed = sum(1 for a in v1_attempts if a["status"] == "PASSED")
        v1_failed = sum(1 for a in v1_attempts if a["status"] == "FAILED")
        v1_total = v1_passed + v1_failed
        v1_rate = v1_passed / v1_total * 100 if v1_total > 0 else 0

        # Months-to-pass analysis for both
        trades_per_month = 2.0

        print(f"\n  {'='*76}")
        print(f"  v1 vs Optimized Comparison")
        print(f"  {'='*76}")
        print(f"  {'Metric':<35} {'v1 (no stop)':>18} {'Optimized':>18}")
        print(f"  {'-'*35} {'-'*18} {'-'*18}")
        print(f"  {'Pass rate':<35} {v1_rate:>17.1f}% {opt_rate:>17.1f}%")
        print(f"  {'Passed / Total':<35} {f'{v1_passed}/{v1_total}':>18} {f'{opt_passed}/{opt_total}':>18}")

        for label, attempts_list, passed_count, total_count in [
            ("v1", v1_attempts, v1_passed, v1_total),
            ("Optimized", opt_attempts, opt_passed, opt_total),
        ]:
            if passed_count > 0:
                pa = [a for a in attempts_list if a["status"] == "PASSED"]
                fa = [a for a in attempts_list if a["status"] == "FAILED"]
                avg_trades_pass = np.mean([a["trades_taken"] for a in pa])
                avg_months_pass = avg_trades_pass / trades_per_month
                avg_attempts = total_count / passed_count
                avg_trades_fail = np.mean([a["trades_taken"] for a in fa]) if fa else 0
                avg_months_fail = avg_trades_fail / trades_per_month
                expected_months = (avg_attempts - 1) * avg_months_fail + avg_months_pass
                expected_cost = expected_months * TOPSTEP_MONTHLY_FEE

                if label == "v1":
                    v1_months = expected_months
                    v1_cost = expected_cost
                else:
                    opt_months = expected_months
                    opt_cost = expected_cost

        if v1_passed > 0 and opt_passed > 0:
            print(f"  {'Expected months to pass':<35} {v1_months:>17.1f} {opt_months:>17.1f}")
            print(f"  {'Expected eval cost':<35} {'${:,.0f}'.format(v1_cost):>18} {'${:,.0f}'.format(opt_cost):>18}")
            delta_rate = opt_rate - v1_rate
            delta_cost = opt_cost - v1_cost
            print(f"  {'-'*35} {'-'*18} {'-'*18}")
            print(f"  {'Delta pass rate':<35} {'':>18} {f'{delta_rate:+.1f}%':>18}")
            print(f"  {'Delta eval cost':<35} {'':>18} {'${:+,.0f}'.format(delta_cost):>18}")
        elif v1_passed > 0:
            print(f"  {'Expected months to pass':<35} {v1_months:>17.1f} {'N/A':>18}")
            print(f"  {'Expected eval cost':<35} {'${:,.0f}'.format(v1_cost):>18} {'N/A':>18}")
        elif opt_passed > 0:
            print(f"  {'Expected months to pass':<35} {'N/A':>18} {opt_months:>17.1f}")
            print(f"  {'Expected eval cost':<35} {'N/A':>18} {'${:,.0f}'.format(opt_cost):>18}")

        print(f"  {'='*76}")
        print(f"{'='*80}\n")

    # Save outputs
    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "week1_week4_topstep_optimized_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
