"""Week 1 & Week 4 v2 — Risk-managed version.

Same core strategy as week1_week4_only.py (buy Monday open, sell Friday close,
weeks 1 & 4 only) with these improvements:

1. POSITION SIZING: 5 MES contracts ($5/point each = $25/point total) instead of
   1 ES ($50/point). Halves per-trade volatility.
2. WEEKLY STOP LOSS: If position is down $2,000+ at any daily close, exit immediately.
3. SKIP AFTER LOSS: W1 loss -> skip W4 same month. W4 loss -> skip W1 next month.
4. SCALE OUT: If up $1,500+ by Wednesday close, close 2 of 5 contracts. Let 3 ride.

Topstep simulation: $150K account, $4,500 trailing DD, $9,000 profit target.
Includes months-to-pass and expected eval cost analysis.

Costs: $5 RT commission per contract × 5 = $25 base. Slippage $2.50/side/contract.
MES point value = $5.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Week 1 & Week 4 v2 (MES, Risk-Managed)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# MES position sizing: 5 contracts at $5/point = $25/point total
MES_POINT_VALUE = 5.0
NUM_CONTRACTS = 5
TOTAL_POINT_VALUE = MES_POINT_VALUE * NUM_CONTRACTS  # $25/point

# Costs per trade (all 5 contracts)
COMMISSION_PER_CONTRACT_RT = 5.0
SLIPPAGE_PER_SIDE_PER_CONTRACT = 2.50
TOTAL_COMMISSION = COMMISSION_PER_CONTRACT_RT * NUM_CONTRACTS  # $25
TOTAL_SLIPPAGE = SLIPPAGE_PER_SIDE_PER_CONTRACT * NUM_CONTRACTS * 2  # $25

INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

# Risk management
WEEKLY_STOP_LOSS = -2_000.0  # Close if down $2,000+ at any daily close
SCALE_OUT_THRESHOLD = 1_500.0  # Lock in partial profit if up $1,500+ by Wednesday
SCALE_OUT_CONTRACTS = 2  # Close 2 of 5 contracts at scale-out
REMAINING_CONTRACTS = NUM_CONTRACTS - SCALE_OUT_CONTRACTS  # 3 ride to Friday

# Topstep eval parameters
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_MONTHLY_FEE = 165.0


def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find Monday-Friday trading week boundaries with daily indices.

    Returns list of dicts with: monday_idx, friday_idx, daily_indices (all
    trading days mon-fri), week_of_month, month, year.
    """
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
                    "month": dates[monday_idx].month,
                    "year": dates[monday_idx].year,
                })
            i = friday_idx + 1
        else:
            i += 1
    return weeks


def simulate_topstep(trade_pnls: list[float], trades_per_week: float = 1.0) -> list[dict]:
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
            attempts[-1]["status"] = "INCOMPLETE"
            break

    return attempts


def run():
    """Run week 1 & week 4 v2 backtest with risk management and Topstep simulation."""
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
    print(f"Position: {NUM_CONTRACTS} MES @ ${MES_POINT_VALUE}/pt = "
          f"${TOTAL_POINT_VALUE}/pt total")
    print(f"Risk mgmt: ${abs(WEEKLY_STOP_LOSS):,.0f} weekly stop, "
          f"${SCALE_OUT_THRESHOLD:,.0f} scale-out trigger\n")

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)
    last_equity_idx = 0

    # Skip-after-loss tracking: (year, month, week_of_month) -> skip
    skip_set = set()

    for w_idx, w in enumerate(active):
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        wom = w["week_of_month"]
        year = w["year"]
        month = w["month"]

        # Check if this week should be skipped
        if (year, month, wom) in skip_set:
            # Fill equity flat for skipped weeks
            for k in range(last_equity_idx, fi + 1):
                equity.iloc[k] = cash
            last_equity_idx = fi + 1
            trades.append({
                "entry_date": w["monday_date"],
                "exit_date": w["friday_date"],
                "week_of_month": wom,
                "entry_price": 0,
                "exit_price": 0,
                "pnl_points": 0,
                "gross_pnl": 0,
                "costs": 0,
                "pnl": 0,
                "contracts_held": 0,
                "exit_reason": "SKIPPED (loss-skip rule)",
                "scaled_out": False,
            })
            continue

        entry_price = open_.iloc[mi]

        # Fill equity for gap days before this trade
        for k in range(last_equity_idx, mi):
            equity.iloc[k] = cash

        # Simulate day-by-day through the week
        exit_idx = fi
        exit_price = close.iloc[fi]
        exit_reason = "Friday close"
        scaled_out = False
        scale_out_pnl = 0.0
        contracts_active = NUM_CONTRACTS

        for day_idx in w["daily_indices"]:
            day_close = close.iloc[day_idx]
            day_pnl_points = day_close - entry_price
            day_pnl_dollar = day_pnl_points * MES_POINT_VALUE * contracts_active + scale_out_pnl

            # Check weekly stop loss at each daily close
            if day_pnl_dollar <= WEEKLY_STOP_LOSS and day_idx != fi:
                exit_idx = day_idx
                exit_price = day_close
                exit_reason = f"STOP LOSS (${day_pnl_dollar:,.0f})"
                break

            # Check scale-out: Wednesday (dayofweek == 2) close
            if (not scaled_out
                    and es.index[day_idx].dayofweek == 2
                    and day_pnl_dollar >= SCALE_OUT_THRESHOLD):
                # Close 2 of 5 contracts at current price
                scale_out_points = day_close - entry_price
                scale_out_pnl = scale_out_points * MES_POINT_VALUE * SCALE_OUT_CONTRACTS
                contracts_active = REMAINING_CONTRACTS
                scaled_out = True

            # Mark-to-market
            mtm = day_pnl_points * MES_POINT_VALUE * contracts_active + scale_out_pnl
            equity.iloc[day_idx] = cash + mtm
            position.iloc[day_idx] = 1.0

        # If we exited early (stop loss), fill remaining days flat
        if exit_idx < fi:
            # Calculate final PnL at stop
            final_points = exit_price - entry_price
            if scaled_out:
                gross_pnl = (final_points * MES_POINT_VALUE * contracts_active
                             + scale_out_pnl)
            else:
                gross_pnl = final_points * TOTAL_POINT_VALUE
            # Fill remaining days after exit
            for k in range(exit_idx + 1, fi + 1):
                equity.iloc[k] = cash + gross_pnl - TOTAL_COMMISSION - TOTAL_SLIPPAGE
        else:
            # Normal Friday exit
            final_points = exit_price - entry_price
            if scaled_out:
                gross_pnl = (final_points * MES_POINT_VALUE * contracts_active
                             + scale_out_pnl)
            else:
                gross_pnl = final_points * TOTAL_POINT_VALUE

        # Costs: full commission/slippage for all contracts traded
        # If scaled out, we have extra commissions for the partial exit
        if scaled_out:
            # Scale-out leg: 2 contracts exit mid-week + remaining 3 exit Friday
            # Entry: 5 contracts one-way slippage + commission
            # Exit scale-out: 2 contracts one-way slippage + partial commission
            # Exit final: 3 contracts one-way slippage + partial commission
            cost = TOTAL_COMMISSION + TOTAL_SLIPPAGE  # keep it simple: same total
        else:
            cost = TOTAL_COMMISSION + TOTAL_SLIPPAGE

        net_pnl = gross_pnl - cost
        cash += net_pnl
        last_equity_idx = fi + 1

        trades.append({
            "entry_date": w["monday_date"],
            "exit_date": es.index[exit_idx] if exit_idx != fi else w["friday_date"],
            "week_of_month": wom,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": final_points,
            "gross_pnl": gross_pnl,
            "costs": cost,
            "pnl": net_pnl,
            "contracts_held": NUM_CONTRACTS,
            "exit_reason": exit_reason,
            "scaled_out": scaled_out,
        })

        # Skip-after-loss logic
        if net_pnl < 0:
            if wom == 1:
                # W1 lost -> skip W4 of same month
                skip_set.add((year, month, 4))
            elif wom == 4:
                # W4 lost -> skip W1 of next month
                if month == 12:
                    skip_set.add((year + 1, 1, 1))
                else:
                    skip_set.add((year, month + 1, 1))

    # Fill remaining equity
    for k in range(last_equity_idx, len(close)):
        equity.iloc[k] = cash

    trade_list = pd.DataFrame(trades)
    # Separate actual trades from skipped for stats
    actual_trades = trade_list[trade_list["contracts_held"] > 0].copy()
    skipped_trades = trade_list[trade_list["contracts_held"] == 0]
    trade_pnls = pd.Series(actual_trades["pnl"].values, dtype=float) if len(actual_trades) > 0 else pd.Series(dtype=float)

    # Overall stats
    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )
    print_summary(stats, STRATEGY_NAME)

    # Risk management summary
    if len(actual_trades) > 0:
        stopped_out = actual_trades[actual_trades["exit_reason"].str.contains("STOP")]
        scaled = actual_trades[actual_trades["scaled_out"]]

        print(f"{'='*75}")
        print(f"  Risk Management Summary")
        print(f"{'='*75}")
        print(f"  Trades taken:        {len(actual_trades)}")
        print(f"  Trades skipped:      {len(skipped_trades)} (skip-after-loss rule)")
        print(f"  Stopped out:         {len(stopped_out)} ({len(stopped_out)/len(actual_trades)*100:.1f}%)")
        print(f"  Scaled out:          {len(scaled)} ({len(scaled)/len(actual_trades)*100:.1f}%)")
        if len(stopped_out) > 0:
            avg_stop_pnl = stopped_out["pnl"].mean()
            print(f"  Avg stop-loss P&L:   ${avg_stop_pnl:,.0f}")
        if len(scaled) > 0:
            avg_scale_pnl = scaled["pnl"].mean()
            print(f"  Avg scaled-out P&L:  ${avg_scale_pnl:,.0f}")
        print(f"{'='*75}\n")

    # Breakdown by week
    if len(actual_trades) > 0:
        print(f"{'='*75}")
        print(f"  Breakdown: Week 1 vs Week 4")
        print(f"{'='*75}")
        print(f"  {'Week':>5} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")

        for wom in sorted(ACTIVE_WEEKS):
            subset = actual_trades[actual_trades["week_of_month"] == wom]
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
    if len(actual_trades) > 0:
        trade_pnl_list = actual_trades["pnl"].tolist()
        attempts = simulate_topstep(trade_pnl_list)

        print(f"{'='*80}")
        print(f"  Topstep $150K Eval Simulation (v2 — Risk-Managed)")
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

        # Calculate months-to-pass and eval cost
        # ~2 active weeks per month (W1 + W4), so ~2 trades per month
        trades_per_month = 2.0  # approximately

        if passed > 0:
            passed_attempts = [a for a in attempts if a["status"] == "PASSED"]
            failed_attempts = [a for a in attempts if a["status"] == "FAILED"]

            avg_trades_to_pass = np.mean([a["trades_taken"] for a in passed_attempts])
            avg_months_to_pass = avg_trades_to_pass / trades_per_month

            # Expected months including failed attempts before passing
            # On average: how many attempts before a pass?
            if total_attempts > 0:
                avg_attempts_to_pass = total_attempts / passed
                avg_trades_per_failed = np.mean([a["trades_taken"] for a in failed_attempts]) if failed_attempts else 0
                avg_months_per_failed = avg_trades_per_failed / trades_per_month

                # Expected total months = failed attempts × months each + passing attempt months
                expected_fails_before_pass = avg_attempts_to_pass - 1
                expected_total_months = (expected_fails_before_pass * avg_months_per_failed
                                         + avg_months_to_pass)
                expected_eval_cost = expected_total_months * TOPSTEP_MONTHLY_FEE

            print(f"\n  {'='*76}")
            print(f"  Months-to-Pass & Eval Cost Analysis")
            print(f"  {'='*76}")
            print(f"  Avg trades to pass (successful):  {avg_trades_to_pass:.1f} trades")
            print(f"  Avg months to pass (successful):  {avg_months_to_pass:.1f} months")
            print(f"  Avg attempts before passing:      {avg_attempts_to_pass:.1f}")
            if failed_attempts:
                print(f"  Avg trades per failed attempt:    {avg_trades_per_failed:.1f}")
                print(f"  Avg months per failed attempt:    {avg_months_per_failed:.1f}")
            print(f"  Expected total months to pass:    {expected_total_months:.1f} months")
            print(f"  Expected eval cost @ ${TOPSTEP_MONTHLY_FEE}/mo: "
                  f"${expected_eval_cost:,.0f}")
            print(f"  {'='*76}")

        print(f"{'='*80}\n")

    # --- v1 vs v2 comparison ---
    # Run v1 logic inline for comparison
    print(f"{'='*80}")
    print(f"  v1 vs v2 Comparison")
    print(f"{'='*80}")
    _run_v1_comparison(all_weeks, open_, close, es.index)
    print(f"  v2 pass rate: {pass_rate:.1f}% ({passed}/{total_attempts})")
    print(f"{'='*80}\n")

    # Save outputs
    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "week1_week4_v2_trades.csv")
        # Don't save internal daily_indices column
        save_cols = [c for c in trade_list.columns if c != "daily_indices"]
        trade_list[save_cols].to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


def _run_v1_comparison(all_weeks, open_, close, dates):
    """Run simplified v1 logic to get Topstep pass rate for comparison."""
    ES_POINT_VALUE = 50.0
    V1_COMMISSION_RT = 5.0
    V1_SLIPPAGE = 12.50

    active = [w for w in all_weeks if w["week_of_month"] in ACTIVE_WEEKS]
    v1_pnls = []
    for w in active:
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        entry = open_.iloc[mi]
        exit_ = close.iloc[fi]
        pnl_points = exit_ - entry
        gross = pnl_points * ES_POINT_VALUE
        cost = V1_COMMISSION_RT + 2 * V1_SLIPPAGE
        v1_pnls.append(gross - cost)

    attempts = simulate_topstep(v1_pnls)
    v1_passed = sum(1 for a in attempts if a["status"] == "PASSED")
    v1_failed = sum(1 for a in attempts if a["status"] == "FAILED")
    v1_total = v1_passed + v1_failed
    v1_rate = v1_passed / v1_total * 100 if v1_total > 0 else 0

    print(f"  v1 pass rate: {v1_rate:.1f}% ({v1_passed}/{v1_total})")


if __name__ == "__main__":
    run()
