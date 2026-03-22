"""ES Week 1 & Week 4 with VIX overlay.

Two independent signals on ES, combined into one account:

Signal 1 (CALENDAR): Buy 1 ES at Monday open, sell at Friday close, only
during week 1 and week 4 of each month. Same as v1.

Signal 2 (VIX): When VIX closes above 25, buy 1 ES at next open. Hold for
10 trading days OR until VIX drops below 20, whichever comes first. Hard
stop at 40 points below entry ($2,000 max loss per VIX trade). Fires
independently of the calendar signal.

Position limits: Max 2 ES contracts at any time (1 calendar + 1 VIX).
ES costs: $5 RT commission, $12.50 slippage per side. Starting capital $100K.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "ES Week 1 & Week 4 + VIX Overlay"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

# VIX signal parameters
VIX_ENTRY_THRESHOLD = 25.0
VIX_EXIT_THRESHOLD = 20.0
VIX_MAX_HOLD_DAYS = 10
VIX_STOP_POINTS = 40.0  # 40 ES points = $2,000

# Topstep eval parameters
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_MONTHLY_FEE = 165.0

ES_V1_PASS_RATE = 41.6


def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find Monday-Friday trading week boundaries."""
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


def run_calendar_signal(es: pd.DataFrame) -> dict:
    """Run the calendar signal: week 1 & week 4, Monday open to Friday close."""
    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    weeks = find_trading_weeks(es.index)
    active = [w for w in weeks if w["week_of_month"] in ACTIVE_WEEKS]

    trades = []
    # Daily P&L series for this signal
    daily_pnl = pd.Series(0.0, index=es.index, dtype=float)

    for w in active:
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        entry_price = open_.iloc[mi]
        exit_price = close.iloc[fi]
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        # Distribute daily mark-to-market
        prev_close = entry_price
        for k in range(mi, fi + 1):
            day_close = close.iloc[k]
            if k == mi:
                day_pnl = (day_close - entry_price) * ES_POINT_VALUE
            else:
                day_pnl = (day_close - prev_close) * ES_POINT_VALUE
            daily_pnl.iloc[k] += day_pnl
            prev_close = day_close

        # Subtract costs on exit day
        daily_pnl.iloc[fi] -= COST_PER_TRADE

        trades.append({
            "signal": "CALENDAR",
            "entry_date": w["monday_date"],
            "exit_date": w["friday_date"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "exit_reason": "Friday close",
        })

    return {"trades": trades, "daily_pnl": daily_pnl}


def run_vix_signal(es: pd.DataFrame, vix: pd.DataFrame) -> dict:
    """Run the VIX signal: buy ES when VIX > 25, exit conditions apply."""
    # Align VIX to ES dates
    vix_close = vix["Close"].reindex(es.index).ffill()
    es_open = es["Open"].astype(float)
    es_close = es["Close"].astype(float)
    es_low = es["Low"].astype(float)

    trades = []
    daily_pnl = pd.Series(0.0, index=es.index, dtype=float)

    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    hold_days = 0

    for i in range(1, len(es)):
        if pd.isna(vix_close.iloc[i - 1]):
            continue

        if not in_trade:
            # Check yesterday's VIX close for entry signal
            if vix_close.iloc[i - 1] > VIX_ENTRY_THRESHOLD:
                in_trade = True
                entry_price = es_open.iloc[i]
                entry_idx = i
                hold_days = 0

        if in_trade:
            hold_days += 1
            day_close = es_close.iloc[i]
            day_low = es_low.iloc[i]

            # Check hard stop: 40 points below entry
            stop_price = entry_price - VIX_STOP_POINTS
            stopped = day_low <= stop_price

            # Determine exit conditions
            exit_trade = False
            exit_price = day_close
            exit_reason = ""

            if stopped:
                exit_trade = True
                exit_price = stop_price  # Assume fill at stop level
                exit_reason = f"STOP (-{VIX_STOP_POINTS:.0f}pts)"
            elif hold_days >= VIX_MAX_HOLD_DAYS:
                exit_trade = True
                exit_price = day_close
                exit_reason = f"MAX HOLD ({VIX_MAX_HOLD_DAYS}d)"
            elif vix_close.iloc[i] < VIX_EXIT_THRESHOLD:
                exit_trade = True
                exit_price = day_close
                exit_reason = "VIX < 20"

            # Daily MTM (use actual exit price if exiting today)
            if i == entry_idx:
                prev_price = entry_price
            else:
                prev_price = es_close.iloc[i - 1]

            if exit_trade:
                day_pnl = (exit_price - prev_price) * ES_POINT_VALUE
            else:
                day_pnl = (day_close - prev_price) * ES_POINT_VALUE
            daily_pnl.iloc[i] += day_pnl

            if exit_trade:
                # Subtract costs on exit
                daily_pnl.iloc[i] -= COST_PER_TRADE

                pnl_points = exit_price - entry_price
                gross_pnl = pnl_points * ES_POINT_VALUE
                net_pnl = gross_pnl - COST_PER_TRADE

                trades.append({
                    "signal": "VIX",
                    "entry_date": es.index[entry_idx],
                    "exit_date": es.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_points": pnl_points,
                    "gross_pnl": gross_pnl,
                    "costs": COST_PER_TRADE,
                    "pnl": net_pnl,
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                })

                in_trade = False

    return {"trades": trades, "daily_pnl": daily_pnl}


def simulate_topstep(trade_pnls: list[float]) -> list[dict]:
    """Simulate Topstep $150K eval attempts."""
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
    """Run ES Week 1 & Week 4 + VIX overlay backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1993-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    # Use ES date range (VIX is longer)
    start = es.index[0]
    end = es.index[-1]
    print(f"Backtest range: {start.date()} to {end.date()}")

    print(f"\nCalendar signal: Week 1 & Week 4, Monday open -> Friday close")
    print(f"VIX signal: Entry VIX>{VIX_ENTRY_THRESHOLD}, Exit VIX<{VIX_EXIT_THRESHOLD} "
          f"or {VIX_MAX_HOLD_DAYS}d hold or {VIX_STOP_POINTS:.0f}pt stop")
    print(f"Costs: ${COST_PER_TRADE:.0f}/trade (${COMMISSION_RT} comm + "
          f"2x${SLIPPAGE_PER_SIDE} slip)\n")

    # Run both signals
    cal_result = run_calendar_signal(es)
    vix_result = run_vix_signal(es, vix)

    cal_trades = pd.DataFrame(cal_result["trades"])
    vix_trades = pd.DataFrame(vix_result["trades"])

    cal_pnls = pd.Series(cal_trades["pnl"].values, dtype=float) if len(cal_trades) > 0 else pd.Series(dtype=float)
    vix_pnls = pd.Series(vix_trades["pnl"].values, dtype=float) if len(vix_trades) > 0 else pd.Series(dtype=float)

    # Build equity curves from daily P&L
    cal_equity = (INITIAL_CAPITAL + cal_result["daily_pnl"].cumsum())
    vix_equity = (INITIAL_CAPITAL + vix_result["daily_pnl"].cumsum())
    combined_daily_pnl = cal_result["daily_pnl"] + vix_result["daily_pnl"]
    combined_equity = (INITIAL_CAPITAL + combined_daily_pnl.cumsum())

    # Position series
    cal_position = pd.Series(0.0, index=es.index)
    for t in cal_result["trades"]:
        mask = (es.index >= t["entry_date"]) & (es.index <= t["exit_date"])
        cal_position[mask] = 1.0

    vix_position = pd.Series(0.0, index=es.index)
    for t in vix_result["trades"]:
        mask = (es.index >= t["entry_date"]) & (es.index <= t["exit_date"])
        vix_position[mask] = 1.0

    combined_position = cal_position + vix_position  # 0, 1, or 2

    # --- 1. Calendar signal alone ---
    print(f"{'='*80}")
    print(f"  Signal 1: CALENDAR (Week 1 & Week 4)")
    print(f"{'='*80}")
    cal_stats = summary(cal_pnls, cal_equity, cal_position)
    print_summary(cal_stats, "Calendar Signal")
    print(f"  Total P&L: ${cal_pnls.sum():,.0f}")

    # --- 2. VIX signal alone ---
    print(f"{'='*80}")
    print(f"  Signal 2: VIX Overlay (entry>{VIX_ENTRY_THRESHOLD}, "
          f"exit<{VIX_EXIT_THRESHOLD}, {VIX_MAX_HOLD_DAYS}d max, "
          f"{VIX_STOP_POINTS:.0f}pt stop)")
    print(f"{'='*80}")
    vix_stats = summary(vix_pnls, vix_equity, vix_position)
    print_summary(vix_stats, "VIX Signal")
    print(f"  Total P&L: ${vix_pnls.sum():,.0f}")

    if len(vix_trades) > 0:
        stopped = vix_trades[vix_trades["exit_reason"].str.contains("STOP")]
        max_hold = vix_trades[vix_trades["exit_reason"].str.contains("MAX HOLD")]
        vix_exit = vix_trades[vix_trades["exit_reason"].str.contains("VIX")]
        print(f"\n  Exit breakdown:")
        print(f"    VIX < 20:      {len(vix_exit):>5} ({len(vix_exit)/len(vix_trades)*100:.1f}%)")
        print(f"    Max hold {VIX_MAX_HOLD_DAYS}d:  {len(max_hold):>5} ({len(max_hold)/len(vix_trades)*100:.1f}%)")
        print(f"    Stop loss:     {len(stopped):>5} ({len(stopped)/len(vix_trades)*100:.1f}%)")
        print(f"    Avg hold days: {vix_trades['hold_days'].mean():.1f}")

    # --- 3. Combined ---
    print(f"\n{'='*80}")
    print(f"  COMBINED: Calendar + VIX Overlay")
    print(f"{'='*80}")

    # For combined trade P&Ls, merge both trade lists sorted by entry date
    all_trades_list = cal_result["trades"] + vix_result["trades"]
    all_trades_list.sort(key=lambda t: t["entry_date"])
    combined_trade_pnls = pd.Series([t["pnl"] for t in all_trades_list], dtype=float)

    combined_stats = summary(combined_trade_pnls, combined_equity, (combined_position > 0).astype(float))
    print_summary(combined_stats, "Combined (Calendar + VIX)")
    print(f"  Total P&L: ${combined_trade_pnls.sum():,.0f}")
    print(f"  Calendar trades: {len(cal_trades)}, VIX trades: {len(vix_trades)}")

    # Overlap analysis
    overlap_days = ((cal_position > 0) & (vix_position > 0)).sum()
    total_active = (combined_position > 0).sum()
    print(f"  Days with 2 contracts: {overlap_days} ({overlap_days/len(es)*100:.1f}% of all days)")
    print(f"  Days with any position: {total_active} ({total_active/len(es)*100:.1f}% of all days)")

    # --- Correlation of returns ---
    print(f"\n{'='*75}")
    print(f"  Return Correlation Between Signals")
    print(f"{'='*75}")

    # Use daily P&L for correlation (only days where at least one signal active)
    active_mask = (cal_result["daily_pnl"] != 0) | (vix_result["daily_pnl"] != 0)
    if active_mask.sum() > 10:
        corr = cal_result["daily_pnl"][active_mask].corr(vix_result["daily_pnl"][active_mask])
        print(f"  Daily P&L correlation (active days): {corr:.3f}")

    # Also compute trade-level correlation using monthly bucketing
    cal_monthly = cal_result["daily_pnl"].resample("ME").sum()
    vix_monthly = vix_result["daily_pnl"].resample("ME").sum()
    monthly_corr_mask = (cal_monthly != 0) | (vix_monthly != 0)
    if monthly_corr_mask.sum() > 10:
        monthly_corr = cal_monthly[monthly_corr_mask].corr(vix_monthly[monthly_corr_mask])
        print(f"  Monthly P&L correlation:             {monthly_corr:.3f}")

    print(f"{'='*75}\n")

    # --- Topstep simulation on combined ---
    combined_attempts = simulate_topstep(combined_trade_pnls.tolist())

    print(f"{'='*80}")
    print(f"  Topstep $150K Eval — Combined (Calendar + VIX)")
    print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing DD, "
          f"${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
    print(f"{'='*80}")
    print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} "
          f"{'Peak Bal':>12} {'Final Bal':>12}")
    print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    comb_passed = 0
    comb_failed = 0
    for a in combined_attempts:
        status_str = a["status"]
        print(f"  {a['attempt']:>3} {status_str:>10} {a['trades_taken']:>8} "
              f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} "
              f"${a['final_balance']:>11,.0f}")
        if status_str == "PASSED":
            comb_passed += 1
        elif status_str == "FAILED":
            comb_failed += 1

    comb_total = comb_passed + comb_failed
    comb_rate = comb_passed / comb_total * 100 if comb_total > 0 else 0

    print(f"{'='*80}")
    print(f"  Combined: {comb_total} attempts | "
          f"Passed: {comb_passed} | Failed: {comb_failed} | "
          f"Pass rate: {comb_rate:.1f}%")

    # Months-to-pass
    trades_per_month = len(combined_trade_pnls) / ((end - start).days / 30.44)

    if comb_passed > 0:
        pa = [a for a in combined_attempts if a["status"] == "PASSED"]
        fa = [a for a in combined_attempts if a["status"] == "FAILED"]
        avg_trades_pass = np.mean([a["trades_taken"] for a in pa])
        avg_months_pass = avg_trades_pass / trades_per_month
        avg_attempts = comb_total / comb_passed
        avg_trades_fail = np.mean([a["trades_taken"] for a in fa]) if fa else 0
        avg_months_fail = avg_trades_fail / trades_per_month
        expected_months = (avg_attempts - 1) * avg_months_fail + avg_months_pass
        expected_cost = expected_months * TOPSTEP_MONTHLY_FEE

        print(f"\n  Avg trades to pass:           {avg_trades_pass:.1f}")
        print(f"  Expected months to pass:      {expected_months:.1f}")
        print(f"  Expected eval cost:           ${expected_cost:,.0f}")

    # ES-only Topstep for comparison
    cal_attempts = simulate_topstep(cal_pnls.tolist())
    cal_passed = sum(1 for a in cal_attempts if a["status"] == "PASSED")
    cal_failed = sum(1 for a in cal_attempts if a["status"] == "FAILED")
    cal_total = cal_passed + cal_failed
    cal_rate = cal_passed / cal_total * 100 if cal_total > 0 else 0

    print(f"\n  {'='*76}")
    print(f"  Combined vs ES-Only (Calendar) Comparison")
    print(f"  {'='*76}")
    print(f"  {'Metric':<35} {'Cal-only':>18} {'Combined':>18}")
    print(f"  {'-'*35} {'-'*18} {'-'*18}")
    print(f"  {'Pass rate (this data)':.<35} {cal_rate:>17.1f}% {comb_rate:>17.1f}%")
    print(f"  {'Passed / Total':.<35} {f'{cal_passed}/{cal_total}':>18} {f'{comb_passed}/{comb_total}':>18}")
    print(f"  {'ES v1 baseline (reference)':.<35} {ES_V1_PASS_RATE:>17.1f}% {'':>18}")

    delta = comb_rate - cal_rate
    print(f"  {'-'*35} {'-'*18} {'-'*18}")
    print(f"  {'Delta (combined vs cal-only)':.<35} {'':>18} {f'{delta:+.1f}%':>18}")
    print(f"  {'='*76}")
    print(f"{'='*80}\n")

    # Save outputs
    plot_equity(combined_equity, STRATEGY_NAME)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save all trades
    all_trades_df = pd.concat([cal_trades, vix_trades], ignore_index=True)
    all_trades_df = all_trades_df.sort_values("entry_date").reset_index(drop=True)
    csv_path = os.path.join(RESULTS_DIR, "es_week14_vix_overlay_trades.csv")
    all_trades_df.to_csv(csv_path, index=False)
    print(f"Trade list saved to {csv_path}")

    return combined_stats


if __name__ == "__main__":
    run()
