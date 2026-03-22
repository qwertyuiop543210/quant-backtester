"""Week 1 & Week 4 Daily Hold — Topstep Compatibility Test.
Tests whether the Week 1/4 + VIX filter edge survives when restructured
from a single Monday→Friday hold into 5 separate daily holds per qualifying week.
Topstep requires all positions closed by 3:10 PM CT each session.
This means Monday open → Friday close is PROHIBITED.
Three variants compared:
  A) Daily Hold: buy open, sell close each day (Topstep compliant)
  B) Session Hold: buy prior close, sell next close (overnight, NOT compliant)
  C) Weekly Hold: buy Monday open, sell Friday close (original backtest, NOT compliant)
Costs per variant:
  A) 5 round trips per week ($25 commission + $125 slippage per week)
  B) 5 round trips per week ($25 commission + $125 slippage per week)
  C) 1 round trip per week ($5 commission + $25 slippage per week)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
STRATEGY_NAME = "Week 1 & Week 4 Daily Hold — Topstep Compatibility Test"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0
# Topstep $150K account parameters
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
def get_week_of_month(date):
    """Return week number (1-5) based on which Monday-Friday block the date falls in.

    Week 1 = days 1-7 of month
    Week 2 = days 8-14
    Week 3 = days 15-21
    Week 4 = days 22-28
    Week 5 = days 29-31
    """
    day = date.day
    if day <= 7:
        return 1
    elif day <= 14:
        return 2
    elif day <= 21:
        return 3
    elif day <= 28:
        return 4
    else:
        return 5
def identify_qualifying_weeks(es_data, vix_data):
    """Identify all qualifying weeks (Week 1 or Week 4, VIX not 15-20).

    Returns list of dicts with:
      - week_start: first trading day (Monday) of the qualifying week
      - week_end: last trading day (Friday) of the qualifying week
      - week_days: list of all trading days in that week
      - vix_friday: VIX close on the prior Friday
      - week_num: 1 or 4
    """
    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")

    # Build a DataFrame with day-of-week and week-of-month info
    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "high": es_data["High"].astype(float),
        "low": es_data["Low"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,  # 0=Mon, 4=Fri
        "week_of_month": [get_week_of_month(d) for d in es_close.index],
    }, index=es_close.index)

    # Group trading days into calendar weeks (Mon-Fri blocks)
    # We define a "trading week" by its ISO week number + year
    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

    qualifying_weeks = []

    for week_key, group in df.groupby("week_key", sort=True):
        if len(group) < 3:  # Need at least 3 trading days to be a real week
            continue

        # The Monday of this week determines the week-of-month
        monday_candidates = group[group["dow"] == 0]
        if len(monday_candidates) == 0:
            # No Monday (holiday) — use first day of week
            first_day = group.index[0]
        else:
            first_day = monday_candidates.index[0]

        week_num = get_week_of_month(first_day)

        # Only Week 1 and Week 4
        if week_num not in [1, 4]:
            continue

        # VIX filter: check prior Friday's VIX close
        # Find the last trading day before this week's first day
        prior_days = df.index[df.index < group.index[0]]
        if len(prior_days) == 0:
            continue
        prior_friday = prior_days[-1]
        vix_val = df.loc[prior_friday, "vix"]

        if pd.isna(vix_val):
            continue

        # Skip if VIX is between 15.0 and 20.0
        if 15.0 <= vix_val <= 20.0:
            continue

        qualifying_weeks.append({
            "week_key": week_key,
            "week_num": week_num,
            "week_days": group.index.tolist(),
            "week_start": group.index[0],
            "week_end": group.index[-1],
            "vix_friday": vix_val,
        })

    return qualifying_weeks, df
def run_variant_a_daily_hold(qualifying_weeks, df):
    """Variant A: Buy open, sell close each day (Topstep compliant).

    5 separate intraday trades per qualifying week.
    """
    trades = []

    for week in qualifying_weeks:
        for day in week["week_days"]:
            row = df.loc[day]
            entry_price = row["open"]
            exit_price = row["close"]

            if pd.isna(entry_price) or pd.isna(exit_price):
                continue

            pnl_points = exit_price - entry_price
            gross_pnl = pnl_points * ES_POINT_VALUE
            cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
            net_pnl = gross_pnl - cost

            trades.append({
                "entry_date": day,
                "exit_date": day,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "hold_days": 0,
                "pnl_points": pnl_points,
                "gross_pnl": gross_pnl,
                "costs": cost,
                "pnl": net_pnl,
                "week_num": week["week_num"],
                "vix_friday": week["vix_friday"],
                "variant": "A_daily_hold",
            })

    return pd.DataFrame(trades)
def run_variant_b_session_hold(qualifying_weeks, df):
    """Variant B: Buy prior close, sell this close (overnight hold, NOT Topstep compliant).

    5 separate close-to-close trades per qualifying week.
    """
    trades = []
    all_dates = df.index.tolist()

    for week in qualifying_weeks:
        for day in week["week_days"]:
            day_idx = all_dates.index(day) if day in all_dates else -1
            if day_idx <= 0:
                continue

            prior_day = all_dates[day_idx - 1]
            entry_price = df.loc[prior_day, "close"]
            exit_price = df.loc[day, "close"]

            if pd.isna(entry_price) or pd.isna(exit_price):
                continue

            pnl_points = exit_price - entry_price
            gross_pnl = pnl_points * ES_POINT_VALUE
            cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
            net_pnl = gross_pnl - cost

            trades.append({
                "entry_date": prior_day,
                "exit_date": day,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "hold_days": (day - prior_day).days,
                "pnl_points": pnl_points,
                "gross_pnl": gross_pnl,
                "costs": cost,
                "pnl": net_pnl,
                "week_num": week["week_num"],
                "vix_friday": week["vix_friday"],
                "variant": "B_session_hold",
            })

    return pd.DataFrame(trades)
def run_variant_c_weekly_hold(qualifying_weeks, df):
    """Variant C: Buy Monday open, sell Friday close (original, NOT Topstep compliant).

    1 trade per qualifying week.
    """
    trades = []

    for week in qualifying_weeks:
        days = week["week_days"]
        if len(days) < 2:
            continue

        entry_day = days[0]  # Monday (or first day if Monday is holiday)
        exit_day = days[-1]   # Friday (or last day)

        entry_price = df.loc[entry_day, "open"]
        exit_price = df.loc[exit_day, "close"]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
        net_pnl = gross_pnl - cost

        trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "hold_days": (exit_day - entry_day).days,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost,
            "pnl": net_pnl,
            "week_num": week["week_num"],
            "vix_friday": week["vix_friday"],
            "variant": "C_weekly_hold",
        })

    return pd.DataFrame(trades)
def simulate_topstep_attempts(trade_pnls):
    """Simulate sequential Topstep evaluation attempts.

    Rules:
    - Start with $150K
    - Trailing drawdown: $4,500 from equity high-water mark
    - Profit target: $9,000 cumulative profit
    - If trailing DD breached → FAIL, restart from next trade
    - If profit target hit → PASS
    """
    attempts = []
    i = 0

    while i < len(trade_pnls):
        balance = TOPSTEP_CAPITAL
        high_water = balance
        start_trade = i
        status = "in_progress"

        while i < len(trade_pnls):
            balance += trade_pnls.iloc[i]
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
def compute_variant_stats(trade_df, variant_name, df):
    """Compute full stats + Topstep sim for a variant."""
    if len(trade_df) == 0:
        return None, None, None

    trade_pnls = pd.Series(trade_df["pnl"].values, dtype=float)

    # Build equity curve
    equity_values = [INITIAL_CAPITAL]
    for pnl in trade_pnls:
        equity_values.append(equity_values[-1] + pnl)

    # Use trade exit dates for equity index
    dates = [trade_df["entry_date"].iloc[0] - pd.Timedelta(days=1)]  # Start date
    dates.extend(trade_df["exit_date"].tolist())

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    position = pd.Series(
        [0] + [1] * len(trade_pnls),
        index=pd.DatetimeIndex(dates)
    )

    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )

    # Topstep simulation
    attempts = simulate_topstep_attempts(trade_pnls)
    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    total = sum(1 for a in attempts if a["status"] in ["PASSED", "FAILED"])
    pass_rate = passed / total * 100 if total > 0 else 0

    topstep_stats = {
        "total_attempts": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "avg_trades_to_pass": np.mean([a["trades_taken"] for a in attempts if a["status"] == "PASSED"]) if passed > 0 else 0,
    }

    return stats, topstep_stats, equity
def run():
    """Run the Topstep compatibility test across all three variants."""
    print("=" * 80)
    print("TOPSTEP COMPATIBILITY TEST")
    print("Week 1 & Week 4 ES Calendar + VIX Filter")
    print("=" * 80)

    print("\nLoading ES and VIX data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    # Identify qualifying weeks
    qualifying_weeks, df = identify_qualifying_weeks(es, vix)
    print(f"\nQualifying weeks found: {len(qualifying_weeks)}")

    week1_count = sum(1 for w in qualifying_weeks if w["week_num"] == 1)
    week4_count = sum(1 for w in qualifying_weeks if w["week_num"] == 4)
    print(f"  Week 1: {week1_count}")
    print(f"  Week 4: {week4_count}")

    # Run all three variants
    print("\n" + "=" * 60)
    print("VARIANT A: Daily Hold (Buy Open → Sell Close)")
    print("Topstep COMPLIANT — no overnight positions")
    print("=" * 60)
    trades_a = run_variant_a_daily_hold(qualifying_weeks, df)
    stats_a, topstep_a, equity_a = compute_variant_stats(trades_a, "A", df)
    if stats_a:
        print_summary(stats_a, "Variant A: Daily Hold")

    print("\n" + "=" * 60)
    print("VARIANT B: Session Hold (Buy Prior Close → Sell Close)")
    print("NOT Topstep compliant — includes overnight gap")
    print("=" * 60)
    trades_b = run_variant_b_session_hold(qualifying_weeks, df)
    stats_b, topstep_b, equity_b = compute_variant_stats(trades_b, "B", df)
    if stats_b:
        print_summary(stats_b, "Variant B: Session Hold")

    print("\n" + "=" * 60)
    print("VARIANT C: Weekly Hold (Buy Mon Open → Sell Fri Close)")
    print("NOT Topstep compliant — original backtest structure")
    print("=" * 60)
    trades_c = run_variant_c_weekly_hold(qualifying_weeks, df)
    stats_c, topstep_c, equity_c = compute_variant_stats(trades_c, "C", df)
    if stats_c:
        print_summary(stats_c, "Variant C: Weekly Hold (Original)")

    # =============================================
    # COMPARISON TABLE
    # =============================================
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    header = f"  {'Metric':<30} {'A: Daily Hold':>16} {'B: Session Hold':>16} {'C: Weekly Hold':>16}"
    print(header)
    print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*16}")

    def fmt(val, fmt_str=".3f"):
        if val is None:
            return "N/A"
        return f"{val:{fmt_str}}"

    if stats_a and stats_b and stats_c:
        rows = [
            ("Profit Factor", stats_a["profit_factor"], stats_b["profit_factor"], stats_c["profit_factor"], ".3f"),
            ("Win Rate", stats_a["win_rate"], stats_b["win_rate"], stats_c["win_rate"], ".1%"),
            ("Total Trades", stats_a["total_trades"], stats_b["total_trades"], stats_c["total_trades"], "d"),
            ("Max Drawdown", stats_a["max_drawdown"], stats_b["max_drawdown"], stats_c["max_drawdown"], ".1%"),
            ("Annualized Return", stats_a["annualized_return"], stats_b["annualized_return"], stats_c["annualized_return"], ".1%"),
            ("Sharpe Ratio", stats_a["sharpe_ratio"], stats_b["sharpe_ratio"], stats_c["sharpe_ratio"], ".3f"),
        ]

        for label, a, b, c, f in rows:
            if f == "d":
                print(f"  {label:<30} {int(a):>16} {int(b):>16} {int(c):>16}")
            elif f == ".1%":
                print(f"  {label:<30} {a:>15.1%} {b:>15.1%} {c:>15.1%}")
            else:
                print(f"  {label:<30} {a:>16{f}} {b:>16{f}} {c:>16{f}}")

        # Cost comparison
        total_cost_a = trades_a["costs"].sum() if len(trades_a) > 0 else 0
        total_cost_b = trades_b["costs"].sum() if len(trades_b) > 0 else 0
        total_cost_c = trades_c["costs"].sum() if len(trades_c) > 0 else 0
        print(f"  {'Total Costs ($)':<30} {'${:,.0f}'.format(total_cost_a):>16} {'${:,.0f}'.format(total_cost_b):>16} {'${:,.0f}'.format(total_cost_c):>16}")

        # Average P&L per trade
        avg_a = trades_a["pnl"].mean() if len(trades_a) > 0 else 0
        avg_b = trades_b["pnl"].mean() if len(trades_b) > 0 else 0
        avg_c = trades_c["pnl"].mean() if len(trades_c) > 0 else 0
        print(f"  {'Avg $ per trade':<30} {'${:,.0f}'.format(avg_a):>16} {'${:,.0f}'.format(avg_b):>16} {'${:,.0f}'.format(avg_c):>16}")

        # Gross P&L per trade (before costs)
        avg_gross_a = trades_a["gross_pnl"].mean() if len(trades_a) > 0 else 0
        avg_gross_b = trades_b["gross_pnl"].mean() if len(trades_b) > 0 else 0
        avg_gross_c = trades_c["gross_pnl"].mean() if len(trades_c) > 0 else 0
        print(f"  {'Avg $ per trade (gross)':<30} {'${:,.0f}'.format(avg_gross_a):>16} {'${:,.0f}'.format(avg_gross_b):>16} {'${:,.0f}'.format(avg_gross_c):>16}")

    # Topstep simulation comparison
    print(f"\n  {'--- TOPSTEP SIMULATION ---':<30}")
    if topstep_a and topstep_b and topstep_c:
        print(f"  {'Pass Rate':<30} {topstep_a['pass_rate']:>15.1f}% {topstep_b['pass_rate']:>15.1f}% {topstep_c['pass_rate']:>15.1f}%")
        print(f"  {'Attempts (pass/total)':<30} {topstep_a['passed']}/{topstep_a['total_attempts']:>13} {topstep_b['passed']}/{topstep_b['total_attempts']:>13} {topstep_c['passed']}/{topstep_c['total_attempts']:>13}")
        print(f"  {'Avg trades to pass':<30} {topstep_a['avg_trades_to_pass']:>16.1f} {topstep_b['avg_trades_to_pass']:>16.1f} {topstep_c['avg_trades_to_pass']:>16.1f}")

    # =============================================
    # VERDICT
    # =============================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if stats_a:
        pf_a = stats_a["profit_factor"]
        pass_rate_a = topstep_a["pass_rate"] if topstep_a else 0

        if pf_a > 1.5 and pass_rate_a > 45:
            print(f"\n  ✅ TOPSTEP COMPATIBLE")
            print(f"  PF {pf_a:.3f} > 1.5 and pass rate {pass_rate_a:.1f}% > 45%")
            print(f"  Proceed with daily hold structure on Topstep.")
            print(f"  The calendar edge SURVIVES daily exit requirements.")
        elif pf_a > 1.2:
            print(f"\n  ⚠️  MARGINAL")
            print(f"  PF {pf_a:.3f} (between 1.2 and 1.5), pass rate {pass_rate_a:.1f}%")
            print(f"  Edge exists but weakened by 5x cost multiplication.")
            print(f"  Consider Apex Trader Funding for weekly holds instead.")
        else:
            print(f"\n  ❌ EDGE DESTROYED BY DAILY COSTS")
            print(f"  PF {pf_a:.3f} < 1.2, pass rate {pass_rate_a:.1f}%")
            print(f"  The calendar effect does not survive 5 daily round trips.")
            print(f"  USE APEX TRADER FUNDING or similar swing-friendly firm.")

        # How much edge is lost
        if stats_c:
            pf_c = stats_c["profit_factor"]
            pf_loss = (1 - pf_a / pf_c) * 100
            print(f"\n  Edge degradation from weekly → daily: {pf_loss:.1f}% PF reduction")

            if stats_b:
                pf_b = stats_b["profit_factor"]
                overnight_contribution = (pf_b - pf_a) / (pf_c - pf_a) * 100 if pf_c != pf_a else 0
                print(f"  Overnight gap contribution to edge: ~{overnight_contribution:.0f}% of the difference")
                print(f"  (If high, the edge is in the overnight gap — Topstep kills it)")
                print(f"  (If low, the edge is in the calendar date — Topstep preserves it)")

    print("\n" + "=" * 80)

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if equity_a is not None:
        plot_equity(equity_a, "Week14 Variant A - Daily Hold (Topstep Compliant)")
    if equity_c is not None:
        plot_equity(equity_c, "Week14 Variant C - Weekly Hold (Original)")

    if len(trades_a) > 0:
        trades_a.to_csv(os.path.join(RESULTS_DIR, "week14_daily_hold_trades.csv"), index=False)
        print(f"Variant A trades saved to results/week14_daily_hold_trades.csv")
    if len(trades_b) > 0:
        trades_b.to_csv(os.path.join(RESULTS_DIR, "week14_session_hold_trades.csv"), index=False)
        print(f"Variant B trades saved to results/week14_session_hold_trades.csv")
    if len(trades_c) > 0:
        trades_c.to_csv(os.path.join(RESULTS_DIR, "week14_weekly_hold_trades.csv"), index=False)
        print(f"Variant C trades saved to results/week14_weekly_hold_trades.csv")
if __name__ == "__main__":
    run()
