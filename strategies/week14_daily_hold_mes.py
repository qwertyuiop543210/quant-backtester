"""Week 1 & Week 4 Daily Hold — MES Micro E-mini Cost Variant.
Tests whether using MES (Micro E-mini S&P 500) instead of full ES
recovers enough edge to make the Topstep daily-hold structure viable.
MES = $5/point (1/10th of ES $50/point)
MES commission = ~$0.62 per side ($1.24 round trip) on Topstep
MES slippage = ~$1.25 per side (1 tick = 0.25 points * $5)
We test multiple contract counts:
  - 2 MES = $10/point exposure, ~$4.98 total cost per trade
  - 3 MES = $15/point exposure, ~$7.47 total cost per trade
  - 5 MES = $25/point exposure, ~$12.45 total cost per trade (half of 1 ES)
Compare against:
  - 1 ES daily hold ($50/point, $30 cost per trade) — the result we already have
  - 1 ES weekly hold ($50/point, $30 cost per trade) — the original backtest
Also runs Topstep sim on each to find optimal contract count.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
STRATEGY_NAME = "Week 1 & Week 4 Daily Hold — MES Variant"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
# MES contract specifications
MES_POINT_VALUE = 5.0       # $5 per point
MES_COMMISSION_RT = 1.24    # $1.24 round trip per contract (Topstep)
MES_SLIPPAGE_PER_SIDE = 1.25  # 1 tick = 0.25 pts * $5 = $1.25
# ES for comparison
ES_POINT_VALUE = 50.0
ES_COMMISSION_RT = 5.00
ES_SLIPPAGE_PER_SIDE = 12.50
# Topstep $150K account
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_DAILY_LOSS_LIMIT = 3_000.0  # Topstep $150K daily loss limit
INITIAL_CAPITAL = 100_000.0
def get_week_of_month(date):
    """Week 1=days 1-7, Week 2=8-14, Week 3=15-21, Week 4=22-28, Week 5=29-31."""
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
    """Identify weeks 1 and 4 where VIX is NOT between 15.0-20.0."""
    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")
    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
        "week_of_month": [get_week_of_month(d) for d in es_close.index],
    }, index=es_close.index)
    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)
    qualifying_weeks = []
    for week_key, group in df.groupby("week_key", sort=True):
        if len(group) < 3:
            continue
        monday_candidates = group[group["dow"] == 0]
        first_day = monday_candidates.index[0] if len(monday_candidates) > 0 else group.index[0]
        week_num = get_week_of_month(first_day)
        if week_num not in [1, 4]:
            continue
        prior_days = df.index[df.index < group.index[0]]
        if len(prior_days) == 0:
            continue
        prior_friday = prior_days[-1]
        vix_val = df.loc[prior_friday, "vix"]
        if pd.isna(vix_val):
            continue
        if 15.0 <= vix_val <= 20.0:
            continue
        qualifying_weeks.append({
            "week_key": week_key,
            "week_num": week_num,
            "week_days": group.index.tolist(),
            "vix_friday": vix_val,
        })
    return qualifying_weeks, df
def run_daily_hold(qualifying_weeks, df, num_contracts, point_value,
                   commission_rt, slippage_per_side, label):
    """Run daily hold variant with specified contract/cost parameters."""
    trades = []
    cost_per_trade = (num_contracts * commission_rt) + (num_contracts * 2 * slippage_per_side)
    for week in qualifying_weeks:
        for day in week["week_days"]:
            row = df.loc[day]
            entry_price = row["open"]
            exit_price = row["close"]
            if pd.isna(entry_price) or pd.isna(exit_price):
                continue
            pnl_points = exit_price - entry_price
            gross_pnl = pnl_points * point_value * num_contracts
            net_pnl = gross_pnl - cost_per_trade
            trades.append({
                "entry_date": day,
                "exit_date": day,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "gross_pnl": gross_pnl,
                "costs": cost_per_trade,
                "pnl": net_pnl,
                "week_num": week["week_num"],
                "label": label,
            })
    return pd.DataFrame(trades)
def run_weekly_hold(qualifying_weeks, df, num_contracts, point_value,
                    commission_rt, slippage_per_side, label):
    """Run weekly hold (Mon open → Fri close) for comparison."""
    trades = []
    cost_per_trade = (num_contracts * commission_rt) + (num_contracts * 2 * slippage_per_side)
    for week in qualifying_weeks:
        days = week["week_days"]
        if len(days) < 2:
            continue
        entry_price = df.loc[days[0], "open"]
        exit_price = df.loc[days[-1], "close"]
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * point_value * num_contracts
        net_pnl = gross_pnl - cost_per_trade
        trades.append({
            "entry_date": days[0],
            "exit_date": days[-1],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost_per_trade,
            "pnl": net_pnl,
            "week_num": week["week_num"],
            "label": label,
        })
    return pd.DataFrame(trades)
def simulate_topstep(trade_pnls):
    """Simulate sequential Topstep evaluation attempts."""
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
            "trades_taken": i - start_trade,
            "profit": balance - TOPSTEP_CAPITAL,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
def compute_stats(trade_df, label):
    """Compute stats + Topstep sim for a variant."""
    if len(trade_df) == 0:
        return None, None
    trade_pnls = pd.Series(trade_df["pnl"].values, dtype=float)
    equity_values = [INITIAL_CAPITAL]
    for pnl in trade_pnls:
        equity_values.append(equity_values[-1] + pnl)
    dates = [trade_df["entry_date"].iloc[0] - pd.Timedelta(days=1)]
    dates.extend(trade_df["exit_date"].tolist())
    equity = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    position = pd.Series([0] + [1] * len(trade_pnls), index=pd.DatetimeIndex(dates))
    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )
    # Topstep simulation
    attempts = simulate_topstep(trade_pnls)
    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    total = sum(1 for a in attempts if a["status"] in ["PASSED", "FAILED"])
    pass_rate = passed / total * 100 if total > 0 else 0
    topstep = {
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate,
        "avg_trades_to_pass": np.mean([a["trades_taken"] for a in attempts if a["status"] == "PASSED"]) if passed > 0 else 0,
    }
    return stats, topstep
def run():
    """Run MES micro variant comparison."""
    print("=" * 80)
    print("MES MICRO E-MINI COST VARIANT TEST")
    print("Week 1 & Week 4 Daily Hold — Finding Optimal Contract Size")
    print("=" * 80)
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    qualifying_weeks, df = identify_qualifying_weeks(es, vix)
    print(f"Qualifying weeks: {len(qualifying_weeks)}")
    # Define all variants to test
    variants = [
        # (label, hold_type, num_contracts, point_value, commission_rt, slippage_per_side)
        ("2 MES Daily",  "daily",  2, MES_POINT_VALUE, MES_COMMISSION_RT, MES_SLIPPAGE_PER_SIDE),
        ("3 MES Daily",  "daily",  3, MES_POINT_VALUE, MES_COMMISSION_RT, MES_SLIPPAGE_PER_SIDE),
        ("5 MES Daily",  "daily",  5, MES_POINT_VALUE, MES_COMMISSION_RT, MES_SLIPPAGE_PER_SIDE),
        ("1 ES Daily",   "daily",  1, ES_POINT_VALUE,  ES_COMMISSION_RT,  ES_SLIPPAGE_PER_SIDE),
        ("1 ES Weekly",  "weekly", 1, ES_POINT_VALUE,  ES_COMMISSION_RT,  ES_SLIPPAGE_PER_SIDE),
        ("2 MES Weekly", "weekly", 2, MES_POINT_VALUE, MES_COMMISSION_RT, MES_SLIPPAGE_PER_SIDE),
        ("5 MES Weekly", "weekly", 5, MES_POINT_VALUE, MES_COMMISSION_RT, MES_SLIPPAGE_PER_SIDE),
    ]
    all_results = {}
    for label, hold_type, n_contracts, pv, comm, slip in variants:
        if hold_type == "daily":
            trades = run_daily_hold(qualifying_weeks, df, n_contracts, pv, comm, slip, label)
        else:
            trades = run_weekly_hold(qualifying_weeks, df, n_contracts, pv, comm, slip, label)
        stats, topstep = compute_stats(trades, label)
        all_results[label] = {
            "trades": trades,
            "stats": stats,
            "topstep": topstep,
        }
    # =============================================
    # COMPARISON TABLE
    # =============================================
    print("\n" + "=" * 120)
    print("FULL COMPARISON TABLE")
    print("=" * 120)
    header_labels = list(all_results.keys())
    col_width = 15
    # Print header
    print(f"  {'Metric':<25}", end="")
    for label in header_labels:
        print(f" {label:>{col_width}}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in header_labels:
        print(f" {'-'*col_width}", end="")
    print()
    # Cost per trade
    print(f"  {'Cost per trade ($)':<25}", end="")
    for label in header_labels:
        t = all_results[label]["trades"]
        avg_cost = t["costs"].mean() if len(t) > 0 else 0
        print(f" ${avg_cost:>{col_width-1}.2f}", end="")
    print()
    # $ per point exposure
    print(f"  {'$/point exposure':<25}", end="")
    for label, hold_type, n_contracts, pv, comm, slip in variants:
        exposure = n_contracts * pv
        print(f" ${exposure:>{col_width-1}.0f}", end="")
    print()
    # Profit Factor
    print(f"  {'Profit Factor':<25}", end="")
    for label in header_labels:
        s = all_results[label]["stats"]
        val = s["profit_factor"] if s else 0
        print(f" {val:>{col_width}.3f}", end="")
    print()
    # Win Rate
    print(f"  {'Win Rate':<25}", end="")
    for label in header_labels:
        s = all_results[label]["stats"]
        val = s["win_rate"] if s else 0
        print(f" {val:>{col_width}.1%}", end="")
    print()
    # Total Trades
    print(f"  {'Total Trades':<25}", end="")
    for label in header_labels:
        s = all_results[label]["stats"]
        val = int(s["total_trades"]) if s else 0
        print(f" {val:>{col_width}}", end="")
    print()
    # Max Drawdown
    print(f"  {'Max Drawdown':<25}", end="")
    for label in header_labels:
        s = all_results[label]["stats"]
        val = s["max_drawdown"] if s else 0
        print(f" {val:>{col_width}.1%}", end="")
    print()
    # Avg $ per trade (net)
    print(f"  {'Avg $/trade (net)':<25}", end="")
    for label in header_labels:
        t = all_results[label]["trades"]
        val = t["pnl"].mean() if len(t) > 0 else 0
        print(f" ${val:>{col_width-1}.0f}", end="")
    print()
    # Avg $ per trade (gross)
    print(f"  {'Avg $/trade (gross)':<25}", end="")
    for label in header_labels:
        t = all_results[label]["trades"]
        val = t["gross_pnl"].mean() if len(t) > 0 else 0
        print(f" ${val:>{col_width-1}.0f}", end="")
    print()
    # Total P&L
    print(f"  {'Total P&L ($)':<25}", end="")
    for label in header_labels:
        t = all_results[label]["trades"]
        val = t["pnl"].sum() if len(t) > 0 else 0
        print(f" ${val:>{col_width-1},.0f}", end="")
    print()
    # Total Costs
    print(f"  {'Total Costs ($)':<25}", end="")
    for label in header_labels:
        t = all_results[label]["trades"]
        val = t["costs"].sum() if len(t) > 0 else 0
        print(f" ${val:>{col_width-1},.0f}", end="")
    print()
    # Cost as % of gross
    print(f"  {'Costs as % of Gross':<25}", end="")
    for label in header_labels:
        t = all_results[label]["trades"]
        gross = t["gross_pnl"].sum() if len(t) > 0 else 0
        costs = t["costs"].sum() if len(t) > 0 else 0
        pct = costs / gross * 100 if gross > 0 else 0
        print(f" {pct:>{col_width}.1f}%", end="")
    print()
    # Topstep section
    print(f"\n  {'--- TOPSTEP SIM ---':<25}")
    print(f"  {'Pass Rate':<25}", end="")
    for label in header_labels:
        tp = all_results[label]["topstep"]
        val = tp["pass_rate"] if tp else 0
        print(f" {val:>{col_width}.1f}%", end="")
    print()
    print(f"  {'Pass/Total':<25}", end="")
    for label in header_labels:
        tp = all_results[label]["topstep"]
        if tp:
            print(f" {tp['passed']}/{tp['total']:>{col_width-len(str(tp['passed']))-1}}", end="")
        else:
            print(f" {'N/A':>{col_width}}", end="")
    print()
    print(f"  {'Avg trades to pass':<25}", end="")
    for label in header_labels:
        tp = all_results[label]["topstep"]
        val = tp["avg_trades_to_pass"] if tp else 0
        print(f" {val:>{col_width}.1f}", end="")
    print()
    # =============================================
    # VERDICT
    # =============================================
    print("\n" + "=" * 80)
    print("VERDICT — OPTIMAL CONFIGURATION FOR TOPSTEP")
    print("=" * 80)
    # Find best daily-hold variant by pass rate
    daily_variants = {k: v for k, v in all_results.items() if "Daily" in k}
    best_daily = max(daily_variants.items(), key=lambda x: x[1]["topstep"]["pass_rate"] if x[1]["topstep"] else 0)
    best_label = best_daily[0]
    best_stats = best_daily[1]["stats"]
    best_topstep = best_daily[1]["topstep"]
    weekly_es = all_results.get("1 ES Weekly", {})
    weekly_stats = weekly_es.get("stats", {})
    weekly_topstep = weekly_es.get("topstep", {})
    print(f"\n  Best daily-hold variant: {best_label}")
    print(f"    PF: {best_stats['profit_factor']:.3f}")
    print(f"    Pass Rate: {best_topstep['pass_rate']:.1f}%")
    print(f"    Avg trades to pass: {best_topstep['avg_trades_to_pass']:.1f}")
    if weekly_topstep:
        print(f"\n  vs Weekly hold (1 ES, Phidias-compatible):")
        print(f"    PF: {weekly_stats['profit_factor']:.3f}")
        print(f"    Pass Rate: {weekly_topstep['pass_rate']:.1f}%")
        print(f"    Avg trades to pass: {weekly_topstep['avg_trades_to_pass']:.1f}")
    print(f"\n  DECISION FRAMEWORK:")
    if best_topstep["pass_rate"] > 45:
        print(f"  ✅ {best_label} on Topstep is VIABLE ({best_topstep['pass_rate']:.1f}% pass rate)")
        print(f"     MES micros recover enough edge to make daily holds work.")
    elif best_topstep["pass_rate"] > 35:
        print(f"  ⚠️  {best_label} on Topstep is MARGINAL ({best_topstep['pass_rate']:.1f}% pass rate)")
        print(f"     Consider Phidias Swing account for weekly holds instead.")
    else:
        print(f"  ❌ Daily holds on Topstep are NOT VIABLE even with MES")
        print(f"     Best pass rate: {best_topstep['pass_rate']:.1f}%")
    if weekly_topstep and weekly_topstep["pass_rate"] > best_topstep["pass_rate"]:
        advantage = weekly_topstep["pass_rate"] - best_topstep["pass_rate"]
        print(f"\n  📊 Phidias weekly hold advantage: +{advantage:.1f}% pass rate")
        print(f"     Weekly hold on Phidias: {weekly_topstep['pass_rate']:.1f}%")
        print(f"     Best daily on Topstep: {best_topstep['pass_rate']:.1f}%")
        print(f"     The weekly hold is structurally superior for this strategy.")
    print("\n" + "=" * 80)
    # Save equity curves for best daily and weekly
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for label in [best_label, "1 ES Weekly"]:
        if label in all_results:
            t = all_results[label]["trades"]
            if len(t) > 0:
                pnls = t["pnl"].values
                eq_vals = [INITIAL_CAPITAL]
                for p in pnls:
                    eq_vals.append(eq_vals[-1] + p)
                dates = [t["entry_date"].iloc[0] - pd.Timedelta(days=1)]
                dates.extend(t["exit_date"].tolist())
                eq = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
                safe_label = label.lower().replace(" ", "_")
                plot_equity(eq, f"Week14 {label}")
                t.to_csv(os.path.join(RESULTS_DIR, f"week14_{safe_label}_trades.csv"), index=False)
                print(f"Saved {label} trades and equity curve")
if __name__ == "__main__":
    run()
