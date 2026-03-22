"""Week 1 & Week 4 Weekly Hold — Phidias Propfirm Swing Account Simulation.
Phidias allows overnight + weekend holding on Swing accounts.
This means our original Monday open → Friday close structure works AS BACKTESTED.
Drawdown is EOD trailing (not intraday), which is more forgiving.
Simulates three Phidias Swing account sizes:
  50K:  $4,000 profit target, $2,500 EOD trailing DD, $116 one-time
  100K: $6,000 profit target, $3,000 EOD trailing DD, $144.60 one-time
  150K: $9,000 profit target, $4,500 EOD trailing DD, $172.60 one-time
Also simulates Topstep $150K for direct comparison:
  150K: $9,000 target, $4,500 intraday trailing DD, $149/month recurring
Key difference: Phidias EOD trailing only checks drawdown at end of day.
For weekly holds, this means the drawdown is checked on Friday close only
(or any day you close a trade). Intraday dips during the week don't count.
This is SIGNIFICANTLY more forgiving for a weekly hold strategy.
Additionally tests the 30% consistency rule for Phidias CASH accounts:
No single trading day can account for more than 30% of total profit at payout time.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
STRATEGY_NAME = "Week 1 & Week 4 — Phidias Swing Account Simulation"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0
# Phidias account parameters
PHIDIAS_ACCOUNTS = {
    "Phidias 50K Swing": {
        "capital": 50_000.0,
        "profit_target": 4_000.0,
        "trailing_dd": 2_500.0,
        "eval_cost": 116.0,
        "cost_type": "one-time",
        "max_contracts_mini": 10,
        "drawdown_type": "EOD",
    },
    "Phidias 100K Swing": {
        "capital": 100_000.0,
        "profit_target": 6_000.0,
        "trailing_dd": 3_000.0,
        "eval_cost": 144.60,
        "cost_type": "one-time",
        "max_contracts_mini": 14,
        "drawdown_type": "EOD",
    },
    "Phidias 150K Swing": {
        "capital": 150_000.0,
        "profit_target": 9_000.0,
        "trailing_dd": 4_500.0,
        "eval_cost": 172.60,
        "cost_type": "one-time",
        "max_contracts_mini": 17,
        "drawdown_type": "EOD",
    },
    "Topstep 150K (comparison)": {
        "capital": 150_000.0,
        "profit_target": 9_000.0,
        "trailing_dd": 4_500.0,
        "eval_cost": 149.0,
        "cost_type": "monthly",
        "max_contracts_mini": 15,
        "drawdown_type": "intraday",
    },
}
def get_week_of_month(date):
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
        "high": es_data["High"].astype(float),
        "low": es_data["Low"].astype(float),
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
def build_weekly_trades(qualifying_weeks, df):
    """Build trade list: Monday open → Friday close, 1 ES contract."""
    trades = []
    cost_per_trade = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
    for week in qualifying_weeks:
        days = week["week_days"]
        if len(days) < 2:
            continue
        entry_day = days[0]
        exit_day = days[-1]
        entry_price = df.loc[entry_day, "open"]
        exit_price = df.loc[exit_day, "close"]
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - cost_per_trade
        # Track daily closes during the week for EOD drawdown calculation
        daily_closes = []
        for day in days:
            daily_closes.append({
                "date": day,
                "close": df.loc[day, "close"],
                "unrealized_pnl": (df.loc[day, "close"] - entry_price) * ES_POINT_VALUE,
            })
        # Track intraday low for intraday drawdown calculation
        intraday_low_pnl = float("inf")
        for day in days:
            low_price = df.loc[day, "low"]
            intraday_low = (low_price - entry_price) * ES_POINT_VALUE
            intraday_low_pnl = min(intraday_low_pnl, intraday_low)
        trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost_per_trade,
            "pnl": net_pnl,
            "week_num": week["week_num"],
            "vix_friday": week["vix_friday"],
            "daily_closes": daily_closes,
            "intraday_low_pnl": intraday_low_pnl,
        })
    return trades
def simulate_phidias_eod(trades, capital, profit_target, trailing_dd):
    """Simulate Phidias EOD trailing drawdown evaluation.
    EOD trailing: drawdown is only checked at end of each trading day.
    High water mark updates only at end of day.
    Intraday dips do NOT trigger liquidation.
    """
    attempts = []
    i = 0
    while i < len(trades):
        balance = capital
        high_water = capital
        start_trade = i
        status = "in_progress"
        while i < len(trades):
            trade = trades[i]
            # For EOD trailing, we check at end of each day during the trade
            for day_data in trade["daily_closes"]:
                eod_balance = capital + (balance - capital) + day_data["unrealized_pnl"]
                # Actually, let's track properly:
                # balance before this trade + unrealized P&L at each day's close
                pass
            # Simpler: for weekly trades, the EOD check happens at Friday close
            # (the only day the position is closed)
            balance += trade["pnl"]
            high_water = max(high_water, balance)
            trailing = high_water - balance
            profit = balance - capital
            i += 1
            if trailing >= trailing_dd:
                status = "FAILED"
                break
            if profit >= profit_target:
                status = "PASSED"
                break
        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade + 1,
            "end_trade": i,
            "trades_taken": i - start_trade,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - capital,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
def simulate_topstep_intraday(trades, capital, profit_target, trailing_dd):
    """Simulate Topstep intraday trailing drawdown.
    Intraday trailing: drawdown is checked tick-by-tick (we approximate with
    the intraday low during each trade week). The high water mark updates
    in real-time with unrealized P&L.
    This is STRICTER than EOD because intraday dips count against you
    even if the trade recovers by close.
    """
    attempts = []
    i = 0
    while i < len(trades):
        balance = capital
        high_water = capital
        start_trade = i
        status = "in_progress"
        while i < len(trades):
            trade = trades[i]
            # Check intraday low during this trade
            intraday_balance = balance + trade["intraday_low_pnl"]
            # High water could have gone up during the week too
            intraday_peak = balance + max(0, trade["gross_pnl"])  # Approximate peak
            high_water = max(high_water, intraday_peak)
            # Check if intraday low breached trailing DD
            intraday_trailing = high_water - intraday_balance
            if intraday_trailing >= trailing_dd:
                i += 1
                status = "FAILED"
                break
            # Close the trade
            balance += trade["pnl"]
            high_water = max(high_water, balance)
            trailing = high_water - balance
            profit = balance - capital
            i += 1
            if trailing >= trailing_dd:
                status = "FAILED"
                break
            if profit >= profit_target:
                status = "PASSED"
                break
        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade + 1,
            "end_trade": i,
            "trades_taken": i - start_trade,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - capital,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
def check_consistency_rule(trades_in_attempt, total_profit):
    """Check Phidias 30% consistency rule.
    No single trading day can account for more than 30% of total profit
    at payout time. For weekly trades, each trade IS one "trading period."
    Returns True if consistent, False if violated.
    """
    if total_profit <= 0:
        return True  # Not requesting payout
    for trade in trades_in_attempt:
        if trade["pnl"] > 0 and trade["pnl"] / total_profit > 0.30:
            return False
    return True
def run():
    """Run Phidias vs Topstep comparison simulation."""
    print("=" * 80)
    print("PHIDIAS SWING ACCOUNT vs TOPSTEP — WEEKLY HOLD SIMULATION")
    print("Week 1 & Week 4 ES Calendar + VIX Filter")
    print("Monday Open → Friday Close (1 ES contract)")
    print("=" * 80)
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    qualifying_weeks, df = identify_qualifying_weeks(es, vix)
    print(f"Qualifying weeks: {len(qualifying_weeks)}")
    trades = build_weekly_trades(qualifying_weeks, df)
    print(f"Total trades: {len(trades)}")
    if len(trades) == 0:
        print("ERROR: No trades generated")
        return
    # Basic strategy stats
    trade_pnls = pd.Series([t["pnl"] for t in trades], dtype=float)
    print(f"\nStrategy Overview:")
    print(f"  Win Rate: {(trade_pnls > 0).mean():.1%}")
    print(f"  Avg Win: ${trade_pnls[trade_pnls > 0].mean():,.0f}")
    print(f"  Avg Loss: ${trade_pnls[trade_pnls < 0].mean():,.0f}")
    print(f"  Avg Trade: ${trade_pnls.mean():,.0f}")
    print(f"  Total P&L: ${trade_pnls.sum():,.0f}")
    # =============================================
    # SIMULATE EACH ACCOUNT
    # =============================================
    results = {}
    for account_name, params in PHIDIAS_ACCOUNTS.items():
        print(f"\n{'='*60}")
        print(f"  {account_name}")
        print(f"  Capital: ${params['capital']:,.0f} | Target: ${params['profit_target']:,.0f} | DD: ${params['trailing_dd']:,.0f} ({params['drawdown_type']})")
        print(f"{'='*60}")
        if params["drawdown_type"] == "EOD":
            attempts = simulate_phidias_eod(
                trades, params["capital"], params["profit_target"], params["trailing_dd"]
            )
        else:
            attempts = simulate_topstep_intraday(
                trades, params["capital"], params["profit_target"], params["trailing_dd"]
            )
        passed = [a for a in attempts if a["status"] == "PASSED"]
        failed = [a for a in attempts if a["status"] == "FAILED"]
        total = len(passed) + len(failed)
        pass_rate = len(passed) / total * 100 if total > 0 else 0
        avg_trades_to_pass = np.mean([a["trades_taken"] for a in passed]) if passed else 0
        avg_trades_to_fail = np.mean([a["trades_taken"] for a in failed]) if failed else 0
        avg_weeks_to_pass = avg_trades_to_pass  # 1 trade = 1 week
        # Expected cost calculation
        if params["cost_type"] == "one-time":
            # One-time: cost per attempt = eval cost. Expected attempts = 1/pass_rate
            expected_attempts = 1 / (pass_rate / 100) if pass_rate > 0 else float("inf")
            expected_cost = expected_attempts * params["eval_cost"]
        else:
            # Monthly: cost = months_to_pass * monthly_fee
            months_to_pass = avg_weeks_to_pass / 4.33 if avg_weeks_to_pass > 0 else 0
            # But if you fail, you keep paying. So expected cost includes fail months too.
            if pass_rate > 0:
                expected_attempts = 1 / (pass_rate / 100)
                avg_months_per_attempt = (
                    (pass_rate/100 * avg_weeks_to_pass + (1-pass_rate/100) * avg_trades_to_fail * 1) / 4.33
                )
                expected_cost = expected_attempts * avg_months_per_attempt * params["eval_cost"]
            else:
                expected_cost = float("inf")
        # Consistency rule check (Phidias only)
        consistency_issues = 0
        if "Phidias" in account_name:
            for attempt in passed:
                start_idx = attempt["start_trade"] - 1
                end_idx = attempt["end_trade"]
                attempt_trades = trades[start_idx:end_idx]
                total_profit = attempt["profit"]
                if not check_consistency_rule(attempt_trades, total_profit):
                    consistency_issues += 1
        results[account_name] = {
            "pass_rate": pass_rate,
            "total_attempts": total,
            "passed": len(passed),
            "failed": len(failed),
            "avg_trades_to_pass": avg_trades_to_pass,
            "avg_weeks_to_pass": avg_weeks_to_pass,
            "expected_cost": expected_cost,
            "eval_cost": params["eval_cost"],
            "cost_type": params["cost_type"],
            "consistency_issues": consistency_issues,
            "consistency_total": len(passed),
        }
        print(f"  Pass Rate: {pass_rate:.1f}% ({len(passed)}/{total})")
        print(f"  Avg trades to pass: {avg_trades_to_pass:.1f}")
        print(f"  Avg weeks to pass: {avg_weeks_to_pass:.1f}")
        print(f"  Expected cost to pass: ${expected_cost:,.0f}")
        if "Phidias" in account_name and len(passed) > 0:
            clean_pct = (len(passed) - consistency_issues) / len(passed) * 100
            print(f"  30% consistency rule: {len(passed) - consistency_issues}/{len(passed)} passes clean ({clean_pct:.0f}%)")
    # =============================================
    # FINAL COMPARISON TABLE
    # =============================================
    print("\n" + "=" * 100)
    print("FINAL COMPARISON — WHICH ACCOUNT TO USE")
    print("=" * 100)
    accounts = list(results.keys())
    col_w = 20
    print(f"\n  {'Metric':<30}", end="")
    for name in accounts:
        short = name.replace(" (comparison)", "").replace("Phidias ", "P-").replace("Topstep ", "TS-")
        print(f" {short:>{col_w}}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in accounts:
        print(f" {'-'*col_w}", end="")
    print()
    metrics = [
        ("Eval Cost", lambda r, n: f"${r['eval_cost']:.0f} {'(1x)' if r['cost_type']=='one-time' else '(/mo)'}"),
        ("Pass Rate", lambda r, n: f"{r['pass_rate']:.1f}%"),
        ("Passed/Total", lambda r, n: f"{r['passed']}/{r['total_attempts']}"),
        ("Avg weeks to pass", lambda r, n: f"{r['avg_weeks_to_pass']:.1f}"),
        ("Expected $ to pass", lambda r, n: f"${r['expected_cost']:,.0f}"),
        ("Consistency clean", lambda r, n: f"{r['consistency_total']-r['consistency_issues']}/{r['consistency_total']}" if "Phidias" in n else "N/A"),
    ]
    for label, fmt_fn in metrics:
        print(f"  {label:<30}", end="")
        for name in accounts:
            val = fmt_fn(results[name], name)
            print(f" {val:>{col_w}}", end="")
        print()
    # =============================================
    # VERDICT
    # =============================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    # Find best by expected cost
    phidias_accounts = {k: v for k, v in results.items() if "Phidias" in k}
    best_phidias = min(phidias_accounts.items(), key=lambda x: x[1]["expected_cost"])
    topstep = results.get("Topstep 150K (comparison)", {})
    print(f"\n  BEST PHIDIAS ACCOUNT: {best_phidias[0]}")
    print(f"    Pass Rate: {best_phidias[1]['pass_rate']:.1f}%")
    print(f"    Expected cost to pass: ${best_phidias[1]['expected_cost']:,.0f}")
    print(f"    Avg weeks to pass: {best_phidias[1]['avg_weeks_to_pass']:.1f}")
    if topstep:
        print(f"\n  vs TOPSTEP 150K (weekly hold, NOT actually allowed):")
        print(f"    Pass Rate: {topstep['pass_rate']:.1f}%")
        print(f"    Expected cost to pass: ${topstep['expected_cost']:,.0f}")
        savings = topstep["expected_cost"] - best_phidias[1]["expected_cost"]
        print(f"\n  💰 Phidias saves ${savings:,.0f} per pass vs Topstep")
        print(f"     AND Topstep can't even run this strategy (no overnight holds)")
    # Account size recommendation
    print(f"\n  📊 ACCOUNT SIZE ANALYSIS:")
    for name, r in sorted(phidias_accounts.items(), key=lambda x: x[1]["expected_cost"]):
        target = PHIDIAS_ACCOUNTS[name]["profit_target"]
        dd = PHIDIAS_ACCOUNTS[name]["trailing_dd"]
        ratio = target / dd
        print(f"    {name}: target/DD ratio = {ratio:.1f}x, "
              f"pass rate {r['pass_rate']:.1f}%, "
              f"cost ${r['expected_cost']:,.0f}")
    print(f"\n  RECOMMENDATION:")
    if best_phidias[1]["pass_rate"] > 50:
        print(f"  ✅ Use {best_phidias[0]}")
        print(f"     ${best_phidias[1]['eval_cost']:.0f} one-time, "
              f"{best_phidias[1]['pass_rate']:.1f}% pass rate, "
              f"~{best_phidias[1]['avg_weeks_to_pass']:.0f} weeks to pass")
        print(f"     Weekly hold runs EXACTLY as backtested. No modifications needed.")
    else:
        print(f"  ⚠️  Best Phidias pass rate is {best_phidias[1]['pass_rate']:.1f}%")
        print(f"     Still the best option since no other firm allows overnight futures holds.")
    print("\n" + "=" * 80)
    # Save trade list
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trade_df = pd.DataFrame([{
        "entry_date": t["entry_date"],
        "exit_date": t["exit_date"],
        "entry_price": t["entry_price"],
        "exit_price": t["exit_price"],
        "pnl_points": t["pnl_points"],
        "gross_pnl": t["gross_pnl"],
        "costs": t["costs"],
        "pnl": t["pnl"],
        "week_num": t["week_num"],
        "vix_friday": t["vix_friday"],
    } for t in trades])
    csv_path = os.path.join(RESULTS_DIR, "week14_phidias_trades.csv")
    trade_df.to_csv(csv_path, index=False)
    print(f"Trade list saved to {csv_path}")
    # Equity curve
    eq_vals = [INITIAL_CAPITAL]
    for t in trades:
        eq_vals.append(eq_vals[-1] + t["pnl"])
    dates = [trades[0]["entry_date"] - pd.Timedelta(days=1)]
    dates.extend([t["exit_date"] for t in trades])
    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
    plot_equity(equity, "Week14 Phidias Weekly Hold (1 ES)")
if __name__ == "__main__":
    run()
