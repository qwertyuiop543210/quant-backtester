"""Phidias Account Comparison Backtest.

Finds the fastest, cheapest path to a funded account.
Optimizes for: fewest trades to pass, lowest expected cost, highest pass rate.

Tests Week 1 & Week 4 (1 ES contract, Monday open -> Friday close) against
all 3 Phidias account tiers, with and without weekly stop losses.

Key difference from Topstep: Phidias drawdown is EOD only (checked after each
trade closes, not intraday trailing). No daily loss limit. Unlimited time.

Costs: $5 RT commission + $12.50 slippage per side = $30 total per trade.
ES point value = $50.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import profit_factor, win_rate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
ACTIVE_WEEKS = {1, 4}

# Phidias account definitions
ACCOUNTS = [
    {"name": "$50K", "capital": 50_000, "profit_target": 4_000, "max_dd": 2_500, "otp_cost": 144.60},
    {"name": "$100K", "capital": 100_000, "profit_target": 6_000, "max_dd": 3_000, "otp_cost": 180.00},
    {"name": "$150K", "capital": 150_000, "profit_target": 9_000, "max_dd": 4_500, "otp_cost": 224.60},
]

# Weekly stop levels to test per account
STOP_LEVELS = {
    "$50K": [None, 1_500, 2_000],
    "$100K": [None, 2_000, 2_500],
    "$150K": [None, 3_000, 3_500],
}


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


def build_trades(open_prices, close_prices, active_weeks, weekly_stop=None):
    """Build trade list for Week 1/4 strategy with optional weekly stop.

    Args:
        open_prices: ES open price series
        close_prices: ES close price series
        active_weeks: list of week dicts from find_trading_weeks
        weekly_stop: if set, close position if down more than this amount
                     at any daily close during the week (dollar amount, positive)

    Returns:
        list of trade dicts with pnl
    """
    trades = []
    for w in active_weeks:
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        entry_price = open_prices.iloc[mi]

        exit_price = close_prices.iloc[fi]
        exit_reason = "Friday close"

        if weekly_stop is not None:
            for day_idx in w["daily_indices"]:
                day_close = close_prices.iloc[day_idx]
                day_pnl_dollar = (day_close - entry_price) * ES_POINT_VALUE
                if day_pnl_dollar <= -weekly_stop and day_idx != fi:
                    exit_price = day_close
                    exit_reason = f"STOP (${day_pnl_dollar:,.0f})"
                    break

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": w["monday_date"],
            "exit_date": w["friday_date"],
            "week_of_month": w["week_of_month"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "exit_reason": exit_reason,
        })
    return trades


def simulate_phidias(trade_pnls, capital, profit_target, max_dd):
    """Simulate Phidias eval attempts with EOD drawdown.

    Drawdown is EOD only: checked once per completed trade (not intraday).
    High-water mark = highest EOD balance seen.
    Fail if (high_water - current_EOD_balance) >= max_dd.

    Args:
        trade_pnls: list of per-trade P&L values
        capital: starting account capital
        profit_target: profit needed to pass
        max_dd: maximum EOD drawdown before failure

    Returns:
        list of attempt dicts
    """
    attempts = []
    i = 0

    while i < len(trade_pnls):
        balance = capital
        high_water = capital
        start_trade = i
        status = "in_progress"

        while i < len(trade_pnls):
            balance += trade_pnls[i]
            # EOD check: update high water and check drawdown after trade closes
            high_water = max(high_water, balance)
            eod_dd = high_water - balance
            profit = balance - capital
            i += 1

            if eod_dd >= max_dd:
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
            "eod_dd_at_end": high_water - balance,
            "status": status,
        })

        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break

    return attempts


def run():
    """Run Phidias account comparison backtest."""
    print("=" * 100)
    print("  PHIDIAS ACCOUNT COMPARISON — Fastest Path to Funded")
    print("  Strategy: Week 1 & Week 4 (1 ES), EOD drawdown")
    print("=" * 100)

    print("\nLoading ES data (2012+ regime only)...")
    es = get_data("ES", start="2012-01-01")
    es = es[es.index >= "2012-01-01"]
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    all_weeks = find_trading_weeks(es.index)
    active = [w for w in all_weeks if w["week_of_month"] in ACTIVE_WEEKS]
    print(f"Total trading weeks: {len(all_weeks)}")
    print(f"Active weeks (1 & 4 only): {len(active)}")
    print(f"Costs: ${COST_PER_TRADE:.0f}/trade (${COMMISSION_RT} comm + ${2*SLIPPAGE_PER_SIDE} slippage)")

    # =========================================================================
    # Build trade lists for each stop level
    # =========================================================================
    all_stop_levels = sorted(set(
        stop for stops in STOP_LEVELS.values() for stop in stops if stop is not None
    ))
    trade_sets = {}

    # No-stop baseline
    baseline_trades = build_trades(open_, close, active, weekly_stop=None)
    trade_sets[None] = [t["pnl"] for t in baseline_trades]
    baseline_pnls = pd.Series(trade_sets[None])

    print(f"\nBaseline (no stop): {len(baseline_trades)} trades, "
          f"PF {profit_factor(baseline_pnls):.3f}, "
          f"WR {win_rate(baseline_pnls):.1%}, "
          f"Total P&L ${baseline_pnls.sum():,.0f}")

    # Stop-loss versions
    for stop in all_stop_levels:
        stop_trades = build_trades(open_, close, active, weekly_stop=stop)
        trade_sets[stop] = [t["pnl"] for t in stop_trades]
        sp = pd.Series(trade_sets[stop])
        print(f"Stop ${stop:,}: {len(stop_trades)} trades, "
              f"PF {profit_factor(sp):.3f}, "
              f"WR {win_rate(sp):.1%}, "
              f"Total P&L ${sp.sum():,.0f}")

    # =========================================================================
    # Run all configurations
    # =========================================================================
    results = []
    all_attempts = []
    trades_per_month = 2.0  # ~2 trades/month (W1 + W4)

    for acct in ACCOUNTS:
        for stop in STOP_LEVELS[acct["name"]]:
            pnls = trade_sets[stop]
            attempts = simulate_phidias(
                pnls,
                capital=acct["capital"],
                profit_target=acct["profit_target"],
                max_dd=acct["max_dd"],
            )

            passed = [a for a in attempts if a["status"] == "PASSED"]
            failed = [a for a in attempts if a["status"] == "FAILED"]
            total_att = len(passed) + len(failed)
            pass_rate = len(passed) / total_att * 100 if total_att > 0 else 0

            avg_trades_pass = np.mean([a["trades_taken"] for a in passed]) if passed else 0
            avg_trades_fail = np.mean([a["trades_taken"] for a in failed]) if failed else 0

            # Expected attempts to pass = 1 / (pass_rate/100)
            if pass_rate > 0:
                exp_attempts = 100.0 / pass_rate
                # Expected weeks = (expected_fails * avg_weeks_per_fail) + avg_weeks_to_pass
                exp_fails = exp_attempts - 1
                exp_weeks = exp_fails * (avg_trades_fail / trades_per_month * 4.3) + \
                            (avg_trades_pass / trades_per_month * 4.3)
                exp_months = exp_weeks / 4.3
                exp_cost = exp_attempts * acct["otp_cost"]
            else:
                exp_attempts = float("inf")
                exp_weeks = float("inf")
                exp_months = float("inf")
                exp_cost = float("inf")

            stop_label = f"${stop:,}" if stop else "None"
            config = {
                "account": acct["name"],
                "stop": stop_label,
                "stop_val": stop,
                "pass_rate": pass_rate,
                "passed": len(passed),
                "failed": len(failed),
                "total_attempts": total_att,
                "avg_trades_pass": avg_trades_pass,
                "avg_trades_fail": avg_trades_fail,
                "exp_attempts": exp_attempts,
                "exp_weeks": exp_weeks,
                "exp_months": exp_months,
                "exp_cost": exp_cost,
                "otp_cost": acct["otp_cost"],
                "capital": acct["capital"],
                "profit_target": acct["profit_target"],
                "max_dd": acct["max_dd"],
            }
            results.append(config)

            # Save individual attempts
            for a in attempts:
                a["account"] = acct["name"]
                a["stop"] = stop_label
                all_attempts.append(a)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"  Phidias Account Comparison — Fastest Path to Funded")
    print(f"  Strategy: Week 1 & Week 4 (1 ES), EOD drawdown")
    print(f"{'='*100}")
    print()

    hdr = (f"  {'Account':<9} {'Stop':<9} {'Pass':>7} {'Passed':>7} {'Failed':>7} "
           f"{'Avg Trades':>11} {'Exp':>10} {'Exp Weeks':>11} {'Exp Months':>11} {'Exp Cost':>11}")
    print(hdr)
    sub = (f"  {'':>9} {'':>9} {'Rate':>7} {'':>7} {'':>7} "
           f"{'To Pass':>11} {'Attempts':>10} {'To Fund':>11} {'To Fund':>11} {'To Fund':>11}")
    print(sub)
    print(f"  {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*11} {'-'*10} {'-'*11} {'-'*11} {'-'*11}")

    for r in results:
        if r["pass_rate"] > 0:
            print(f"  {r['account']:<9} {r['stop']:<9} {r['pass_rate']:>6.1f}% "
                  f"{r['passed']:>7} {r['failed']:>7} "
                  f"{r['avg_trades_pass']:>11.1f} {r['exp_attempts']:>10.1f} "
                  f"{r['exp_weeks']:>11.1f} {r['exp_months']:>11.1f} "
                  f"${r['exp_cost']:>9,.0f}")
        else:
            print(f"  {r['account']:<9} {r['stop']:<9} {r['pass_rate']:>6.1f}% "
                  f"{r['passed']:>7} {r['failed']:>7} "
                  f"{'N/A':>11} {'N/A':>10} {'N/A':>11} {'N/A':>11} {'N/A':>11}")

    print(f"{'='*100}")

    # =========================================================================
    # DETAILED ATTEMPT LOG per configuration
    # =========================================================================
    for r in results:
        acct_attempts = [a for a in all_attempts
                         if a["account"] == r["account"] and a["stop"] == r["stop"]]
        print(f"\n  {r['account']} | Stop: {r['stop']} | "
              f"Target: ${r['profit_target']:,} | DD Limit: ${r['max_dd']:,}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'EOD DD':>10}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*10}")
        for a in acct_attempts:
            print(f"  {a['attempt']:>3} {a['status']:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['eod_dd_at_end']:>9,.0f}")

    # =========================================================================
    # RECOMMENDATION
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"  RECOMMENDATION — Fastest & Cheapest Path to Funded")
    print(f"{'='*100}")

    # Filter to configs with at least 1 pass
    viable = [r for r in results if r["pass_rate"] > 0]

    if not viable:
        print("\n  No configuration produced any passes. Strategy may not be viable for Phidias.")
        print(f"{'='*100}")
        return

    # Rank by fastest (lowest exp months)
    by_speed = sorted(viable, key=lambda r: r["exp_months"])
    # Rank by cheapest (lowest exp cost)
    by_cost = sorted(viable, key=lambda r: r["exp_cost"])

    print(f"\n  RANKED BY SPEED (fastest to fund):")
    print(f"  {'Rank':>5} {'Account':<9} {'Stop':<9} {'Months':>8} {'Cost':>10} {'Pass Rate':>10}")
    print(f"  {'-'*5} {'-'*9} {'-'*9} {'-'*8} {'-'*10} {'-'*10}")
    for i, r in enumerate(by_speed):
        print(f"  {i+1:>5} {r['account']:<9} {r['stop']:<9} "
              f"{r['exp_months']:>8.1f} ${r['exp_cost']:>9,.0f} {r['pass_rate']:>9.1f}%")

    print(f"\n  RANKED BY COST (cheapest to fund):")
    print(f"  {'Rank':>5} {'Account':<9} {'Stop':<9} {'Cost':>10} {'Months':>8} {'Pass Rate':>10}")
    print(f"  {'-'*5} {'-'*9} {'-'*9} {'-'*10} {'-'*8} {'-'*10}")
    for i, r in enumerate(by_cost):
        print(f"  {i+1:>5} {r['account']:<9} {r['stop']:<9} "
              f"${r['exp_cost']:>9,.0f} {r['exp_months']:>8.1f} {r['pass_rate']:>9.1f}%")

    # Flags
    print(f"\n  FLAGS:")
    for r in results:
        if 0 < r["pass_rate"] < 30:
            print(f"  !! HIGH RISK: {r['account']} Stop={r['stop']} — "
                  f"{r['pass_rate']:.1f}% pass rate, likely multiple expensive failures")

    # Check if cheaper tier costs more than next tier due to retries
    account_best = {}
    for r in viable:
        acct = r["account"]
        if acct not in account_best or r["exp_cost"] < account_best[acct]["exp_cost"]:
            account_best[acct] = r

    acct_order = ["$50K", "$100K", "$150K"]
    for i in range(len(acct_order) - 1):
        lower = acct_order[i]
        higher = acct_order[i + 1]
        if lower in account_best and higher in account_best:
            if account_best[lower]["exp_cost"] > account_best[higher]["exp_cost"]:
                print(f"  !! SKIP {lower}: Best {lower} config costs "
                      f"${account_best[lower]['exp_cost']:,.0f} vs "
                      f"${account_best[higher]['exp_cost']:,.0f} for {higher}. "
                      f"Go straight to {higher}.")

    # Best overall = best combined rank (speed + cost)
    rank_speed = {id(r): i for i, r in enumerate(by_speed)}
    rank_cost = {id(r): i for i, r in enumerate(by_cost)}
    combined = sorted(viable, key=lambda r: rank_speed[id(r)] + rank_cost[id(r)])
    best = combined[0]

    print(f"\n  SWEET SPOT: {best['account']} with stop={best['stop']}")
    print(f"    Pass rate:       {best['pass_rate']:.1f}%")
    print(f"    Exp attempts:    {best['exp_attempts']:.1f}")
    print(f"    Exp months:      {best['exp_months']:.1f}")
    print(f"    Exp cost:        ${best['exp_cost']:,.0f}")
    print(f"    Avg trades/pass: {best['avg_trades_pass']:.1f}")

    # Check for tradeoffs between top 2
    if len(combined) >= 2:
        second = combined[1]
        if best["exp_cost"] > second["exp_cost"] and best["exp_months"] < second["exp_months"]:
            delta_cost = best["exp_cost"] - second["exp_cost"]
            delta_months = second["exp_months"] - best["exp_months"]
            print(f"\n  TRADEOFF: {best['account']}+{best['stop']} is "
                  f"${delta_cost:,.0f} more expensive but {delta_months:.1f} months faster "
                  f"than {second['account']}+{second['stop']}")
        elif best["exp_months"] > second["exp_months"] and best["exp_cost"] < second["exp_cost"]:
            delta_cost = second["exp_cost"] - best["exp_cost"]
            delta_months = best["exp_months"] - second["exp_months"]
            print(f"\n  TRADEOFF: {best['account']}+{best['stop']} is "
                  f"${delta_cost:,.0f} cheaper but {delta_months:.1f} months slower "
                  f"than {second['account']}+{second['stop']}")

    print(f"\n{'='*100}")

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save all attempts
    attempts_df = pd.DataFrame(all_attempts)
    csv_path = os.path.join(RESULTS_DIR, "phidias_account_comparison.csv")
    attempts_df.to_csv(csv_path, index=False)
    print(f"\nAttempt details saved to {csv_path}")

    # Save summary table
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("  Phidias Account Comparison — Fastest Path to Funded")
    summary_lines.append("  Strategy: Week 1 & Week 4 (1 ES), EOD drawdown")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    summary_lines.append(hdr)
    summary_lines.append(sub)
    summary_lines.append(f"  {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7} "
                         f"{'-'*11} {'-'*10} {'-'*11} {'-'*11} {'-'*11}")
    for r in results:
        if r["pass_rate"] > 0:
            summary_lines.append(
                f"  {r['account']:<9} {r['stop']:<9} {r['pass_rate']:>6.1f}% "
                f"{r['passed']:>7} {r['failed']:>7} "
                f"{r['avg_trades_pass']:>11.1f} {r['exp_attempts']:>10.1f} "
                f"{r['exp_weeks']:>11.1f} {r['exp_months']:>11.1f} "
                f"${r['exp_cost']:>9,.0f}")
        else:
            summary_lines.append(
                f"  {r['account']:<9} {r['stop']:<9} {r['pass_rate']:>6.1f}% "
                f"{r['passed']:>7} {r['failed']:>7} "
                f"{'N/A':>11} {'N/A':>10} {'N/A':>11} {'N/A':>11} {'N/A':>11}")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    summary_lines.append(f"SWEET SPOT: {best['account']} with stop={best['stop']}")
    summary_lines.append(f"  Pass rate: {best['pass_rate']:.1f}%, "
                         f"Exp months: {best['exp_months']:.1f}, "
                         f"Exp cost: ${best['exp_cost']:,.0f}")

    txt_path = os.path.join(RESULTS_DIR, "phidias_recommendation.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Summary saved to {txt_path}")


if __name__ == "__main__":
    run()
