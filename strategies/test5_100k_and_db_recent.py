"""Test 5 — Recent DB trades + Phidias $100K vs $50K comparison.

Task 1: Print detailed DB trades in 2025-2026 and CO trades in 2026.
Task 2: Phidias $100K Swing simulation (2 contracts).
Task 3: Side-by-side $50K vs $100K comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
np.random.seed(42)

import pandas as pd

from strategies.validation_helpers import (
    load_data, generate_co_trades, generate_db_trades, build_combined_trades,
    ES_POINT_VALUE,
)


# ── Simulation engine (parameterized) ────────────────────────────────────────

def simulate_phidias(trades_df, starting_balance, profit_target, eod_drawdown):
    """Simulate sequential Phidias evaluation attempts with closed-P&L drawdown."""
    trades = trades_df.sort_values("exit_date").reset_index(drop=True)
    attempts = []
    trade_idx = 0

    while trade_idx < len(trades):
        balance = starting_balance
        hwm = starting_balance
        closed_pnl = 0.0
        attempt_start = trades.iloc[trade_idx]["entry_date"]
        attempt_trades = 0
        attempt_max_dd = 0.0
        status = None
        end_date = None

        while trade_idx < len(trades):
            trade = trades.iloc[trade_idx]
            trade_pnl = trade["pnl"]

            closed_pnl += trade_pnl
            balance = starting_balance + closed_pnl
            hwm = max(hwm, balance)
            dd = hwm - balance
            attempt_max_dd = max(attempt_max_dd, dd)
            attempt_trades += 1
            trade_idx += 1

            if dd >= eod_drawdown:
                status = "FAILED"
                end_date = trade["exit_date"]
                break

            if balance >= starting_balance + profit_target:
                status = "PASSED"
                end_date = trade["exit_date"]
                break

        if status is None:
            status = "INCOMPLETE"
            end_date = trades.iloc[trade_idx - 1]["exit_date"]

        attempts.append({
            "attempt": len(attempts) + 1,
            "status": status,
            "trades": attempt_trades,
            "pnl": closed_pnl,
            "max_drawdown": attempt_max_dd,
            "start_date": attempt_start,
            "end_date": end_date,
            "final_balance": balance,
            "high_water": hwm,
        })

    return attempts


def analyze_attempts(attempts, label, trades_per_month, otp_cost):
    """Compute summary stats from attempt list."""
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]

    total = len(passed) + len(failed)
    pass_rate = len(passed) / total if total > 0 else 0.0
    avg_trades_to_pass = np.mean([a["trades"] for a in passed]) if passed else 0.0
    avg_months_to_pass = avg_trades_to_pass / trades_per_month if trades_per_month > 0 else 0.0
    expected_cost = (1.0 / pass_rate) * otp_cost if pass_rate > 0 else float("inf")

    max_consec = 0
    cur_consec = 0
    for a in attempts:
        if a["status"] == "FAILED":
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    return {
        "label": label,
        "total": total,
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": pass_rate,
        "avg_trades_to_pass": avg_trades_to_pass,
        "avg_months_to_pass": avg_months_to_pass,
        "expected_cost": expected_cost,
        "max_consec_fails": max_consec,
    }


# ── Recompute P&L for N contracts ────────────────────────────────────────────

def recompute_pnl(trades_df, contracts, cost_per_contract=30.0):
    """Recompute P&L column for a given number of contracts.

    P&L = pnl_points × $50 × contracts - cost_per_contract × contracts
    """
    df = trades_df.copy()
    df["pnl"] = df["pnl_points"] * ES_POINT_VALUE * contracts - cost_per_contract * contracts
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    print("Loading data...")
    es, vix = load_data()

    # Generate base trades (1 contract, $30 cost — default)
    co_trades = generate_co_trades(es, vix)
    db_trades = generate_db_trades(es, vix)
    combined_trades = build_combined_trades(es, vix)

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1: RECENT DB TRADES
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TASK 1: RECENT DB TRADES (2025-2026)")
    print("=" * 90)

    db_recent = db_trades[
        (db_trades["entry_date"].dt.year >= 2025) &
        (db_trades["entry_date"].dt.year <= 2026)
    ].copy()

    if len(db_recent) > 0:
        print(f"\n  DB Trades in 2025-2026: {len(db_recent)}")
        print(f"\n  {'Date (entry)':>14s} | {'Date (exit)':>14s} | {'Entry Price':>12s} | "
              f"{'Exit Price':>11s} | {'Hold Days':>9s} | {'Exit Reason':>11s} | "
              f"{'VIX at Sig':>10s} | {'P&L':>10s}")
        print("  " + "-" * 105)
        for _, t in db_recent.iterrows():
            print(f"  {t['entry_date'].strftime('%Y-%m-%d'):>14s} | "
                  f"{t['exit_date'].strftime('%Y-%m-%d'):>14s} | "
                  f"{t['entry_price']:>12,.2f} | "
                  f"{t['exit_price']:>11,.2f} | "
                  f"{int(t['hold_days']):>9d} | "
                  f"{t['exit_reason']:>11s} | "
                  f"{t['vix_at_entry']:>10.2f} | "
                  f"${t['pnl']:>+9,.2f}")
        print(f"\n  Total DB P&L (2025-2026): ${db_recent['pnl'].sum():+,.2f}")
    else:
        print("\n  No DB trades found in 2025-2026.")

    # CO trades in 2026
    print(f"\n  {'─' * 50}")
    co_2026 = co_trades[co_trades["entry_date"].dt.year == 2026].copy()
    if len(co_2026) > 0:
        print(f"\n  CO Trades in 2026 (partial year): {len(co_2026)}")
        print(f"\n  {'Date (entry)':>14s} | {'Date (exit)':>14s} | {'Entry Price':>12s} | "
              f"{'Exit Price':>11s} | {'Hold Days':>9s} | {'Exit Reason':>11s} | "
              f"{'VIX at Sig':>10s} | {'P&L':>10s}")
        print("  " + "-" * 105)
        for _, t in co_2026.iterrows():
            hold = (t["exit_date"] - t["entry_date"]).days
            print(f"  {t['entry_date'].strftime('%Y-%m-%d'):>14s} | "
                  f"{t['exit_date'].strftime('%Y-%m-%d'):>14s} | "
                  f"{t['entry_price']:>12,.2f} | "
                  f"{t['exit_price']:>11,.2f} | "
                  f"{hold:>9d} | "
                  f"{'week_hold':>11s} | "
                  f"{t['vix_at_entry']:>10.2f} | "
                  f"${t['pnl']:>+9,.2f}")
        print(f"\n  Total CO P&L (2026): ${co_2026['pnl'].sum():+,.2f}")
    else:
        print("\n  No CO trades found in 2026.")

    print(f"\n  Next potential DB signal requires: RSI(2) < 10 on ES close AND "
          f"VIX between 20-35 AND not in Week 1 or Week 4.")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 2: PHIDIAS $100K SWING SIMULATION (2 contracts)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TASK 2: PHIDIAS $100K SWING SIMULATION (2 contracts)")
    print("=" * 90)

    BALANCE_100K = 100_000.0
    TARGET_100K = 6_000.0
    DRAWDOWN_100K = 3_000.0
    OTP_100K = 180.0
    CONTRACTS_100K = 2

    print(f"\n  Rules: ${BALANCE_100K:,.0f} starting balance")
    print(f"         ${TARGET_100K:,.0f} profit target")
    print(f"         ${DRAWDOWN_100K:,.0f} trailing drawdown on CLOSED P&L (from HWM)")
    print(f"         ${OTP_100K:.2f} per evaluation attempt")
    print(f"         {CONTRACTS_100K} mini ES contracts")
    print(f"         P&L per trade = point_change × $100 - $60")

    # Recompute P&L for 2 contracts
    co_2c = recompute_pnl(co_trades, CONTRACTS_100K)
    db_2c = recompute_pnl(db_trades, CONTRACTS_100K)

    # Build combined from recomputed trades (merge by exit_date)
    comb_2c = pd.concat([co_2c, db_2c], ignore_index=True)
    comb_2c = comb_2c.sort_values("exit_date").reset_index(drop=True)

    print(f"\n  CO trades (2c): {len(co_2c)}, avg P&L: ${co_2c['pnl'].mean():+,.2f}")
    print(f"  DB trades (2c): {len(db_2c)}, avg P&L: ${db_2c['pnl'].mean():+,.2f}")
    print(f"  Combined trades (2c): {len(comb_2c)}, avg P&L: ${comb_2c['pnl'].mean():+,.2f}")

    # Run simulations
    print("\n  Running $100K simulations...")
    co_100k_attempts = simulate_phidias(co_2c, BALANCE_100K, TARGET_100K, DRAWDOWN_100K)
    db_100k_attempts = simulate_phidias(db_2c, BALANCE_100K, TARGET_100K, DRAWDOWN_100K)
    comb_100k_attempts = simulate_phidias(comb_2c, BALANCE_100K, TARGET_100K, DRAWDOWN_100K)

    # Trades per month
    date_range_months = (es.index[-1] - es.index[0]).days / 365.25 * 12
    co_tpm = len(co_trades) / date_range_months
    db_tpm = len(db_trades) / date_range_months
    comb_tpm = len(comb_2c) / date_range_months

    co_100k = analyze_attempts(co_100k_attempts, "CO-Only", co_tpm, OTP_100K)
    db_100k = analyze_attempts(db_100k_attempts, "DB-Only", db_tpm, OTP_100K)
    comb_100k = analyze_attempts(comb_100k_attempts, "Combined", comb_tpm, OTP_100K)

    # Print comparison table
    print(f"\n  $100K Account — 2 Contracts")
    print(f"\n  {'Metric':<26s} | {'CO-Only':>10s} | {'DB-Only':>10s} | {'Combined':>10s}")
    print("  " + "-" * 66)

    for metric_name, fmt_fn in [
        ("Total attempts", lambda s: f"{s['total']:>10d}"),
        ("Passed", lambda s: f"{s['passed']:>10d}"),
        ("Failed", lambda s: f"{s['failed']:>10d}"),
        ("Pass rate", lambda s: f"{s['pass_rate']:>9.1%} "),
        ("Avg trades to pass", lambda s: f"{s['avg_trades_to_pass']:>10.1f}" if s['passed'] else f"{'N/A':>10s}"),
        ("Avg months to pass", lambda s: f"{s['avg_months_to_pass']:>10.1f}" if s['passed'] else f"{'N/A':>10s}"),
        ("Expected eval cost", lambda s: "${:,.0f}".format(s["expected_cost"]).rjust(10) if s['pass_rate'] > 0 else "       $--"),
        ("Max consecutive fails", lambda s: f"{s['max_consec_fails']:>10d}"),
    ]:
        print(f"  {metric_name:<26s} | {fmt_fn(co_100k)} | {fmt_fn(db_100k)} | {fmt_fn(comb_100k)}")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 3: SIDE-BY-SIDE $50K vs $100K COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TASK 3: $50K vs $100K SIDE-BY-SIDE COMPARISON")
    print("=" * 90)

    BALANCE_50K = 50_000.0
    TARGET_50K = 4_000.0
    DRAWDOWN_50K = 2_500.0
    OTP_50K = 144.60
    CONTRACTS_50K = 1

    # $50K uses 1-contract trades (original P&L)
    print("\n  Running $50K simulations (1 contract)...")
    co_50k_attempts = simulate_phidias(co_trades, BALANCE_50K, TARGET_50K, DRAWDOWN_50K)
    db_50k_attempts = simulate_phidias(db_trades, BALANCE_50K, TARGET_50K, DRAWDOWN_50K)
    comb_50k_attempts = simulate_phidias(combined_trades, BALANCE_50K, TARGET_50K, DRAWDOWN_50K)

    co_50k = analyze_attempts(co_50k_attempts, "CO-Only", co_tpm, OTP_50K)
    db_50k = analyze_attempts(db_50k_attempts, "DB-Only", db_tpm, OTP_50K)
    comb_50k = analyze_attempts(comb_50k_attempts, "Combined", comb_tpm, OTP_50K)

    # Print $50K table for reference
    print(f"\n  $50K Account — 1 Contract")
    print(f"\n  {'Metric':<26s} | {'CO-Only':>10s} | {'DB-Only':>10s} | {'Combined':>10s}")
    print("  " + "-" * 66)

    for metric_name, fmt_fn in [
        ("Total attempts", lambda s: f"{s['total']:>10d}"),
        ("Passed", lambda s: f"{s['passed']:>10d}"),
        ("Failed", lambda s: f"{s['failed']:>10d}"),
        ("Pass rate", lambda s: f"{s['pass_rate']:>9.1%} "),
        ("Avg trades to pass", lambda s: f"{s['avg_trades_to_pass']:>10.1f}" if s['passed'] else f"{'N/A':>10s}"),
        ("Avg months to pass", lambda s: f"{s['avg_months_to_pass']:>10.1f}" if s['passed'] else f"{'N/A':>10s}"),
        ("Expected eval cost", lambda s: "${:,.0f}".format(s["expected_cost"]).rjust(10) if s['pass_rate'] > 0 else "       $--"),
        ("Max consecutive fails", lambda s: f"{s['max_consec_fails']:>10d}"),
    ]:
        print(f"  {metric_name:<26s} | {fmt_fn(co_50k)} | {fmt_fn(db_50k)} | {fmt_fn(comb_50k)}")

    # ── Final Comparison Table (Combined strategy) ────────────────────────
    print(f"\n  {'─' * 90}")
    print(f"  FINAL COMPARISON — Combined Strategy")
    print(f"  {'─' * 90}")

    # Expected cost to fund = eval_cost / pass_rate = expected_eval_cost
    cost_to_fund_50k = comb_50k["expected_cost"]
    cost_to_fund_100k = comb_100k["expected_cost"]

    # Annual P&L: trades per year × avg P&L per trade
    trades_per_year = comb_tpm * 12
    avg_pnl_1c = combined_trades["pnl"].mean()
    avg_pnl_2c = comb_2c["pnl"].mean()
    annual_pnl_50k = trades_per_year * avg_pnl_1c
    annual_pnl_100k = trades_per_year * avg_pnl_2c

    # Time to $75K withdrawal cap
    months_to_75k_50k = 75_000 / (annual_pnl_50k / 12) if annual_pnl_50k > 0 else float("inf")
    months_to_75k_100k = 75_000 / (annual_pnl_100k / 12) if annual_pnl_100k > 0 else float("inf")

    print(f"\n  {'Account':<10s} | {'Contracts':>9s} | {'Target':>8s} | {'Drawdown':>8s} | "
          f"{'Pass Rate':>9s} | {'Eval Cost':>10s} | {'Cost to Fund':>13s} | {'Avg Months':>10s}")
    print("  " + "-" * 95)
    print(f"  {'$50K':<10s} | {CONTRACTS_50K:>9d} | {'$4,000':>8s} | {'$2,500':>8s} | "
          f"{comb_50k['pass_rate']:>8.1%} | {'$144.60':>10s} | "
          f"${cost_to_fund_50k:>11,.0f} | {comb_50k['avg_months_to_pass']:>10.1f}")
    print(f"  {'$100K':<10s} | {CONTRACTS_100K:>9d} | {'$6,000':>8s} | {'$3,000':>8s} | "
          f"{comb_100k['pass_rate']:>8.1%} | {'$180.00':>10s} | "
          f"${cost_to_fund_100k:>11,.0f} | {comb_100k['avg_months_to_pass']:>10.1f}")

    # ── Recommendation ────────────────────────────────────────────────────
    print(f"\n  {'─' * 90}")
    print("  RECOMMENDATION")
    print(f"  {'─' * 90}")

    print(f"\n  Annual P&L once funded:")
    print(f"    $50K  (1 contract): ~${annual_pnl_50k:,.0f}/yr")
    print(f"    $100K (2 contracts): ~${annual_pnl_100k:,.0f}/yr")

    print(f"\n  Time to $75K cumulative withdrawal (triggers LIVE upgrade):")
    print(f"    $50K:  {months_to_75k_50k:.1f} months ({months_to_75k_50k / 12:.1f} years)")
    print(f"    $100K: {months_to_75k_100k:.1f} months ({months_to_75k_100k / 12:.1f} years)")

    print(f"\n  Cost to get funded (eval cost / pass rate):")
    print(f"    $50K:  ${cost_to_fund_50k:,.0f}")
    print(f"    $100K: ${cost_to_fund_100k:,.0f}")

    # Determine winner
    if comb_100k["pass_rate"] > 0 and comb_50k["pass_rate"] > 0:
        roi_50k = annual_pnl_50k / cost_to_fund_50k if cost_to_fund_50k > 0 else 0
        roi_100k = annual_pnl_100k / cost_to_fund_100k if cost_to_fund_100k > 0 else 0

        print(f"\n  ROI (annual P&L / cost to fund):")
        print(f"    $50K:  {roi_50k:.1f}x")
        print(f"    $100K: {roi_100k:.1f}x")

        if cost_to_fund_100k <= cost_to_fund_50k * 1.5 and annual_pnl_100k >= annual_pnl_50k * 1.5:
            winner = "$100K"
            reason = ("The $100K account earns ~2× per trade with 2 contracts. "
                      "Even if the pass rate is slightly lower due to the tighter "
                      "drawdown-to-target ratio, the doubled earning power once funded "
                      "and faster path to the $75K LIVE upgrade make it the better value.")
        elif comb_100k["pass_rate"] >= comb_50k["pass_rate"] * 0.8:
            winner = "$100K"
            reason = ("The $100K account maintains a comparable pass rate while doubling "
                      "per-trade earnings. The higher eval cost ($180 vs $144.60) is offset "
                      "by reaching the $75K LIVE upgrade in roughly half the time.")
        else:
            winner = "$50K"
            reason = ("The $50K account has a significantly higher pass rate, making it "
                      "cheaper to get funded. While $100K earns more per trade, the higher "
                      "failure rate and tighter drawdown-to-target ratio erode the advantage.")

        print(f"\n  >>> RECOMMENDATION: {winner} account")
        print(f"  {reason}")
    else:
        print("\n  Cannot compare — one or both accounts have 0% pass rate.")

    print()


if __name__ == "__main__":
    main()
