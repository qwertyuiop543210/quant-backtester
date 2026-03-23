"""Test 4 — Phidias $50K Swing account pass rate simulation.

Simulates sequential evaluation attempts for CO-only, DB-only, and Combined
strategies under Phidias $50K Swing account rules:
  - $50,000 starting balance
  - $4,000 profit target (pass)
  - $2,500 trailing drawdown on CLOSED P&L from high-water mark (fail)
  - 1 mini ES contract, $144.60 OTP per attempt

IMPORTANT: Phidias assesses the drawdown on closed P&L only.  Unrealized
mark-to-market losses during a multi-day hold are invisible.  The balance
only updates when a trade is closed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
np.random.seed(42)

import pandas as pd

from strategies.validation_helpers import (
    load_data, generate_co_trades, generate_db_trades, build_combined_trades,
)

# ── Phidias $50K Swing Constants ─────────────────────────────────────────────

STARTING_BALANCE = 50_000.0
PROFIT_TARGET = 4_000.0
EOD_DRAWDOWN = 2_500.0
OTP_COST = 144.60
MONTHLY_FEE = 0.0
MAX_CONTRACTS = 1
ES_POINT_VALUE = 50.0
COST_PER_TRADE = 30.0


# ── Phidias simulation engine (closed P&L only) ─────────────────────────────

def simulate_phidias(trades_df):
    """Simulate sequential Phidias $50K Swing evaluation attempts.

    Iterates through trades sorted by exit_date (the date P&L is realized).
    Balance only updates when a trade closes.  Drawdown is assessed on
    closed P&L from the high-water mark — unrealized intraday/intraweek
    losses are invisible.

    Args:
        trades_df: DataFrame with at least exit_date and pnl columns,
                   sorted by exit_date.

    Returns:
        List of attempt dicts.
    """
    trades = trades_df.sort_values("exit_date").reset_index(drop=True)

    attempts = []
    trade_idx = 0

    while trade_idx < len(trades):
        balance = STARTING_BALANCE
        hwm = STARTING_BALANCE
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
            balance = STARTING_BALANCE + closed_pnl
            hwm = max(hwm, balance)
            dd = hwm - balance
            attempt_max_dd = max(attempt_max_dd, dd)
            attempt_trades += 1
            trade_idx += 1

            # Check FAIL (closed P&L drawdown from HWM)
            if dd >= EOD_DRAWDOWN:
                status = "FAILED"
                end_date = trade["exit_date"]
                break

            # Check PASS (profit target)
            if balance >= STARTING_BALANCE + PROFIT_TARGET:
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


# ── Analysis helpers ─────────────────────────────────────────────────────────

def analyze_attempts(attempts, label, trades_per_month):
    """Compute summary stats from attempt list."""
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    incomplete = [a for a in attempts if a["status"] == "INCOMPLETE"]

    total = len(passed) + len(failed)  # exclude incomplete
    pass_rate = len(passed) / total if total > 0 else 0.0
    avg_trades_to_pass = np.mean([a["trades"] for a in passed]) if passed else 0.0
    avg_pnl_at_fail = np.mean([a["pnl"] for a in failed]) if failed else 0.0
    expected_cost = (1.0 / pass_rate) * OTP_COST if pass_rate > 0 else float("inf")
    avg_months_to_pass = avg_trades_to_pass / trades_per_month if trades_per_month > 0 else 0.0

    # Max consecutive fails
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
        "incomplete": len(incomplete),
        "pass_rate": pass_rate,
        "avg_trades_to_pass": avg_trades_to_pass,
        "avg_months_to_pass": avg_months_to_pass,
        "avg_pnl_at_fail": avg_pnl_at_fail,
        "expected_cost": expected_cost,
        "max_consec_fails": max_consec,
        "attempts": attempts,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TEST 4 — Phidias $50K Swing Account Pass Rate Simulation")
    print("=" * 70)
    print(f"\n  Rules: ${STARTING_BALANCE:,.0f} starting balance")
    print(f"         ${PROFIT_TARGET:,.0f} profit target")
    print(f"         ${EOD_DRAWDOWN:,.0f} trailing drawdown on CLOSED P&L (from HWM)")
    print(f"         ${OTP_COST:.2f} per evaluation attempt")
    print(f"         {MAX_CONTRACTS} mini ES contract")

    # Load data
    print("\n  Loading data...")
    es, vix = load_data()

    # Generate trades
    print("\n  Generating trades...")
    co_trades = generate_co_trades(es, vix)
    db_trades = generate_db_trades(es, vix)
    combined_trades = build_combined_trades(es, vix)

    print(f"  CO trades: {len(co_trades)}")
    print(f"  DB trades: {len(db_trades)}")
    print(f"  Combined trades: {len(combined_trades)}")

    # ── Run simulations (closed P&L, sorted by exit_date) ────────────────
    print("\n  Running Phidias simulations (closed P&L only)...")

    co_attempts = simulate_phidias(co_trades)
    db_attempts = simulate_phidias(db_trades)
    comb_attempts = simulate_phidias(combined_trades)

    # ── Debug: first 3 failed attempts for Combined ──────────────────────
    print("\n  DEBUG — Trade-by-trade trace for first 3 FAILED Combined attempts:")
    printed = 0
    comb_sorted = combined_trades.sort_values("exit_date").reset_index(drop=True)
    # Replay to find failing trades
    tidx = 0
    for a in comb_attempts:
        if a["status"] == "FAILED" and printed < 3:
            print(f"\n  Attempt #{a['attempt']}: "
                  f"{a['start_date'].strftime('%Y-%m-%d')} to "
                  f"{a['end_date'].strftime('%Y-%m-%d')} "
                  f"({a['trades']} trades, max_dd=${a['max_drawdown']:,.0f})")
            bal = STARTING_BALANCE
            hwm_d = STARTING_BALANCE
            for k in range(tidx, tidx + a["trades"]):
                t = comb_sorted.iloc[k]
                bal += t["pnl"]
                hwm_d = max(hwm_d, bal)
                dd_d = hwm_d - bal
                tag_str = t.get("tag", "?")
                print(f"    {t['exit_date'].strftime('%Y-%m-%d')} [{tag_str:>2s}]: "
                      f"pnl=${t['pnl']:>+10,.2f}  "
                      f"bal=${bal:>12,.2f}  "
                      f"hwm=${hwm_d:>12,.2f}  "
                      f"dd=${dd_d:>8,.2f}  "
                      f"{'FAIL' if dd_d >= EOD_DRAWDOWN else 'ok'}")
            printed += 1
        tidx += a["trades"]

    # ── Verification: avg P&L at failure ─────────────────────────────────
    failed_comb = [a for a in comb_attempts if a["status"] == "FAILED"]
    if failed_comb:
        avg_fail_pnl = np.mean([a["pnl"] for a in failed_comb])
        max_dd_values = [a["max_drawdown"] for a in failed_comb]
        print(f"\n  Verification (Combined):")
        print(f"    Avg P&L at failure:  ${avg_fail_pnl:+,.0f}")
        print(f"    Avg max DD at fail:  ${np.mean(max_dd_values):,.0f}")
        print(f"    Max max DD at fail:  ${max(max_dd_values):,.0f}")
        print(f"    Min max DD at fail:  ${min(max_dd_values):,.0f}")

    # Compute trades per month
    date_range_years = (es.index[-1] - es.index[0]).days / 365.25
    date_range_months = date_range_years * 12
    co_tpm = len(co_trades) / date_range_months
    db_tpm = len(db_trades) / date_range_months
    comb_tpm = len(combined_trades) / date_range_months

    print(f"\n  CO trades/month: {co_tpm:.2f}")
    print(f"  DB trades/month: {db_tpm:.2f}")
    print(f"  Combined trades/month: {comb_tpm:.2f}")

    co_stats = analyze_attempts(co_attempts, "CO-Only", co_tpm)
    db_stats = analyze_attempts(db_attempts, "DB-Only", db_tpm)
    comb_stats = analyze_attempts(comb_attempts, "Combined", comb_tpm)

    # ── Output Table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PASS RATE COMPARISON")
    print("=" * 70)

    header = f"  {'Metric':<26s} | {'CO-Only':>10s} | {'DB-Only':>10s} | {'Combined':>10s}"
    sep = "  " + "-" * 66
    print(header)
    print(sep)

    for stats_list in [
        ("Total attempts", lambda s: f"{s['total']:>10d}"),
        ("Passed", lambda s: f"{s['passed']:>10d}"),
        ("Failed", lambda s: f"{s['failed']:>10d}"),
        ("Pass rate", lambda s: f"{s['pass_rate']:>9.1%} "),
        ("Avg trades to pass", lambda s: f"{s['avg_trades_to_pass']:>10.1f}" if s['passed'] else f"{'N/A':>10s}"),
        ("Avg months to pass", lambda s: f"{s['avg_months_to_pass']:>10.1f}" if s['passed'] else f"{'N/A':>10s}"),
        ("Expected eval cost", lambda s: "${:,.0f}".format(s["expected_cost"]).rjust(10) if s['pass_rate'] > 0 else "       $--"),
        ("Max consecutive fails", lambda s: f"{s['max_consec_fails']:>10d}"),
        ("Avg P&L at failure", lambda s: "${:,.0f}".format(s["avg_pnl_at_fail"]).rjust(10) if s['failed'] else f"{'N/A':>10s}"),
    ]:
        metric_name, fmt_fn = stats_list
        print(f"  {metric_name:<26s} | {fmt_fn(co_stats)} | {fmt_fn(db_stats)} | {fmt_fn(comb_stats)}")

    # ── Detailed Attempt Log (Combined, first 20) ───────────────────────
    print("\n" + "=" * 70)
    print("  DETAILED ATTEMPT LOG — Combined (first 20)")
    print("=" * 70)
    print(f"  {'#':>3s} | {'Status':<10s} | {'Trades':>6s} | {'P&L':>10s} | {'Max DD':>8s} | {'Start':>12s} | {'End':>12s}")
    print("  " + "-" * 76)

    for a in comb_stats["attempts"][:20]:
        print(f"  {a['attempt']:>3d} | {a['status']:<10s} | {a['trades']:>6d} | "
              f"${a['pnl']:>+9,.0f} | ${a['max_drawdown']:>7,.0f} | "
              f"{a['start_date'].strftime('%Y-%m-%d'):>12s} | {a['end_date'].strftime('%Y-%m-%d'):>12s}")

    # ── Yearly Breakdown (Combined) ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  YEARLY BREAKDOWN — Combined")
    print("=" * 70)
    print(f"  {'Year':>6s} | {'CO Trades':>9s} | {'DB Trades':>9s} | {'Total P&L':>12s} | {'Would Pass?':>12s}")
    print("  " + "-" * 58)

    years = sorted(combined_trades["entry_date"].dt.year.unique())
    for year in years:
        year_trades = combined_trades[combined_trades["entry_date"].dt.year == year]
        co_count = len(year_trades[year_trades["tag"] == "CO"])
        db_count = len(year_trades[year_trades["tag"] == "DB"])
        total_pnl = year_trades["pnl"].sum()

        # Check if cumulative P&L hit $4K at some point during the year
        year_pnls = year_trades.sort_values("entry_date")["pnl"].values
        cum = 0.0
        would_pass = False
        for p in year_pnls:
            cum += p
            if cum >= PROFIT_TARGET:
                would_pass = True
                break

        print(f"  {year:>6d} | {co_count:>9d} | {db_count:>9d} | ${total_pnl:>+11,.0f} | "
              f"{'YES' if would_pass else 'NO':>12s}")

    # ── Final Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    co_pr = co_stats["pass_rate"]
    comb_pr = comb_stats["pass_rate"]
    delta = comb_pr - co_pr
    cost_savings = co_stats["expected_cost"] - comb_stats["expected_cost"]

    print(f"\n  CO-only pass rate:   {co_pr:.1%}")
    print(f"  Combined pass rate:  {comb_pr:.1%}")
    print(f"  Delta:               {delta:+.1%}")
    if co_stats["pass_rate"] > 0 and comb_stats["pass_rate"] > 0:
        print(f"  Expected eval cost savings: ${cost_savings:+,.0f} "
              f"(${co_stats['expected_cost']:,.0f} → ${comb_stats['expected_cost']:,.0f})")

    print()
    if comb_pr >= co_pr:
        print(f"  Combined pass rate: {comb_pr:.1%} — better than or equal to CO-only ({co_pr:.1%})")
        print("  Adding Dip Buyer improves or maintains the pass rate.")
    else:
        print(f"  Combined pass rate: {comb_pr:.1%} — worse than CO-only ({co_pr:.1%})")
        print("  WARNING: Adding Dip Buyer reduced the pass rate. The extra trades")
        print("  may be introducing drawdown risk during off-weeks.")

    print()


if __name__ == "__main__":
    main()
