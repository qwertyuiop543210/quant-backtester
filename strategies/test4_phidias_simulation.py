"""Test 4 — Phidias $50K Swing account pass rate simulation.

Simulates sequential evaluation attempts for CO-only, DB-only, and Combined
strategies under Phidias $50K Swing account rules:
  - $50,000 starting balance
  - $4,000 profit target (pass)
  - $2,500 EOD trailing drawdown from high-water mark (fail)
  - 1 mini ES contract, $144.60 OTP per attempt
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


# ── Build daily P&L series from trades ───────────────────────────────────────

def build_daily_pnl(trades_df, es_df):
    """Build a daily P&L Series from trade DataFrame and ES price data.

    For each trade, walks day-by-day from entry to exit computing mark-to-market
    changes. Returns a Series indexed by date with daily P&L values.
    Also returns a list of (start_date, end_date) tuples marking trade boundaries.
    """
    dates = es_df.index
    daily_pnl = pd.Series(0.0, index=dates)
    trade_boundaries = []

    for _, trade in trades_df.iterrows():
        entry_date = trade["entry_date"]
        exit_date = trade["exit_date"]

        # Find indices in ES data
        entry_loc = dates.get_loc(entry_date)
        exit_loc = dates.get_loc(exit_date)

        entry_price = float(es_df["Open"].iloc[entry_loc])
        trade_boundaries.append((entry_date, exit_date))

        for day_loc in range(entry_loc, exit_loc + 1):
            day_date = dates[day_loc]
            day_close = float(es_df["Close"].iloc[day_loc])

            if day_loc == entry_loc:
                # First day: mark-to-market from entry price
                day_pnl = (day_close - entry_price) * ES_POINT_VALUE
            else:
                # Subsequent days: change from previous close
                prev_close = float(es_df["Close"].iloc[day_loc - 1])
                day_pnl = (day_close - prev_close) * ES_POINT_VALUE

            # Subtract cost on exit day
            if day_loc == exit_loc:
                day_pnl -= COST_PER_TRADE

            daily_pnl.iloc[day_loc] += day_pnl

    return daily_pnl, trade_boundaries


# ── Build trade segments (merge overlapping boundaries) ──────────────────────

def _build_trade_segments(trade_boundaries, dates):
    """Merge overlapping trade boundaries into non-overlapping segments.

    Each segment is a contiguous block of trading days where at least one
    position is open.  Returns list of dicts:
        {start_loc, end_loc, start_date, end_date, trade_count}
    """
    if not trade_boundaries:
        return []

    # Convert to (start_loc, end_loc) pairs
    loc_pairs = []
    for entry_date, exit_date in trade_boundaries:
        loc_pairs.append((dates.get_loc(entry_date), dates.get_loc(exit_date)))

    # Sort by start_loc
    loc_pairs.sort()

    # Merge overlapping ranges and count trades per segment
    segments = []
    cur_start, cur_end = loc_pairs[0]
    cur_count = 1
    for s, e in loc_pairs[1:]:
        if s <= cur_end:
            # Overlapping — extend and increment count
            cur_end = max(cur_end, e)
            cur_count += 1
        else:
            segments.append({
                "start_loc": cur_start, "end_loc": cur_end,
                "start_date": dates[cur_start], "end_date": dates[cur_end],
                "trade_count": cur_count,
            })
            cur_start, cur_end = s, e
            cur_count = 1
    segments.append({
        "start_loc": cur_start, "end_loc": cur_end,
        "start_date": dates[cur_start], "end_date": dates[cur_end],
        "trade_count": cur_count,
    })
    return segments


# ── Phidias simulation engine ────────────────────────────────────────────────

def simulate_phidias(daily_pnl_series, trade_boundaries):
    """Simulate sequential Phidias $50K Swing evaluation attempts.

    Merges overlapping trade boundaries into segments, then walks day-by-day
    through the daily P&L series.  After FAIL/PASS, restarts at the next
    segment boundary.

    Args:
        daily_pnl_series: Series indexed by date with daily P&L values.
        trade_boundaries: List of (entry_date, exit_date) tuples.

    Returns:
        List of attempt dicts with status, trade count, P&L, dates, etc.
    """
    dates = daily_pnl_series.index
    segments = _build_trade_segments(trade_boundaries, dates)

    attempts = []
    seg_idx = 0

    while seg_idx < len(segments):
        balance = STARTING_BALANCE
        high_water = STARTING_BALANCE
        attempt_start_date = segments[seg_idx]["start_date"]
        attempt_trades = 0
        attempt_max_dd = 0.0
        status = "in_progress"

        while seg_idx < len(segments):
            seg = segments[seg_idx]

            breached = False
            for day_loc in range(seg["start_loc"], seg["end_loc"] + 1):
                day_pnl = daily_pnl_series.iloc[day_loc]
                balance += day_pnl
                high_water = max(high_water, balance)
                dd = high_water - balance
                attempt_max_dd = max(attempt_max_dd, dd)

                if dd >= EOD_DRAWDOWN:
                    breached = True
                    break

            attempt_trades += seg["trade_count"]
            seg_idx += 1

            if breached:
                status = "FAILED"
                break

            if balance >= STARTING_BALANCE + PROFIT_TARGET:
                status = "PASSED"
                break

        attempt_end_date = segments[seg_idx - 1]["end_date"]

        attempts.append({
            "attempt": len(attempts) + 1,
            "status": status,
            "trades": attempt_trades,
            "pnl": balance - STARTING_BALANCE,
            "max_drawdown": attempt_max_dd,
            "start_date": attempt_start_date,
            "end_date": attempt_end_date,
            "final_balance": balance,
            "high_water": high_water,
        })

        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"

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
    print(f"         ${EOD_DRAWDOWN:,.0f} EOD trailing drawdown (from HWM)")
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

    # Build daily P&L series for each
    print("\n  Building daily P&L series...")
    co_daily, co_bounds = build_daily_pnl(co_trades, es)
    db_daily, db_bounds = build_daily_pnl(db_trades, es)
    comb_daily, comb_bounds = build_daily_pnl(combined_trades, es)

    # Run simulations
    print("  Running Phidias simulations...")
    co_attempts = simulate_phidias(co_daily, co_bounds)
    db_attempts = simulate_phidias(db_daily, db_bounds)
    comb_attempts = simulate_phidias(comb_daily, comb_bounds)

    # Compute trades per month
    date_range_years = (es.index[-1] - es.index[0]).days / 365.25
    date_range_months = date_range_years * 12
    co_tpm = len(co_trades) / date_range_months
    db_tpm = len(db_trades) / date_range_months
    comb_tpm = len(combined_trades) / date_range_months

    print(f"  CO trades/month: {co_tpm:.2f}")
    print(f"  DB trades/month: {db_tpm:.2f}")
    print(f"  Combined trades/month: {comb_tpm:.2f}")

    co_stats = analyze_attempts(co_attempts, "CO-Only", co_tpm)
    db_stats = analyze_attempts(db_attempts, "DB-Only", db_tpm)
    comb_stats = analyze_attempts(comb_attempts, "Combined", comb_tpm)

    # ── Output Table ─────────────────────────────────────────────────────────
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

    # ── Detailed Attempt Log (Combined, first 20) ───────────────────────────
    print("\n" + "=" * 70)
    print("  DETAILED ATTEMPT LOG — Combined (first 20)")
    print("=" * 70)
    print(f"  {'#':>3s} | {'Status':<10s} | {'Trades':>6s} | {'P&L':>10s} | {'Max DD':>8s} | {'Start':>12s} | {'End':>12s}")
    print("  " + "-" * 76)

    for a in comb_stats["attempts"][:20]:
        print(f"  {a['attempt']:>3d} | {a['status']:<10s} | {a['trades']:>6d} | "
              f"${a['pnl']:>+9,.0f} | ${a['max_drawdown']:>7,.0f} | "
              f"{a['start_date'].strftime('%Y-%m-%d'):>12s} | {a['end_date'].strftime('%Y-%m-%d'):>12s}")

    # ── Yearly Breakdown (Combined) ──────────────────────────────────────────
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

    # ── Final Verdict ────────────────────────────────────────────────────────
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
