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
    """Build a COMPLETE daily P&L Series from trade DataFrame and ES data.

    For each trade, walks every trading day from entry to exit computing
    mark-to-market changes.  Returns a Series indexed by date with daily
    P&L values (summed when trades overlap on the same day).
    """
    dates = es_df.index
    daily_pnl = pd.Series(0.0, index=dates)

    for _, trade in trades_df.iterrows():
        entry_date = trade["entry_date"]
        exit_date = trade["exit_date"]

        entry_price = float(es_df.loc[entry_date, "Open"])

        # Get ALL trading days from entry to exit (inclusive)
        mask = (dates >= entry_date) & (dates <= exit_date)
        trade_days = dates[mask]

        prev = entry_price  # first "previous" is the entry price
        for day in trade_days:
            today_close = float(es_df.loc[day, "Close"])
            daily_pnl[day] += (today_close - prev) * ES_POINT_VALUE
            prev = today_close

        # Subtract cost on exit day
        daily_pnl[exit_date] -= COST_PER_TRADE

    return daily_pnl


# ── Phidias simulation engine ────────────────────────────────────────────────

def simulate_phidias(daily_pnl, trades_df):
    """Simulate sequential Phidias $50K Swing evaluation attempts.

    Walks through every date in daily_pnl in order.  Tracks which trades
    are active (entered but not exited).  Checks EOD drawdown and profit
    target at the end of EVERY trading day.

    On FAIL/PASS at day D: the attempt ends immediately.  Remaining days
    of any active trade are abandoned.  The next attempt starts at the
    first trade whose entry_date > D.

    Args:
        daily_pnl: Series indexed by date with daily P&L values.
        trades_df: DataFrame with entry_date and exit_date columns.

    Returns:
        List of attempt dicts.
    """
    trades = trades_df.sort_values("entry_date").reset_index(drop=True)
    dates = daily_pnl.index

    # Build entry/exit date -> trade index lookups
    entry_map = {}  # date -> list of trade indices entering
    exit_map = {}   # date -> list of trade indices exiting
    for i in range(len(trades)):
        ed = trades.iloc[i]["entry_date"]
        xd = trades.iloc[i]["exit_date"]
        entry_map.setdefault(ed, []).append(i)
        exit_map.setdefault(xd, []).append(i)

    attempts = []
    min_next_entry = None  # after fail/pass, trades must start strictly after this
    balance = STARTING_BALANCE
    hwm = STARTING_BALANCE
    attempt_start = None
    attempt_trades = 0
    attempt_max_dd = 0.0
    active = set()         # currently held trade indices
    in_attempt = False
    debug_logs = {}        # attempt_num -> list of day records

    for date in dates:
        # --- Enter new trades ---
        if date in entry_map:
            for tidx in entry_map[date]:
                # Skip trades that fall within a previous attempt's consumed range
                if min_next_entry is not None and date <= min_next_entry:
                    continue
                if not in_attempt:
                    # Fresh attempt
                    balance = STARTING_BALANCE
                    hwm = STARTING_BALANCE
                    attempt_start = date
                    attempt_trades = 0
                    attempt_max_dd = 0.0
                    in_attempt = True
                    min_next_entry = None
                active.add(tidx)
                attempt_trades += 1

        # Skip days where we have no active position
        if not in_attempt:
            continue
        if not active and daily_pnl[date] == 0.0:
            continue

        # --- Apply daily P&L ---
        day_pnl = daily_pnl[date]
        balance += day_pnl
        hwm = max(hwm, balance)
        dd = hwm - balance
        attempt_max_dd = max(attempt_max_dd, dd)

        # Debug logging (kept for first 3 fails per run)
        anum = len(attempts) + 1
        if anum not in debug_logs:
            debug_logs[anum] = []
        debug_logs[anum].append({
            "date": date, "pnl": day_pnl, "balance": balance,
            "hwm": hwm, "dd": dd,
        })

        # --- Exit trades that close today ---
        if date in exit_map:
            for tidx in exit_map[date]:
                active.discard(tidx)

        # --- Check FAIL (EOD drawdown) ---
        if dd >= EOD_DRAWDOWN:
            attempts.append({
                "attempt": len(attempts) + 1,
                "status": "FAILED",
                "trades": attempt_trades,
                "pnl": balance - STARTING_BALANCE,
                "max_drawdown": dd,  # exact DD at moment of failure
                "start_date": attempt_start,
                "end_date": date,
                "final_balance": balance,
                "high_water": hwm,
            })
            active.clear()
            in_attempt = False
            min_next_entry = date  # next trade must enter strictly after today
            continue

        # --- Check PASS (profit target) ---
        if balance >= STARTING_BALANCE + PROFIT_TARGET:
            attempts.append({
                "attempt": len(attempts) + 1,
                "status": "PASSED",
                "trades": attempt_trades,
                "pnl": balance - STARTING_BALANCE,
                "max_drawdown": attempt_max_dd,
                "start_date": attempt_start,
                "end_date": date,
                "final_balance": balance,
                "high_water": hwm,
            })
            active.clear()
            in_attempt = False
            min_next_entry = date  # next trade must enter strictly after today
            continue

    # Handle incomplete attempt (ran out of data mid-trade)
    if in_attempt:
        attempts.append({
            "attempt": len(attempts) + 1,
            "status": "INCOMPLETE",
            "trades": attempt_trades,
            "pnl": balance - STARTING_BALANCE,
            "max_drawdown": attempt_max_dd,
            "start_date": attempt_start,
            "end_date": dates[-1],
            "final_balance": balance,
            "high_water": hwm,
        })

    # --- Debug: day-by-day trace for first 3 FAILED attempts ---
    print("\n  DEBUG — Day-by-day trace for first 3 FAILED attempts:")
    printed = 0
    for a in attempts:
        if a["status"] == "FAILED" and printed < 3:
            num = a["attempt"]
            print(f"\n  Attempt #{num}: {a['start_date'].strftime('%Y-%m-%d')} "
                  f"to {a['end_date'].strftime('%Y-%m-%d')} "
                  f"({a['trades']} trades)")
            if num in debug_logs:
                for d in debug_logs[num]:
                    tag = "FAIL" if d["dd"] >= EOD_DRAWDOWN else "ok"
                    print(f"    {d['date'].strftime('%Y-%m-%d')}: "
                          f"pnl=${d['pnl']:>+10,.2f}  "
                          f"bal=${d['balance']:>12,.2f}  "
                          f"hwm=${d['hwm']:>12,.2f}  "
                          f"dd=${d['dd']:>8,.2f}  {tag}")
            printed += 1

    # --- Sanity check: flag any failed attempt with extreme DD ---
    extreme = [a for a in attempts
               if a["status"] == "FAILED" and a["max_drawdown"] > 3000]
    if extreme:
        print(f"\n  NOTE: {len(extreme)} failed attempts have max_dd > $3,000.")
        print("  This is expected with mini ES ($50/pt) — a single day's move")
        print("  can overshoot the $2,500 threshold. The EOD check fires on the")
        print("  first day DD >= $2,500; the overshoot is the day's full loss.")

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
    co_daily = build_daily_pnl(co_trades, es)
    db_daily = build_daily_pnl(db_trades, es)
    comb_daily = build_daily_pnl(combined_trades, es)

    # Verify: print daily P&L for the first CO trade
    t0 = co_trades.iloc[0]
    mask = (co_daily.index >= t0["entry_date"]) & (co_daily.index <= t0["exit_date"])
    t0_days = co_daily[mask]
    print(f"\n  Verify first CO trade ({t0['entry_date'].strftime('%Y-%m-%d')} to "
          f"{t0['exit_date'].strftime('%Y-%m-%d')}): {len(t0_days)} daily P&L values")
    for date, pnl in t0_days.items():
        print(f"    {date.strftime('%Y-%m-%d')}: ${pnl:+,.2f}")
    print(f"    Sum: ${t0_days.sum():+,.2f}  (trade P&L: ${t0['pnl']:+,.2f})")

    # Run simulations
    print("\n  Running Phidias simulations...")
    print("\n  --- CO-Only ---")
    co_attempts = simulate_phidias(co_daily, co_trades)
    print("\n  --- DB-Only ---")
    db_attempts = simulate_phidias(db_daily, db_trades)
    print("\n  --- Combined ---")
    comb_attempts = simulate_phidias(comb_daily, combined_trades)

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
