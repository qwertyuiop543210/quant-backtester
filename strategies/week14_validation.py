"""Validation tests for ES Week 1 & Week 4 VIX 15-20 skip filter.

TEST 1 — BOUNDARY SENSITIVITY: Run with multiple VIX skip ranges to confirm
the edge is stable and not an artifact of exact threshold choice.

TEST 2 — WALK-FORWARD: Split at Jan 1 2013 (midpoint). Run Skip VIX 15-20
on each half independently. If both halves show improvement, the edge is real.

ES costs: $5 RT commission, $12.50 slippage per side = $30/trade.
Topstep: $150K account, $4,500 trailing DD, $9,000 profit target.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import profit_factor

STRATEGY_NAME = "Week 1 & Week 4 — Validation"

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0

WALK_FORWARD_SPLIT = "2013-01-01"

BOUNDARY_RANGES = [
    (13, 18),
    (14, 19),
    (15, 20),  # current filter
    (16, 21),
    (17, 22),
    (13, 22),  # wide band
    (None, None),  # no skip (v1 baseline)
]


def get_week_of_month(date: pd.Timestamp) -> int:
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
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


def simulate_topstep(trade_pnls: list[float]) -> dict:
    """Run Topstep simulation, return summary stats."""
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

        attempts.append({"status": status, "trades_taken": i - start_trade})
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break

    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    failed = sum(1 for a in attempts if a["status"] == "FAILED")
    total = passed + failed
    rate = passed / total * 100 if total > 0 else 0
    return {"passed": passed, "failed": failed, "total": total,
            "rate": rate, "attempts": attempts}


def build_all_trades(es: pd.DataFrame, vix_close: pd.Series) -> pd.DataFrame:
    """Build all Week 1 & Week 4 trades with VIX at entry."""
    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    weeks = find_trading_weeks(es.index)
    active = [w for w in weeks if w["week_of_month"] in ACTIVE_WEEKS]

    trades = []
    for w in active:
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        entry_price = open_.iloc[mi]
        exit_price = close.iloc[fi]

        vix_at_entry = vix_close.iloc[mi - 1] if mi > 0 else vix_close.iloc[mi]
        if pd.isna(vix_at_entry):
            vix_at_entry = vix_close.iloc[mi]

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": w["monday_date"],
            "exit_date": w["friday_date"],
            "week_of_month": w["week_of_month"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": net_pnl,
            "vix_at_entry": vix_at_entry,
        })

    return pd.DataFrame(trades)


def filter_trades(all_trades: pd.DataFrame, skip_lo, skip_hi) -> pd.DataFrame:
    """Filter out trades where VIX is in [skip_lo, skip_hi)."""
    if skip_lo is None:
        return all_trades.copy()
    mask = ~((all_trades["vix_at_entry"] >= skip_lo) & (all_trades["vix_at_entry"] < skip_hi))
    return all_trades[mask].reset_index(drop=True)


def compute_stats(trades_df: pd.DataFrame) -> dict:
    """Compute PF, trades, total P&L, avg P&L, win rate."""
    pnls = pd.Series(trades_df["pnl"].values, dtype=float)
    n = len(pnls)
    if n == 0:
        return {"pf": 0, "trades": 0, "total_pnl": 0, "avg_pnl": 0, "win_rate": 0}
    pf = profit_factor(pnls)
    total = pnls.sum()
    avg = pnls.mean()
    wr = (pnls > 0).mean()
    return {"pf": pf, "trades": n, "total_pnl": total, "avg_pnl": avg, "win_rate": wr}


def run_boundary_sensitivity(all_trades: pd.DataFrame):
    """TEST 1: Boundary sensitivity across VIX skip ranges."""
    print(f"\n{'='*90}")
    print(f"  TEST 1 — BOUNDARY SENSITIVITY")
    print(f"  Does the edge survive shifting the VIX skip range?")
    print(f"{'='*90}\n")

    results = []
    for skip_lo, skip_hi in BOUNDARY_RANGES:
        filtered = filter_trades(all_trades, skip_lo, skip_hi)
        stats = compute_stats(filtered)
        topstep = simulate_topstep(filtered["pnl"].tolist())

        label = f"Skip {skip_lo}-{skip_hi}" if skip_lo is not None else "No skip (v1)"
        if (skip_lo, skip_hi) == (15, 20):
            label += " *"
        if (skip_lo, skip_hi) == (13, 22):
            label += " (wide)"

        results.append({
            "label": label,
            "skip_lo": skip_lo,
            "skip_hi": skip_hi,
            **stats,
            "topstep_rate": topstep["rate"],
            "topstep_passed": topstep["passed"],
            "topstep_total": topstep["total"],
        })

    # Print table
    print(f"  {'VIX Skip Range':<22} {'PF':>8} {'Trades':>8} {'Total P&L':>14} "
          f"{'Avg P&L':>10} {'Win%':>7} {'TS Pass%':>10} {'TS P/T':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*14} {'-'*10} {'-'*7} {'-'*10} {'-'*8}")

    for r in results:
        pf_str = f"{r['pf']:.3f}" if r['pf'] != float('inf') else "inf"
        print(f"  {r['label']:<22} {pf_str:>8} {r['trades']:>8} "
              f"{'${:,.0f}'.format(r['total_pnl']):>14} "
              f"{'${:,.0f}'.format(r['avg_pnl']):>10} "
              f"{r['win_rate']:>6.1%} {r['topstep_rate']:>9.1f}% "
              f"{r['topstep_passed']}/{r['topstep_total']:>6}")

    print(f"\n  * = current filter")
    print()

    # Assess stability
    baseline = [r for r in results if r["skip_lo"] is None][0]
    narrow_ranges = [r for r in results
                     if r["skip_lo"] is not None and (r["skip_lo"], r["skip_hi"]) != (13, 22)]
    improvements = sum(1 for r in narrow_ranges if r["pf"] > baseline["pf"])
    ts_improvements = sum(1 for r in narrow_ranges
                          if r["topstep_rate"] > baseline["topstep_rate"])

    print(f"  BOUNDARY ASSESSMENT:")
    print(f"    Narrow ranges tested:              {len(narrow_ranges)}")
    print(f"    Ranges with PF > v1 baseline:      {improvements}/{len(narrow_ranges)}")
    print(f"    Ranges with TS pass > v1 baseline:  {ts_improvements}/{len(narrow_ranges)}")

    if improvements >= 4 and ts_improvements >= 3:
        verdict = "STABLE — filter improves results across most nearby boundaries"
    elif improvements >= 3:
        verdict = "MODERATE — some sensitivity to exact boundaries"
    else:
        verdict = "UNSTABLE — edge depends on exact threshold choice (likely overfit)"

    print(f"    Verdict: {verdict}")
    print()

    return results


def run_walk_forward(all_trades: pd.DataFrame):
    """TEST 2: Walk-forward split at Jan 1 2013."""
    print(f"\n{'='*90}")
    print(f"  TEST 2 — WALK-FORWARD VALIDATION")
    print(f"  Split at {WALK_FORWARD_SPLIT}. Does Skip VIX 15-20 work in both halves?")
    print(f"{'='*90}\n")

    split_date = pd.Timestamp(WALK_FORWARD_SPLIT)

    first_half = all_trades[all_trades["entry_date"] < split_date].copy()
    second_half = all_trades[all_trades["entry_date"] >= split_date].copy()

    periods = [
        ("First half (2000-2012)", first_half),
        ("Second half (2013-2026)", second_half),
        ("Full period (reference)", all_trades),
    ]

    results = []
    for period_label, period_trades in periods:
        # v1 (no skip) for this period
        v1_stats = compute_stats(period_trades)
        v1_topstep = simulate_topstep(period_trades["pnl"].tolist())

        # Filtered (skip 15-20) for this period
        filtered = filter_trades(period_trades, 15, 20)
        f_stats = compute_stats(filtered)
        f_topstep = simulate_topstep(filtered["pnl"].tolist())

        results.append({
            "period": period_label,
            "v1_pf": v1_stats["pf"],
            "v1_trades": v1_stats["trades"],
            "v1_total_pnl": v1_stats["total_pnl"],
            "v1_topstep_rate": v1_topstep["rate"],
            "v1_topstep_passed": v1_topstep["passed"],
            "v1_topstep_total": v1_topstep["total"],
            "f_pf": f_stats["pf"],
            "f_trades": f_stats["trades"],
            "f_total_pnl": f_stats["total_pnl"],
            "f_topstep_rate": f_topstep["rate"],
            "f_topstep_passed": f_topstep["passed"],
            "f_topstep_total": f_topstep["total"],
            "skipped": v1_stats["trades"] - f_stats["trades"],
        })

    # Print comparison table
    print(f"  {'Period':<28} │ {'--- v1 (no skip) ---':^30} │ {'--- Skip VIX 15-20 ---':^30}")
    print(f"  {'':28} │ {'PF':>7} {'Trades':>7} {'P&L':>10} {'TS%':>6} │ "
          f"{'PF':>7} {'Trades':>7} {'P&L':>10} {'TS%':>6}")
    print(f"  {'-'*28}-+-{'-'*30}-+-{'-'*30}")

    for r in results:
        v1_pf = f"{r['v1_pf']:.3f}" if r['v1_pf'] != float('inf') else "inf"
        f_pf = f"{r['f_pf']:.3f}" if r['f_pf'] != float('inf') else "inf"
        print(f"  {r['period']:<28} │ {v1_pf:>7} {r['v1_trades']:>7} "
              f"{'${:,.0f}'.format(r['v1_total_pnl']):>10} {r['v1_topstep_rate']:>5.1f}% │ "
              f"{f_pf:>7} {r['f_trades']:>7} "
              f"{'${:,.0f}'.format(r['f_total_pnl']):>10} {r['f_topstep_rate']:>5.1f}%")

    print()

    # Detailed per-period analysis
    for r in results:
        pf_delta = r["f_pf"] - r["v1_pf"] if r["v1_pf"] != float("inf") and r["f_pf"] != float("inf") else 0
        ts_delta = r["f_topstep_rate"] - r["v1_topstep_rate"]
        print(f"  {r['period']}:")
        print(f"    Skipped {r['skipped']} trades | "
              f"PF delta: {pf_delta:+.3f} | "
              f"Topstep delta: {ts_delta:+.1f}%")

    # Assess walk-forward
    first = results[0]
    second = results[1]

    first_improves_pf = first["f_pf"] > first["v1_pf"]
    second_improves_pf = second["f_pf"] > second["v1_pf"]
    first_improves_ts = first["f_topstep_rate"] >= first["v1_topstep_rate"]
    second_improves_ts = second["f_topstep_rate"] >= second["v1_topstep_rate"]

    print(f"\n  WALK-FORWARD ASSESSMENT:")
    print(f"    First half  PF improvement:  {'YES' if first_improves_pf else 'NO'} "
          f"({first['v1_pf']:.3f} -> {first['f_pf']:.3f})")
    print(f"    Second half PF improvement:  {'YES' if second_improves_pf else 'NO'} "
          f"({second['v1_pf']:.3f} -> {second['f_pf']:.3f})")
    print(f"    First half  TS improvement:  {'YES' if first_improves_ts else 'NO'} "
          f"({first['v1_topstep_rate']:.1f}% -> {first['f_topstep_rate']:.1f}%)")
    print(f"    Second half TS improvement:  {'YES' if second_improves_ts else 'NO'} "
          f"({second['v1_topstep_rate']:.1f}% -> {second['f_topstep_rate']:.1f}%)")

    both_pf = first_improves_pf and second_improves_pf
    both_ts = first_improves_ts and second_improves_ts

    if both_pf and both_ts:
        verdict = "CONFIRMED — filter improves PF and Topstep in both halves"
    elif both_pf:
        verdict = "PARTIAL — PF improves in both halves, Topstep mixed"
    elif both_ts:
        verdict = "PARTIAL — Topstep improves in both halves, PF mixed"
    else:
        verdict = "FAILED — filter does not consistently improve across time periods (likely overfit)"

    print(f"    Verdict: {verdict}")
    print()

    return results


def run():
    print(f"\n{'#'*90}")
    print(f"  VALIDATION SUITE — ES Week 1 & Week 4, Skip Mid-VIX Filter")
    print(f"{'#'*90}")
    print("Loading data...")

    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1993-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    vix_close = vix["Close"].reindex(es.index).ffill()

    all_trades = build_all_trades(es, vix_close)
    print(f"Total Week 1 & Week 4 trades: {len(all_trades)}")
    print(f"Date range: {all_trades['entry_date'].iloc[0].date()} to "
          f"{all_trades['entry_date'].iloc[-1].date()}")

    # TEST 1
    boundary_results = run_boundary_sensitivity(all_trades)

    # TEST 2
    walkforward_results = run_walk_forward(all_trades)

    # Final verdict
    print(f"\n{'#'*90}")
    print(f"  OVERALL VALIDATION VERDICT")
    print(f"{'#'*90}")

    # Boundary: count narrow ranges that beat baseline
    baseline = [r for r in boundary_results if r["skip_lo"] is None][0]
    narrow = [r for r in boundary_results
              if r["skip_lo"] is not None and (r["skip_lo"], r["skip_hi"]) != (13, 22)]
    pf_beats = sum(1 for r in narrow if r["pf"] > baseline["pf"])
    ts_beats = sum(1 for r in narrow if r["topstep_rate"] > baseline["topstep_rate"])
    boundary_pass = pf_beats >= 4

    # Walk-forward: both halves improve PF
    first_wf = walkforward_results[0]
    second_wf = walkforward_results[1]
    wf_pass = (first_wf["f_pf"] > first_wf["v1_pf"] and
               second_wf["f_pf"] > second_wf["v1_pf"])

    print(f"\n  Boundary sensitivity: {'PASS' if boundary_pass else 'FAIL'} "
          f"({pf_beats}/{len(narrow)} ranges beat v1 PF)")
    print(f"  Walk-forward:         {'PASS' if wf_pass else 'FAIL'} "
          f"(both halves PF improvement: {wf_pass})")

    if boundary_pass and wf_pass:
        print(f"\n  >>> EDGE CONFIRMED — VIX 15-20 skip filter is robust <<<")
    elif boundary_pass or wf_pass:
        print(f"\n  >>> EDGE PARTIALLY CONFIRMED — one test passed, one failed <<<")
        print(f"  >>> Proceed with caution; filter may have limited robustness <<<")
    else:
        print(f"\n  >>> EDGE NOT CONFIRMED — filter is likely overfit <<<")
        print(f"  >>> Recommend trading v1 baseline without VIX filter <<<")

    print()


if __name__ == "__main__":
    run()
