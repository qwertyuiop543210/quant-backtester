"""Test 2 — Walk-forward optimization with expanding windows.

Optimizes only the Chosen One VIX skip range across 12 expanding windows.
Dip Buyer parameters remain fixed at defaults throughout.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
np.random.seed(42)

import pandas as pd

from strategies.validation_helpers import (
    load_data, generate_co_trades, generate_db_trades,
    pf_from_array, wr_from_array,
)


# VIX skip ranges to test (None = no skip)
VIX_RANGES = [
    (None, None, "none"),
    (13.0, 18.0, "13-18"),
    (14.0, 19.0, "14-19"),
    (15.0, 20.0, "15-20"),
    (16.0, 21.0, "16-21"),
    (17.0, 22.0, "17-22"),
]

# Tiebreak preference order: closest to 15-20
TIEBREAK_ORDER = ["15-20", "14-19", "16-21", "13-18", "17-22", "none"]


def filter_by_date(df, start, end):
    """Filter trades where entry_date falls within [start, end]."""
    if len(df) == 0:
        return df
    mask = (df["entry_date"] >= pd.Timestamp(start)) & (df["entry_date"] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def main():
    print("=" * 80)
    print("  TEST 2 — WALK-FORWARD OPTIMIZATION (EXPANDING WINDOWS)")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────
    es, vix = load_data()

    # ── Define windows ────────────────────────────────────────────────────
    windows = []
    for i in range(12):
        train_start = "2012-01-01"
        train_end = f"{2013 + i}-12-31"
        test_year = 2014 + i
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"
        windows.append((train_start, train_end, test_start, test_end, test_year))

    # ── Pre-generate CO trades for each VIX range on full dataset ────────
    co_by_range = {}
    for lo, hi, label in VIX_RANGES:
        co_by_range[label] = generate_co_trades(
            es, vix,
            vix_skip_lo=lo if lo is not None else None,
            vix_skip_hi=hi if hi is not None else None,
        )

    # ── Pre-generate DB trades on full dataset (fixed params) ────────────
    db_all = generate_db_trades(es, vix)

    # ── Walk-forward loop ─────────────────────────────────────────────────
    results = []
    all_test_pnls = []

    print(f"\n{'=' * 80}")
    print("  WALK-FORWARD WINDOWS")
    print(f"{'=' * 80}\n")

    for train_start, train_end, test_start, test_end, test_year in windows:
        # --- TRAINING: find best VIX skip range ---
        best_pf = -1.0
        best_label = "none"

        for lo, hi, label in VIX_RANGES:
            train_trades = filter_by_date(co_by_range[label], train_start, train_end)
            if len(train_trades) == 0:
                pf = 0.0
            else:
                pf = pf_from_array(train_trades["pnl"].values.astype(float))

            # Pick highest PF; tiebreak by closeness to 15-20
            if pf > best_pf or (pf == best_pf and
                                TIEBREAK_ORDER.index(label) < TIEBREAK_ORDER.index(best_label)):
                best_pf = pf
                best_label = label

        # Handle edge case: all ranges produced 0 trades
        if best_pf == 0.0:
            best_label = "none"

        # --- TESTING: generate test-year trades with winning range ---
        co_test = filter_by_date(co_by_range[best_label], test_start, test_end)
        db_test = filter_by_date(db_all, test_start, test_end)

        co_pnls = co_test["pnl"].values.astype(float) if len(co_test) > 0 else np.array([])
        db_pnls = db_test["pnl"].values.astype(float) if len(db_test) > 0 else np.array([])
        combined = np.concatenate([co_pnls, db_pnls])

        if len(combined) > 0:
            test_pf = pf_from_array(combined)
            test_total = float(combined.sum())
            test_wr = wr_from_array(combined)
        else:
            test_pf = 0.0
            test_total = 0.0
            test_wr = 0.0

        all_test_pnls.append(combined)

        results.append({
            "train_window": f"2012-{2013 + (test_year - 2014)}",
            "vix_skip": best_label,
            "test_year": test_year,
            "co_trades": len(co_pnls),
            "db_trades": len(db_pnls),
            "combined_pf": test_pf,
            "combined_pnl": test_total,
            "wr": test_wr,
        })

    # ── Print table ───────────────────────────────────────────────────────
    header = (f"  {'Train Window':<16}| {'VIX Skip':>10} | {'Test Year':>9} | "
              f"{'CO Trades':>9} | {'DB Trades':>9} | {'Comb PF':>9} | "
              f"{'Comb P&L':>12} | {'WR':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        pf_str = f"{r['combined_pf']:.3f}" if np.isfinite(r['combined_pf']) else "inf"
        print(f"  {r['train_window']:<16}| {r['vix_skip']:>10} | {r['test_year']:>9} | "
              f"{r['co_trades']:>9} | {r['db_trades']:>9} | {pf_str:>9} | "
              f"${r['combined_pnl']:>11,.0f} | {r['wr']:>5.1%}")

    # ── Walk-forward aggregate metrics ────────────────────────────────────
    wf_all = np.concatenate(all_test_pnls)
    wf_pf = pf_from_array(wf_all)
    wf_total = float(wf_all.sum())
    wf_wr = wr_from_array(wf_all)

    print(f"\n{'=' * 80}")
    print("  WALK-FORWARD AGGREGATE")
    print(f"{'=' * 80}")
    print(f"  Walk-forward PF:       {wf_pf:.3f}")
    print(f"  Walk-forward total P&L: ${wf_total:,.0f}")
    print(f"  Walk-forward win rate:  {wf_wr:.1%}")
    print(f"  Total test trades:      {len(wf_all)}")

    # ── Fixed-parameter baseline (same test years) ────────────────────────
    co_fixed = generate_co_trades(es, vix, vix_skip_lo=15.0, vix_skip_hi=20.0)
    db_fixed = generate_db_trades(es, vix)

    co_fixed_test = filter_by_date(co_fixed, "2014-01-01", "2025-12-31")
    db_fixed_test = filter_by_date(db_fixed, "2014-01-01", "2025-12-31")

    fixed_pnls = np.concatenate([
        co_fixed_test["pnl"].values.astype(float) if len(co_fixed_test) > 0 else np.array([]),
        db_fixed_test["pnl"].values.astype(float) if len(db_fixed_test) > 0 else np.array([]),
    ])
    fixed_pf = pf_from_array(fixed_pnls)

    print(f"  Fixed-parameter PF:     {fixed_pf:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  VERDICT")
    print(f"{'=' * 80}")

    ratio = wf_pf / fixed_pf if fixed_pf > 0 else 0.0
    cond1 = wf_pf > 1.0
    cond2 = wf_pf >= fixed_pf * 0.70

    print(f"  Walk-forward PF:    {wf_pf:.3f}")
    print(f"  Fixed-parameter PF: {fixed_pf:.3f}")
    print(f"  Ratio (WF/Fixed):   {ratio:.3f}")
    print()
    print(f"  Condition 1: WF PF > 1.0              → {'PASS' if cond1 else 'FAIL'}  ({wf_pf:.3f})")
    print(f"  Condition 2: WF PF >= Fixed × 0.70    → {'PASS' if cond2 else 'FAIL'}  "
          f"({wf_pf:.3f} >= {fixed_pf * 0.70:.3f})")
    print()

    if cond1 and cond2:
        print("  *** PASS — Walk-forward optimization is profitable out of sample")
        print(f"    and within 30% of fixed-parameter performance (ratio={ratio:.3f}).")
    else:
        reasons = []
        if not cond1:
            reasons.append(f"WF PF {wf_pf:.3f} <= 1.0 (not profitable out of sample)")
        if not cond2:
            reasons.append(f"WF PF {wf_pf:.3f} < {fixed_pf * 0.70:.3f} (more than 30% below fixed)")
        print("  *** FAIL — " + "; ".join(reasons))

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
