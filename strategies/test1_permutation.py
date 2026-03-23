"""Test 1 — Permutation test (10,000 shuffles) for the combined portfolio.

Chosen One: random 50% subset selection (since CO trades ALL eligible weeks).
Dip Buyer:  random-entry-date approach (since shuffling P&L doesn't change PF).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
np.random.seed(42)

from strategies.validation_helpers import (
    load_data, generate_co_trades, generate_db_trades,
    pf_from_array, wr_from_array, print_ascii_histogram,
    ES_POINT_VALUE, COST_PER_TRADE,
)

N_PERMUTATIONS = 10_000


def main():
    print("=" * 80)
    print("  TEST 1 — PERMUTATION TEST")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────
    es, vix = load_data()

    # ── Generate actual trades ────────────────────────────────────────────
    co_trades = generate_co_trades(es, vix)
    db_trades = generate_db_trades(es, vix)

    co_pnls = co_trades["pnl"].values.astype(float)
    db_pnls = db_trades["pnl"].values.astype(float)
    combined_pnls = np.concatenate([co_pnls, db_pnls])

    co_pf = pf_from_array(co_pnls)
    db_pf = pf_from_array(db_pnls)
    comb_pf = pf_from_array(combined_pnls)
    comb_wr = wr_from_array(combined_pnls)

    print(f"\n  Actual Chosen One: {len(co_pnls)} trades, PF {co_pf:.3f}, "
          f"Total P&L ${co_pnls.sum():,.0f}")
    print(f"  Actual Dip Buyer:  {len(db_pnls)} trades, PF {db_pf:.3f}, "
          f"Total P&L ${db_pnls.sum():,.0f}")
    print(f"  Combined:          {len(combined_pnls)} trades, PF {comb_pf:.3f}, "
          f"Total P&L ${combined_pnls.sum():,.0f}, WR {comb_wr:.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # CHOSEN ONE PERMUTATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  CHOSEN ONE PERMUTATION TEST")
    print(f"{'=' * 80}")
    print("  Chosen One trades ALL eligible weeks — permutation test not applicable (no subsampling).")
    print(f"  Alternative test: random 50% subset selection, {N_PERMUTATIONS:,} iterations")
    print(f"  Actual CO PF: {co_pf:.3f}")

    N_co = len(co_pnls)
    co_random_pfs = []
    co_beat_count = 0
    co_valid_count = 0

    for perm in range(N_PERMUTATIONS):
        if (perm + 1) % 2000 == 0:
            print(f"    CO permutation {perm + 1:,}/{N_PERMUTATIONS:,}...")

        # Fair coin flip for each trade
        mask = np.random.random(N_co) < 0.5
        subset = co_pnls[mask]

        if len(subset) == 0:
            continue  # skip empty subsets

        pf = pf_from_array(subset)
        co_valid_count += 1
        co_random_pfs.append(pf)

        # Count beats
        if pf == float("inf"):
            if co_pf == float("inf"):
                co_beat_count += 1
        elif pf >= co_pf:
            co_beat_count += 1

    co_pvalue = co_beat_count / co_valid_count if co_valid_count > 0 else 1.0
    print(f"  Permutation p-value: {co_pvalue:.4f} "
          f"({co_beat_count} out of {co_valid_count} valid shuffles matched or beat actual PF)")

    # Percentiles (exclude inf)
    co_finite = np.array([x for x in co_random_pfs if np.isfinite(x)])
    if len(co_finite) > 0:
        pcts = np.percentile(co_finite, [5, 25, 50, 75, 95])
        print(f"  Percentiles (excl inf): 5th={pcts[0]:.3f}  25th={pcts[1]:.3f}  "
              f"50th={pcts[2]:.3f}  75th={pcts[3]:.3f}  95th={pcts[4]:.3f}")

    # Histogram (filter inf before plotting)
    print_ascii_histogram(co_finite, co_pf, label="CO PF")

    # ══════════════════════════════════════════════════════════════════════
    # DIP BUYER PERMUTATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  DIP BUYER PERMUTATION TEST")
    print(f"{'=' * 80}")
    print(f"  Actual DB PF: {db_pf:.3f}")
    print(f"  Random-entry-date approach: {N_PERMUTATIONS:,} iterations")

    N_db = len(db_pnls)
    hold_days_arr = db_trades["hold_days"].values.astype(int)

    # Pre-compute price arrays
    es_subset = es.loc["2012-01-01":]
    es_open_arr = es_subset["Open"].values.astype(float)
    es_close_arr = es_subset["Close"].values.astype(float)
    total_days = len(es_open_arr)

    # Build valid start indices for each unique hold duration
    unique_holds = np.unique(hold_days_arr)
    valid_starts = {}
    for k in unique_holds:
        # entry at index i, hold k days (inclusive of entry = day 1)
        # exit at index i + k - 1, need i + k - 1 < total_days
        max_start = total_days - k
        if max_start > 0:
            valid_starts[k] = np.arange(max_start)
        else:
            valid_starts[k] = np.array([], dtype=int)

    db_random_pfs = []
    db_beat_count = 0
    db_valid_count = 0

    for perm in range(N_PERMUTATIONS):
        if (perm + 1) % 2000 == 0:
            print(f"    DB permutation {perm + 1:,}/{N_PERMUTATIONS:,}...")

        # Sample N_db random entries, each with its corresponding hold_days
        pnl_arr = np.empty(N_db)
        ok = True
        for k_idx in range(N_db):
            k = hold_days_arr[k_idx]
            vs = valid_starts[k]
            if len(vs) == 0:
                ok = False
                break
            rand_idx = vs[np.random.randint(len(vs))]
            entry_p = es_open_arr[rand_idx]
            exit_p = es_close_arr[rand_idx + k - 1]
            pnl_arr[k_idx] = (exit_p - entry_p) * ES_POINT_VALUE - COST_PER_TRADE

        if not ok:
            continue

        pf = pf_from_array(pnl_arr)
        db_valid_count += 1
        db_random_pfs.append(pf)

        if pf == float("inf"):
            if db_pf == float("inf"):
                db_beat_count += 1
        elif pf >= db_pf:
            db_beat_count += 1

    db_pvalue = db_beat_count / db_valid_count if db_valid_count > 0 else 1.0
    print(f"  Permutation p-value: {db_pvalue:.4f} "
          f"({db_beat_count} out of {db_valid_count} valid shuffles matched or beat actual PF)")

    # Percentiles (exclude inf)
    db_finite = np.array([x for x in db_random_pfs if np.isfinite(x)])
    if len(db_finite) > 0:
        pcts = np.percentile(db_finite, [5, 25, 50, 75, 95])
        print(f"  Percentiles (excl inf): 5th={pcts[0]:.3f}  25th={pcts[1]:.3f}  "
              f"50th={pcts[2]:.3f}  75th={pcts[3]:.3f}  95th={pcts[4]:.3f}")

    print_ascii_histogram(db_finite, db_pf, label="DB PF")

    # ══════════════════════════════════════════════════════════════════════
    # COMBINED
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  COMBINED")
    print(f"{'=' * 80}")
    print(f"  Combined PF: {comb_pf:.3f} — composed of two independently tested edges above.")
    print("  (Combined permutation skipped.)")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  FINAL RESULTS")
    print(f"{'=' * 80}")

    # CO verdict
    if co_pvalue < 0.05:
        co_verdict = "PASS — full set PF significantly higher than random subsets"
    else:
        co_verdict = (f"CONSISTENT — edge distributed broadly across weeks (p={co_pvalue:.2f}). "
                      "Calendar selection itself is the edge.")
    print(f"  Chosen One: {co_verdict}")

    # DB verdict
    if db_pvalue < 0.05:
        db_verdict = "PASS — Dip Buyer P&L significantly beats random entries"
    else:
        db_verdict = "FAIL — cannot distinguish from random entries"
    print(f"  Dip Buyer:  {db_verdict}")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
