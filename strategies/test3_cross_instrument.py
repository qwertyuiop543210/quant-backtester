"""Test 3 — Cross-instrument validation on NQ, YM, RTY/IWM.

Applies the Chosen One and Dip Buyer strategies with ZERO parameter changes
to other equity-index instruments. Checks if the calendar/reversion edges
transfer beyond ES.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
np.random.seed(42)

import pandas as pd

from core.data_loader import get_data
from strategies.validation_helpers import (
    load_data, generate_co_trades, generate_db_trades,
    pf_from_array, wr_from_array,
)

POINT_VALUES = {
    "ES": 50.0,
    "NQ": 20.0,
    "YM": 5.0,
    "RTY": 50.0,
    "IWM": 1.0,
}


def directional_hit_rate(df):
    """Fraction of trades where exit_price > entry_price (ignoring costs)."""
    if len(df) == 0:
        return 0.0
    return float((df["exit_price"] > df["entry_price"]).sum() / len(df))


def main():
    print("=" * 80)
    print("  TEST 3 — CROSS-INSTRUMENT VALIDATION")
    print("=" * 80)

    # ── Load ES and VIX ───────────────────────────────────────────────────
    es, vix = load_data()

    # ── Load other instruments ────────────────────────────────────────────
    print()
    nq = get_data("NQ", start="2012-01-01")
    print(f"NQ  range: {nq.index[0].date()} to {nq.index[-1].date()} ({len(nq)} days)")

    ym = get_data("YM", start="2012-01-01")
    print(f"YM  range: {ym.index[0].date()} to {ym.index[-1].date()} ({len(ym)} days)")

    # RTY with IWM fallback
    rty_is_etf = False
    rty_label = "RTY"
    try:
        rty = get_data("RTY", start="2012-01-01")
        if len(rty) < 1000:
            raise ValueError(f"RTY has only {len(rty)} rows, falling back to IWM")
    except Exception as e:
        print(f"  RTY unavailable ({e}), falling back to IWM...")
        rty = get_data("IWM", start="2012-01-01")
        rty_is_etf = True
        rty_label = "IWM"
    print(f"{rty_label}  range: {rty.index[0].date()} to {rty.index[-1].date()} ({len(rty)} days)")

    rty_pv = POINT_VALUES["IWM"] if rty_is_etf else POINT_VALUES["RTY"]

    # ── Build instrument list ─────────────────────────────────────────────
    instruments = [
        ("ES", es, POINT_VALUES["ES"]),
        ("NQ", nq, POINT_VALUES["NQ"]),
        ("YM", ym, POINT_VALUES["YM"]),
        (rty_label, rty, rty_pv),
    ]

    # ── Run strategies on each instrument ─────────────────────────────────
    rows = []
    for name, df, pv in instruments:
        # Chosen One
        co = generate_co_trades(df, vix, vix_skip_lo=15.0, vix_skip_hi=20.0,
                                point_value=pv, cost=30.0, start_date="2012-01-01")
        co_pnls = co["pnl"].values.astype(float) if len(co) > 0 else np.array([])
        co_pf = pf_from_array(co_pnls)
        co_wr = wr_from_array(co_pnls)
        co_dir = directional_hit_rate(co)
        co_total = float(co_pnls.sum()) if len(co_pnls) > 0 else 0.0

        # Dip Buyer
        db = generate_db_trades(df, vix, point_value=pv, cost=30.0,
                                start_date="2012-01-01")
        db_pnls = db["pnl"].values.astype(float) if len(db) > 0 else np.array([])
        db_pf = pf_from_array(db_pnls)
        db_wr = wr_from_array(db_pnls)
        db_dir = directional_hit_rate(db)
        db_total = float(db_pnls.sum()) if len(db_pnls) > 0 else 0.0

        rows.append({
            "name": name, "pv": pv,
            "co_trades": len(co), "co_pf": co_pf, "co_wr": co_wr,
            "co_dir": co_dir, "co_total": co_total,
            "db_trades": len(db), "db_pf": db_pf, "db_wr": db_wr,
            "db_dir": db_dir, "db_total": db_total,
        })

    # ── Print results table ───────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  RESULTS")
    print(f"{'=' * 80}\n")

    header = (f"  {'Instrument':<12}| {'Pt Val':>6} | {'CO Trds':>7} | {'CO PF':>7} | "
              f"{'CO WR':>6} | {'CO Dir%':>7} | {'DB Trds':>7} | {'DB PF':>7} | "
              f"{'DB WR':>6} | {'DB Dir%':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in rows:
        co_pf_s = f"{r['co_pf']:.3f}" if np.isfinite(r['co_pf']) else "inf"
        db_pf_s = f"{r['db_pf']:.3f}" if np.isfinite(r['db_pf']) else "inf"
        print(f"  {r['name']:<12}| {r['pv']:>6.0f} | {r['co_trades']:>7} | {co_pf_s:>7} | "
              f"{r['co_wr']:>5.1%} | {r['co_dir']:>6.1%} | {r['db_trades']:>7} | {db_pf_s:>7} | "
              f"{r['db_wr']:>5.1%} | {r['db_dir']:>6.1%}")

    # ── P&L summary ──────────────────────────────────────────────────────
    print(f"\n  {'Instrument':<12}| {'CO Total P&L':>14} | {'DB Total P&L':>14}")
    print("  " + "-" * 44)
    for r in rows:
        print(f"  {r['name']:<12}| ${r['co_total']:>13,.0f} | ${r['db_total']:>13,.0f}")

    # ── Verdicts per instrument ───────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  INSTRUMENT VERDICTS (based on Chosen One)")
    print(f"{'=' * 80}\n")

    verdicts = {}
    for r in rows:
        name = r["name"]
        co_dir = r["co_dir"]
        co_pf = r["co_pf"]
        is_etf = (name == "IWM")

        if co_dir > 0.53 and (co_pf > 1.0 or is_etf):
            verdict = "EDGE CONFIRMED"
        elif co_dir > 0.53:
            verdict = "DIRECTIONAL EDGE"
        else:
            verdict = "NO EDGE"

        verdicts[name] = verdict
        note = ""
        if is_etf:
            note = "  (ETF: PF ignored due to $1/pt vs $30 cost)"
        print(f"  {name:<6} CO Dir={co_dir:.1%}  CO PF={co_pf:.3f}  → {verdict}{note}")

    # ── IWM caveat ────────────────────────────────────────────────────────
    if rty_is_etf:
        print(f"\n  NOTE: IWM is an ETF proxy ($1/point). The $30 round-trip cost")
        print(f"  dominates at this scale, so PF is not meaningful. Directional")
        print(f"  hit rate is the appropriate metric for IWM.")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  FINAL CROSS-INSTRUMENT VERDICT")
    print(f"{'=' * 80}\n")

    non_es = [name for name in verdicts if name != "ES"]
    edge_count = sum(1 for name in non_es if verdicts[name] in ("EDGE CONFIRMED", "DIRECTIONAL EDGE"))

    print(f"  Non-ES instruments with edge: {edge_count} / {len(non_es)}")
    for name in non_es:
        print(f"    {name}: {verdicts[name]}")

    print()
    if edge_count >= 2:
        print(f"  *** PASS — {edge_count}/{len(non_es)} non-ES instruments show a calendar edge.")
        print(f"    The Chosen One effect transfers across equity-index instruments.")
    else:
        print(f"  *** FAIL — Only {edge_count}/{len(non_es)} non-ES instruments show an edge.")
        print(f"    Insufficient evidence that the calendar effect is universal.")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
