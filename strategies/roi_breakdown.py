"""Year-by-year ROI breakdown for CO, DB, and Combined strategies."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
np.random.seed(42)

import pandas as pd

from strategies.validation_helpers import (
    load_data, generate_co_trades, generate_db_trades,
)

ACCOUNT_SIZE = 50_000.0


def main():
    print("=" * 100)
    print("  YEAR-BY-YEAR ROI BREAKDOWN — CO, DB, Combined")
    print("=" * 100)

    es, vix = load_data()

    co = generate_co_trades(es, vix)
    db = generate_db_trades(es, vix)

    print(f"\n  CO trades: {len(co)}   DB trades: {len(db)}   "
          f"Combined: {len(co) + len(db)}")

    # ── Build year-by-year stats ─────────────────────────────────────────
    years = range(2012, 2026)  # 2012–2025 full years

    rows = []
    for year in years:
        co_yr = co[co["entry_date"].dt.year == year]
        db_yr = db[db["entry_date"].dt.year == year]

        co_n = len(co_yr)
        co_pnl = co_yr["pnl"].sum() if co_n else 0.0
        db_n = len(db_yr)
        db_pnl = db_yr["pnl"].sum() if db_n else 0.0

        comb_n = co_n + db_n
        comb_pnl = co_pnl + db_pnl
        roi_raw = comb_pnl / ACCOUNT_SIZE * 100
        roi_split = comb_pnl * 0.80 / ACCOUNT_SIZE * 100

        rows.append({
            "year": year, "co_n": co_n, "co_pnl": co_pnl,
            "db_n": db_n, "db_pnl": db_pnl,
            "comb_n": comb_n, "comb_pnl": comb_pnl,
            "roi_raw": roi_raw, "roi_split": roi_split,
        })

    # ── Print table ──────────────────────────────────────────────────────
    hdr = (f"  {'Year':>4s} | {'CO #':>4s} | {'CO P&L':>11s} | "
           f"{'DB #':>4s} | {'DB P&L':>11s} | "
           f"{'Comb #':>6s} | {'Comb P&L':>11s} | "
           f"{'ROI':>7s} | {'ROI 80%':>7s}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"\n{hdr}")
    print(sep)

    for r in rows:
        print(f"  {r['year']:>4d} | {r['co_n']:>4d} | ${r['co_pnl']:>+10,.0f} | "
              f"{r['db_n']:>4d} | ${r['db_pnl']:>+10,.0f} | "
              f"{r['comb_n']:>6d} | ${r['comb_pnl']:>+10,.0f} | "
              f"{r['roi_raw']:>6.1f}% | {r['roi_split']:>6.1f}%")

    # ── Totals / Averages ────────────────────────────────────────────────
    print(sep)

    tot_co_n = sum(r["co_n"] for r in rows)
    tot_co_pnl = sum(r["co_pnl"] for r in rows)
    tot_db_n = sum(r["db_n"] for r in rows)
    tot_db_pnl = sum(r["db_pnl"] for r in rows)
    tot_comb_n = sum(r["comb_n"] for r in rows)
    tot_comb_pnl = sum(r["comb_pnl"] for r in rows)
    tot_roi_raw = tot_comb_pnl / ACCOUNT_SIZE * 100
    tot_roi_split = tot_comb_pnl * 0.80 / ACCOUNT_SIZE * 100

    n_years = len(rows)
    avg_co_n = tot_co_n / n_years
    avg_co_pnl = tot_co_pnl / n_years
    avg_db_n = tot_db_n / n_years
    avg_db_pnl = tot_db_pnl / n_years
    avg_comb_n = tot_comb_n / n_years
    avg_comb_pnl = tot_comb_pnl / n_years
    avg_roi_raw = tot_roi_raw / n_years
    avg_roi_split = tot_roi_split / n_years

    print(f"  {'TOT':>4s} | {tot_co_n:>4d} | ${tot_co_pnl:>+10,.0f} | "
          f"{tot_db_n:>4d} | ${tot_db_pnl:>+10,.0f} | "
          f"{tot_comb_n:>6d} | ${tot_comb_pnl:>+10,.0f} | "
          f"{tot_roi_raw:>6.1f}% | {tot_roi_split:>6.1f}%")
    print(f"  {'AVG':>4s} | {avg_co_n:>4.1f} | ${avg_co_pnl:>+10,.0f} | "
          f"{avg_db_n:>4.1f} | ${avg_db_pnl:>+10,.0f} | "
          f"{avg_comb_n:>6.1f} | ${avg_comb_pnl:>+10,.0f} | "
          f"{avg_roi_raw:>6.1f}% | {avg_roi_split:>6.1f}%")

    # ── Summary stats ────────────────────────────────────────────────────
    comb_pnls = np.array([r["comb_pnl"] for r in rows])
    median_pnl = float(np.median(comb_pnls))
    worst = min(rows, key=lambda r: r["comb_pnl"])
    best = max(rows, key=lambda r: r["comb_pnl"])
    profitable = sum(1 for r in rows if r["comb_pnl"] > 0)

    print(f"\n  Median year P&L:       ${median_pnl:>+10,.0f}")
    print(f"  Worst year:            {worst['year']} (${worst['comb_pnl']:>+10,.0f})")
    print(f"  Best year:             {best['year']} (${best['comb_pnl']:>+10,.0f})")
    print(f"  Profitable years:      {profitable} / {n_years}")
    print(f"  Avg annual ROI (raw):  {avg_roi_raw:>+.1f}%")
    print(f"  Avg annual ROI (80%):  {avg_roi_split:>+.1f}%")
    print()


if __name__ == "__main__":
    main()
