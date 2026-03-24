"""Portfolio drawdown analysis for CO, DB, and Combined portfolios.

Computes:
  - Trade-level (closed P&L) max drawdown for each portfolio
  - Daily mark-to-market drawdown for Combined
  - Year-by-year drawdown for Combined
  - Worst individual trades
  - Worst consecutive losing streaks
"""

import pandas as pd
import numpy as np
from validation_helpers import (
    load_data, generate_co_trades, generate_db_trades,
    ES_POINT_VALUE, COST_PER_TRADE,
)

# ── Trade-level drawdown computation ─────────────────────────────────────────

def compute_trade_drawdown(trades: pd.DataFrame) -> dict:
    """Walk trades in exit_date order, track closed-P&L drawdown."""
    if trades.empty:
        return {
            "max_dd": 0.0, "peak_date": None, "trough_date": None,
            "recovery_date": None, "trades_to_recover": 0,
            "cum_pnl_series": [], "dd_series": [],
        }

    trades = trades.sort_values("exit_date").reset_index(drop=True)
    cum_pnl = 0.0
    hwm = 0.0
    max_dd = 0.0

    # For the max drawdown episode
    dd_peak_date = None       # exit_date that SET the HWM before DD began
    dd_trough_date = None     # exit_date of deepest point
    dd_recovery_date = None   # exit_date when cum_pnl >= hwm again
    trades_to_recover = 0

    # Track the HWM that was active when the worst DD started
    hwm_at_max_dd_start = 0.0
    trough_idx = -1

    cum_pnls = []
    dds = []

    for idx, row in trades.iterrows():
        cum_pnl += row["pnl"]
        cum_pnls.append(cum_pnl)

        if cum_pnl > hwm:
            hwm = cum_pnl

        dd = hwm - cum_pnl
        dds.append(dd)

        if dd > max_dd:
            max_dd = dd
            dd_trough_date = row["exit_date"]
            trough_idx = idx
            hwm_at_max_dd_start = hwm

    # Now find dd_peak_date: the exit_date of the trade that SET the HWM
    # Walk backwards from trough to find the last trade where cum_pnl == hwm_at_max_dd_start
    if trough_idx >= 0:
        for i in range(trough_idx, -1, -1):
            if cum_pnls[i] >= hwm_at_max_dd_start:
                dd_peak_date = trades.iloc[i]["exit_date"]
                break
        # If HWM was 0 (never had a winning trade before DD), peak is start
        if dd_peak_date is None and hwm_at_max_dd_start == 0.0:
            dd_peak_date = trades.iloc[0]["exit_date"]

        # Find recovery: first trade AFTER trough where cum_pnl >= hwm_at_max_dd_start
        recovered = False
        for i in range(trough_idx + 1, len(cum_pnls)):
            if cum_pnls[i] >= hwm_at_max_dd_start:
                dd_recovery_date = trades.iloc[i]["exit_date"]
                trades_to_recover = i - trough_idx
                recovered = True
                break

        if not recovered:
            trades_to_recover = 0  # still in DD or never recovered

    return {
        "max_dd": max_dd,
        "peak_date": dd_peak_date,
        "trough_date": dd_trough_date,
        "recovery_date": dd_recovery_date,
        "trades_to_recover": trades_to_recover,
        "cum_pnl_series": cum_pnls,
        "dd_series": dds,
    }


# ── Year-by-year drawdown (HWM carries forward) ─────────────────────────────

def compute_yearly_drawdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year max drawdown with carry-forward HWM."""
    if trades.empty:
        return pd.DataFrame()

    trades = trades.sort_values("exit_date").reset_index(drop=True)

    cum_pnl = 0.0
    hwm = 0.0
    rows = []

    # Pre-compute per-trade cumulative info
    trade_info = []
    for _, row in trades.iterrows():
        cum_pnl += row["pnl"]
        if cum_pnl > hwm:
            hwm = cum_pnl
        dd = hwm - cum_pnl
        trade_info.append({
            "exit_date": row["exit_date"],
            "pnl": row["pnl"],
            "cum_pnl": cum_pnl,
            "hwm": hwm,
            "dd": dd,
            "year": row["exit_date"].year,
        })

    info_df = pd.DataFrame(trade_info)

    for year, grp in info_df.groupby("year"):
        n_trades = len(grp)
        total_pnl = grp["pnl"].sum()
        max_dd = grp["dd"].max()

        # Find the trough trade for this year's max DD
        trough_row = grp.loc[grp["dd"].idxmax()]
        trough_date = trough_row["exit_date"]

        # Find the peak date: walk backward from trough to find where HWM was set
        trough_global_idx = trough_row.name
        hwm_val = trough_row["hwm"]
        dd_start_date = None
        for i in range(trough_global_idx, -1, -1):
            if info_df.iloc[i]["cum_pnl"] >= hwm_val:
                dd_start_date = info_df.iloc[i]["exit_date"]
                break
        if dd_start_date is None:
            dd_start_date = info_df.iloc[0]["exit_date"]

        rows.append({
            "year": year,
            "trades": n_trades,
            "total_pnl": total_pnl,
            "max_dd": max_dd,
            "dd_start": dd_start_date,
            "dd_trough": trough_date,
        })

    return pd.DataFrame(rows)


# ── Consecutive losing streaks ───────────────────────────────────────────────

def find_losing_streaks(trades: pd.DataFrame) -> list[dict]:
    """Find all consecutive losing streaks (pnl < 0) in exit_date order."""
    if trades.empty:
        return []

    trades = trades.sort_values("exit_date").reset_index(drop=True)
    streaks = []
    current_streak = 0
    current_loss = 0.0
    start_date = None

    for _, row in trades.iterrows():
        if row["pnl"] < 0:
            if current_streak == 0:
                start_date = row["exit_date"]
            current_streak += 1
            current_loss += row["pnl"]
            end_date = row["exit_date"]
        else:
            if current_streak > 0:
                streaks.append({
                    "consecutive_losses": current_streak,
                    "total_loss": current_loss,
                    "start_date": start_date,
                    "end_date": end_date,
                })
            current_streak = 0
            current_loss = 0.0
            start_date = None

    # Close any trailing streak
    if current_streak > 0:
        streaks.append({
            "consecutive_losses": current_streak,
            "total_loss": current_loss,
            "start_date": start_date,
            "end_date": end_date,
        })

    return streaks


# ── Daily MTM drawdown ───────────────────────────────────────────────────────

def compute_daily_mtm_drawdown(combined: pd.DataFrame, es: pd.DataFrame) -> dict:
    """Build daily P&L series for combined portfolio, compute MTM drawdown."""
    es_close = es["Close"].astype(float)
    es_open = es["Open"].astype(float)
    trading_days = es.index

    daily_pnl = pd.Series(0.0, index=trading_days)

    for _, trade in combined.iterrows():
        entry_date = trade["entry_date"]
        exit_date = trade["exit_date"]
        entry_price = float(es_open.loc[entry_date]) if entry_date in es.index else float(trade["entry_price"])

        trade_days = trading_days[(trading_days >= entry_date) & (trading_days <= exit_date)]

        prev = entry_price
        for day in trade_days:
            today_close = float(es_close.loc[day])
            daily_pnl.loc[day] += (today_close - prev) * ES_POINT_VALUE
            prev = today_close

        # Subtract cost on exit day
        daily_pnl.loc[exit_date] -= COST_PER_TRADE

    cum_equity = daily_pnl.cumsum()
    running_peak = cum_equity.cummax()
    dd_series = running_peak - cum_equity
    max_mtm_dd = dd_series.max()

    # Find the peak and trough dates for the max MTM drawdown
    trough_date = dd_series.idxmax()
    # Peak is the date of the running peak value at trough
    peak_val = running_peak.loc[trough_date]
    # Find the last date where cum_equity == peak_val, before trough
    candidates = cum_equity.loc[:trough_date]
    peak_date = candidates[candidates == peak_val].index[-1] if (candidates == peak_val).any() else candidates.idxmax()

    return {
        "max_mtm_dd": max_mtm_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "cum_equity": cum_equity,
        "dd_series": dd_series,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PORTFOLIO DRAWDOWN ANALYSIS")
    print("=" * 80)

    es, vix = load_data("2012-01-01")
    print()

    # Generate trades
    co_trades = generate_co_trades(es, vix, start_date="2012-01-01")
    db_trades = generate_db_trades(es, vix, start_date="2012-01-01")

    # Combined sorted by exit_date
    combined = pd.concat([co_trades, db_trades], ignore_index=True)
    combined = combined.sort_values("exit_date").reset_index(drop=True)

    print(f"CO trades: {len(co_trades)}   DB trades: {len(db_trades)}   Combined: {len(combined)}")
    print()

    # ── Table 1: Max drawdown summary ────────────────────────────────────────
    print("=" * 100)
    print("TABLE 1 — MAX DRAWDOWN SUMMARY")
    print("=" * 100)

    co_dd = compute_trade_drawdown(co_trades)
    db_dd = compute_trade_drawdown(db_trades)
    comb_dd = compute_trade_drawdown(combined)
    mtm = compute_daily_mtm_drawdown(combined, es)

    def fmt_dd(val):
        return f"${val:,.0f}"

    def fmt_date(d):
        if d is None:
            return "—"
        return pd.Timestamp(d).strftime("%Y-%m-%d")

    def fmt_recovery(dd_info):
        if dd_info["recovery_date"] is None:
            if dd_info["trough_date"] is not None:
                return "still in DD"
            return "—"
        return fmt_date(dd_info["recovery_date"])

    def fmt_trades_to_recover(dd_info):
        if dd_info["recovery_date"] is None:
            if dd_info["trough_date"] is not None:
                return "still in DD"
            return "—"
        return str(dd_info["trades_to_recover"])

    header = f"  {'Portfolio':<12} | {'Max DD (closed)':>16} | {'Max DD (MTM)':>16} | {'Peak Date':>12} | {'Trough Date':>12} | {'Recovery Date':>14} | {'Trades to Recover':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, dd_info, show_mtm in [
        ("CO-only", co_dd, False),
        ("DB-only", db_dd, False),
        ("Combined", comb_dd, True),
    ]:
        mtm_str = fmt_dd(mtm["max_mtm_dd"]) if show_mtm else "—"
        print(f"  {label:<12} | {fmt_dd(dd_info['max_dd']):>16} | {mtm_str:>16} | {fmt_date(dd_info['peak_date']):>12} | {fmt_date(dd_info['trough_date']):>12} | {fmt_recovery(dd_info):>14} | {fmt_trades_to_recover(dd_info):>18}")

    if mtm["peak_date"] is not None:
        print(f"\n  Daily MTM drawdown peak: {fmt_date(mtm['peak_date'])}  →  trough: {fmt_date(mtm['trough_date'])}")

    # ── Table 2: Year-by-year drawdown ───────────────────────────────────────
    print()
    print("=" * 100)
    print("TABLE 2 — YEAR-BY-YEAR MAX DRAWDOWN (Combined, closed P&L, HWM carries forward)")
    print("=" * 100)

    yearly = compute_yearly_drawdown(combined)

    header2 = f"  {'Year':>6} | {'Trades':>7} | {'Total P&L':>12} | {'Max DD (closed)':>16} | {'DD Start → Trough':>30}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    # Fill in years with no trades
    if not yearly.empty:
        all_years = range(int(yearly["year"].min()), int(yearly["year"].max()) + 1)
        for y in all_years:
            row = yearly[yearly["year"] == y]
            if row.empty:
                print(f"  {y:>6} | {'—':>7} | {'—':>12} | {'—':>16} | {'—':>30}")
            else:
                r = row.iloc[0]
                dd_range = f"{fmt_date(r['dd_start'])} → {fmt_date(r['dd_trough'])}"
                print(f"  {int(r['year']):>6} | {int(r['trades']):>7} | ${r['total_pnl']:>10,.0f} | {fmt_dd(r['max_dd']):>16} | {dd_range:>30}")

    # ── Table 3: Five worst individual trades ────────────────────────────────
    print()
    print("=" * 100)
    print("TABLE 3 — FIVE WORST INDIVIDUAL TRADES (Combined)")
    print("=" * 100)

    worst5 = combined.nsmallest(5, "pnl")

    header3 = f"  {'#':>3} | {'Date (entry→exit)':>28} | {'Strategy':>8} | {'Entry':>10} | {'Exit':>10} | {'P&L':>10} | {'VIX':>6}"
    print(header3)
    print("  " + "-" * (len(header3) - 2))

    for rank, (_, t) in enumerate(worst5.iterrows(), 1):
        dates = f"{fmt_date(t['entry_date'])} → {fmt_date(t['exit_date'])}"
        print(f"  {rank:>3} | {dates:>28} | {t['tag']:>8} | {t['entry_price']:>10,.2f} | {t['exit_price']:>10,.2f} | ${t['pnl']:>9,.0f} | {t['vix_at_entry']:>5.1f}")

    # ── Table 4: Five worst consecutive losing streaks ───────────────────────
    print()
    print("=" * 100)
    print("TABLE 4 — FIVE WORST CONSECUTIVE LOSING STREAKS (Combined)")
    print("=" * 100)

    streaks = find_losing_streaks(combined)
    # Sort by total loss (most negative first)
    streaks.sort(key=lambda s: s["total_loss"])
    top_streaks = streaks[:5]

    header4 = f"  {'#':>3} | {'Consecutive Losses':>19} | {'Total Loss':>12} | {'Start Date':>12} | {'End Date':>12}"
    print(header4)
    print("  " + "-" * (len(header4) - 2))

    if not top_streaks:
        print("  No losing streaks found.")
    else:
        for rank, s in enumerate(top_streaks, 1):
            print(f"  {rank:>3} | {s['consecutive_losses']:>19} | ${s['total_loss']:>11,.0f} | {fmt_date(s['start_date']):>12} | {fmt_date(s['end_date']):>12}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
