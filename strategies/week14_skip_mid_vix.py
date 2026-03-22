"""Week 1 & Week 4 ES — skip mid-VIX (15-20) regime.

Same as v1: 1 ES contract, Monday open to Friday close, weeks 1 and 4 only.
BUT skip any week where VIX is between 15 and 20 at Monday open.
Trade only when VIX < 15 OR VIX > 20.

ES costs: $5 RT commission, $12.50 slippage per side.
Topstep: $150K account, $4,500 trailing DD, $9,000 profit target.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Week 1 & Week 4 — Skip Mid-VIX (15-20)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

VIX_SKIP_LO = 15.0
VIX_SKIP_HI = 20.0

TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_MONTHLY_FEE = 165.0

ES_V1_PASS_RATE = 41.6


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


def simulate_topstep(trade_pnls: list[float]) -> list[dict]:
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

        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade + 1,
            "end_trade": i,
            "trades_taken": i - start_trade,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - TOPSTEP_CAPITAL,
            "max_trailing_dd": high_water - balance if status == "FAILED" else trailing_dd,
            "status": status,
        })

        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break

    return attempts


def run():
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1993-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    vix_close = vix["Close"].reindex(es.index).ffill()
    open_ = es["Open"].astype(float)
    close = es["Close"].astype(float)

    weeks = find_trading_weeks(es.index)
    active = [w for w in weeks if w["week_of_month"] in ACTIVE_WEEKS]

    print(f"\nTotal trading weeks: {len(weeks)}")
    print(f"Active weeks (1 & 4): {len(active)}")
    print(f"Filter: skip when VIX {VIX_SKIP_LO}-{VIX_SKIP_HI} at Monday open")
    print(f"Costs: ${COST_PER_TRADE:.0f}/trade\n")

    # Build trades for both v1 (all) and filtered
    v1_trades = []
    filtered_trades = []

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

        trade = {
            "entry_date": w["monday_date"],
            "exit_date": w["friday_date"],
            "week_of_month": w["week_of_month"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "vix_at_entry": vix_at_entry,
            "monday_idx": mi,
            "friday_idx": fi,
        }

        v1_trades.append(trade)

        # Skip mid-VIX
        if not (VIX_SKIP_LO <= vix_at_entry < VIX_SKIP_HI):
            filtered_trades.append(trade)

    v1_df = pd.DataFrame(v1_trades)
    filt_df = pd.DataFrame(filtered_trades)
    skipped = len(v1_df) - len(filt_df)

    print(f"v1 trades (all):     {len(v1_df)}")
    print(f"Filtered trades:     {len(filt_df)}")
    print(f"Skipped (VIX 15-20): {skipped} ({skipped/len(v1_df)*100:.1f}%)")

    # VIX regime breakdown of skipped vs kept
    print(f"\n{'='*70}")
    print(f"  Trade Distribution by VIX at Entry")
    print(f"{'='*70}")
    print(f"  {'VIX Range':<15} {'v1':>6} {'Filtered':>10} {'Skipped':>9}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*9}")
    for label, lo, hi in [("< 15", 0, 15), ("15-20", 15, 20), ("20-25", 20, 25),
                           ("25-30", 25, 30), ("> 30", 30, 999)]:
        v1_n = len(v1_df[(v1_df["vix_at_entry"] >= lo) & (v1_df["vix_at_entry"] < hi)])
        f_n = len(filt_df[(filt_df["vix_at_entry"] >= lo) & (filt_df["vix_at_entry"] < hi)])
        s_n = v1_n - f_n
        marker = " <-- SKIPPED" if s_n > 0 else ""
        print(f"  {label:<15} {v1_n:>6} {f_n:>10} {s_n:>9}{marker}")
    print(f"{'='*70}\n")

    # Build equity curves
    def build_equity(trade_list, label):
        eq = pd.Series(INITIAL_CAPITAL, index=es.index, dtype=float)
        pos = pd.Series(0.0, index=es.index, dtype=float)
        cash = INITIAL_CAPITAL
        last_idx = 0

        for _, t in trade_list.iterrows():
            mi = t["monday_idx"]
            fi = t["friday_idx"]
            entry_price = open_.iloc[mi]

            for k in range(last_idx, mi):
                eq.iloc[k] = cash
            for k in range(mi, fi + 1):
                mtm = (close.iloc[k] - entry_price) * ES_POINT_VALUE
                eq.iloc[k] = cash + mtm
                pos.iloc[k] = 1.0

            exit_price = close.iloc[fi]
            cash += (exit_price - entry_price) * ES_POINT_VALUE - COST_PER_TRADE
            last_idx = fi + 1

        for k in range(last_idx, len(close)):
            eq.iloc[k] = cash

        return eq, pos

    v1_equity, v1_position = build_equity(v1_df, "v1")
    filt_equity, filt_position = build_equity(filt_df, "filtered")

    v1_pnls = pd.Series(v1_df["pnl"].values, dtype=float)
    filt_pnls = pd.Series(filt_df["pnl"].values, dtype=float)

    # Stats comparison
    v1_stats = summary(v1_pnls, v1_equity, v1_position)
    filt_stats = summary(filt_pnls, filt_equity, filt_position)

    print(f"{'='*80}")
    print(f"  v1 (All Trades) vs Filtered (Skip VIX 15-20)")
    print(f"{'='*80}")
    print(f"  {'Metric':<25} {'v1 (all)':>18} {'Skip 15-20':>18}")
    print(f"  {'-'*25} {'-'*18} {'-'*18}")

    for label, key, fmt in [
        ("Trades", "total_trades", "d"),
        ("Profit Factor", "profit_factor", ".3f"),
        ("Win Rate", "win_rate", ".1%"),
        ("Sharpe Ratio", "sharpe_ratio", ".3f"),
        ("Max Drawdown", "max_drawdown", ".1%"),
        ("Annualized Return", "annualized_return", ".1%"),
        ("Time in Market", "time_in_market", ".1%"),
    ]:
        v1_val = v1_stats[key]
        f_val = filt_stats[key]
        v1_str = f"{v1_val:{fmt}}" if v1_val != float("inf") else "inf"
        f_str = f"{f_val:{fmt}}" if f_val != float("inf") else "inf"
        print(f"  {label:<25} {v1_str:>18} {f_str:>18}")

    print(f"  {'Total P&L':<25} {'${:,.0f}'.format(v1_pnls.sum()):>18} "
          f"{'${:,.0f}'.format(filt_pnls.sum()):>18}")
    print(f"  {'Avg P&L/trade':<25} {'${:,.0f}'.format(v1_pnls.mean()):>18} "
          f"{'${:,.0f}'.format(filt_pnls.mean()):>18}")
    print(f"{'='*80}\n")

    # P&L of skipped trades
    skipped_mask = v1_df["vix_at_entry"].between(VIX_SKIP_LO, VIX_SKIP_HI, inclusive="left")
    skipped_pnls = v1_df.loc[skipped_mask, "pnl"]
    if len(skipped_pnls) > 0:
        print(f"  Skipped trades analysis:")
        print(f"    Count:      {len(skipped_pnls)}")
        print(f"    Total P&L:  ${skipped_pnls.sum():,.0f}")
        print(f"    Avg P&L:    ${skipped_pnls.mean():,.0f}")
        print(f"    Win rate:   {(skipped_pnls > 0).mean():.1%}")
        sk_pf = profit_factor(pd.Series(skipped_pnls.values, dtype=float))
        print(f"    PF:         {sk_pf:.3f}")
        print()

    # --- Topstep simulation ---
    v1_attempts = simulate_topstep(v1_pnls.tolist())
    filt_attempts = simulate_topstep(filt_pnls.tolist())

    for label, attempts in [("v1 (All Trades)", v1_attempts),
                             ("Filtered (Skip VIX 15-20)", filt_attempts)]:
        passed = sum(1 for a in attempts if a["status"] == "PASSED")
        failed = sum(1 for a in attempts if a["status"] == "FAILED")
        total = passed + failed
        rate = passed / total * 100 if total > 0 else 0

        print(f"{'='*80}")
        print(f"  Topstep $150K Eval — {label}")
        print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing DD, "
              f"${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
        print(f"{'='*80}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} "
              f"{'Peak Bal':>12} {'Final Bal':>12}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

        for a in attempts:
            print(f"  {a['attempt']:>3} {a['status']:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} "
                  f"${a['final_balance']:>11,.0f}")

        print(f"{'='*80}")
        print(f"  {label}: {total} attempts | Passed: {passed} | "
              f"Failed: {failed} | Pass rate: {rate:.1f}%\n")

    # Summary
    v1_passed = sum(1 for a in v1_attempts if a["status"] == "PASSED")
    v1_failed = sum(1 for a in v1_attempts if a["status"] == "FAILED")
    v1_total = v1_passed + v1_failed
    v1_rate = v1_passed / v1_total * 100 if v1_total > 0 else 0

    f_passed = sum(1 for a in filt_attempts if a["status"] == "PASSED")
    f_failed = sum(1 for a in filt_attempts if a["status"] == "FAILED")
    f_total = f_passed + f_failed
    f_rate = f_passed / f_total * 100 if f_total > 0 else 0

    print(f"{'='*76}")
    print(f"  Topstep Pass Rate Summary")
    print(f"{'='*76}")
    print(f"  {'Version':<35} {'Pass Rate':>12} {'Passed/Total':>15}")
    print(f"  {'-'*35} {'-'*12} {'-'*15}")
    print(f"  {'v1 (all trades)':.<35} {v1_rate:>11.1f}% {f'{v1_passed}/{v1_total}':>15}")
    print(f"  {'Skip VIX 15-20':.<35} {f_rate:>11.1f}% {f'{f_passed}/{f_total}':>15}")
    print(f"  {'ES v1 baseline (reference)':.<35} {ES_V1_PASS_RATE:>11.1f}%")
    delta = f_rate - v1_rate
    print(f"  {'-'*35} {'-'*12} {'-'*15}")
    print(f"  {'Delta (filtered vs v1)':.<35} {f'{delta:+.1f}%':>12}")
    print(f"{'='*76}\n")

    # Save outputs
    plot_equity(filt_equity, STRATEGY_NAME)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "week14_skip_mid_vix_trades.csv")
    filt_df.to_csv(csv_path, index=False)
    print(f"Trade list saved to {csv_path}")

    return filt_stats


if __name__ == "__main__":
    run()
