"""Multi-instrument Week 1 & Week 4 strategy.

Runs the Week 1 & Week 4 strategy (buy Monday open, sell Friday close) simultaneously
across three uncorrelated futures: ES (S&P 500), ZB (30-year bonds), GC (gold).
Each instrument gets its own independent signal. Uses the overlapping date range
where all three have data.

Reports:
1. Per-instrument stats (PF, win rate, trades, max DD, annualized return)
2. Combined portfolio stats (sum of all three P&Ls)
3. Correlation matrix of weekly returns across instruments
4. Topstep simulation on combined portfolio ($150K, $4,500 trailing DD, $9,000 target)
5. Combined pass rate vs ES-only v1 baseline

Instrument specs:
  ES=F: 1 contract, $50/point, $5 RT commission, $12.50 slippage/side
  ZB=F: 1 contract, $1000/point, $5 RT commission, $15.625 slippage/side
  GC=F: 1 contract, $100/point, $5 RT commission, $10 slippage/side
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Multi-Instrument Week 1 & Week 4 (ES+ZB+GC)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

# Instrument specifications
INSTRUMENTS = {
    "ES": {
        "symbol": "ES",
        "point_value": 50.0,
        "commission_rt": 5.0,
        "slippage_per_side": 12.50,
    },
    "ZB": {
        "symbol": "ZB",
        "point_value": 1000.0,
        "commission_rt": 5.0,
        "slippage_per_side": 15.625,
    },
    "GC": {
        "symbol": "GC",
        "point_value": 100.0,
        "commission_rt": 5.0,
        "slippage_per_side": 10.0,
    },
}

# Topstep eval parameters
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_MONTHLY_FEE = 165.0

# ES-only v1 baseline pass rate for comparison
ES_V1_PASS_RATE = 41.6


def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Find Monday-Friday trading week boundaries."""
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


def backtest_single_instrument(name: str, spec: dict, common_dates: pd.DatetimeIndex,
                               data: pd.DataFrame) -> dict:
    """Run week 1 & week 4 backtest on a single instrument.

    Returns dict with trade_list DataFrame, per-week pnl Series keyed by
    (year, month, wom), and summary stats.
    """
    point_value = spec["point_value"]
    cost = spec["commission_rt"] + 2 * spec["slippage_per_side"]

    # Reindex data to common dates
    df = data.reindex(common_dates).ffill().dropna()
    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)

    weeks = find_trading_weeks(df.index)
    active = [w for w in weeks if w["week_of_month"] in ACTIVE_WEEKS]

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=df.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=df.index)
    last_equity_idx = 0

    # Weekly returns keyed by monday_date for correlation
    weekly_returns = {}

    for w in active:
        mi = w["monday_idx"]
        fi = w["friday_idx"]

        entry_price = open_.iloc[mi]
        exit_price = close.iloc[fi]
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * point_value
        net_pnl = gross_pnl - cost

        for k in range(last_equity_idx, mi):
            equity.iloc[k] = cash
        for k in range(mi, fi + 1):
            mtm = (close.iloc[k] - entry_price) * point_value
            equity.iloc[k] = cash + mtm
            position.iloc[k] = 1.0

        cash += net_pnl
        last_equity_idx = fi + 1

        monday_date = w["monday_date"]
        weekly_returns[monday_date] = net_pnl

        trades.append({
            "entry_date": monday_date,
            "exit_date": w["friday_date"],
            "week_of_month": w["week_of_month"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost,
            "pnl": net_pnl,
        })

    for k in range(last_equity_idx, len(df)):
        equity.iloc[k] = cash

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)

    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )

    return {
        "name": name,
        "trade_list": trade_list,
        "trade_pnls": trade_pnls,
        "equity": equity,
        "position": position,
        "stats": stats,
        "weekly_returns": weekly_returns,
    }


def simulate_topstep(trade_pnls: list[float]) -> list[dict]:
    """Simulate Topstep $150K eval attempts."""
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
    """Run multi-instrument week 1 & week 4 backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data for ES, ZB, GC...")

    # Load all three instruments
    raw_data = {}
    for name, spec in INSTRUMENTS.items():
        raw_data[name] = get_data(spec["symbol"], start="1997-01-01")
        print(f"  {name}: {raw_data[name].index[0].date()} to "
              f"{raw_data[name].index[-1].date()} ({len(raw_data[name])} days)")

    # Find overlapping date range
    start_date = max(df.index[0] for df in raw_data.values())
    end_date = min(df.index[-1] for df in raw_data.values())
    print(f"\nOverlapping range: {start_date.date()} to {end_date.date()}")

    # Build common business day index from the union of all instrument dates
    all_dates = raw_data["ES"].index
    for name in ["ZB", "GC"]:
        all_dates = all_dates.union(raw_data[name].index)
    common_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    print(f"Common trading days: {len(common_dates)}\n")

    # Print instrument specs
    print(f"{'='*75}")
    print(f"  Instrument Specifications")
    print(f"{'='*75}")
    print(f"  {'Instr':>6} {'Pt Value':>10} {'Comm RT':>10} {'Slip/Side':>10} {'Cost/Trade':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for name, spec in INSTRUMENTS.items():
        cost = spec["commission_rt"] + 2 * spec["slippage_per_side"]
        print(f"  {name:>6} ${spec['point_value']:>8,.0f} ${spec['commission_rt']:>8.2f} "
              f"${spec['slippage_per_side']:>8.3f} ${cost:>10.2f}")
    print(f"{'='*75}\n")

    # Run backtest for each instrument
    results = {}
    for name, spec in INSTRUMENTS.items():
        results[name] = backtest_single_instrument(name, spec, common_dates, raw_data[name])

    # --- 1. Per-instrument stats ---
    print(f"{'='*80}")
    print(f"  Per-Instrument Results")
    print(f"{'='*80}")
    print(f"  {'Instr':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'MaxDD':>8} "
          f"{'AnnRet':>9} {'TotalPnL':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*12}")

    for name in INSTRUMENTS:
        s = results[name]["stats"]
        total_pnl = results[name]["trade_pnls"].sum()
        pf_str = f"{s['profit_factor']:.3f}" if s["profit_factor"] != float("inf") else "inf"
        print(f"  {name:>6} {s['total_trades']:>8} {pf_str:>8} {s['win_rate']:>8.1%} "
              f"{s['max_drawdown']:>7.1%} {s['annualized_return']:>8.1%} ${total_pnl:>11,.0f}")

    print(f"{'='*80}\n")

    # --- 2. Combined portfolio ---
    # Build combined equity curve by summing daily P&L across instruments
    combined_equity = pd.Series(INITIAL_CAPITAL, index=common_dates, dtype=float)
    combined_position = pd.Series(0.0, index=common_dates, dtype=float)

    # Sum the per-instrument equity changes
    for name in INSTRUMENTS:
        inst_eq = results[name]["equity"].reindex(common_dates, method="ffill")
        inst_pos = results[name]["position"].reindex(common_dates, fill_value=0.0)
        # Add the P&L component (equity - initial capital)
        combined_equity += (inst_eq - INITIAL_CAPITAL)
        combined_position = combined_position.clip(lower=0) + inst_pos

    # Combined position: 1 if any instrument has a position
    combined_position = (combined_position > 0).astype(float)

    # Combine trade P&Ls aligned by week
    # Get all unique monday dates across instruments
    all_mondays = set()
    for name in INSTRUMENTS:
        all_mondays.update(results[name]["weekly_returns"].keys())
    all_mondays = sorted(all_mondays)

    combined_weekly_pnls = []
    for monday in all_mondays:
        week_pnl = sum(
            results[name]["weekly_returns"].get(monday, 0.0)
            for name in INSTRUMENTS
        )
        combined_weekly_pnls.append(week_pnl)

    combined_trade_pnls = pd.Series(combined_weekly_pnls, dtype=float)

    combined_stats = summary(
        trade_pnls=combined_trade_pnls,
        equity_curve=combined_equity,
        position_series=combined_position,
    )

    print(f"{'='*80}")
    print(f"  Combined Portfolio Results (ES + ZB + GC)")
    print(f"{'='*80}")
    print_summary(combined_stats, "Combined Portfolio")

    # --- 3. Correlation matrix ---
    print(f"{'='*75}")
    print(f"  Weekly Return Correlations")
    print(f"{'='*75}")

    # Build DataFrame of weekly returns
    corr_data = {}
    for name in INSTRUMENTS:
        wr = results[name]["weekly_returns"]
        corr_data[name] = pd.Series(wr)

    corr_df = pd.DataFrame(corr_data).dropna()
    corr_matrix = corr_df.corr()

    # Print correlation matrix
    names = list(INSTRUMENTS.keys())
    header = "        " + "  ".join(f"{n:>8}" for n in names)
    print(f"  {header}")
    for row_name in names:
        vals = "  ".join(f"{corr_matrix.loc[row_name, col]:>8.3f}" for col in names)
        print(f"    {row_name:>4}  {vals}")

    print(f"\n  Avg pairwise correlation: {_avg_pairwise_corr(corr_matrix, names):.3f}")
    print(f"{'='*75}\n")

    # --- 4. Topstep simulation on combined portfolio ---
    combined_attempts = simulate_topstep(combined_weekly_pnls)

    print(f"{'='*80}")
    print(f"  Topstep $150K Eval — Combined Portfolio (ES+ZB+GC)")
    print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing DD, "
          f"${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
    print(f"{'='*80}")
    print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} "
          f"{'Peak Bal':>12} {'Final Bal':>12}")
    print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    comb_passed = 0
    comb_failed = 0
    for a in combined_attempts:
        status_str = a["status"]
        print(f"  {a['attempt']:>3} {status_str:>10} {a['trades_taken']:>8} "
              f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} "
              f"${a['final_balance']:>11,.0f}")
        if status_str == "PASSED":
            comb_passed += 1
        elif status_str == "FAILED":
            comb_failed += 1

    comb_total = comb_passed + comb_failed
    comb_rate = comb_passed / comb_total * 100 if comb_total > 0 else 0

    print(f"{'='*80}")
    print(f"  Combined: {comb_total} attempts | "
          f"Passed: {comb_passed} | Failed: {comb_failed} | "
          f"Pass rate: {comb_rate:.1f}%")

    # Months-to-pass analysis
    trades_per_month = 2.0  # ~2 active weeks per month

    if comb_passed > 0:
        pa = [a for a in combined_attempts if a["status"] == "PASSED"]
        fa = [a for a in combined_attempts if a["status"] == "FAILED"]
        avg_trades_pass = np.mean([a["trades_taken"] for a in pa])
        avg_months_pass = avg_trades_pass / trades_per_month
        avg_attempts = comb_total / comb_passed
        avg_trades_fail = np.mean([a["trades_taken"] for a in fa]) if fa else 0
        avg_months_fail = avg_trades_fail / trades_per_month
        expected_months = (avg_attempts - 1) * avg_months_fail + avg_months_pass
        expected_cost = expected_months * TOPSTEP_MONTHLY_FEE

        print(f"\n  Avg trades to pass:           {avg_trades_pass:.1f}")
        print(f"  Avg attempts before passing:  {avg_attempts:.1f}")
        print(f"  Expected months to pass:      {expected_months:.1f}")
        print(f"  Expected eval cost:           ${expected_cost:,.0f}")

    # --- 5. Combined vs ES-only comparison ---
    # Run ES-only Topstep for direct comparison on same date range
    es_pnl_list = results["ES"]["trade_pnls"].tolist()
    es_attempts = simulate_topstep(es_pnl_list)
    es_passed = sum(1 for a in es_attempts if a["status"] == "PASSED")
    es_failed = sum(1 for a in es_attempts if a["status"] == "FAILED")
    es_total = es_passed + es_failed
    es_rate = es_passed / es_total * 100 if es_total > 0 else 0

    print(f"\n  {'='*76}")
    print(f"  Combined vs ES-Only Comparison")
    print(f"  {'='*76}")
    print(f"  {'Metric':<35} {'ES-only':>18} {'Combined':>18}")
    print(f"  {'-'*35} {'-'*18} {'-'*18}")
    print(f"  {'Pass rate (this data)':.<35} {es_rate:>17.1f}% {comb_rate:>17.1f}%")
    print(f"  {'Passed / Total':.<35} {f'{es_passed}/{es_total}':>18} {f'{comb_passed}/{comb_total}':>18}")
    print(f"  {'ES v1 baseline (reference)':.<35} {ES_V1_PASS_RATE:>17.1f}% {'':>18}")

    delta = comb_rate - es_rate
    print(f"  {'-'*35} {'-'*18} {'-'*18}")
    print(f"  {'Delta (combined vs ES-only)':.<35} {'':>18} {f'{delta:+.1f}%':>18}")
    print(f"  {'='*76}")
    print(f"{'='*80}\n")

    # Save outputs
    plot_equity(combined_equity, STRATEGY_NAME)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Save combined trades
    all_trades = []
    for name in INSTRUMENTS:
        tl = results[name]["trade_list"].copy()
        tl.insert(0, "instrument", name)
        all_trades.append(tl)
    combined_trades_df = pd.concat(all_trades, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, "multi_instrument_week14_trades.csv")
    combined_trades_df.to_csv(csv_path, index=False)
    print(f"Trade list saved to {csv_path}")

    # Save correlation matrix
    corr_path = os.path.join(RESULTS_DIR, "multi_instrument_week14_correlations.csv")
    corr_matrix.to_csv(corr_path)
    print(f"Correlation matrix saved to {corr_path}")

    return combined_stats


def _avg_pairwise_corr(corr_matrix: pd.DataFrame, names: list[str]) -> float:
    """Average of off-diagonal pairwise correlations."""
    total = 0.0
    count = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            total += corr_matrix.loc[names[i], names[j]]
            count += 1
    return total / count if count > 0 else 0.0


if __name__ == "__main__":
    run()
