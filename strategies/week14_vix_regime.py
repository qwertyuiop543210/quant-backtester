"""Week 1 & Week 4 ES strategy split by VIX regime at entry.

Same base strategy as v1: 1 ES contract, Monday open to Friday close,
weeks 1 and 4 only. Results split by VIX level at Monday open.

Regimes:
  1: VIX < 15
  2: VIX 15-20
  3: VIX 20-25
  4: VIX 25-30
  5: VIX > 30

Filtered versions:
  A: Trade all regimes (baseline = v1)
  B: Skip when VIX > 25
  C: Skip when VIX > 20
  D: Only trade when VIX < 20, use 2 contracts instead of 1

Topstep simulation on all four: $150K, $4,500 trailing DD, $9,000 target.
ES costs: $5 RT commission, $12.50 slippage per side.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate, max_drawdown
from core.plotting import plot_equity

STRATEGY_NAME = "Week 1 & Week 4 — VIX Regime Analysis"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
INITIAL_CAPITAL = 100_000.0
ACTIVE_WEEKS = {1, 4}

TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
TOPSTEP_MONTHLY_FEE = 165.0

REGIMES = [
    (1, "VIX < 15",  0.0, 15.0),
    (2, "VIX 15-20", 15.0, 20.0),
    (3, "VIX 20-25", 20.0, 25.0),
    (4, "VIX 25-30", 25.0, 30.0),
    (5, "VIX > 30",  30.0, 999.0),
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


def classify_regime(vix_level: float) -> int:
    for regime_id, _, lo, hi in REGIMES:
        if lo <= vix_level < hi:
            return regime_id
    return 5


def build_trades(es: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Build all week 1 & 4 trades with VIX regime at entry."""
    vix_close = vix["Close"].reindex(es.index).ffill()
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

        # VIX at Monday open — use previous trading day's close
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
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "vix_at_entry": vix_at_entry,
            "regime": classify_regime(vix_at_entry),
        })

    return pd.DataFrame(trades)


def build_equity_curve(trade_list: pd.DataFrame, es: pd.DataFrame,
                       contracts: int = 1) -> tuple[pd.Series, pd.Series]:
    """Build equity curve and position series from filtered trade list."""
    close = es["Close"].astype(float)
    open_ = es["Open"].astype(float)
    equity = pd.Series(INITIAL_CAPITAL, index=es.index, dtype=float)
    position = pd.Series(0.0, index=es.index, dtype=float)
    cash = INITIAL_CAPITAL

    weeks = find_trading_weeks(es.index)
    week_lookup = {w["monday_date"]: w for w in weeks}

    last_idx = 0
    for _, t in trade_list.iterrows():
        w = week_lookup.get(t["entry_date"])
        if w is None:
            continue
        mi = w["monday_idx"]
        fi = w["friday_idx"]
        entry_price = open_.iloc[mi]

        for k in range(last_idx, mi):
            equity.iloc[k] = cash

        for k in range(mi, fi + 1):
            mtm = (close.iloc[k] - entry_price) * ES_POINT_VALUE * contracts
            equity.iloc[k] = cash + mtm
            position.iloc[k] = float(contracts)

        exit_price = close.iloc[fi]
        pnl = (exit_price - entry_price) * ES_POINT_VALUE * contracts - COST_PER_TRADE * contracts
        cash += pnl
        last_idx = fi + 1

    for k in range(last_idx, len(close)):
        equity.iloc[k] = cash

    return equity, position


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


def topstep_summary(attempts: list[dict]) -> tuple[int, int, float]:
    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    failed = sum(1 for a in attempts if a["status"] == "FAILED")
    total = passed + failed
    rate = passed / total * 100 if total > 0 else 0
    return passed, total, rate


def run():
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1993-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")
    print(f"Costs: ${COST_PER_TRADE:.0f}/trade\n")

    all_trades = build_trades(es, vix)
    print(f"Total week 1 & 4 trades: {len(all_trades)}")

    # --- Regime breakdown ---
    print(f"\n{'='*85}")
    print(f"  VIX Regime Breakdown at Entry")
    print(f"{'='*85}")
    print(f"  {'Regime':<12} {'Trades':>7} {'PF':>8} {'WinRate':>9} "
          f"{'AvgPnL':>10} {'TotalPnL':>12} {'MaxDD':>10}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*9} {'-'*10} {'-'*12} {'-'*10}")

    for regime_id, regime_label, _, _ in REGIMES:
        subset = all_trades[all_trades["regime"] == regime_id]
        if len(subset) == 0:
            print(f"  {regime_label:<12} {0:>7} {'---':>8} {'---':>9} "
                  f"{'---':>10} {'---':>12} {'---':>10}")
            continue

        sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
        pf = profit_factor(sub_pnls)
        wr = win_rate(sub_pnls)
        avg = sub_pnls.mean()
        total = sub_pnls.sum()

        # Max drawdown from cumulative P&L
        cum = INITIAL_CAPITAL + sub_pnls.cumsum()
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min()
        dd_pct = abs(dd)

        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        print(f"  {regime_label:<12} {len(subset):>7} {pf_str:>8} {wr:>8.1%} "
              f"${avg:>9,.0f} ${total:>11,.0f} {dd_pct:>9.1%}")

    print(f"{'='*85}\n")

    # --- Define four versions ---
    versions = {}

    # Version A: all regimes (baseline)
    versions["A"] = {
        "label": "A: All regimes (v1 baseline)",
        "trades": all_trades,
        "contracts": 1,
    }

    # Version B: skip VIX > 25
    versions["B"] = {
        "label": "B: Skip VIX > 25",
        "trades": all_trades[all_trades["vix_at_entry"] <= 25].reset_index(drop=True),
        "contracts": 1,
    }

    # Version C: skip VIX > 20
    versions["C"] = {
        "label": "C: Skip VIX > 20",
        "trades": all_trades[all_trades["vix_at_entry"] <= 20].reset_index(drop=True),
        "contracts": 1,
    }

    # Version D: only VIX < 20, 2 contracts
    d_trades = all_trades[all_trades["vix_at_entry"] < 20].reset_index(drop=True)
    versions["D"] = {
        "label": "D: VIX < 20 only, 2 contracts",
        "trades": d_trades,
        "contracts": 2,
    }

    # --- Per-version stats ---
    print(f"{'='*90}")
    print(f"  Filtered Version Comparison")
    print(f"{'='*90}")
    print(f"  {'Version':<35} {'Trades':>7} {'PF':>8} {'WinRate':>9} "
          f"{'AnnRet':>8} {'MaxDD':>8} {'TotalPnL':>12}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*12}")

    version_results = {}
    for key in ["A", "B", "C", "D"]:
        v = versions[key]
        tl = v["trades"]
        contracts = v["contracts"]

        if len(tl) == 0:
            print(f"  {v['label']:<35} {0:>7} {'---':>8}")
            continue

        # Scale P&L by contracts
        scaled_pnls = tl["pnl"].values * contracts
        trade_pnls = pd.Series(scaled_pnls, dtype=float)

        equity, position = build_equity_curve(tl, es, contracts)
        stats = summary(trade_pnls, equity, position)

        total_pnl = trade_pnls.sum()
        pf_str = f"{stats['profit_factor']:.3f}" if stats["profit_factor"] != float("inf") else "inf"
        print(f"  {v['label']:<35} {stats['total_trades']:>7} {pf_str:>8} "
              f"{stats['win_rate']:>8.1%} {stats['annualized_return']:>7.1%} "
              f"{stats['max_drawdown']:>7.1%} ${total_pnl:>11,.0f}")

        version_results[key] = {
            "stats": stats,
            "trade_pnls": trade_pnls,
            "equity": equity,
        }

    print(f"{'='*90}\n")

    # --- Topstep simulation for all versions ---
    print(f"{'='*80}")
    print(f"  Topstep $150K Eval Comparison")
    print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing DD, "
          f"${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
    print(f"{'='*80}\n")

    topstep_results = {}
    for key in ["A", "B", "C", "D"]:
        v = versions[key]
        if key not in version_results:
            continue

        pnl_list = version_results[key]["trade_pnls"].tolist()
        attempts = simulate_topstep(pnl_list)
        passed, total, rate = topstep_summary(attempts)
        topstep_results[key] = {"passed": passed, "total": total, "rate": rate, "attempts": attempts}

        print(f"  --- {v['label']} ---")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} "
              f"{'Peak Bal':>12} {'Final Bal':>12}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

        for a in attempts:
            print(f"  {a['attempt']:>3} {a['status']:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} "
                  f"${a['final_balance']:>11,.0f}")

        print(f"  Pass rate: {rate:.1f}% ({passed}/{total})\n")

    # --- Summary table ---
    print(f"{'='*80}")
    print(f"  Topstep Pass Rate Summary")
    print(f"{'='*80}")
    print(f"  {'Version':<35} {'Pass Rate':>12} {'Passed':>10} {'Total':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10}")

    for key in ["A", "B", "C", "D"]:
        if key not in topstep_results:
            continue
        v = versions[key]
        tr = topstep_results[key]
        print(f"  {v['label']:<35} {tr['rate']:>11.1f}% {tr['passed']:>10} {tr['total']:>10}")

    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  {'ES v1 baseline (reference)':.<35} {'41.6%':>12}")
    print(f"{'='*80}\n")

    # Save outputs
    best_key = max(topstep_results, key=lambda k: topstep_results[k]["rate"]) if topstep_results else "A"
    if best_key in version_results:
        plot_equity(version_results[best_key]["equity"],
                    f"{STRATEGY_NAME} — Best: {versions[best_key]['label']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "week14_vix_regime_trades.csv")
    all_trades.to_csv(csv_path, index=False)
    print(f"Trade list saved to {csv_path}")

    return version_results


if __name__ == "__main__":
    run()
