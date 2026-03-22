"""Two critical tests:
TEST 1: MES CONSISTENCY FIX
The Phidias 50K had 0/60 passes clean under the 30% consistency rule with 1 ES.
Does switching to 2 MES ($10/point instead of $50/point) fix this?
2 MES means avg trade ~$115 instead of ~$575, requiring ~35 trades to hit $4K target.
No single $115 trade can be 30% of $4,000 ($1,200 threshold).
Tests: 1 MES, 2 MES, 3 MES on Phidias 50K, 100K, 150K Swing accounts.
TEST 2: ALL-WEEKS MEAN REVERSION PERIOD BREAKDOWN
Variant F from the standalone backtest showed PF 1.398, 557 trades, Sharpe 1.755.
But we need to see if it's durable across all time periods, or if it's another
regime artifact like the week 2/3 version was.
If both pass: we have a two-account Phidias strategy.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
MES_POINT_VALUE = 5.0
COMMISSION_RT_ES = 5.0
COMMISSION_RT_MES = 1.24
SLIPPAGE_PER_SIDE_ES = 12.50
SLIPPAGE_PER_SIDE_MES = 1.25
PHIDIAS_ACCOUNTS = {
    "50K": {"capital": 50_000.0, "profit_target": 4_000.0, "trailing_dd": 2_500.0, "eval_cost": 116.0},
    "100K": {"capital": 100_000.0, "profit_target": 6_000.0, "trailing_dd": 3_000.0, "eval_cost": 144.60},
    "150K": {"capital": 150_000.0, "profit_target": 9_000.0, "trailing_dd": 4_500.0, "eval_cost": 172.60},
}
def get_week_of_month(date):
    day = date.day
    if day <= 7:
        return 1
    elif day <= 14:
        return 2
    elif day <= 21:
        return 3
    elif day <= 28:
        return 4
    else:
        return 5
# =============================================================================
# TEST 1: MES CONSISTENCY FIX
# =============================================================================
def build_calendar_trades(es_data, vix_data, num_contracts, point_value,
                          commission_rt, slippage_per_side):
    """Build Week 1/4 + VIX filter trades with specified sizing."""
    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")
    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
        "week_of_month": [get_week_of_month(d) for d in es_close.index],
    }, index=es_close.index)
    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)
    cost_per_trade = (num_contracts * commission_rt) + (num_contracts * 2 * slippage_per_side)
    trades = []
    for week_key, group in df.groupby("week_key", sort=True):
        if len(group) < 3:
            continue
        monday_candidates = group[group["dow"] == 0]
        first_day = monday_candidates.index[0] if len(monday_candidates) > 0 else group.index[0]
        week_num = get_week_of_month(first_day)
        if week_num not in [1, 4]:
            continue
        prior_days = df.index[df.index < group.index[0]]
        if len(prior_days) == 0:
            continue
        prior_friday = prior_days[-1]
        vix_val = df.loc[prior_friday, "vix"]
        if pd.isna(vix_val) or 15.0 <= vix_val <= 20.0:
            continue
        entry_day = group.index[0]
        exit_day = group.index[-1]
        entry_price = df.loc[entry_day, "open"]
        exit_price = df.loc[exit_day, "close"]
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * point_value * num_contracts
        net_pnl = gross_pnl - cost_per_trade
        trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "costs": cost_per_trade,
        })
    return trades
def simulate_phidias_with_consistency(trades, capital, profit_target, trailing_dd):
    """Simulate Phidias eval + check 30% consistency on each pass."""
    attempts = []
    i = 0
    while i < len(trades):
        balance = capital
        high_water = capital
        start_trade = i
        status = "in_progress"
        attempt_trades = []
        while i < len(trades):
            trade = trades[i]
            balance += trade["pnl"]
            high_water = max(high_water, balance)
            trailing = high_water - balance
            profit = balance - capital
            attempt_trades.append(trade)
            i += 1
            if trailing >= trailing_dd:
                status = "FAILED"
                break
            if profit >= profit_target:
                status = "PASSED"
                break
        # Check consistency rule on passes
        consistency_clean = True
        if status == "PASSED":
            total_profit = balance - capital
            if total_profit > 0:
                for t in attempt_trades:
                    if t["pnl"] > 0 and t["pnl"] / total_profit > 0.30:
                        consistency_clean = False
                        break
        attempts.append({
            "trades_taken": len(attempt_trades),
            "profit": balance - capital,
            "status": status,
            "consistency_clean": consistency_clean if status == "PASSED" else None,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
def run_test_1(es_data, vix_data):
    """TEST 1: MES consistency fix across all account sizes and contract counts."""
    print("=" * 80)
    print("TEST 1: MES SIZING — CONSISTENCY RULE FIX")
    print("=" * 80)
    sizing_options = [
        ("1 ES",  1, ES_POINT_VALUE,  COMMISSION_RT_ES,  SLIPPAGE_PER_SIDE_ES),
        ("1 MES", 1, MES_POINT_VALUE, COMMISSION_RT_MES, SLIPPAGE_PER_SIDE_MES),
        ("2 MES", 2, MES_POINT_VALUE, COMMISSION_RT_MES, SLIPPAGE_PER_SIDE_MES),
        ("3 MES", 3, MES_POINT_VALUE, COMMISSION_RT_MES, SLIPPAGE_PER_SIDE_MES),
        ("5 MES", 5, MES_POINT_VALUE, COMMISSION_RT_MES, SLIPPAGE_PER_SIDE_MES),
    ]
    results = {}
    for sizing_label, n_contracts, pv, comm, slip in sizing_options:
        trades = build_calendar_trades(es_data, vix_data, n_contracts, pv, comm, slip)
        avg_trade = np.mean([t["pnl"] for t in trades]) if trades else 0
        for acct_name, acct in PHIDIAS_ACCOUNTS.items():
            key = f"{sizing_label} / {acct_name}"
            attempts = simulate_phidias_with_consistency(
                trades, acct["capital"], acct["profit_target"], acct["trailing_dd"]
            )
            passed = [a for a in attempts if a["status"] == "PASSED"]
            failed = [a for a in attempts if a["status"] == "FAILED"]
            total = len(passed) + len(failed)
            pass_rate = len(passed) / total * 100 if total > 0 else 0
            clean = sum(1 for a in passed if a["consistency_clean"])
            clean_rate = clean / len(passed) * 100 if passed else 0
            avg_trades_to_pass = np.mean([a["trades_taken"] for a in passed]) if passed else 0
            expected_attempts = 1 / (pass_rate / 100) if pass_rate > 0 else float("inf")
            # For consistency: we need CLEAN passes, not just passes
            clean_pass_rate = (clean / total * 100) if total > 0 else 0
            expected_attempts_clean = 1 / (clean_pass_rate / 100) if clean_pass_rate > 0 else float("inf")
            expected_cost_clean = expected_attempts_clean * acct["eval_cost"]
            results[key] = {
                "sizing": sizing_label,
                "account": acct_name,
                "pass_rate": pass_rate,
                "passed": len(passed),
                "total": total,
                "clean": clean,
                "clean_rate": clean_rate,
                "clean_pass_rate": clean_pass_rate,
                "avg_trades_to_pass": avg_trades_to_pass,
                "avg_trade_pnl": avg_trade,
                "expected_cost_clean": expected_cost_clean,
                "eval_cost": acct["eval_cost"],
            }
    # Print results table
    print(f"\n  {'Config':<20} {'Account':<8} {'Pass%':>7} {'P/Tot':>7} "
          f"{'Clean':>7} {'Cln%':>7} {'ClnPR%':>7} {'AvgTr':>7} "
          f"{'Avg$':>8} {'$ToPass':>9}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*8} {'-'*9}")
    for key, r in sorted(results.items(), key=lambda x: (x[1]["sizing"], x[1]["account"])):
        cost_str = f"${r['expected_cost_clean']:,.0f}" if r['expected_cost_clean'] < 100000 else "N/A"
        print(f"  {r['sizing']:<20} {r['account']:<8} "
              f"{r['pass_rate']:>6.1f}% "
              f"{r['passed']}/{r['total']:>4} "
              f"{r['clean']:>5}/{r['passed']:<1} "
              f"{r['clean_rate']:>6.1f}% "
              f"{r['clean_pass_rate']:>6.1f}% "
              f"{r['avg_trades_to_pass']:>6.1f} "
              f"${r['avg_trade_pnl']:>6.0f} "
              f"{cost_str:>9}")
    # Find best config
    best = None
    best_key = None
    for key, r in results.items():
        if r["clean_pass_rate"] > 0:
            if best is None or r["expected_cost_clean"] < best["expected_cost_clean"]:
                best = r
                best_key = key
    print(f"\n  BEST CONFIG: {best_key}")
    if best:
        print(f"    Clean pass rate: {best['clean_pass_rate']:.1f}%")
        print(f"    Expected cost per clean pass: ${best['expected_cost_clean']:,.0f}")
        print(f"    Avg trades to pass: {best['avg_trades_to_pass']:.1f}")
        print(f"    Avg $ per trade: ${best['avg_trade_pnl']:,.0f}")
    return results
# =============================================================================
# TEST 2: ALL-WEEKS MEAN REVERSION PERIOD BREAKDOWN
# =============================================================================
def run_test_2(es_data):
    """TEST 2: All-weeks mean reversion (Variant F) — period breakdown."""
    print("\n\n" + "=" * 80)
    print("TEST 2: ALL-WEEKS MEAN REVERSION — PERIOD DURABILITY CHECK")
    print("Variant F: Down week only, no volume filter, ALL weeks")
    print("=" * 80)
    es_close = es_data["Close"].astype(float)
    es_volume = es_data["Volume"].astype(float)
    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "high": es_data["High"].astype(float),
        "low": es_data["Low"].astype(float),
        "volume": es_volume,
        "dow": es_close.index.dayofweek,
    }, index=es_close.index)
    # Get Fridays
    fridays = df[df["dow"] == 4].copy()
    fridays["prior_close"] = fridays["close"].shift(1)
    fridays["down_week"] = fridays["close"] < fridays["prior_close"]
    fridays["next_friday_close"] = fridays["close"].shift(-1)
    fridays = fridays.dropna(subset=["prior_close", "next_friday_close"])
    # Variant F: all weeks, down week only, no filters
    signal = fridays["down_week"]
    cost_per_trade = COMMISSION_RT_ES + 2 * SLIPPAGE_PER_SIDE_ES
    trades = []
    for friday_date in fridays.index[signal]:
        row = fridays.loc[friday_date]
        entry_price = row["close"]
        exit_price = row["next_friday_close"]
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - cost_per_trade
        trades.append({
            "entry_date": friday_date,
            "exit_date": friday_date + pd.Timedelta(days=7),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost_per_trade,
            "pnl": net_pnl,
            "year": friday_date.year,
            "week_of_month": get_week_of_month(friday_date + pd.Timedelta(days=3)),
        })
    trade_df = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_df["pnl"].values, dtype=float)
    # Overall stats
    print(f"\n  Overall: {len(trade_df)} trades")
    wins = (trade_pnls > 0).sum()
    losses = (trade_pnls <= 0).sum()
    gross_w = trade_pnls[trade_pnls > 0].sum()
    gross_l = abs(trade_pnls[trade_pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = wins / len(trade_pnls) if len(trade_pnls) > 0 else 0
    print(f"  PF: {pf:.3f} | Win Rate: {wr:.1%} | Avg: ${trade_pnls.mean():,.0f} | Total: ${trade_pnls.sum():,.0f}")
    # Period breakdown
    print(f"\n  PERIOD BREAKDOWN:")
    print(f"  {'Period':<12} {'Trades':>7} {'PF':>8} {'WR':>8} {'Avg$':>10} {'Total$':>12} {'Verdict':>12}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
    periods = [
        ("2000-2005", 2000, 2005),
        ("2006-2010", 2006, 2010),
        ("2011-2015", 2011, 2015),
        ("2016-2020", 2016, 2020),
        ("2021-2026", 2021, 2026),
    ]
    period_results = []
    for label, y1, y2 in periods:
        subset = trade_df[(trade_df["year"] >= y1) & (trade_df["year"] <= y2)]
        if len(subset) == 0:
            continue
        s_pnls = pd.Series(subset["pnl"].values, dtype=float)
        s_wins = (s_pnls > 0).sum()
        s_losses = (s_pnls <= 0).sum()
        s_gw = s_pnls[s_pnls > 0].sum()
        s_gl = abs(s_pnls[s_pnls < 0].sum())
        s_pf = s_gw / s_gl if s_gl > 0 else float("inf")
        s_wr = s_wins / len(s_pnls)
        s_avg = s_pnls.mean()
        s_total = s_pnls.sum()
        verdict = "✅" if s_pf > 1.0 else "❌"
        period_results.append({"label": label, "pf": s_pf, "profitable": s_pf > 1.0})
        print(f"  {label:<12} {len(subset):>7} {s_pf:>8.2f} {s_wr:>7.1%} "
              f"${s_avg:>9,.0f} ${s_total:>11,.0f} {verdict:>12}")
    # Year by year
    print(f"\n  YEAR-BY-YEAR:")
    print(f"  {'Year':<6} {'Trades':>7} {'PF':>8} {'WR':>8} {'Avg$':>10} {'Total$':>12}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
    for year in sorted(trade_df["year"].unique()):
        subset = trade_df[trade_df["year"] == year]
        s_pnls = pd.Series(subset["pnl"].values, dtype=float)
        s_gw = s_pnls[s_pnls > 0].sum()
        s_gl = abs(s_pnls[s_pnls < 0].sum())
        s_pf = s_gw / s_gl if s_gl > 0 else float("inf")
        s_wr = (s_pnls > 0).mean()
        print(f"  {year:<6} {len(subset):>7} {s_pf:>8.2f} {s_wr:>7.1%} "
              f"${s_pnls.mean():>9,.0f} ${s_pnls.sum():>11,.0f}")
    # By week of month
    print(f"\n  BY WEEK OF MONTH:")
    for wk in sorted(trade_df["week_of_month"].unique()):
        subset = trade_df[trade_df["week_of_month"] == wk]
        s_pnls = pd.Series(subset["pnl"].values, dtype=float)
        s_gw = s_pnls[s_pnls > 0].sum()
        s_gl = abs(s_pnls[s_pnls < 0].sum())
        s_pf = s_gw / s_gl if s_gl > 0 else float("inf")
        s_wr = (s_pnls > 0).mean()
        print(f"    Week {wk}: {len(subset)} trades, PF {s_pf:.2f}, WR {s_wr:.1%}, "
              f"avg ${s_pnls.mean():,.0f}")
    # Phidias sim for Variant F as independent account
    print(f"\n  PHIDIAS SIM — Variant F as independent 50K Swing account:")
    attempts = simulate_phidias_with_consistency(
        trades, 50_000.0, 4_000.0, 2_500.0
    )
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    total_att = len(passed) + len(failed)
    pass_rate = len(passed) / total_att * 100 if total_att > 0 else 0
    clean = sum(1 for a in passed if a["consistency_clean"])
    clean_rate = clean / len(passed) * 100 if passed else 0
    print(f"    Pass Rate: {pass_rate:.1f}% ({len(passed)}/{total_att})")
    print(f"    Consistency clean: {clean}/{len(passed)} ({clean_rate:.0f}%)")
    print(f"    Avg trades to pass: {np.mean([a['trades_taken'] for a in passed]):.1f}" if passed else "    N/A")
    # Verdict
    profitable_periods = sum(1 for p in period_results if p["profitable"])
    total_periods = len(period_results)
    print(f"\n  DURABILITY VERDICT:")
    print(f"  Profitable periods: {profitable_periods}/{total_periods}")
    if profitable_periods >= 4:
        print(f"  ✅ DURABLE — Edge present in {profitable_periods}/{total_periods} periods")
        print(f"     Suitable as independent Phidias account (Account 2)")
    elif profitable_periods >= 3:
        print(f"  ⚠️  MIXED — Only {profitable_periods}/{total_periods} periods profitable")
        print(f"     Use with caution. May be regime-dependent.")
    else:
        print(f"  ❌ NOT DURABLE — Only {profitable_periods}/{total_periods} periods profitable")
        print(f"     Do not deploy as independent strategy.")
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "mr_allweeks_variant_f_trades.csv")
    trade_df.to_csv(csv_path, index=False)
    print(f"\n  Trades saved to {csv_path}")
    pnls = trade_df["pnl"].values
    eq_vals = [100_000.0]
    for p in pnls:
        eq_vals.append(eq_vals[-1] + p)
    dates = [trade_df["entry_date"].iloc[0] - pd.Timedelta(days=7)]
    dates.extend(trade_df["entry_date"].tolist())
    eq = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
    plot_equity(eq, "All-Weeks Mean Reversion (Variant F)")
    return trade_df
def run():
    """Run both tests."""
    print("Loading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()}")
    # TEST 1
    test1_results = run_test_1(es, vix)
    # TEST 2
    test2_trades = run_test_2(es)
    # FINAL SUMMARY
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY — TWO-ACCOUNT STRATEGY FEASIBILITY")
    print("=" * 80)
    print(f"\n  Account 1: Week 1/4 Calendar + VIX Filter")
    print(f"    → Check TEST 1 above for optimal MES sizing and consistency")
    print(f"\n  Account 2: All-Weeks Mean Reversion (Variant F)")
    print(f"    → Check TEST 2 above for period durability")
    print(f"\n  If both pass: run two Phidias 50K Swing accounts simultaneously")
    print(f"  Total eval cost: $232 one-time ($116 × 2)")
    print(f"  Strategies are uncorrelated (calendar effect vs mean reversion)")
    print(f"  Combined weekly coverage: near 100% of trading weeks")
    print("\n" + "=" * 80)
if __name__ == "__main__":
    run()
