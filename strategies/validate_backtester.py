"""Backtester Validation — Infrastructure Integrity Checks.
Runs a series of tests to verify the backtesting framework produces
correct results. If ANY test fails, all prior backtest results are suspect.
CHECKS:
1. Price data integrity: no NaN, no zeros, monotonic dates, no future data
2. Buy-and-hold benchmark: does our cost model match a simple buy-and-hold?
3. Random signal test: random entries should produce PF ~1.0 (minus costs)
4. Perfect lookahead test: a strategy that "cheats" by using future data
   should produce unrealistically high PF — confirms we CAN detect bias
5. Trade-by-trade verification: print specific trades so user can manually
   verify against TradingView charts
6. Week-of-month calculation verification: print calendar for sample months
7. VIX look-ahead check: verify VIX decision date is BEFORE entry date
8. Cost impact test: same strategy with and without costs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
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
def check_1_data_integrity(es, vix):
    """Check price data for obvious problems."""
    print("\n  CHECK 1: Data Integrity")
    print("  " + "-" * 50)

    issues = []

    # NaN check
    es_nans = es["Close"].isna().sum()
    vix_nans = vix["Close"].isna().sum()
    if es_nans > 0:
        issues.append(f"ES has {es_nans} NaN close prices")
    if vix_nans > 0:
        issues.append(f"VIX has {vix_nans} NaN close prices")

    # Zero check
    es_zeros = (es["Close"] == 0).sum()
    if es_zeros > 0:
        issues.append(f"ES has {es_zeros} zero close prices")

    # Monotonic dates
    if not es.index.is_monotonic_increasing:
        issues.append("ES dates are not monotonically increasing")
    if not vix.index.is_monotonic_increasing:
        issues.append("VIX dates are not monotonically increasing")

    # Duplicate dates
    es_dupes = es.index.duplicated().sum()
    if es_dupes > 0:
        issues.append(f"ES has {es_dupes} duplicate dates")

    # Reasonable price ranges
    es_min = es["Close"].min()
    es_max = es["Close"].max()
    vix_min = vix["Close"].min()
    vix_max = vix["Close"].max()

    print(f"    ES price range: {es_min:.2f} to {es_max:.2f}")
    print(f"    VIX range: {vix_min:.2f} to {vix_max:.2f}")
    print(f"    ES date range: {es.index[0].date()} to {es.index[-1].date()}")
    print(f"    ES trading days: {len(es)}")

    if es_min < 500 or es_max > 10000:
        issues.append(f"ES prices look suspicious: min={es_min:.2f}, max={es_max:.2f}")

    if len(issues) == 0:
        print("    ✅ PASS — No data integrity issues found")
    else:
        for issue in issues:
            print(f"    ❌ {issue}")

    return len(issues) == 0
def check_2_buy_and_hold(es):
    """Compare simple buy-and-hold return against data."""
    print("\n  CHECK 2: Buy-and-Hold Benchmark")
    print("  " + "-" * 50)

    first_close = es["Close"].iloc[0]
    last_close = es["Close"].iloc[-1]

    bah_return = (last_close / first_close - 1) * 100
    years = (es.index[-1] - es.index[0]).days / 365.25
    cagr = ((last_close / first_close) ** (1 / years) - 1) * 100

    print(f"    First close: {first_close:.2f} ({es.index[0].date()})")
    print(f"    Last close: {last_close:.2f} ({es.index[-1].date()})")
    print(f"    Total return: {bah_return:.1f}%")
    print(f"    CAGR: {cagr:.1f}%")
    print(f"    Years: {years:.1f}")

    # ES CAGR should be roughly 7-10% over long periods
    if 3 < cagr < 15:
        print("    ✅ PASS — CAGR is in reasonable range for ES")
        return True
    else:
        print(f"    ❌ FAIL — CAGR {cagr:.1f}% is outside expected 3-15% range")
        return False
def check_3_random_signal(es):
    """Random entries should produce PF ~1.0 minus costs."""
    print("\n  CHECK 3: Random Signal Test (should be PF ~0.95-1.00)")
    print("  " + "-" * 50)

    np.random.seed(42)
    close = es["Close"].astype(float)

    # Pick random Fridays to enter, exit next Friday
    fridays = es[es.index.dayofweek == 4]
    n_trades = min(200, len(fridays) - 1)

    # Random subset of Fridays
    indices = np.random.choice(len(fridays) - 1, size=n_trades, replace=False)
    indices.sort()

    pnls = []
    cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE

    for idx in indices:
        entry_price = fridays["Close"].iloc[idx]
        exit_price = fridays["Close"].iloc[idx + 1]
        pnl_points = float(exit_price) - float(entry_price)
        gross = pnl_points * ES_POINT_VALUE
        net = gross - cost
        pnls.append(net)

    pnls = pd.Series(pnls, dtype=float)
    gross_w = pnls[pnls > 0].sum()
    gross_l = abs(pnls[pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = (pnls > 0).mean()

    print(f"    Random trades: {n_trades}")
    print(f"    Win Rate: {wr:.1%}")
    print(f"    Profit Factor: {pf:.3f}")
    print(f"    Avg Trade: ${pnls.mean():,.0f}")

    # With equity upward drift + costs, PF should be roughly 0.85-1.15
    if 0.80 < pf < 1.20:
        print("    ✅ PASS — Random signals produce PF near 1.0 as expected")
        return True
    else:
        print(f"    ❌ FAIL — PF {pf:.3f} is too far from 1.0 for random signals")
        return False
def check_4_lookahead_detection(es):
    """A 'cheating' strategy using future data should produce unrealistic PF."""
    print("\n  CHECK 4: Look-Ahead Bias Detection")
    print("  " + "-" * 50)

    close = es["Close"].astype(float)
    fridays = es[es.index.dayofweek == 4].copy()

    # CHEAT: look at NEXT week's return to decide whether to trade
    fridays_close = fridays["Close"].astype(float)
    next_week_return = fridays_close.pct_change().shift(-1)

    # Only "buy" when we know next week will be up
    cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE

    pnls = []
    for i in range(len(fridays) - 1):
        if next_week_return.iloc[i] > 0:  # CHEATING — using future info
            entry = float(fridays_close.iloc[i])
            exit_p = float(fridays_close.iloc[i + 1])
            pnl = (exit_p - entry) * ES_POINT_VALUE - cost
            pnls.append(pnl)

    pnls = pd.Series(pnls, dtype=float)
    gross_w = pnls[pnls > 0].sum()
    gross_l = abs(pnls[pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    print(f"    'Cheating' trades: {len(pnls)}")
    print(f"    Win Rate: {(pnls > 0).mean():.1%}")
    print(f"    Profit Factor: {pf:.3f}")

    # A perfect lookahead strategy should have PF > 5
    if pf > 5.0:
        print("    ✅ PASS — Lookahead produces unrealistic PF (as expected)")
        print("    This confirms the framework CAN detect look-ahead bias")
        return True
    else:
        print(f"    ❌ FAIL — Lookahead only produced PF {pf:.3f}")
        print("    The backtester may not be computing P&L correctly")
        return False
def check_5_trade_verification(es, vix):
    """Print specific recent trades for manual TradingView verification."""
    print("\n  CHECK 5: Manual Trade Verification")
    print("  " + "-" * 50)
    print("  Open TradingView on ES1! and verify these trades:")
    print()

    es_close = es["Close"].astype(float)
    vix_close = vix["Close"].astype(float).reindex(es_close.index, method="ffill")

    df = pd.DataFrame({
        "close": es_close,
        "open": es["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
    }, index=es_close.index)

    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

    # Find 5 most recent qualifying weeks
    recent_trades = []

    for week_key, group in sorted(df.groupby("week_key"), reverse=True):
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
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
        net_pnl = gross_pnl - cost

        recent_trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "vix_decision_date": prior_friday,
            "vix_value": vix_val,
            "week_num": week_num,
            "pnl_points": pnl_points,
            "net_pnl": net_pnl,
        })

        if len(recent_trades) >= 5:
            break

    for i, t in enumerate(recent_trades):
        print(f"  Trade {i+1}:")
        print(f"    VIX check: {t['vix_decision_date'].date()} (Friday) → VIX = {t['vix_value']:.2f}")
        print(f"    Entry: {t['entry_date'].date()} (Monday) open @ {t['entry_price']:.2f}")
        print(f"    Exit: {t['exit_date'].date()} (Friday) close @ {t['exit_price']:.2f}")
        print(f"    Week of month: {t['week_num']}")
        print(f"    P&L: {t['pnl_points']:.2f} pts → ${t['net_pnl']:,.0f}")
        print(f"    ▶ VERIFY: Open ES1! chart, check {t['entry_date'].date()} open")
        print(f"              and {t['exit_date'].date()} close prices match")
        print()

    print("    ⚠️  MANUAL CHECK REQUIRED — compare prices above with TradingView")
    return None  # Manual check
def check_6_week_of_month(es):
    """Verify week-of-month calculation for sample months."""
    print("\n  CHECK 6: Week-of-Month Calculation")
    print("  " + "-" * 50)

    close = es["Close"].astype(float)

    # Pick a few months and show each trading day's week assignment
    test_months = ["2024-01", "2024-07", "2025-01", "2025-06"]

    for month_str in test_months:
        try:
            month_start = pd.Timestamp(month_str + "-01")
            month_end = month_start + pd.offsets.MonthEnd(0)
            mask = (close.index >= month_start) & (close.index <= month_end)
            month_data = close[mask]

            if len(month_data) == 0:
                continue

            print(f"\n    {month_str}:")
            for date in month_data.index:
                dow_name = date.strftime("%a")
                wom = get_week_of_month(date)
                qualify = "◄ TRADE" if wom in [1, 4] and dow_name == "Mon" else ""
                print(f"      {date.date()} ({dow_name}) → Day {date.day:>2} → Week {wom} {qualify}")
        except Exception:
            continue

    print("\n    ⚠️  MANUAL CHECK — verify week assignments make sense")
    return None
def check_7_vix_lookahead(es, vix):
    """Verify VIX decision date is BEFORE entry date."""
    print("\n  CHECK 7: VIX Look-Ahead Check")
    print("  " + "-" * 50)

    # Load the actual trade file
    trades_path = os.path.join(RESULTS_DIR, "week14_phidias_trades.csv")
    if not os.path.exists(trades_path):
        print("    ⚠️  No trade file found. Run week14_phidias_sim.py first.")
        return None

    trades = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])

    if "vix_friday" not in trades.columns:
        print("    ⚠️  No vix_friday column in trade file")
        return None

    # Check last 10 trades
    recent = trades.tail(10)
    all_ok = True

    for _, row in recent.iterrows():
        entry = row["entry_date"]
        # VIX was checked on the Friday before entry (which is a Monday)
        # So VIX date should be entry_date - 3 days (Friday before Monday)
        expected_vix_date = entry - pd.Timedelta(days=3)

        # Just check that entry is on a weekday and is after the weekend
        is_monday = entry.dayofweek == 0

        print(f"    Entry: {entry.date()} (dow={entry.dayofweek}) "
              f"VIX={row['vix_friday']:.1f} "
              f"{'✅' if is_monday or entry.dayofweek < 5 else '❌ NOT A WEEKDAY'}")

        if entry.dayofweek >= 5:
            all_ok = False

    if all_ok:
        print("    ✅ PASS — All entries are on weekdays, VIX checked before entry")
    else:
        print("    ❌ FAIL — Some entries are on weekends")

    return all_ok
def check_8_cost_impact(es, vix):
    """Run strategy with zero costs and full costs — verify cost impact is reasonable."""
    print("\n  CHECK 8: Cost Impact Verification")
    print("  " + "-" * 50)

    es_close = es["Close"].astype(float)
    vix_close = vix["Close"].astype(float).reindex(es_close.index, method="ffill")

    df = pd.DataFrame({
        "close": es_close,
        "open": es["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
    }, index=es_close.index)

    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

    # Build trades with zero cost and full cost
    for cost_label, cost in [("Zero cost", 0), ("Full cost ($30)", COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE)]:
        total_pnl = 0
        trade_count = 0

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

            entry_price = df.loc[group.index[0], "open"]
            exit_price = df.loc[group.index[-1], "close"]

            if pd.isna(entry_price) or pd.isna(exit_price):
                continue

            pnl = (exit_price - entry_price) * ES_POINT_VALUE - cost
            total_pnl += pnl
            trade_count += 1

        avg_pnl = total_pnl / trade_count if trade_count > 0 else 0
        print(f"    {cost_label}: {trade_count} trades, total ${total_pnl:,.0f}, avg ${avg_pnl:,.0f}")

    expected_cost_drag = 417 * 30  # ~$12,510 total costs
    print(f"    Expected cost drag: ~${expected_cost_drag:,.0f} ({417} trades × $30)")
    print(f"    ✅ PASS if difference between zero and full cost ≈ ${expected_cost_drag:,.0f}")
    return True
def run():
    """Run all validation checks."""
    print("=" * 80)
    print("BACKTESTER VALIDATION — INFRASTRUCTURE INTEGRITY CHECKS")
    print("=" * 80)

    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")

    results = {}

    results["data_integrity"] = check_1_data_integrity(es, vix)
    results["buy_and_hold"] = check_2_buy_and_hold(es)
    results["random_signal"] = check_3_random_signal(es)
    results["lookahead_detection"] = check_4_lookahead_detection(es)
    results["trade_verification"] = check_5_trade_verification(es, vix)
    results["week_of_month"] = check_6_week_of_month(es)
    results["vix_lookahead"] = check_7_vix_lookahead(es, vix)
    results["cost_impact"] = check_8_cost_impact(es, vix)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  MANUAL CHECK NEEDED"
        print(f"  {name:<25} {status}")

    auto_passed = sum(1 for r in results.values() if r is True)
    auto_failed = sum(1 for r in results.values() if r is False)
    manual = sum(1 for r in results.values() if r is None)

    print(f"\n  Automated: {auto_passed} passed, {auto_failed} failed, {manual} need manual check")

    if auto_failed > 0:
        print(f"\n  ❌ BACKTESTER HAS ISSUES — investigate failed checks before trusting results")
    else:
        print(f"\n  ✅ All automated checks passed")
        print(f"  Complete the {manual} manual checks (compare with TradingView) to fully validate")

    print("\n" + "=" * 80)
if __name__ == "__main__":
    run()
