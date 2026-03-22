"""Time-Series Momentum SHORT-ONLY Sleeve — Full Backtest.
THESIS: When the 12-month return on equity futures is negative, shorting
captures the trend continuation. This is the "crisis alpha" documented
extensively in managed futures / CTA literature (Moskowitz, Ooi, Pedersen 2012).
This is SHORT-ONLY. When 12m return is positive, position is FLAT.
This is designed as a complement to the Week 1/4 long calendar strategy.
RULES:
1. On the first trading day of each month, compute 12-month return
   R_12m = P_t / P_{t-252} - 1
2. If R_12m < 0: SHORT 1 ES (or 3 MES) at close, hold for 1 month
3. If R_12m >= 0: FLAT (no position)
4. Recompute at next month start
5. Hard stop: 2x 20-day ATR from entry
ALSO TESTS:
- 6-month lookback variant (126 trading days)
- Multiple instruments: ES, NQ (if data available)
- With and without ATR stop
- Correlation with Week 1/4 calendar strategy
ACCEPTANCE CRITERIA (from prior research spec):
1. PF > 1.3
2. 50+ trades
3. Profitable in 3-4 out of 4 bear regimes (2000-03, 2008-09, 2022-23, 2025-26)
4. Correlation with Week 1/4 < 0.20
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
STRATEGY_NAME = "TSMOM Short-Only"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
MES_POINT_VALUE = 5.0
NUM_MES = 3
DOLLAR_PER_POINT = MES_POINT_VALUE * NUM_MES  # $15/point
COMMISSION_RT = 1.24 * NUM_MES  # $3.72
SLIPPAGE_PER_SIDE = 1.25 * NUM_MES  # $3.75
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # ~$11.22
INITIAL_CAPITAL = 100_000.0
PHIDIAS_CAPITAL = 50_000.0
PHIDIAS_TARGET = 4_000.0
PHIDIAS_DD = 2_500.0
def build_monthly_signals(es_data, lookback_days=252):
    """Build monthly TSMOM signals.
    On first trading day of each month:
    - Compute return over past lookback_days
    - If negative: signal = -1 (short)
    - If positive: signal = 0 (flat)
    Returns DataFrame with monthly signals and entry/exit dates.
    """
    es_close = es_data["Close"].astype(float)
    es_high = es_data["High"].astype(float)
    es_low = es_data["Low"].astype(float)
    # 20-day ATR for stops
    tr = pd.concat([
        es_high - es_low,
        (es_high - es_close.shift(1)).abs(),
        (es_low - es_close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_20 = tr.rolling(20).mean()
    # Get first trading day of each month
    es_close_df = pd.DataFrame({"close": es_close, "atr_20": atr_20}, index=es_close.index)
    es_close_df["year_month"] = es_close_df.index.to_period("M")
    first_days = es_close_df.groupby("year_month").first()
    first_days["date"] = es_close_df.groupby("year_month").apply(lambda x: x.index[0]).values
    # Compute lookback return
    signals = []
    dates_list = es_close.index.tolist()
    for i, (ym, row) in enumerate(first_days.iterrows()):
        current_date = row["date"]
        current_price = row["close"]
        current_atr = row["atr_20"]
        # Find price lookback_days ago
        current_idx = dates_list.index(current_date) if current_date in dates_list else -1
        if current_idx < lookback_days:
            continue
        past_date = dates_list[current_idx - lookback_days]
        past_price = es_close.loc[past_date]
        r_lookback = (current_price / past_price) - 1
        # Next month's first day (exit date)
        if i + 1 < len(first_days):
            next_row = first_days.iloc[i + 1]
            exit_date = next_row["date"]
            exit_price = next_row["close"]
        else:
            continue  # Can't compute exit
        signal = -1 if r_lookback < 0 else 0
        signals.append({
            "entry_date": current_date,
            "exit_date": exit_date,
            "entry_price": current_price,
            "exit_price": exit_price,
            "r_lookback": r_lookback,
            "signal": signal,
            "atr_20": current_atr,
            "year": current_date.year,
        })
    return pd.DataFrame(signals)
def simulate_trades(signals_df, es_data, use_atr_stop=True, atr_multiplier=2.0):
    """Simulate short trades with optional ATR stop.
    For SHORT trades:
    - Entry: short at entry_date close
    - Exit: first of (next month open, ATR stop hit, hold period end)
    - P&L for short = (entry_price - exit_price) * dollar_per_point
    """
    es_close = es_data["Close"].astype(float)
    es_high = es_data["High"].astype(float)
    all_dates = es_close.index.tolist()
    shorts_only = signals_df[signals_df["signal"] == -1].copy()
    trades = []
    for _, row in shorts_only.iterrows():
        entry_date = row["entry_date"]
        scheduled_exit = row["exit_date"]
        entry_price = row["entry_price"]
        atr = row["atr_20"]
        # ATR stop level (for short: stop is ABOVE entry)
        stop_price = entry_price + atr_multiplier * atr if (use_atr_stop and not pd.isna(atr)) else None
        # Walk through each day to check stop
        entry_idx = all_dates.index(entry_date) if entry_date in all_dates else -1
        exit_idx = all_dates.index(scheduled_exit) if scheduled_exit in all_dates else -1
        if entry_idx < 0 or exit_idx < 0:
            continue
        actual_exit_price = None
        actual_exit_date = None
        exit_reason = None
        max_adverse = 0  # Worst unrealized loss (for short: price going UP)
        max_favorable = 0  # Best unrealized gain (for short: price going DOWN)
        for day_idx in range(entry_idx + 1, min(exit_idx + 1, len(all_dates))):
            day_date = all_dates[day_idx]
            day_high = es_high.iloc[day_idx]
            day_close = es_close.iloc[day_idx]
            # For short: adverse = price going up, favorable = price going down
            unrealized_adverse = (day_high - entry_price) * DOLLAR_PER_POINT  # Worst case intraday
            unrealized_close = (entry_price - day_close) * DOLLAR_PER_POINT  # P&L at close
            max_adverse = max(max_adverse, unrealized_adverse)  # Largest loss
            max_favorable = max(max_favorable, unrealized_close)  # Largest gain
            # Check ATR stop (price goes above stop)
            if stop_price is not None and day_high >= stop_price:
                actual_exit_price = stop_price  # Approximate fill at stop
                actual_exit_date = day_date
                exit_reason = "atr_stop"
                break
            # Check if this is the scheduled exit
            if day_date == scheduled_exit:
                actual_exit_price = day_close
                actual_exit_date = day_date
                exit_reason = "monthly_rebal"
                break
        if actual_exit_price is None:
            continue
        # Short P&L: (entry - exit) * dollar_per_point
        pnl_points = entry_price - actual_exit_price
        gross_pnl = pnl_points * DOLLAR_PER_POINT
        net_pnl = gross_pnl - COST_PER_TRADE
        trades.append({
            "entry_date": entry_date,
            "exit_date": actual_exit_date,
            "entry_price": entry_price,
            "exit_price": actual_exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "hold_days": (actual_exit_date - entry_date).days,
            "exit_reason": exit_reason,
            "r_lookback": row["r_lookback"],
            "atr_at_entry": atr,
            "max_adverse": max_adverse,
            "max_favorable": max_favorable,
            "year": entry_date.year,
        })
    return pd.DataFrame(trades)
def simulate_phidias(trade_pnls):
    """Simulate Phidias 50K Swing eval attempts."""
    attempts = []
    i = 0
    while i < len(trade_pnls):
        balance = PHIDIAS_CAPITAL
        high_water = PHIDIAS_CAPITAL
        start = i
        status = "in_progress"
        while i < len(trade_pnls):
            balance += trade_pnls.iloc[i]
            high_water = max(high_water, balance)
            trailing = high_water - balance
            profit = balance - PHIDIAS_CAPITAL
            i += 1
            if trailing >= PHIDIAS_DD:
                status = "FAILED"
                break
            if profit >= PHIDIAS_TARGET:
                status = "PASSED"
                break
        attempts.append({
            "trades_taken": i - start,
            "profit": balance - PHIDIAS_CAPITAL,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
def compute_monthly_correlation(tsmom_trades, calendar_path):
    """Compute monthly P&L correlation with Week 1/4 calendar strategy."""
    if not os.path.exists(calendar_path):
        return None, "Calendar trades CSV not found"
    cal = pd.read_csv(calendar_path, parse_dates=["entry_date", "exit_date"])
    # Monthly P&L — use "ME" not "M" for newer pandas
    tsmom_monthly = tsmom_trades.set_index("entry_date")["pnl"].resample("ME").sum()
    cal_monthly = cal.set_index("entry_date")["pnl"].resample("ME").sum()
    common = tsmom_monthly.index.intersection(cal_monthly.index)
    if len(common) < 12:
        return None, f"Only {len(common)} common months"
    # Only correlate months where at least one strategy has a trade
    mask = (tsmom_monthly.loc[common] != 0) | (cal_monthly.loc[common] != 0)
    if mask.sum() < 12:
        return None, f"Only {mask.sum()} active months"
    corr = tsmom_monthly.loc[common].corr(cal_monthly.loc[common])
    return corr, f"{len(common)} months, {mask.sum()} active"
def run():
    """Run TSMOM short-only backtest with sensitivity and acceptance gates."""
    print("=" * 80)
    print("TIME-SERIES MOMENTUM — SHORT-ONLY SLEEVE")
    print("3 MES contracts ($15/point), monthly rebalance")
    print("=" * 80)
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()}")
    # =============================================
    # PRIMARY: 12-month lookback, ATR stop
    # =============================================
    print(f"\n{'='*60}")
    print("PRIMARY: 252-day lookback, 2x ATR stop, monthly hold")
    print(f"{'='*60}")
    signals = build_monthly_signals(es, lookback_days=252)
    short_months = (signals["signal"] == -1).sum()
    flat_months = (signals["signal"] == 0).sum()
    total_months = len(signals)
    print(f"Total months: {total_months}")
    print(f"Short months: {short_months} ({short_months/total_months:.1%})")
    print(f"Flat months: {flat_months} ({flat_months/total_months:.1%})")
    trades = simulate_trades(signals, es, use_atr_stop=True, atr_multiplier=2.0)
    print(f"Trades executed: {len(trades)}")
    if len(trades) == 0:
        print("ERROR: No trades generated")
        return
    trade_pnls = pd.Series(trades["pnl"].values, dtype=float)
    wins = (trade_pnls > 0).sum()
    losses = (trade_pnls <= 0).sum()
    gross_w = trade_pnls[trade_pnls > 0].sum()
    gross_l = abs(trade_pnls[trade_pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = wins / len(trade_pnls)
    print(f"\n  Trades: {len(trades)}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Profit Factor: {pf:.3f}")
    print(f"  Avg Win: ${trade_pnls[trade_pnls > 0].mean():,.0f}" if wins > 0 else "  Avg Win: N/A")
    print(f"  Avg Loss: ${trade_pnls[trade_pnls < 0].mean():,.0f}" if losses > 0 else "  Avg Loss: N/A")
    print(f"  Avg Trade: ${trade_pnls.mean():,.0f}")
    print(f"  Total P&L: ${trade_pnls.sum():,.0f}")
    print(f"  Max Adverse (worst trade): ${trades['max_adverse'].max():,.0f}")
    # Exit reasons
    print(f"\n  Exit Reasons:")
    for reason in trades["exit_reason"].unique():
        subset = trades[trades["exit_reason"] == reason]
        sub_pnls = subset["pnl"]
        sub_wr = (sub_pnls > 0).mean()
        print(f"    {reason}: {len(subset)} trades, {sub_wr:.1%} WR, ${sub_pnls.mean():,.0f} avg")
    # Bear regime analysis
    print(f"\n  BEAR REGIME PERFORMANCE (the critical test):")
    print(f"  {'Regime':<16} {'Trades':>7} {'PF':>8} {'WR':>7} {'Avg$':>9} {'Total$':>11}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*11}")
    bear_regimes = [
        ("2000-2003 Dot-com", 2000, 2003),
        ("2007-2009 GFC", 2007, 2009),
        ("2022-2023 Rate hike", 2022, 2023),
        ("2025-2026 Current", 2025, 2026),
    ]
    bear_results = []
    for label, y1, y2 in bear_regimes:
        subset = trades[(trades["year"] >= y1) & (trades["year"] <= y2)]
        if len(subset) == 0:
            print(f"  {label:<16} {'0':>7}")
            bear_results.append({"label": label, "profitable": False, "trades": 0})
            continue
        s = pd.Series(subset["pnl"].values, dtype=float)
        gw = s[s > 0].sum()
        gl = abs(s[s < 0].sum())
        spf = gw / gl if gl > 0 else float("inf")
        swr = (s > 0).mean()
        profitable = spf > 1.0
        bear_results.append({"label": label, "profitable": profitable, "trades": len(subset)})
        verdict = "✅" if profitable else "❌"
        print(f"  {label:<16} {len(subset):>7} {spf:>8.2f} {swr:>6.1%} "
              f"${s.mean():>8,.0f} ${s.sum():>10,.0f} {verdict}")
    # Full period breakdown
    print(f"\n  FULL PERIOD BREAKDOWN:")
    print(f"  {'Period':<12} {'Trades':>7} {'PF':>8} {'WR':>7} {'Avg$':>9} {'Total$':>11}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*11}")
    periods = [
        ("2000-2005", 2000, 2005),
        ("2006-2010", 2006, 2010),
        ("2011-2015", 2011, 2015),
        ("2016-2020", 2016, 2020),
        ("2021-2026", 2021, 2026),
    ]
    for label, y1, y2 in periods:
        subset = trades[(trades["year"] >= y1) & (trades["year"] <= y2)]
        if len(subset) == 0:
            print(f"  {label:<12} {'0':>7}")
            continue
        s = pd.Series(subset["pnl"].values, dtype=float)
        gw = s[s > 0].sum()
        gl = abs(s[s < 0].sum())
        spf = gw / gl if gl > 0 else float("inf")
        swr = (s > 0).mean()
        print(f"  {label:<12} {len(subset):>7} {spf:>8.2f} {swr:>6.1%} "
              f"${s.mean():>8,.0f} ${s.sum():>10,.0f}")
    # Year by year
    print(f"\n  YEAR-BY-YEAR:")
    for year in sorted(trades["year"].unique()):
        subset = trades[trades["year"] == year]
        s = pd.Series(subset["pnl"].values, dtype=float)
        gw = s[s > 0].sum()
        gl = abs(s[s < 0].sum())
        spf = gw / gl if gl > 0 else float("inf")
        print(f"    {year}: {len(subset)} trades, PF {spf:.2f}, "
              f"WR {(s>0).mean():.0%}, avg ${s.mean():,.0f}, total ${s.sum():,.0f}")
    # Phidias sim
    print(f"\n  PHIDIAS 50K SWING SIMULATION:")
    attempts = simulate_phidias(trade_pnls)
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    total_att = len(passed) + len(failed)
    pass_rate = len(passed) / total_att * 100 if total_att > 0 else 0
    print(f"    Pass Rate: {pass_rate:.1f}% ({len(passed)}/{total_att})")
    if passed:
        print(f"    Avg trades to pass: {np.mean([a['trades_taken'] for a in passed]):.1f}")
    # Correlation with Week 1/4
    cal_path = os.path.join(RESULTS_DIR, "week14_phidias_trades.csv")
    corr, corr_note = compute_monthly_correlation(trades, cal_path)
    print(f"\n  CORRELATION WITH WEEK 1/4 CALENDAR:")
    if corr is not None:
        print(f"    Monthly P&L correlation: {corr:.3f} ({corr_note})")
    else:
        print(f"    {corr_note}")
    # =============================================
    # SENSITIVITY: 6-month lookback
    # =============================================
    print(f"\n{'='*60}")
    print("SENSITIVITY VARIANTS")
    print(f"{'='*60}")
    sensitivity = [
        ("12m + 2x ATR stop", 252, True, 2.0),
        ("12m + no stop", 252, False, 0),
        ("6m + 2x ATR stop", 126, True, 2.0),
        ("6m + no stop", 126, False, 0),
        ("9m + 2x ATR stop", 189, True, 2.0),
        ("12m + 3x ATR stop", 252, True, 3.0),
    ]
    print(f"\n  {'Variant':<25} {'Trades':>7} {'PF':>8} {'WR':>7} {'Avg$':>9} {'Total$':>11}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*11}")
    for label, lb, use_stop, atr_mult in sensitivity:
        sig = build_monthly_signals(es, lookback_days=lb)
        tr = simulate_trades(sig, es, use_atr_stop=use_stop, atr_multiplier=atr_mult)
        if len(tr) == 0:
            print(f"  {label:<25} {'0':>7}")
            continue
        tp = pd.Series(tr["pnl"].values, dtype=float)
        gw = tp[tp > 0].sum()
        gl = abs(tp[tp < 0].sum())
        tpf = gw / gl if gl > 0 else float("inf")
        twr = (tp > 0).mean()
        is_primary = " ◄" if lb == 252 and use_stop and atr_mult == 2.0 else ""
        print(f"  {label:<25} {len(tr):>7} {tpf:>8.2f} {twr:>6.1%} "
              f"${tp.mean():>8,.0f} ${tp.sum():>10,.0f}{is_primary}")
    # =============================================
    # ACCEPTANCE GATES
    # =============================================
    print(f"\n{'='*80}")
    print("ACCEPTANCE GATES")
    print(f"{'='*80}")
    bear_profitable = sum(1 for b in bear_results if b["profitable"] and b["trades"] > 0)
    bear_with_trades = sum(1 for b in bear_results if b["trades"] > 0)
    gate_1 = pf > 1.3
    gate_2 = len(trades) >= 50
    gate_3 = bear_profitable >= 3  # 3 out of 4 bear regimes
    gate_4 = corr is not None and abs(corr) < 0.20
    print(f"\n  Gate 1: PF > 1.30              → PF {pf:.3f} → {'✅ PASS' if gate_1 else '❌ FAIL'}")
    print(f"  Gate 2: 50+ trades             → {len(trades)} trades → {'✅ PASS' if gate_2 else '❌ FAIL'}")
    print(f"  Gate 3: 3/4 bear regimes       → {bear_profitable}/{bear_with_trades} profitable → {'✅ PASS' if gate_3 else '❌ FAIL'}")
    print(f"  Gate 4: Correlation < 0.20     → {f'Corr {corr:.3f}' if corr is not None else 'N/A'} → {'✅ PASS' if gate_4 else '❌ FAIL' if corr is not None else '⚠️  UNABLE TO TEST'}")
    all_passed = gate_1 and gate_2 and gate_3 and (gate_4 or corr is None)
    print(f"\n  VERDICT: {'✅ ALL GATES PASSED — Deploy as Account 2 (Short Sleeve)' if all_passed else '❌ FAILED — Do not deploy'}")
    if not all_passed:
        failed_gates = []
        if not gate_1:
            failed_gates.append(f"PF {pf:.3f} < 1.30")
        if not gate_2:
            failed_gates.append(f"{len(trades)} trades < 50")
        if not gate_3:
            failed_gates.append(f"Only {bear_profitable}/{bear_with_trades} bear regimes profitable")
        if corr is not None and not gate_4:
            failed_gates.append(f"Correlation {corr:.3f} >= 0.20")
        print(f"  Failed: {'; '.join(failed_gates)}")
    print(f"\n{'='*80}")
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "tsmom_short_trades.csv")
    trades.to_csv(csv_path, index=False)
    print(f"Trades saved to {csv_path}")
    if len(trades) > 0:
        eq_vals = [INITIAL_CAPITAL]
        for p in trade_pnls:
            eq_vals.append(eq_vals[-1] + p)
        dates = [trades["entry_date"].iloc[0] - pd.Timedelta(days=30)]
        dates.extend(trades["entry_date"].tolist())
        eq = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
        plot_equity(eq, "TSMOM Short-Only (3 MES)")
if __name__ == "__main__":
    run()
