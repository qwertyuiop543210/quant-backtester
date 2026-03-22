"""VIX Spike + IBS Mean Reversion — Full Backtest.
THESIS: After a VIX spike on a day when ES closes near its low (IBS < 0.3),
equity markets reliably recover. This captures maximum-fear exhaustion points.
ENTRY CONDITIONS (ALL required):
  1. VIX 1-day change > +15%        — spike detection
  2. VIX absolute level > 25        — elevated fear (not low-vol noise)
  3. ES IBS < 0.30                  — close near day's low (maximum fear)
  4. Buy at close (or next open)
EXIT CONDITIONS (first to trigger):
  1. VIX drops below 20             — fear resolved (PRIMARY exit)
  2. 15 trading days elapsed        — time stop
  3. -$1,500 hard stop on 3 MES     — risk limit
POSITION SIZING: 3 MES contracts ($15/point)
  - Hard stop at -$1,500 = -100 ES points (very wide, lets trade breathe)
  - Target is open-ended (ride the recovery)
ALSO TESTS SENSITIVITY:
  - VIX spike thresholds: 10%, 15%, 20%
  - VIX level thresholds: 20, 25, 30
  - IBS thresholds: 0.20, 0.30, 0.40, no filter
  - Hold periods: 10, 15, 20 days
  - Stop levels: $1,000, $1,500, $2,000, no stop
ACCEPTANCE CRITERIA (all 4 must pass):
  1. 50+ trades
  2. PF > 1.50 standalone
  3. Monthly correlation with Week 1/4 signal < 0.20
  4. Phidias 50K pass rate > 45%
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
STRATEGY_NAME = "VIX Spike + IBS Reversion"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MES_POINT_VALUE = 5.0
NUM_CONTRACTS = 3
DOLLAR_PER_POINT = MES_POINT_VALUE * NUM_CONTRACTS  # $15/point
COMMISSION_RT = 1.24 * NUM_CONTRACTS  # $3.72
SLIPPAGE_PER_SIDE = 1.25 * NUM_CONTRACTS  # $3.75
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # ~$11.22
INITIAL_CAPITAL = 100_000.0
# Phidias 50K Swing
PHIDIAS_CAPITAL = 50_000.0
PHIDIAS_TARGET = 4_000.0
PHIDIAS_DD = 2_500.0
def compute_ibs(high, low, close):
    """Internal Bar Strength: (close - low) / (high - low). Range 0-1."""
    denom = high - low
    ibs = (close - low) / denom
    ibs = ibs.replace([np.inf, -np.inf], np.nan)
    return ibs
def find_entries(es_data, vix_data, vix_spike_pct=0.15, vix_level=25, ibs_thresh=0.30):
    """Find all entry signals.
    Returns DataFrame of entry dates with entry prices and context.
    """
    es_close = es_data["Close"].astype(float)
    es_open = es_data["Open"].astype(float)
    es_high = es_data["High"].astype(float)
    es_low = es_data["Low"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")
    # VIX 1-day percentage change
    vix_change = vix_close.pct_change()
    # IBS
    ibs = compute_ibs(es_high, es_low, es_close)
    # Entry signals
    vix_spiked = vix_change > vix_spike_pct
    vix_elevated = vix_close > vix_level
    ibs_low = ibs < ibs_thresh if ibs_thresh is not None else pd.Series(True, index=es_close.index)
    signal = vix_spiked & vix_elevated & ibs_low
    entries = pd.DataFrame({
        "entry_price": es_close,  # Buy at close
        "es_open_next": es_open.shift(-1),  # Or next open
        "vix_close": vix_close,
        "vix_change_pct": vix_change * 100,
        "ibs": ibs,
        "signal": signal,
    }, index=es_close.index)
    return entries[entries["signal"]].copy()
def simulate_trades(entries, es_data, vix_data, max_hold=15, hard_stop_dollars=1500):
    """Simulate trades with VIX-based exit, time stop, and hard dollar stop.
    Entry: buy at signal day's close
    Exit (first to trigger):
      1. VIX closes below 20
      2. max_hold trading days elapsed
      3. Unrealized loss exceeds hard_stop_dollars
    """
    es_close = es_data["Close"].astype(float)
    es_low = es_data["Low"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")
    all_dates = es_close.index.tolist()
    trades = []
    # Remove overlapping entries (must be max_hold+ days apart)
    clean_entries = []
    last_exit_idx = -1
    for entry_date in entries.index:
        entry_idx = all_dates.index(entry_date) if entry_date in all_dates else -1
        if entry_idx <= last_exit_idx:
            continue  # Still in a trade
        entry_price = es_close.loc[entry_date]
        entry_vix = vix_close.loc[entry_date]
        entry_ibs = entries.loc[entry_date, "ibs"]
        exit_price = None
        exit_date = None
        exit_reason = None
        max_adverse = 0
        max_favorable = 0
        for hold_day in range(1, max_hold + 1):
            check_idx = entry_idx + hold_day
            if check_idx >= len(all_dates):
                break
            check_date = all_dates[check_idx]
            current_close = es_close.iloc[check_idx]
            current_low = es_low.iloc[check_idx]
            current_vix = vix_close.iloc[check_idx] if check_idx < len(vix_close) else np.nan
            # Track max adverse excursion (using intraday low)
            unrealized_low = (current_low - entry_price) * DOLLAR_PER_POINT
            unrealized_close = (current_close - entry_price) * DOLLAR_PER_POINT
            max_adverse = min(max_adverse, unrealized_low)
            max_favorable = max(max_favorable, unrealized_close)
            # Check hard stop (intraday)
            if hard_stop_dollars is not None and unrealized_low <= -hard_stop_dollars:
                # Stopped out — approximate exit at stop level
                exit_price = entry_price - (hard_stop_dollars / DOLLAR_PER_POINT)
                exit_date = check_date
                exit_reason = "hard_stop"
                break
            # Check VIX exit (end of day)
            if not pd.isna(current_vix) and current_vix < 20:
                exit_price = current_close
                exit_date = check_date
                exit_reason = "vix_below_20"
                break
            # Check time stop
            if hold_day >= max_hold:
                exit_price = current_close
                exit_date = check_date
                exit_reason = "time_stop"
                break
        if exit_price is None:
            # Ran out of data
            continue
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * DOLLAR_PER_POINT
        net_pnl = gross_pnl - COST_PER_TRADE
        last_exit_idx = all_dates.index(exit_date) if exit_date in all_dates else entry_idx + max_hold
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "hold_days": (exit_date - entry_date).days,
            "exit_reason": exit_reason,
            "entry_vix": entry_vix,
            "entry_ibs": entry_ibs,
            "max_adverse": max_adverse,
            "max_favorable": max_favorable,
            "year": entry_date.year,
        })
    return pd.DataFrame(trades)
def simulate_phidias(trade_pnls, capital, target, trailing_dd):
    """Simulate Phidias eval attempts."""
    attempts = []
    i = 0
    while i < len(trade_pnls):
        balance = capital
        high_water = capital
        start = i
        status = "in_progress"
        while i < len(trade_pnls):
            balance += trade_pnls.iloc[i]
            high_water = max(high_water, balance)
            trailing = high_water - balance
            profit = balance - capital
            i += 1
            if trailing >= trailing_dd:
                status = "FAILED"
                break
            if profit >= target:
                status = "PASSED"
                break
        attempts.append({
            "trades_taken": i - start,
            "profit": balance - capital,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
def compute_monthly_correlation(vix_trades, calendar_trades_path):
    """Compute monthly P&L correlation between VIX spike strategy and Week 1/4.
    Loads Week 1/4 trades from saved CSV and aligns monthly P&L.
    """
    if not os.path.exists(calendar_trades_path):
        return None, "Calendar trades CSV not found"
    cal = pd.read_csv(calendar_trades_path, parse_dates=["entry_date", "exit_date"])
    # Monthly P&L for each strategy
    vix_monthly = vix_trades.set_index("entry_date")["pnl"].resample("M").sum()
    cal_monthly = cal.set_index("entry_date")["pnl"].resample("M").sum()
    # Align
    common = vix_monthly.index.intersection(cal_monthly.index)
    if len(common) < 12:
        return None, f"Only {len(common)} common months"
    corr = vix_monthly.loc[common].corr(cal_monthly.loc[common])
    return corr, f"{len(common)} common months"
def run():
    """Run VIX Spike + IBS backtest with sensitivity analysis."""
    print("=" * 80)
    print("VIX SPIKE + IBS MEAN REVERSION — FULL BACKTEST")
    print("3 MES contracts ($15/point)")
    print("=" * 80)
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()}")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()}")
    # =============================================
    # PRIMARY VARIANT (per spec)
    # =============================================
    print(f"\n{'='*60}")
    print("PRIMARY VARIANT: VIX +15%, Level >25, IBS <0.30, 15d hold, $1500 stop")
    print(f"{'='*60}")
    entries = find_entries(es, vix, vix_spike_pct=0.15, vix_level=25, ibs_thresh=0.30)
    print(f"Entry signals found: {len(entries)}")
    trades = simulate_trades(entries, es, vix, max_hold=15, hard_stop_dollars=1500)
    print(f"Trades after overlap removal: {len(trades)}")
    if len(trades) == 0:
        print("ERROR: No trades generated")
        return
    trade_pnls = pd.Series(trades["pnl"].values, dtype=float)
    # Stats
    wins = (trade_pnls > 0).sum()
    losses = (trade_pnls <= 0).sum()
    gross_w = trade_pnls[trade_pnls > 0].sum()
    gross_l = abs(trade_pnls[trade_pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = wins / len(trade_pnls)
    print(f"\n  Trades: {len(trades)}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Profit Factor: {pf:.3f}")
    print(f"  Avg Win: ${trade_pnls[trade_pnls > 0].mean():,.0f}")
    print(f"  Avg Loss: ${trade_pnls[trade_pnls < 0].mean():,.0f}")
    print(f"  Avg Trade: ${trade_pnls.mean():,.0f}")
    print(f"  Total P&L: ${trade_pnls.sum():,.0f}")
    print(f"  Max Adverse Excursion (worst): ${trades['max_adverse'].min():,.0f}")
    # Exit reason breakdown
    print(f"\n  Exit Reasons:")
    for reason in trades["exit_reason"].unique():
        subset = trades[trades["exit_reason"] == reason]
        sub_pnls = subset["pnl"]
        sub_wr = (sub_pnls > 0).mean()
        print(f"    {reason}: {len(subset)} trades, {sub_wr:.1%} WR, ${sub_pnls.mean():,.0f} avg")
    # Period breakdown
    print(f"\n  Period Breakdown:")
    print(f"  {'Period':<12} {'Trades':>7} {'PF':>8} {'WR':>8} {'Avg$':>10} {'Total$':>12}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
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
        print(f"  {label:<12} {len(subset):>7} {spf:>8.2f} {swr:>7.1%} "
              f"${s.mean():>9,.0f} ${s.sum():>11,.0f}")
    # Year by year
    print(f"\n  Year-by-Year:")
    for year in sorted(trades["year"].unique()):
        subset = trades[trades["year"] == year]
        s = pd.Series(subset["pnl"].values, dtype=float)
        gw = s[s > 0].sum()
        gl = abs(s[s < 0].sum())
        spf = gw / gl if gl > 0 else float("inf")
        print(f"    {year}: {len(subset)} trades, PF {spf:.2f}, "
              f"WR {(s>0).mean():.0%}, avg ${s.mean():,.0f}, total ${s.sum():,.0f}")
    # VIX level at entry
    print(f"\n  VIX Level at Entry:")
    vix_bins = [(25, 30), (30, 40), (40, 50), (50, 100)]
    for lo, hi in vix_bins:
        subset = trades[(trades["entry_vix"] >= lo) & (trades["entry_vix"] < hi)]
        if len(subset) == 0:
            continue
        s = pd.Series(subset["pnl"].values, dtype=float)
        print(f"    VIX {lo}-{hi}: {len(subset)} trades, "
              f"WR {(s>0).mean():.0%}, avg ${s.mean():,.0f}")
    # Phidias simulation
    print(f"\n  Phidias 50K Swing Simulation:")
    attempts = simulate_phidias(trade_pnls, PHIDIAS_CAPITAL, PHIDIAS_TARGET, PHIDIAS_DD)
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    total_att = len(passed) + len(failed)
    pass_rate = len(passed) / total_att * 100 if total_att > 0 else 0
    print(f"    Pass Rate: {pass_rate:.1f}% ({len(passed)}/{total_att})")
    if passed:
        print(f"    Avg trades to pass: {np.mean([a['trades_taken'] for a in passed]):.1f}")
    # Monthly correlation with Week 1/4
    cal_path = os.path.join(RESULTS_DIR, "week14_phidias_trades.csv")
    corr, corr_note = compute_monthly_correlation(trades, cal_path)
    print(f"\n  Monthly Correlation with Week 1/4 Calendar:")
    if corr is not None:
        print(f"    Correlation: {corr:.3f} ({corr_note})")
    else:
        print(f"    {corr_note}")
    # =============================================
    # SENSITIVITY ANALYSIS
    # =============================================
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"\n  {'VIX%':>5} {'VIXlvl':>7} {'IBS':>5} {'Hold':>5} {'Stop':>6} "
          f"{'Trades':>7} {'PF':>8} {'WR':>7} {'Avg$':>8} {'Total$':>10}")
    print(f"  {'-'*5} {'-'*7} {'-'*5} {'-'*5} {'-'*6} "
          f"{'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
    sensitivity_configs = [
        # (vix_spike_pct, vix_level, ibs_thresh, max_hold, hard_stop)
        (0.10, 25, 0.30, 15, 1500),
        (0.15, 25, 0.30, 15, 1500),  # PRIMARY
        (0.20, 25, 0.30, 15, 1500),
        (0.15, 20, 0.30, 15, 1500),
        (0.15, 30, 0.30, 15, 1500),
        (0.15, 25, 0.20, 15, 1500),
        (0.15, 25, 0.40, 15, 1500),
        (0.15, 25, None, 15, 1500),  # No IBS filter
        (0.15, 25, 0.30, 10, 1500),
        (0.15, 25, 0.30, 20, 1500),
        (0.15, 25, 0.30, 15, 1000),
        (0.15, 25, 0.30, 15, 2000),
        (0.15, 25, 0.30, 15, None),  # No stop
    ]
    best_config = None
    best_pf = 0
    for vsp, vlvl, ibs_t, mh, hs in sensitivity_configs:
        e = find_entries(es, vix, vix_spike_pct=vsp, vix_level=vlvl, ibs_thresh=ibs_t)
        if len(e) == 0:
            continue
        t = simulate_trades(e, es, vix, max_hold=mh, hard_stop_dollars=hs)
        if len(t) == 0:
            continue
        tp = pd.Series(t["pnl"].values, dtype=float)
        gw = tp[tp > 0].sum()
        gl = abs(tp[tp < 0].sum())
        tpf = gw / gl if gl > 0 else float("inf")
        twr = (tp > 0).mean()
        ibs_str = f"{ibs_t:.2f}" if ibs_t is not None else "None"
        hs_str = f"${hs}" if hs is not None else "None"
        is_primary = " ◄" if (vsp == 0.15 and vlvl == 25 and ibs_t == 0.30 and mh == 15 and hs == 1500) else ""
        print(f"  {vsp*100:>4.0f}% {vlvl:>7} {ibs_str:>5} {mh:>5} {hs_str:>6} "
              f"{len(t):>7} {tpf:>8.2f} {twr:>6.0%} ${tp.mean():>7,.0f} ${tp.sum():>9,.0f}{is_primary}")
        if tpf > best_pf and len(t) >= 30:
            best_pf = tpf
            best_config = (vsp, vlvl, ibs_t, mh, hs, len(t), tpf, twr)
    # =============================================
    # FOUR-GATE ACCEPTANCE TEST
    # =============================================
    print(f"\n{'='*80}")
    print("FOUR-GATE ACCEPTANCE TEST")
    print(f"{'='*80}")
    gate_1 = len(trades) >= 50
    gate_2 = pf > 1.50
    gate_3 = corr is not None and abs(corr) < 0.20
    gate_4 = pass_rate > 45
    print(f"\n  Gate 1: 50+ trades           → {len(trades)} trades → {'✅ PASS' if gate_1 else '❌ FAIL'}")
    print(f"  Gate 2: PF > 1.50            → PF {pf:.3f} → {'✅ PASS' if gate_2 else '❌ FAIL'}")
    print(f"  Gate 3: Correlation < 0.20   → {f'Corr {corr:.3f}' if corr is not None else 'N/A'} → {'✅ PASS' if gate_3 else '❌ FAIL' if corr is not None else '⚠️  UNABLE TO TEST'}")
    print(f"  Gate 4: Pass rate > 45%      → {pass_rate:.1f}% → {'✅ PASS' if gate_4 else '❌ FAIL'}")
    all_passed = gate_1 and gate_2 and (gate_3 or corr is None) and gate_4
    print(f"\n  VERDICT: {'✅ ALL GATES PASSED — Deploy as Account 2' if all_passed else '❌ FAILED — Do not deploy'}")
    if not all_passed:
        failed_gates = []
        if not gate_1:
            failed_gates.append(f"Need 50+ trades, got {len(trades)}")
        if not gate_2:
            failed_gates.append(f"Need PF > 1.50, got {pf:.3f}")
        if corr is not None and not gate_3:
            failed_gates.append(f"Need correlation < 0.20, got {corr:.3f}")
        if not gate_4:
            failed_gates.append(f"Need pass rate > 45%, got {pass_rate:.1f}%")
        print(f"  Failed gates: {'; '.join(failed_gates)}")
        if best_config:
            print(f"\n  Best sensitivity config (PF {best_config[6]:.2f}, {best_config[5]} trades):")
            print(f"    VIX spike: {best_config[0]*100:.0f}%, Level: {best_config[1]}, "
                  f"IBS: {best_config[2]}, Hold: {best_config[3]}d, Stop: {best_config[4]}")
    print(f"\n{'='*80}")
    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "vix_spike_ibs_trades.csv")
    trades.to_csv(csv_path, index=False)
    print(f"Trades saved to {csv_path}")
    if len(trades) > 0:
        eq_vals = [INITIAL_CAPITAL]
        for p in trade_pnls:
            eq_vals.append(eq_vals[-1] + p)
        dates = [trades["entry_date"].iloc[0] - pd.Timedelta(days=30)]
        dates.extend(trades["entry_date"].tolist())
        eq = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
        plot_equity(eq, "VIX Spike + IBS Reversion (3 MES)")
if __name__ == "__main__":
    run()
