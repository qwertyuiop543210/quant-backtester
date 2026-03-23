"""Dip Buyer strategy — Short-term mean reversion on ES futures.
Academic basis:
- De Bondt & Thaler (1985): Investor overreaction creates predictable reversals.
- Jegadeesh (1990): Short-term return predictability at 1-month horizon.
- Cooper, Gutierrez & Hameed (2004): Reversal profits are state-dependent,
  strongest following volatile market conditions.
- Nagel (2012): Short-term reversal returns = compensation for liquidity provision.
- Giot (2005): High VIX predicts positive subsequent equity returns.
Rules:
- Entry: 2-day RSI of ES closes below 10. Buy 1 ES at next day's open.
- Exit: 2-day RSI rises above 65 OR 5 trading days elapse (whichever first).
- VIX filter: VIX must be > 20 (no ceiling; prior day close).
- No overlap: Skip if Chosen One is active (Week 1 or Week 4, Mon-Fri).
- Position: 1 ES contract, long only.
Phidias $50K Swing simulation: $50K account, $4,000 profit target,
$2,500 EOD drawdown, 1 mini overnight allowed.
Costs: $5 round trip commission, $12.50 slippage per side per contract.
ES point value = $50, starting capital $100K, data from 2012-01-01.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary, profit_factor, win_rate
from core.plotting import plot_equity
STRATEGY_NAME = "Dip Buyer (ES) — RSI Mean Reversion"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30 total
INITIAL_CAPITAL = 100_000.0
# Signal parameters
RSI_PERIOD = 2
RSI_ENTRY_THRESHOLD = 10      # Buy when 2-day RSI < 10
RSI_EXIT_THRESHOLD = 65       # Sell when 2-day RSI > 65
MAX_HOLD_DAYS = 5             # Time-stop: exit after 5 trading days
# VIX filters
VIX_FLOOR = 20.0              # Must be above 20 (below = dead zone)
VIX_CEILING = 999.0           # No ceiling (effectively disabled)
# Chosen One overlap: weeks 1 & 4
CHOSEN_ONE_WEEKS = {1, 4}
# Phidias $50K Swing eval parameters
PHIDIAS_CAPITAL = 50_000.0
PHIDIAS_PROFIT_TARGET = 4_000.0
PHIDIAS_EOD_DRAWDOWN = 2_500.0
# ---------------------------------------------------------------------------
# RSI calculation (Wilder's smoothing, no external TA library needed)
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Compute RSI using Wilder's exponential smoothing method.
    Args:
        close: Close price series.
        period: RSI lookback period (default 2 for short-term mean reversion).
    Returns:
        RSI series (0-100), with NaN for initial warmup period.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    # Wilder's smoothing: first value is SMA, then EMA with alpha = 1/period
    avg_gain = gains.copy()
    avg_loss = losses.copy()
    # Seed with SMA of first `period` values
    avg_gain.iloc[:period] = np.nan
    avg_loss.iloc[:period] = np.nan
    avg_gain.iloc[period] = gains.iloc[1:period + 1].mean()
    avg_loss.iloc[period] = losses.iloc[1:period + 1].mean()
    # Wilder's smoothing for rest
    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gains.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + losses.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
# ---------------------------------------------------------------------------
# Chosen One overlap detection
# ---------------------------------------------------------------------------
def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1
def build_chosen_one_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Return boolean Series: True on days when Chosen One might be active.
    Chosen One is active Monday-Friday of Week 1 and Week 4 each month.
    We mark the full Mon-Fri trading week as occupied.
    """
    mask = pd.Series(False, index=dates)
    i = 0
    while i < len(dates):
        date = dates[i]
        if date.dayofweek == 0:  # Monday
            wom = get_week_of_month(date)
            if wom in CHOSEN_ONE_WEEKS:
                # Mark Monday through Friday of this week
                j = i
                while j < len(dates) and dates[j].isocalendar()[1] == date.isocalendar()[1]:
                    mask.iloc[j] = True
                    j += 1
                i = j
                continue
        i += 1
    return mask
# ---------------------------------------------------------------------------
# Phidias $50K Swing evaluation simulation
# ---------------------------------------------------------------------------
def simulate_phidias(trade_pnls: list[float], daily_pnls_per_trade: list[list[float]]) -> list[dict]:
    """Simulate Phidias $50K Swing eval attempts.
    Rules:
    - Start with $50K.
    - EOD drawdown: $2,500 from starting balance (NOT trailing — from $50K).
    - Profit target: $4,000 cumulative profit.
    - If EOD drawdown breached at any daily close, attempt fails -> restart.
    - If profit target hit, attempt passes.
    Args:
        trade_pnls: List of net P&L per trade.
        daily_pnls_per_trade: List of lists — each inner list contains
            daily P&L values within that trade (for EOD drawdown checks).
    Returns:
        List of attempt dicts.
    """
    attempts = []
    i = 0
    while i < len(trade_pnls):
        balance = PHIDIAS_CAPITAL
        high_water = balance
        start_trade = i
        status = "in_progress"
        while i < len(trade_pnls):
            # Check each day within the trade for EOD drawdown
            breached = False
            for daily_pnl in daily_pnls_per_trade[i]:
                balance += daily_pnl
                high_water = max(high_water, balance)
                # EOD drawdown from high water mark
                if high_water - balance >= PHIDIAS_EOD_DRAWDOWN:
                    breached = True
                    break
            if not breached:
                # If trade didn't breach intraday, apply full trade P&L
                # (balance already updated day by day above)
                pass
            profit = balance - PHIDIAS_CAPITAL
            i += 1
            if breached:
                status = "FAILED"
                break
            if profit >= PHIDIAS_PROFIT_TARGET:
                status = "PASSED"
                break
        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade + 1,
            "end_trade": i,
            "trades_taken": i - start_trade,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - PHIDIAS_CAPITAL,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts
# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------
def run():
    """Run Dip Buyer backtest with Phidias simulation."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES and VIX data...")
    es = get_data("ES", start="2000-01-01")
    vix = get_data("VIX", start="2000-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")
    # Align VIX to ES trading days
    vix_close = vix["Close"].reindex(es.index).ffill()
    es_open = es["Open"].astype(float)
    es_close = es["Close"].astype(float)
    es_high = es["High"].astype(float)
    es_low = es["Low"].astype(float)
    # Compute 2-day RSI on ES close
    rsi = compute_rsi(es_close, period=RSI_PERIOD)
    # Build Chosen One overlap mask
    chosen_one_mask = build_chosen_one_mask(es.index)
    print(f"\nStrategy parameters:")
    print(f"  RSI period: {RSI_PERIOD}")
    print(f"  Entry: RSI < {RSI_ENTRY_THRESHOLD}")
    print(f"  Exit: RSI > {RSI_EXIT_THRESHOLD} or {MAX_HOLD_DAYS}-day time stop")
    print(f"  VIX filter: > {VIX_FLOOR} (no ceiling)")
    print(f"  Chosen One overlap: skip Week 1 & Week 4 (ACTIVE)")
    print(f"  Cost per trade: ${COST_PER_TRADE:.2f}")
    print()
    # --- Generate trades ---
    trades = []
    daily_pnls_per_trade = []  # For Phidias simulation
    equity = pd.Series(INITIAL_CAPITAL, index=es_close.index, dtype=float)
    position = pd.Series(0.0, index=es_close.index)
    cash = INITIAL_CAPITAL
    last_equity_idx = 0
    in_trade = False
    entry_idx = None
    entry_price = None
    hold_days = 0
    for i in range(1, len(es_close)):
        date = es.index[i]
        if in_trade:
            hold_days += 1
            current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50.0
            position.iloc[i] = 1.0
            # Check exit conditions
            exit_signal = False
            exit_reason = ""
            if current_rsi > RSI_EXIT_THRESHOLD:
                exit_signal = True
                exit_reason = "RSI > 65"
            elif hold_days >= MAX_HOLD_DAYS:
                exit_signal = True
                exit_reason = f"{MAX_HOLD_DAYS}-day time stop"
            if exit_signal:
                # Exit at today's close
                exit_price = es_close.iloc[i]
                pnl_points = exit_price - entry_price
                gross_pnl = pnl_points * ES_POINT_VALUE
                net_pnl = gross_pnl - COST_PER_TRADE
                # Compute daily P&Ls within this trade for Phidias sim
                trade_daily_pnls = []
                for d in range(entry_idx, i + 1):
                    if d == entry_idx:
                        # Entry day: bought at open, mark to close
                        day_pnl = (es_close.iloc[d] - entry_price) * ES_POINT_VALUE
                        if d == i:
                            day_pnl -= COST_PER_TRADE  # Costs on exit day
                    elif d == i:
                        # Exit day: mark from prior close to exit close
                        day_pnl = (es_close.iloc[d] - es_close.iloc[d - 1]) * ES_POINT_VALUE
                        day_pnl -= COST_PER_TRADE  # Apply costs on exit
                    else:
                        # Middle days: close-to-close
                        day_pnl = (es_close.iloc[d] - es_close.iloc[d - 1]) * ES_POINT_VALUE
                    trade_daily_pnls.append(day_pnl)
                cash += net_pnl
                # VIX at entry for regime tracking
                vix_at_entry = vix_close.iloc[entry_idx - 1] if entry_idx > 0 else vix_close.iloc[entry_idx]
                trades.append({
                    "entry_date": es.index[entry_idx],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_points": pnl_points,
                    "gross_pnl": gross_pnl,
                    "costs": COST_PER_TRADE,
                    "pnl": net_pnl,
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                    "rsi_at_entry_signal": rsi.iloc[entry_idx - 1] if entry_idx > 0 else np.nan,
                    "rsi_at_exit": current_rsi,
                    "vix_at_entry": vix_at_entry,
                    "year": es.index[entry_idx].year,
                    "entry_dow": es.index[entry_idx].dayofweek,
                    "entry_month": es.index[entry_idx].month,
                    "chosen_one_overlap": bool(chosen_one_mask.iloc[entry_idx]),
                })
                daily_pnls_per_trade.append(trade_daily_pnls)
                in_trade = False
                entry_idx = None
                entry_price = None
                hold_days = 0
        else:
            # Check for entry signal
            # RSI signal fires on day i-1 close, we enter at day i open
            prev_rsi = rsi.iloc[i - 1] if not pd.isna(rsi.iloc[i - 1]) else 50.0
            prev_vix = vix_close.iloc[i - 1] if not pd.isna(vix_close.iloc[i - 1]) else 15.0
            valid_signal = (
                prev_rsi < RSI_ENTRY_THRESHOLD
                and prev_vix > VIX_FLOOR
                and prev_vix < VIX_CEILING
            )
            signal_fires = valid_signal and not chosen_one_mask.iloc[i]
            if signal_fires:
                entry_price = es_open.iloc[i]
                entry_idx = i
                hold_days = 0
                in_trade = True
                position.iloc[i] = 1.0
        # Update equity
        if in_trade and entry_price is not None:
            mtm = (es_close.iloc[i] - entry_price) * ES_POINT_VALUE
            equity.iloc[i] = cash + mtm
        else:
            equity.iloc[i] = cash
    # If still in a trade at end, force close
    if in_trade and entry_idx is not None:
        exit_price = es_close.iloc[-1]
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE
        cash += net_pnl
        equity.iloc[-1] = cash
        trade_daily_pnls = []
        for d in range(entry_idx, len(es_close)):
            if d == entry_idx:
                day_pnl = (es_close.iloc[d] - entry_price) * ES_POINT_VALUE
            else:
                day_pnl = (es_close.iloc[d] - es_close.iloc[d - 1]) * ES_POINT_VALUE
            trade_daily_pnls.append(day_pnl)
        trade_daily_pnls[-1] -= COST_PER_TRADE
        vix_at_entry = vix_close.iloc[entry_idx - 1] if entry_idx > 0 else vix_close.iloc[entry_idx]
        trades.append({
            "entry_date": es.index[entry_idx],
            "exit_date": es.index[-1],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "hold_days": hold_days,
            "exit_reason": "FORCED CLOSE (end of data)",
            "rsi_at_entry_signal": rsi.iloc[entry_idx - 1] if entry_idx > 0 else np.nan,
            "rsi_at_exit": rsi.iloc[-1],
            "vix_at_entry": vix_at_entry,
            "year": es.index[entry_idx].year,
            "entry_dow": es.index[entry_idx].dayofweek,
            "entry_month": es.index[entry_idx].month,
            "chosen_one_overlap": bool(chosen_one_mask.iloc[entry_idx]),
        })
        daily_pnls_per_trade.append(trade_daily_pnls)
    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)
    # --- Overall stats ---
    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )
    print_summary(stats, STRATEGY_NAME)
    # --- Yearly breakdown ---
    if len(trade_list) > 0:
        print(f"{'='*80}")
        print(f"  Yearly Breakdown")
        print(f"{'='*80}")
        print(f"  {'Year':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
        for year in sorted(trade_list["year"].unique()):
            subset = trade_list[trade_list["year"] == year]
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            total = sub_pnls.sum()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {year:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}")
        print(f"{'='*80}")
    # --- VIX regime breakdown ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  VIX Regime Breakdown at Entry")
        print(f"{'='*80}")
        print(f"  {'VIX Range':>12} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
        vix_bins = [(20, 25), (25, 30), (30, 35)]
        for lo, hi in vix_bins:
            subset = trade_list[(trade_list["vix_at_entry"] >= lo) &
                                (trade_list["vix_at_entry"] < hi)]
            if len(subset) == 0:
                continue
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            total = sub_pnls.sum()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            label = f"{lo}-{hi}"
            print(f"  {label:>12} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}")
        print(f"{'='*80}")
    # --- Exit reason breakdown ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  Exit Reason Breakdown")
        print(f"{'='*80}")
        print(f"  {'Reason':>20} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
        for reason in trade_list["exit_reason"].unique():
            subset = trade_list[trade_list["exit_reason"] == reason]
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {reason:>20} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f}")
        print(f"{'='*80}")
    # --- Hold duration stats ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  Hold Duration Distribution")
        print(f"{'='*80}")
        print(f"  {'Days':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
        for d in sorted(trade_list["hold_days"].unique()):
            subset = trade_list[trade_list["hold_days"] == d]
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {d:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f}")
        avg_hold = trade_list["hold_days"].mean()
        print(f"\n  Average hold: {avg_hold:.1f} trading days")
        print(f"{'='*80}")
    # --- Chosen One overlap analysis ---
    # Count valid RSI+VIX signals that were blocked by Chosen One filter
    blocked_by_chosen_one = 0
    total_valid_signals = 0
    scan_in_trade = False
    scan_hold = 0
    for i in range(1, len(es_close)):
        if scan_in_trade:
            scan_hold += 1
            cur_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50.0
            if cur_rsi > RSI_EXIT_THRESHOLD or scan_hold >= MAX_HOLD_DAYS:
                scan_in_trade = False
                scan_hold = 0
        else:
            p_rsi = rsi.iloc[i - 1] if not pd.isna(rsi.iloc[i - 1]) else 50.0
            p_vix = vix_close.iloc[i - 1] if not pd.isna(vix_close.iloc[i - 1]) else 15.0
            if p_rsi < RSI_ENTRY_THRESHOLD and p_vix > VIX_FLOOR and p_vix < VIX_CEILING:
                total_valid_signals += 1
                if chosen_one_mask.iloc[i]:
                    blocked_by_chosen_one += 1
                else:
                    scan_in_trade = True
                    scan_hold = 0
    print(f"\n{'='*80}")
    print(f"  Chosen One Overlap Analysis")
    print(f"{'='*80}")
    if len(trade_list) > 0:
        print(f"  Total trades taken: {len(trade_list)}")
    print(f"  Signals blocked by Chosen One filter: {blocked_by_chosen_one} out of {total_valid_signals} total valid signals")
    print(f"{'='*80}")
    # --- Sensitivity analysis: vary RSI entry threshold ---
    print(f"\n{'='*80}")
    print(f"  Sensitivity Analysis: RSI Entry Threshold")
    print(f"  (VIX {VIX_FLOOR}-{VIX_CEILING}, exit RSI>{RSI_EXIT_THRESHOLD} or {MAX_HOLD_DAYS}d stop)")
    print(f"{'='*80}")
    print(f"  {'RSI<':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
    for entry_thresh in [5, 10, 15, 20, 25]:
        test_trades = _quick_backtest(
            es_open, es_close, rsi, vix_close, chosen_one_mask,
            rsi_entry=entry_thresh,
            rsi_exit=RSI_EXIT_THRESHOLD,
            vix_floor=VIX_FLOOR,
            vix_ceiling=VIX_CEILING,
            max_hold=MAX_HOLD_DAYS,
        )
        if len(test_trades) == 0:
            print(f"  {entry_thresh:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
            continue
        t_pnls = pd.Series([t["pnl"] for t in test_trades], dtype=float)
        pf = profit_factor(t_pnls)
        wr = win_rate(t_pnls)
        avg = t_pnls.mean()
        total = t_pnls.sum()
        n = len(t_pnls)
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        marker = " <-- base" if entry_thresh == RSI_ENTRY_THRESHOLD else ""
        print(f"  {entry_thresh:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}{marker}")
    print(f"{'='*80}")
    # --- Sensitivity analysis: VIX floor ---
    print(f"\n{'='*80}")
    print(f"  Sensitivity Analysis: VIX Floor")
    print(f"  (RSI<{RSI_ENTRY_THRESHOLD}, VIX ceiling={VIX_CEILING}, exit RSI>{RSI_EXIT_THRESHOLD} or {MAX_HOLD_DAYS}d stop)")
    print(f"{'='*80}")
    print(f"  {'VIX>':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
    for vix_fl in [0, 15, 20, 25, 30]:
        test_trades = _quick_backtest(
            es_open, es_close, rsi, vix_close, chosen_one_mask,
            rsi_entry=RSI_ENTRY_THRESHOLD,
            rsi_exit=RSI_EXIT_THRESHOLD,
            vix_floor=vix_fl,
            vix_ceiling=VIX_CEILING,
            max_hold=MAX_HOLD_DAYS,
        )
        if len(test_trades) == 0:
            print(f"  {vix_fl:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
            continue
        t_pnls = pd.Series([t["pnl"] for t in test_trades], dtype=float)
        pf = profit_factor(t_pnls)
        wr = win_rate(t_pnls)
        avg = t_pnls.mean()
        total = t_pnls.sum()
        n = len(t_pnls)
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        marker = " <-- base" if vix_fl == VIX_FLOOR else ""
        print(f"  {vix_fl:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}{marker}")
    print(f"{'='*80}")
    # --- Sensitivity analysis: max hold days ---
    print(f"\n{'='*80}")
    print(f"  Sensitivity Analysis: Max Hold Days")
    print(f"  (RSI<{RSI_ENTRY_THRESHOLD}, exit RSI>{RSI_EXIT_THRESHOLD}, VIX {VIX_FLOOR}-{VIX_CEILING})")
    print(f"{'='*80}")
    print(f"  {'Days':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
    for hold_d in [1, 2, 3, 4, 5, 6, 7]:
        test_trades = _quick_backtest(
            es_open, es_close, rsi, vix_close, chosen_one_mask,
            rsi_entry=RSI_ENTRY_THRESHOLD,
            rsi_exit=RSI_EXIT_THRESHOLD,
            vix_floor=VIX_FLOOR,
            vix_ceiling=VIX_CEILING,
            max_hold=hold_d,
        )
        if len(test_trades) == 0:
            print(f"  {hold_d:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
            continue
        t_pnls = pd.Series([t["pnl"] for t in test_trades], dtype=float)
        pf = profit_factor(t_pnls)
        wr = win_rate(t_pnls)
        avg = t_pnls.mean()
        total = t_pnls.sum()
        n = len(t_pnls)
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        marker = " <-- base" if hold_d == MAX_HOLD_DAYS else ""
        print(f"  {hold_d:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}{marker}")
    print(f"{'='*80}")
    # --- Sensitivity analysis: RSI exit threshold ---
    print(f"\n{'='*80}")
    print(f"  Sensitivity Analysis: RSI Exit Threshold")
    print(f"  (RSI entry<{RSI_ENTRY_THRESHOLD}, VIX {VIX_FLOOR}-{VIX_CEILING}, max hold {MAX_HOLD_DAYS}d)")
    print(f"{'='*80}")
    print(f"  {'RSI>':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
    for exit_thresh in [30, 40, 50, 60, 65, 70, 80]:
        test_trades = _quick_backtest(
            es_open, es_close, rsi, vix_close, chosen_one_mask,
            rsi_entry=RSI_ENTRY_THRESHOLD,
            rsi_exit=exit_thresh,
            vix_floor=VIX_FLOOR,
            vix_ceiling=VIX_CEILING,
            max_hold=MAX_HOLD_DAYS,
        )
        if len(test_trades) == 0:
            print(f"  {exit_thresh:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
            continue
        t_pnls = pd.Series([t["pnl"] for t in test_trades], dtype=float)
        pf = profit_factor(t_pnls)
        wr = win_rate(t_pnls)
        avg = t_pnls.mean()
        total = t_pnls.sum()
        n = len(t_pnls)
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        marker = " <-- base" if exit_thresh == RSI_EXIT_THRESHOLD else ""
        print(f"  {exit_thresh:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}{marker}")
    print(f"{'='*80}")
    # --- Sensitivity analysis: VIX ceiling ---
    print(f"\n{'='*80}")
    print(f"  Sensitivity Analysis: VIX Ceiling")
    print(f"  (RSI entry<{RSI_ENTRY_THRESHOLD}, VIX floor={VIX_FLOOR}, exit RSI>{RSI_EXIT_THRESHOLD}, max hold {MAX_HOLD_DAYS}d)")
    print(f"{'='*80}")
    print(f"  {'VIX<':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
    for vix_ceil in [25, 30, 35, 40, 50, 100]:
        test_trades = _quick_backtest(
            es_open, es_close, rsi, vix_close, chosen_one_mask,
            rsi_entry=RSI_ENTRY_THRESHOLD,
            rsi_exit=RSI_EXIT_THRESHOLD,
            vix_floor=VIX_FLOOR,
            vix_ceiling=vix_ceil,
            max_hold=MAX_HOLD_DAYS,
        )
        if len(test_trades) == 0:
            print(f"  {vix_ceil:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
            continue
        t_pnls = pd.Series([t["pnl"] for t in test_trades], dtype=float)
        pf = profit_factor(t_pnls)
        wr = win_rate(t_pnls)
        avg = t_pnls.mean()
        total = t_pnls.sum()
        n = len(t_pnls)
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        marker = " <-- base" if vix_ceil == VIX_CEILING else ""
        label = "none" if vix_ceil == 100 else str(vix_ceil)
        print(f"  {label:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}{marker}")
    print(f"{'='*80}")
    # --- No VIX filter comparison ---
    print(f"\n{'='*80}")
    print(f"  Comparison: With vs Without VIX Filter")
    print(f"{'='*80}")
    no_filter_trades = _quick_backtest(
        es_open, es_close, rsi, vix_close, chosen_one_mask,
        rsi_entry=RSI_ENTRY_THRESHOLD,
        rsi_exit=RSI_EXIT_THRESHOLD,
        vix_floor=0,
        vix_ceiling=999,
        max_hold=MAX_HOLD_DAYS,
    )
    if len(no_filter_trades) > 0:
        nf_pnls = pd.Series([t["pnl"] for t in no_filter_trades], dtype=float)
        nf_pf = profit_factor(nf_pnls)
        nf_wr = win_rate(nf_pnls)
        print(f"  No VIX filter:   {len(no_filter_trades):>4} trades, PF {nf_pf:.3f}, "
              f"WR {nf_wr:.1%}, Total ${nf_pnls.sum():,.0f}")
    if len(trade_list) > 0:
        print(f"  With VIX filter: {len(trade_list):>4} trades, PF {stats['profit_factor']:.3f}, "
              f"WR {stats['win_rate']:.1%}, Total ${trade_pnls.sum():,.0f}")
    print(f"{'='*80}")
    # --- Day-of-week breakdown ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  Entry Day-of-Week Breakdown")
        print(f"{'='*80}")
        print(f"  {'Day':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        for dow in range(5):
            subset = trade_list[trade_list["entry_dow"] == dow]
            if len(subset) == 0:
                print(f"  {day_names[dow]:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
                continue
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            total = sub_pnls.sum()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {day_names[dow]:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}")
        print(f"{'='*80}")
    # --- Monthly seasonality ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  Monthly Seasonality")
        print(f"{'='*80}")
        print(f"  {'Month':>6} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}")
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        for m in range(1, 13):
            subset = trade_list[trade_list["entry_month"] == m]
            if len(subset) == 0:
                print(f"  {month_names[m]:>6} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12}")
                continue
            sub_pnls = pd.Series(subset["pnl"].values, dtype=float)
            pf = profit_factor(sub_pnls)
            wr = win_rate(sub_pnls)
            avg = sub_pnls.mean()
            total = sub_pnls.sum()
            n = len(sub_pnls)
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            print(f"  {month_names[m]:>6} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f}")
        print(f"{'='*80}")
    # --- Stop loss analysis ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  Stop Loss Analysis")
        print(f"{'='*80}")
        print(f"  {'Stop':>8} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12} {'WorstTrade':>12} {'MaxConsecL':>12}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        for stop_level in [None, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
            test_trades = _quick_backtest_with_stop(
                es_open, es_close, rsi, vix_close, chosen_one_mask, stop_level,
            )
            if len(test_trades) == 0:
                label = "None" if stop_level is None else f"${stop_level}"
                print(f"  {label:>8} {'0':>8} {'n/a':>8} {'n/a':>9} {'n/a':>10} {'n/a':>12} {'n/a':>12} {'n/a':>12}")
                continue
            t_pnls = pd.Series([t["pnl"] for t in test_trades], dtype=float)
            pf = profit_factor(t_pnls)
            wr = win_rate(t_pnls)
            avg = t_pnls.mean()
            total = t_pnls.sum()
            worst = t_pnls.min()
            n = len(t_pnls)
            # Max consecutive losses
            max_consec = 0
            cur_consec = 0
            for p in t_pnls:
                if p < 0:
                    cur_consec += 1
                    max_consec = max(max_consec, cur_consec)
                else:
                    cur_consec = 0
            pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
            label = "None" if stop_level is None else f"${stop_level}"
            marker = " <-- base" if stop_level is None else ""
            print(f"  {label:>8} {n:>8} {pf_str:>8} {wr:>8.1%} {avg:>10.0f} {total:>12.0f} {worst:>12.0f} {max_consec:>12}{marker}")
        print(f"{'='*80}")
    # --- Win/Loss streaks ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  Win/Loss Streaks")
        print(f"{'='*80}")
        pnls = trade_pnls.values
        # Longest winning/losing streaks
        max_win_streak = 0
        max_loss_streak = 0
        cur_win = 0
        cur_loss = 0
        for p in pnls:
            if p >= 0:
                cur_win += 1
                max_win_streak = max(max_win_streak, cur_win)
                cur_loss = 0
            else:
                cur_loss += 1
                max_loss_streak = max(max_loss_streak, cur_loss)
                cur_win = 0
        # Max consecutive losses (same as max_loss_streak)
        # Max drawdown from equity peak in dollar terms
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - peak
        max_dd = drawdowns.min()
        max_dd_end_idx = np.argmin(drawdowns)
        # Find peak before the max drawdown
        max_dd_peak_idx = np.argmax(cum_pnl[:max_dd_end_idx + 1]) if max_dd_end_idx > 0 else 0
        # Recovery: how many trades after max_dd_end_idx to get back to peak level
        peak_level = peak[max_dd_end_idx]
        recovery_trades = None
        for ri in range(max_dd_end_idx + 1, len(cum_pnl)):
            if cum_pnl[ri] >= peak_level:
                recovery_trades = ri - max_dd_end_idx
                break
        print(f"  Longest winning streak:  {max_win_streak} trades")
        print(f"  Longest losing streak:   {max_loss_streak} trades")
        print(f"  Max drawdown (from peak): ${max_dd:,.0f}")
        print(f"  Drawdown occurred: trade {max_dd_peak_idx + 1} to trade {max_dd_end_idx + 1}")
        if recovery_trades is not None:
            print(f"  Trades to recover:       {recovery_trades}")
        else:
            print(f"  Trades to recover:       Never recovered")
        print(f"{'='*80}")
    # --- In-sample vs out-of-sample ---
    if len(trade_list) > 0:
        print(f"\n{'='*80}")
        print(f"  In-Sample vs Out-of-Sample")
        print(f"{'='*80}")
        mid = len(trade_list) // 2
        is_trades = trade_list.iloc[:mid]
        oos_trades = trade_list.iloc[mid:]
        is_pnls = pd.Series(is_trades["pnl"].values, dtype=float)
        oos_pnls = pd.Series(oos_trades["pnl"].values, dtype=float)
        is_pf = profit_factor(is_pnls)
        oos_pf = profit_factor(oos_pnls)
        is_wr = win_rate(is_pnls)
        oos_wr = win_rate(oos_pnls)
        is_pf_str = f"{is_pf:.3f}" if is_pf != float("inf") else "inf"
        oos_pf_str = f"{oos_pf:.3f}" if oos_pf != float("inf") else "inf"
        is_date_range = f"{is_trades['entry_date'].iloc[0].date()} to {is_trades['entry_date'].iloc[-1].date()}"
        oos_date_range = f"{oos_trades['entry_date'].iloc[0].date()} to {oos_trades['entry_date'].iloc[-1].date()}"
        print(f"  {'Split':>12} {'Trades':>8} {'PF':>8} {'WinRate':>9} {'AvgPnL':>10} {'TotalPnL':>12}  {'Date Range'}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}  {'-'*25}")
        print(f"  {'In-Sample':>12} {len(is_pnls):>8} {is_pf_str:>8} {is_wr:>8.1%} {is_pnls.mean():>10.0f} {is_pnls.sum():>12.0f}  {is_date_range}")
        print(f"  {'Out-of-Samp':>12} {len(oos_pnls):>8} {oos_pf_str:>8} {oos_wr:>8.1%} {oos_pnls.mean():>10.0f} {oos_pnls.sum():>12.0f}  {oos_date_range}")
        print(f"{'='*80}")
    # --- Chosen One + Dip Buyer combined preview ---
    print(f"\n{'='*80}")
    print(f"  Chosen One + Dip Buyer Combined (Preview)")
    print(f"{'='*80}")
    co_csv = os.path.join(RESULTS_DIR, "week1_week4_only_trades.csv")
    if os.path.exists(co_csv) and len(trade_list) > 0:
        co_trades = pd.read_csv(co_csv, parse_dates=["entry_date", "exit_date"])
        # Yearly P&L for each strategy
        db_yearly = trade_list.groupby("year")["pnl"].sum()
        co_yearly = co_trades.groupby(co_trades["entry_date"].dt.year)["pnl"].sum()
        all_years = sorted(set(db_yearly.index) | set(co_yearly.index))
        print(f"  {'Year':>6} {'CO_PnL':>12} {'DB_PnL':>12} {'Combined':>12} {'Flag':>8}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
        total_co = 0
        total_db = 0
        for year in all_years:
            co_pnl = co_yearly.get(year, 0.0)
            db_pnl = db_yearly.get(year, 0.0)
            combined = co_pnl + db_pnl
            total_co += co_pnl
            total_db += db_pnl
            flag = "BOTH -" if co_pnl < 0 and db_pnl < 0 else ""
            print(f"  {year:>6} {co_pnl:>12,.0f} {db_pnl:>12,.0f} {combined:>12,.0f} {flag:>8}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'TOTAL':>6} {total_co:>12,.0f} {total_db:>12,.0f} {total_co + total_db:>12,.0f}")
        print(f"{'='*80}")
    else:
        print(f"  Chosen One trade list not found — run week1_week4_only.py first")
        print(f"{'='*80}")
    # --- Phidias $50K Swing Evaluation Simulation ---
    if len(trades) > 0 and len(daily_pnls_per_trade) > 0:
        attempts = simulate_phidias(
            [t["pnl"] for t in trades],
            daily_pnls_per_trade,
        )
        print(f"\n{'='*80}")
        print(f"  Phidias $50K Swing Eval Simulation")
        print(f"  Rules: ${PHIDIAS_EOD_DRAWDOWN:,.0f} EOD drawdown, "
              f"${PHIDIAS_PROFIT_TARGET:,.0f} profit target")
        print(f"{'='*80}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'Peak Bal':>12} {'Final Bal':>12}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        passed = 0
        failed = 0
        for a in attempts:
            status_str = a["status"]
            print(f"  {a['attempt']:>3} {status_str:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} ${a['final_balance']:>11,.0f}")
            if status_str == "PASSED":
                passed += 1
            elif status_str == "FAILED":
                failed += 1
        total_attempts = passed + failed
        pass_rate = passed / total_attempts * 100 if total_attempts > 0 else 0
        print(f"{'='*80}")
        print(f"  Total attempts: {total_attempts} | "
              f"Passed: {passed} | Failed: {failed} | "
              f"Pass rate: {pass_rate:.1f}%")
        if passed > 0:
            passed_attempts = [a for a in attempts if a["status"] == "PASSED"]
            avg_trades_to_pass = np.mean([a["trades_taken"] for a in passed_attempts])
            print(f"  Avg trades to pass: {avg_trades_to_pass:.1f}")
        print(f"{'='*80}\n")
    # --- Save outputs ---
    plot_equity(equity, STRATEGY_NAME)
    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "dip_buyer_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")
    return stats
def _quick_backtest(es_open, es_close, rsi, vix_close, chosen_one_mask,
                    rsi_entry, rsi_exit, vix_floor, vix_ceiling, max_hold):
    """Lightweight backtest for sensitivity analysis. Returns list of trade dicts."""
    trades = []
    in_trade = False
    entry_price = None
    hold_days = 0
    for i in range(1, len(es_close)):
        if in_trade:
            hold_days += 1
            current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50.0
            exit_signal = current_rsi > rsi_exit or hold_days >= max_hold
            if exit_signal:
                exit_price = es_close.iloc[i]
                pnl_points = exit_price - entry_price
                gross_pnl = pnl_points * ES_POINT_VALUE
                net_pnl = gross_pnl - COST_PER_TRADE
                trades.append({"pnl": net_pnl})
                in_trade = False
                entry_price = None
                hold_days = 0
        else:
            prev_rsi = rsi.iloc[i - 1] if not pd.isna(rsi.iloc[i - 1]) else 50.0
            prev_vix = vix_close.iloc[i - 1] if not pd.isna(vix_close.iloc[i - 1]) else 15.0
            signal = (
                prev_rsi < rsi_entry
                and prev_vix > vix_floor
                and prev_vix < vix_ceiling
                and not chosen_one_mask.iloc[i]
            )
            if signal:
                entry_price = es_open.iloc[i]
                hold_days = 0
                in_trade = True
    return trades
def _quick_backtest_with_stop(es_open, es_close, rsi, vix_close, chosen_one_mask,
                              stop_level):
    """Backtest with optional fixed dollar stop loss checked at daily close."""
    trades = []
    in_trade = False
    entry_price = None
    entry_idx = None
    hold_days = 0
    for i in range(1, len(es_close)):
        if in_trade:
            hold_days += 1
            current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50.0
            # Daily close P&L from entry
            daily_pnl = (es_close.iloc[i] - entry_price) * ES_POINT_VALUE - COST_PER_TRADE
            # Check stop loss at daily close
            if stop_level is not None and daily_pnl < -stop_level:
                trades.append({"pnl": daily_pnl})
                in_trade = False
                entry_price = None
                entry_idx = None
                hold_days = 0
                continue
            # Normal exit conditions
            exit_signal = current_rsi > RSI_EXIT_THRESHOLD or hold_days >= MAX_HOLD_DAYS
            if exit_signal:
                exit_price = es_close.iloc[i]
                pnl_points = exit_price - entry_price
                gross_pnl = pnl_points * ES_POINT_VALUE
                net_pnl = gross_pnl - COST_PER_TRADE
                trades.append({"pnl": net_pnl})
                in_trade = False
                entry_price = None
                entry_idx = None
                hold_days = 0
        else:
            prev_rsi = rsi.iloc[i - 1] if not pd.isna(rsi.iloc[i - 1]) else 50.0
            prev_vix = vix_close.iloc[i - 1] if not pd.isna(vix_close.iloc[i - 1]) else 15.0
            signal = (
                prev_rsi < RSI_ENTRY_THRESHOLD
                and prev_vix > VIX_FLOOR
                and prev_vix < VIX_CEILING
                and not chosen_one_mask.iloc[i]
            )
            if signal:
                entry_price = es_open.iloc[i]
                entry_idx = i
                hold_days = 0
                in_trade = True
    return trades
if __name__ == "__main__":
    run()
