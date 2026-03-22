"""Rico Bot Fractal Swing Strategy — 5-MINUTE ES BARS.
Same logic as Rico Bot, tested on 5m bars (the timeframe the bot runs best on).
yfinance provides ~60 days of 5m data.
2L/2R fractal swings for primary signals.
HTF gating: 10L/10R fractals (~50 bars each side = ~4H proxy on 5m).
MTF gating: 3L/3R fractals (~15 bars each side = ~1H proxy on 5m).
Sweep, BOS, CHoCH detection with original Rico Bot parameters.
SL at swing invalidation, TP at 1.5R, max 16-tick SL.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
import yfinance as yf
STRATEGY_NAME = "Rico Fractal Swings 5m (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
TICK_SIZE = 0.25
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE  # $30
INITIAL_CAPITAL = 100_000.0
SWEEP_MIN_RANGE_TICKS = 8
BOS_MIN_BODY_TICKS = 6
BOS_CLOSE_EXTREME_PCT = 0.25
MAX_SL_TICKS = 16
TP_RATIO = 1.5
REGIME_SMA_PERIOD = 20
# HTF fractal lookbacks for 5m bars
HTF_LEFT_RIGHT = 10   # 10 bars each side = ~50 min (~4H proxy)
MTF_LEFT_RIGHT = 3    # 3 bars each side = ~15 min (~1H proxy)
PRIMARY_LEFT_RIGHT = 2 # standard 2L/2R
TOPSTEP_CAPITAL = 150_000.0
TOPSTEP_TRAILING_DD = 4_500.0
TOPSTEP_PROFIT_TARGET = 9_000.0
def download_5m_data():
    """Download 5-minute ES futures data from yfinance."""
    print("Downloading 5m ES=F data from yfinance (max ~60 days)...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(period="60d", interval="5m")
    if df.empty:
        print("  WARNING: ES=F returned no data, trying SPY as proxy...")
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="60d", interval="5m")
        if df.empty:
            raise ValueError("Could not download any 5m data")
        print("  Using SPY as ES proxy — price levels differ but structure is similar")
    df = df.dropna()
    print(f"  Downloaded {len(df)} bars")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    trading_days = df.index.normalize().nunique()
    print(f"  Trading days: {trading_days}")
    return df
def find_fractals(highs, lows, left, right):
    n = len(highs)
    frac_highs = []
    frac_lows = []
    for i in range(left, n - right):
        is_high = True
        for j in range(1, left + 1):
            if highs[i] <= highs[i - j]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if highs[i] <= highs[i + j]:
                    is_high = False
                    break
        if is_high:
            frac_highs.append((i, highs[i]))
        is_low = True
        for j in range(1, left + 1):
            if lows[i] >= lows[i - j]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if lows[i] >= lows[i + j]:
                    is_low = False
                    break
        if is_low:
            frac_lows.append((i, lows[i]))
    return frac_highs, frac_lows
def detect_sweep(bar_open, bar_high, bar_low, bar_close, swing_highs, swing_lows):
    bar_range = bar_high - bar_low
    min_range = SWEEP_MIN_RANGE_TICKS * TICK_SIZE
    if bar_range < min_range:
        return None
    for _, sw_price in swing_lows:
        if bar_low < sw_price and bar_close > sw_price:
            return 'bull_sweep'
    for _, sw_price in swing_highs:
        if bar_high > sw_price and bar_close < sw_price:
            return 'bear_sweep'
    return None
def detect_bos(bar_open, bar_high, bar_low, bar_close, swing_highs, swing_lows):
    body = abs(bar_close - bar_open)
    bar_range = bar_high - bar_low
    min_body = BOS_MIN_BODY_TICKS * TICK_SIZE
    if body < min_body or bar_range == 0:
        return None
    close_pct = (bar_close - bar_low) / bar_range
    if close_pct >= (1.0 - BOS_CLOSE_EXTREME_PCT):
        for _, sw_price in swing_highs:
            if bar_close > sw_price and bar_open <= sw_price:
                return 'bull_bos'
    if close_pct <= BOS_CLOSE_EXTREME_PCT:
        for _, sw_price in swing_lows:
            if bar_close < sw_price and bar_open >= sw_price:
                return 'bear_bos'
    return None
def detect_choch(swing_highs, swing_lows, bar_close):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    last_low = swing_lows[-1][1]
    prev_low = swing_lows[-2][1]
    last_high = swing_highs[-1][1]
    if last_low < prev_low and bar_close > last_high:
        return 'bull_choch'
    last_high_val = swing_highs[-1][1]
    prev_high_val = swing_highs[-2][1]
    last_low_val = swing_lows[-1][1]
    if last_high_val > prev_high_val and bar_close < last_low_val:
        return 'bear_choch'
    return None
def calc_bias(bar_close, swing_highs, swing_lows):
    if len(swing_highs) == 0 or len(swing_lows) == 0:
        return 'neutral'
    last_sh = swing_highs[-1][1]
    last_sl = swing_lows[-1][1]
    midpoint = (last_sh + last_sl) / 2.0
    if bar_close > midpoint:
        return 'bull'
    elif bar_close < midpoint:
        return 'bear'
    return 'neutral'
def run_strategy(es):
    opens = es["Open"].values.astype(float)
    highs = es["High"].values.astype(float)
    lows = es["Low"].values.astype(float)
    closes = es["Close"].values.astype(float)
    dates = es.index
    n = len(closes)
    print(f"  Computing fractals on {n} bars...")
    sma20 = pd.Series(closes).rolling(REGIME_SMA_PERIOD).mean().values
    fh_2, fl_2 = find_fractals(highs, lows, PRIMARY_LEFT_RIGHT, PRIMARY_LEFT_RIGHT)
    print(f"  2L/2R fractals: {len(fh_2)} highs, {len(fl_2)} lows")
    fh_mtf, fl_mtf = find_fractals(highs, lows, MTF_LEFT_RIGHT, MTF_LEFT_RIGHT)
    print(f"  {MTF_LEFT_RIGHT}L/{MTF_LEFT_RIGHT}R fractals (1H proxy): {len(fh_mtf)} highs, {len(fl_mtf)} lows")
    fh_htf, fl_htf = find_fractals(highs, lows, HTF_LEFT_RIGHT, HTF_LEFT_RIGHT)
    print(f"  {HTF_LEFT_RIGHT}L/{HTF_LEFT_RIGHT}R fractals (4H proxy): {len(fh_htf)} highs, {len(fl_htf)} lows")
    trades = []
    equity = np.full(n, INITIAL_CAPITAL, dtype=float)
    position = np.zeros(n)
    cash = INITIAL_CAPITAL
    in_trade = False
    entry_price = 0.0
    entry_date = None
    stop_loss = 0.0
    take_profit = 0.0
    trade_direction = 0
    cooldown_until = 0
    sweep_count = 0
    bos_count = 0
    choch_count = 0
    gated_long = 0
    gated_short = 0
    sl_filtered = 0
    start_bar = max(REGIME_SMA_PERIOD, HTF_LEFT_RIGHT + 1)
    for i in range(start_bar, n):
        active_sh_2 = [(idx, p) for idx, p in fh_2 if idx <= i - PRIMARY_LEFT_RIGHT]
        active_sl_2 = [(idx, p) for idx, p in fl_2 if idx <= i - PRIMARY_LEFT_RIGHT]
        active_sh_mtf = [(idx, p) for idx, p in fh_mtf if idx <= i - MTF_LEFT_RIGHT]
        active_sl_mtf = [(idx, p) for idx, p in fl_mtf if idx <= i - MTF_LEFT_RIGHT]
        active_sh_htf = [(idx, p) for idx, p in fh_htf if idx <= i - HTF_LEFT_RIGHT]
        active_sl_htf = [(idx, p) for idx, p in fl_htf if idx <= i - HTF_LEFT_RIGHT]
        recent_sh = active_sh_2[-20:]
        recent_sl = active_sl_2[-20:]
        if in_trade:
            hit_sl = False
            hit_tp = False
            exit_price = 0.0
            exit_price_tp = 0.0
            if trade_direction == 1:
                if lows[i] <= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                if highs[i] >= take_profit:
                    hit_tp = True
                    exit_price_tp = take_profit
            else:
                if highs[i] >= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                if lows[i] <= take_profit:
                    hit_tp = True
                    exit_price_tp = take_profit
            if hit_sl and hit_tp:
                hit_tp = False
            if hit_sl:
                pnl_points = (exit_price - entry_price) * trade_direction
                pnl_dollars = pnl_points * ES_POINT_VALUE - COST_PER_TRADE
                cash += pnl_dollars
                trades.append({
                    "entry_date": entry_date, "exit_date": dates[i],
                    "direction": "long" if trade_direction == 1 else "short",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "stop_loss": stop_loss, "take_profit": take_profit,
                    "pnl_points": pnl_points,
                    "gross_pnl": pnl_points * ES_POINT_VALUE,
                    "costs": COST_PER_TRADE, "pnl": pnl_dollars,
                    "exit_reason": "STOP LOSS",
                })
                in_trade = False
                cooldown_until = i + 3
            elif hit_tp:
                pnl_points = (exit_price_tp - entry_price) * trade_direction
                pnl_dollars = pnl_points * ES_POINT_VALUE - COST_PER_TRADE
                cash += pnl_dollars
                trades.append({
                    "entry_date": entry_date, "exit_date": dates[i],
                    "direction": "long" if trade_direction == 1 else "short",
                    "entry_price": entry_price, "exit_price": exit_price_tp,
                    "stop_loss": stop_loss, "take_profit": take_profit,
                    "pnl_points": pnl_points,
                    "gross_pnl": pnl_points * ES_POINT_VALUE,
                    "costs": COST_PER_TRADE, "pnl": pnl_dollars,
                    "exit_reason": "TAKE PROFIT",
                })
                in_trade = False
                cooldown_until = i + 2
            if in_trade:
                position[i] = trade_direction
                equity[i] = cash + (closes[i] - entry_price) * trade_direction * ES_POINT_VALUE
                continue
        equity[i] = cash
        position[i] = 0
        if i < cooldown_until:
            continue
        if len(recent_sh) < 2 or len(recent_sl) < 2:
            continue
        if np.isnan(sma20[i]):
            continue
        sma_dist = abs(closes[i] - sma20[i])
        avg_range = np.mean(highs[max(0, i-20):i] - lows[max(0, i-20):i])
        if avg_range == 0:
            continue
        if sma_dist > 2.0 * avg_range:
            continue
        htf_bias = calc_bias(closes[i], active_sh_htf[-10:], active_sl_htf[-10:])
        mtf_bias = calc_bias(closes[i], active_sh_mtf[-10:], active_sl_mtf[-10:])
        sweep = detect_sweep(opens[i], highs[i], lows[i], closes[i],
                             recent_sh[-5:], recent_sl[-5:])
        bos = detect_bos(opens[i], highs[i], lows[i], closes[i],
                         recent_sh[-5:], recent_sl[-5:])
        choch = detect_choch(recent_sh, recent_sl, closes[i])
        if sweep:
            sweep_count += 1
        if bos:
            bos_count += 1
        if choch:
            choch_count += 1
        long_signal = (sweep == 'bull_sweep' or bos == 'bull_bos'
                       or choch == 'bull_choch')
        long_gated = long_signal and htf_bias != 'bear' and mtf_bias == 'bull'
        short_signal = (sweep == 'bear_sweep' or bos == 'bear_bos'
                        or choch == 'bear_choch')
        short_gated = short_signal and htf_bias != 'bull' and mtf_bias == 'bear'
        if long_gated:
            gated_long += 1
            sl_level = recent_sl[-1][1]
            risk_points = closes[i] - sl_level
            risk_ticks = risk_points / TICK_SIZE
            if risk_ticks <= 0 or risk_ticks > MAX_SL_TICKS:
                sl_filtered += 1
                continue
            entry_price = closes[i]
            stop_loss = sl_level
            take_profit = entry_price + risk_points * TP_RATIO
            trade_direction = 1
            in_trade = True
            entry_date = dates[i]
            position[i] = 1
            equity[i] = cash
        elif short_gated:
            gated_short += 1
            sl_level = recent_sh[-1][1]
            risk_points = sl_level - closes[i]
            risk_ticks = risk_points / TICK_SIZE
            if risk_ticks <= 0 or risk_ticks > MAX_SL_TICKS:
                sl_filtered += 1
                continue
            entry_price = closes[i]
            stop_loss = sl_level
            take_profit = entry_price - risk_points * TP_RATIO
            trade_direction = -1
            in_trade = True
            entry_date = dates[i]
            position[i] = -1
            equity[i] = cash
    if in_trade:
        exit_price = closes[-1]
        pnl_points = (exit_price - entry_price) * trade_direction
        pnl_dollars = pnl_points * ES_POINT_VALUE - COST_PER_TRADE
        cash += pnl_dollars
        trades.append({
            "entry_date": entry_date, "exit_date": dates[-1],
            "direction": "long" if trade_direction == 1 else "short",
            "entry_price": entry_price, "exit_price": exit_price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "pnl_points": pnl_points,
            "gross_pnl": pnl_points * ES_POINT_VALUE,
            "costs": COST_PER_TRADE, "pnl": pnl_dollars,
            "exit_reason": "END OF DATA",
        })
        equity[-1] = cash
    print(f"\n  Signal diagnostics:")
    print(f"    Raw sweeps detected:  {sweep_count}")
    print(f"    Raw BOS detected:     {bos_count}")
    print(f"    Raw CHoCH detected:   {choch_count}")
    print(f"    Gated long signals:   {gated_long}")
    print(f"    Gated short signals:  {gated_short}")
    print(f"    Filtered by SL size:  {sl_filtered}")
    print(f"    Trades taken:         {len(trades)}")
    trade_list = pd.DataFrame(trades)
    trade_pnls = (pd.Series(trade_list["pnl"].values, dtype=float)
                  if len(trades) > 0 else pd.Series(dtype=float))
    return {
        "trades": trade_list,
        "trade_pnls": trade_pnls,
        "equity_curve": pd.Series(equity, index=dates),
        "position": pd.Series(position, index=dates),
    }
def simulate_topstep(trade_pnls):
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
    print(f"\n{'='*80}")
    print(f"  {STRATEGY_NAME}")
    print(f"{'='*80}")
    es = download_5m_data()
    print(f"\nStrategy parameters:")
    print(f"  Fractal: {PRIMARY_LEFT_RIGHT}L/{PRIMARY_LEFT_RIGHT}R (primary), "
          f"{MTF_LEFT_RIGHT}L/{MTF_LEFT_RIGHT}R (1H proxy), "
          f"{HTF_LEFT_RIGHT}L/{HTF_LEFT_RIGHT}R (4H proxy)")
    print(f"  Sweep min range: {SWEEP_MIN_RANGE_TICKS} ticks ({SWEEP_MIN_RANGE_TICKS * TICK_SIZE} pts)")
    print(f"  BOS min body: {BOS_MIN_BODY_TICKS} ticks, close extreme: {BOS_CLOSE_EXTREME_PCT:.0%}")
    print(f"  Max SL: {MAX_SL_TICKS} ticks ({MAX_SL_TICKS * TICK_SIZE} pts)")
    print(f"  TP ratio: {TP_RATIO}R")
    print(f"  Costs: ${COST_PER_TRADE:.0f}/trade "
          f"(${COMMISSION_RT} comm + 2x${SLIPPAGE_PER_SIDE} slip)")
    print(f"  Regime filter: SMA{REGIME_SMA_PERIOD} distance < 2x avg range\n")
    result = run_strategy(es)
    trade_list = result["trades"]
    trade_pnls = result["trade_pnls"]
    equity = result["equity_curve"]
    pos = result["position"]
    if len(trade_list) == 0:
        print("\n  NO TRADES GENERATED on 5-minute bars.")
        print("  Check signal diagnostics above — if gated signals > 0 but")
        print("  trades = 0, the SL filter is still too tight.")
        print("\n  VERDICT: NO EDGE (insufficient trades)")
        return
    # Summary stats
    total_pnl = trade_pnls.sum()
    winners = trade_pnls[trade_pnls > 0]
    losers = trade_pnls[trade_pnls <= 0]
    gross_wins = winners.sum() if len(winners) > 0 else 0
    gross_losses = abs(losers.sum()) if len(losers) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    wr = len(winners) / len(trade_pnls) if len(trade_pnls) > 0 else 0
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    eq = result["equity_curve"]
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    print(f"\n{'='*70}")
    print(f"  {STRATEGY_NAME} — Results")
    print(f"{'='*70}")
    print(f"  Total trades:     {len(trade_pnls):>8}")
    print(f"  Winners:          {len(winners):>8}")
    print(f"  Losers:           {len(losers):>8}")
    print(f"  Win rate:         {wr:>7.1%}")
    print(f"  Profit factor:    {pf:>8.3f}")
    print(f"  Total P&L:        ${total_pnl:>10,.0f}")
    print(f"  Avg winner:       ${avg_win:>10,.0f}")
    print(f"  Avg loser:        ${avg_loss:>10,.0f}")
    print(f"  Max drawdown:     {max_dd:>7.1%}")
    print(f"  Final equity:     ${equity.iloc[-1]:>10,.0f}")
    expectancy = total_pnl / len(trade_pnls)
    print(f"  Expectancy/trade: ${expectancy:>10,.0f}")
    if pf > 1.2 and len(trade_pnls) > 30:
        print(f"\n  VERDICT: POTENTIALLY TRADEABLE (PF {pf:.2f}, {len(trade_pnls)} trades)")
        print(f"  WARNING: Only ~60 days of data. Need 6+ months for confidence.")
    elif len(trade_pnls) < 30:
        print(f"\n  VERDICT: INSUFFICIENT DATA ({len(trade_pnls)} trades, need 30+ minimum)")
    else:
        print(f"\n  VERDICT: NO EDGE (PF {pf:.2f})")
    # Trade breakdown
    print(f"\n{'='*70}")
    print(f"  Trade Breakdown")
    print(f"{'='*70}")
    longs = trade_list[trade_list["direction"] == "long"]
    shorts = trade_list[trade_list["direction"] == "short"]
    print(f"  Long trades:  {len(longs):>5}  |  Short trades: {len(shorts):>5}")
    if len(longs) > 0:
        long_wr = (longs["pnl"] > 0).mean()
        print(f"  Long win rate:  {long_wr:.1%}  |  Avg long P&L: ${longs['pnl'].mean():,.0f}")
    if len(shorts) > 0:
        short_wr = (shorts["pnl"] > 0).mean()
        print(f"  Short win rate: {short_wr:.1%}  |  Avg short P&L: ${shorts['pnl'].mean():,.0f}")
    print(f"\n  Exit reasons:")
    for reason in trade_list["exit_reason"].unique():
        count = (trade_list["exit_reason"] == reason).sum()
        subset = trade_list[trade_list["exit_reason"] == reason]
        avg_pnl = subset["pnl"].mean()
        print(f"    {reason:<15} {count:>5} trades  avg P&L: ${avg_pnl:>8,.0f}")
    if len(trade_list) > 0:
        trade_list = trade_list.copy()
        trade_list["week"] = pd.to_datetime(trade_list["entry_date"]).dt.isocalendar().week.values
        trade_list["year"] = pd.to_datetime(trade_list["entry_date"]).dt.year
        trade_list["yr_wk"] = trade_list["year"].astype(str) + "-W" + trade_list["week"].astype(str).str.zfill(2)
        print(f"\n  Week-by-week:")
        print(f"  {'Week':>10} {'Trades':>8} {'Win%':>8} {'Total P&L':>12} {'Avg P&L':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
        for wk, grp in trade_list.groupby("yr_wk"):
            wk_trades = len(grp)
            wk_wr = (grp["pnl"] > 0).mean()
            wk_total = grp["pnl"].sum()
            wk_avg = grp["pnl"].mean()
            print(f"  {wk:>10} {wk_trades:>8} {wk_wr:>7.1%} ${wk_total:>11,.0f} ${wk_avg:>9,.0f}")
    print(f"{'='*70}")
    # Topstep simulation
    if len(trade_pnls) >= 10:
        pnl_list = trade_pnls.tolist()
        attempts = simulate_topstep(pnl_list)
        print(f"\n{'='*80}")
        print(f"  Topstep $150K Eval Simulation")
        print(f"  Rules: ${TOPSTEP_TRAILING_DD:,.0f} trailing DD, ${TOPSTEP_PROFIT_TARGET:,.0f} profit target")
        print(f"{'='*80}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'Peak Bal':>12} {'Final Bal':>12}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        passed = 0
        failed = 0
        for a in attempts:
            status_str = a["status"]
            print(f"  {a['attempt']:>3} {status_str:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['peak_balance']:>11,.0f} "
                  f"${a['final_balance']:>11,.0f}")
            if status_str == "PASSED":
                passed += 1
            elif status_str == "FAILED":
                failed += 1
        total_attempts = passed + failed
        pass_rate = passed / total_attempts * 100 if total_attempts > 0 else 0
        print(f"{'='*80}")
        print(f"  Total attempts: {total_attempts} | Passed: {passed} | Failed: {failed} | Pass rate: {pass_rate:.1f}%")
        print(f"{'='*80}\n")
    # Save trade list
    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "rico_fractal_5m_trades.csv")
        save_cols = [c for c in trade_list.columns if c not in ("week", "year", "yr_wk")]
        trade_list[save_cols].to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")
    return result
if __name__ == "__main__":
    run()
