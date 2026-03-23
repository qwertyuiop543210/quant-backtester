"""Intraday Fundamental Account Candidates — Phidias $50K.

Tests three intraday strategies suitable for a Phidias Fundamental account
(must close all positions by 3:59 PM ET, no overnight holds).

STRATEGY 1: Opening Range Breakout (ORB)
  - If price breaks X points above open within first 30 min -> buy
  - If price breaks X points below open within first 30 min -> sell short
  - Exit at 3:50 PM ET (simulated as daily close)
  - Uses daily High/Low as proxy for whether breakout threshold was hit
  - Tests X = 5, 10, 15, 20 points

STRATEGY 2: Gap Fade / Mean Reversion
  - Gap = today open - yesterday close
  - Gap down > X points -> buy at open, sell at close
  - Gap up > X points -> sell at open, buy at close
  - Tests X = 10, 15, 20, 25 with optional VIX > 20 filter

STRATEGY 3: Tuesday Reversal
  - If Monday was down (close < open): buy Tuesday open, sell Tuesday close
  - If Monday was up (close > open): sell Tuesday open, buy Tuesday close
  - Tests with and without VIX > 20 filter

All strategies: 1 ES contract, $30/trade costs, 2013-01-01 start.
Phidias Fundamental: $50K, $4,000 target, $2,500 EOD DD, ~$116 OTP.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import profit_factor, win_rate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COST_PER_TRADE = 30.0  # $5 comm + $25 slippage
START_DATE = "2013-01-01"

# Phidias Fundamental $50K
PHIDIAS_CAPITAL = 50_000
PHIDIAS_TARGET = 4_000
PHIDIAS_MAX_DD = 2_500
PHIDIAS_OTP_COST = 116.00


# =============================================================================
# CHOSEN ONE BUILDER (for correlation analysis)
# =============================================================================

def get_week_of_month(date):
    return (date.day - 1) // 7 + 1


def build_chosen_one_trades(es_data, vix_data):
    """Build Chosen One trades using validated ISO-week approach."""
    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")

    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
    }, index=es_close.index)

    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

    trades = []
    for week_key, group in df.groupby("week_key", sort=True):
        if len(group) < 3 or group.index[0] < pd.Timestamp(START_DATE):
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
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE
        trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "pnl": net_pnl,
        })
    return pd.DataFrame(trades)


# =============================================================================
# STRATEGY 1: OPENING RANGE BREAKOUT (ORB)
# =============================================================================

def build_orb_trades(es_data, threshold):
    """ORB using daily OHLC as proxy for intraday breakout.

    Logic: If High - Open >= threshold -> long breakout happened, entry = Open + threshold.
           If Open - Low >= threshold -> short breakout happened, entry = Open - threshold.
           If both triggered -> skip (ambiguous day, both directions hit).
           Exit = Close (proxy for 3:50 PM exit).

    P&L for long:  (Close - entry) * $50 - $30
    P&L for short: (entry - Close) * $50 - $30
    """
    open_ = es_data["Open"].astype(float)
    high = es_data["High"].astype(float)
    low = es_data["Low"].astype(float)
    close = es_data["Close"].astype(float)

    trades = []
    for i in range(len(es_data)):
        date = es_data.index[i]
        o, h, l, c = open_.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]

        long_triggered = (h - o) >= threshold
        short_triggered = (o - l) >= threshold

        if long_triggered and short_triggered:
            # Both sides hit — ambiguous, skip
            continue
        elif long_triggered:
            entry = o + threshold
            pnl_points = c - entry
            direction = "LONG"
        elif short_triggered:
            entry = o - threshold
            pnl_points = entry - c
            direction = "SHORT"
        else:
            continue

        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": date,
            "exit_date": date,
            "entry_price": entry,
            "exit_price": c,
            "direction": direction,
            "pnl_points": pnl_points,
            "pnl": net_pnl,
        })

    return pd.DataFrame(trades)


# =============================================================================
# STRATEGY 2: GAP FADE / MEAN REVERSION
# =============================================================================

def build_gap_fade_trades(es_data, vix_data, gap_threshold, vix_filter=False):
    """Gap fade: buy on gap down, sell on gap up, exit at close same day.

    Gap = today's open - prior day's close.
    With VIX filter: only trade when prior day VIX close > 20.
    """
    open_ = es_data["Open"].astype(float)
    close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(close.index, method="ffill")

    trades = []
    for i in range(1, len(es_data)):
        date = es_data.index[i]
        prev_date = es_data.index[i - 1]

        gap = open_.iloc[i] - close.iloc[i - 1]

        if abs(gap) < gap_threshold:
            continue

        if vix_filter:
            vix_val = vix_close.iloc[i - 1]
            if pd.isna(vix_val) or vix_val <= 20.0:
                continue

        if gap < -gap_threshold:
            # Gap down -> buy at open, sell at close (fade the gap)
            entry = open_.iloc[i]
            exit_ = close.iloc[i]
            pnl_points = exit_ - entry
            direction = "LONG"
        elif gap > gap_threshold:
            # Gap up -> sell at open, buy at close (fade the gap)
            entry = open_.iloc[i]
            exit_ = close.iloc[i]
            pnl_points = entry - exit_
            direction = "SHORT"
        else:
            continue

        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": date,
            "exit_date": date,
            "entry_price": entry,
            "exit_price": exit_,
            "direction": direction,
            "gap_points": gap,
            "pnl_points": pnl_points,
            "pnl": net_pnl,
        })

    return pd.DataFrame(trades)


# =============================================================================
# STRATEGY 3: TUESDAY REVERSAL
# =============================================================================

def build_tuesday_reversal_trades(es_data, vix_data, vix_filter=False):
    """Tuesday reversal: fade Monday's direction on Tuesday.

    If Monday was down (close < open): buy Tuesday open, sell Tuesday close.
    If Monday was up (close > open): sell Tuesday open, buy Tuesday close.
    """
    open_ = es_data["Open"].astype(float)
    close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(close.index, method="ffill")
    dow = es_data.index.dayofweek  # 0=Mon, 1=Tue, ...

    trades = []
    for i in range(1, len(es_data)):
        date = es_data.index[i]

        # Must be Tuesday
        if dow[i] != 1:
            continue

        # Prior day must be Monday
        prev_date = es_data.index[i - 1]
        if dow[i - 1] != 0:
            continue

        if vix_filter:
            # Use Monday's VIX close
            vix_val = vix_close.iloc[i - 1]
            if pd.isna(vix_val) or vix_val <= 20.0:
                continue

        mon_open = open_.iloc[i - 1]
        mon_close = close.iloc[i - 1]
        monday_direction = mon_close - mon_open

        if monday_direction == 0:
            continue  # Flat Monday, skip

        tue_open = open_.iloc[i]
        tue_close = close.iloc[i]

        if monday_direction < 0:
            # Monday was down -> buy Tuesday (expect reversal up)
            pnl_points = tue_close - tue_open
            direction = "LONG"
        else:
            # Monday was up -> sell Tuesday (expect reversal down)
            pnl_points = tue_open - tue_close
            direction = "SHORT"

        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": date,
            "exit_date": date,
            "entry_price": tue_open,
            "exit_price": tue_close,
            "direction": direction,
            "monday_move": monday_direction,
            "pnl_points": pnl_points,
            "pnl": net_pnl,
        })

    return pd.DataFrame(trades)


# =============================================================================
# PHIDIAS SIMULATION
# =============================================================================

def simulate_phidias(trade_pnls, capital, profit_target, max_dd):
    """Simulate Phidias eval attempts with EOD drawdown."""
    attempts = []
    i = 0

    while i < len(trade_pnls):
        balance = capital
        high_water = capital
        start_trade = i
        status = "in_progress"

        while i < len(trade_pnls):
            balance += trade_pnls[i]
            high_water = max(high_water, balance)
            eod_dd = high_water - balance
            profit = balance - capital
            i += 1

            if eod_dd >= max_dd:
                status = "FAILED"
                break
            if profit >= profit_target:
                status = "PASSED"
                break

        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade + 1,
            "end_trade": i,
            "trades_taken": i - start_trade,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - capital,
            "eod_dd_at_end": high_water - balance,
            "status": status,
        })

        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break

    return attempts


# =============================================================================
# PRINT / ANALYSIS HELPERS
# =============================================================================

def compute_stats(trades_df):
    """Compute summary stats dict from a trades DataFrame."""
    if len(trades_df) == 0:
        return {"trades": 0, "pf": 0, "wr": 0, "total_pnl": 0, "avg_pnl": 0,
                "best": 0, "worst": 0, "std": 0, "trades_per_year": 0}
    pnls = pd.Series(trades_df["pnl"].values, dtype=float)
    gross_w = pnls[pnls > 0].sum()
    gross_l = abs(pnls[pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    years = max((trades_df["exit_date"].max() - trades_df["entry_date"].min()).days / 365.25, 1)
    return {
        "trades": len(pnls),
        "pf": pf,
        "wr": (pnls > 0).mean(),
        "total_pnl": pnls.sum(),
        "avg_pnl": pnls.mean(),
        "best": pnls.max(),
        "worst": pnls.min(),
        "std": pnls.std(),
        "trades_per_year": len(pnls) / years,
    }


def print_strategy_stats(trades_df, label):
    """Print basic stats + year-by-year for a strategy."""
    s = compute_stats(trades_df)
    print(f"\n  {label}")
    print(f"  {'-'*60}")
    if s["trades"] == 0:
        print(f"  NO TRADES")
        return s
    print(f"  Trades:          {s['trades']}")
    print(f"  Trades/year:     {s['trades_per_year']:.1f}")
    print(f"  Profit Factor:   {s['pf']:.3f}")
    print(f"  Win Rate:        {s['wr']:.1%}")
    print(f"  Total P&L:       ${s['total_pnl']:>12,.0f}")
    print(f"  Avg P&L/trade:   ${s['avg_pnl']:>12,.0f}")
    print(f"  Best trade:      ${s['best']:>12,.0f}")
    print(f"  Worst trade:     ${s['worst']:>12,.0f}")
    print(f"  Std dev:         ${s['std']:>12,.0f}")

    # Year-by-year
    df_copy = trades_df.copy()
    df_copy["year"] = df_copy["entry_date"].dt.year
    print(f"\n  Year-by-year:")
    for year in sorted(df_copy["year"].unique()):
        subset = df_copy[df_copy["year"] == year]
        yr_pnls = subset["pnl"]
        yr_gw = yr_pnls[yr_pnls > 0].sum()
        yr_gl = abs(yr_pnls[yr_pnls < 0].sum())
        yr_pf = yr_gw / yr_gl if yr_gl > 0 else float("inf")
        verdict = "+" if yr_pf > 1.0 else "-"
        pf_str = f"{yr_pf:>6.2f}" if yr_pf < 100 else "   inf"
        print(f"    {year}: {len(subset):>4} trades, PF {pf_str}, "
              f"WR {(yr_pnls>0).mean():>5.1%}, total ${yr_pnls.sum():>10,.0f} {verdict}")
    return s


def compute_correlation_with_chosen(intraday_df, chosen_df):
    """Compute return correlation for trades that fall within the same week.

    Since intraday trades are single-day and Chosen One trades span a week,
    we sum intraday P&L during each Chosen One trade window.
    """
    if len(intraday_df) == 0 or len(chosen_df) == 0:
        return None

    co_pnls = []
    intra_pnls = []

    for _, co_trade in chosen_df.iterrows():
        co_start = co_trade["entry_date"]
        co_end = co_trade["exit_date"]

        # Find intraday trades within this Chosen One window
        mask = (intraday_df["entry_date"] >= co_start) & (intraday_df["entry_date"] <= co_end)
        overlap = intraday_df[mask]
        if len(overlap) > 0:
            co_pnls.append(co_trade["pnl"])
            intra_pnls.append(overlap["pnl"].sum())

    if len(co_pnls) < 10:
        return None
    return float(np.corrcoef(co_pnls, intra_pnls)[0, 1])


def phidias_summary(attempts, label):
    """Compute summary from Phidias attempts."""
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    total = len(passed) + len(failed)
    pass_rate = len(passed) / total * 100 if total > 0 else 0.0
    avg_trades_pass = np.mean([a["trades_taken"] for a in passed]) if passed else 0
    avg_trades_fail = np.mean([a["trades_taken"] for a in failed]) if failed else 0

    if pass_rate > 0:
        exp_attempts = 100.0 / pass_rate
        exp_cost = exp_attempts * PHIDIAS_OTP_COST
    else:
        exp_attempts = float("inf")
        exp_cost = float("inf")

    return {
        "label": label,
        "pass_rate": pass_rate,
        "passed": len(passed),
        "failed": len(failed),
        "total": total,
        "avg_trades_pass": avg_trades_pass,
        "avg_trades_fail": avg_trades_fail,
        "exp_attempts": exp_attempts,
        "exp_cost": exp_cost,
        "attempts": attempts,
    }


# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 90)
    print("  INTRADAY FUNDAMENTAL CANDIDATES — Phidias $50K")
    print("  Must close by 3:59 PM ET. $4,000 target, $2,500 EOD DD, ~$116 OTP")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    es_post = es[es.index >= START_DATE]
    vix_post = vix[vix.index >= START_DATE]
    print(f"Using data from {START_DATE}: {len(es_post)} trading days")

    # Build Chosen One trades for correlation reference
    chosen_trades = build_chosen_one_trades(es, vix)
    chosen_pnls = pd.Series(chosen_trades["pnl"].values, dtype=float)
    chosen_pf = profit_factor(chosen_pnls)
    print(f"\nChosen One reference: {len(chosen_trades)} trades, PF {chosen_pf:.3f}")

    # =========================================================================
    # STRATEGY 1: ORB
    # =========================================================================
    print(f"\n{'='*90}")
    print("  STRATEGY 1: OPENING RANGE BREAKOUT (ORB)")
    print("  Daily OHLC proxy: High/Low determines if breakout was hit, Close = exit")
    print(f"{'='*90}")

    orb_results = {}
    for threshold in [5, 10, 15, 20]:
        label = f"ORB X={threshold}pt"
        trades = build_orb_trades(es_post, threshold)
        stats = print_strategy_stats(trades, label)
        corr = compute_correlation_with_chosen(trades, chosen_trades)
        if corr is not None:
            print(f"  Correlation with Chosen One: {corr:.3f}")
        else:
            print(f"  Correlation with Chosen One: insufficient overlap")
        orb_results[label] = {"trades": trades, "stats": stats, "corr": corr}

    # =========================================================================
    # STRATEGY 2: GAP FADE
    # =========================================================================
    print(f"\n{'='*90}")
    print("  STRATEGY 2: GAP FADE / MEAN REVERSION")
    print("  Fade overnight gap > X points. Test with and without VIX > 20 filter.")
    print(f"{'='*90}")

    gap_results = {}
    for threshold in [10, 15, 20, 25]:
        for vix_filter in [False, True]:
            vix_tag = " VIX>20" if vix_filter else ""
            label = f"Gap Fade X={threshold}pt{vix_tag}"
            trades = build_gap_fade_trades(es_post, vix_post, threshold, vix_filter=vix_filter)
            stats = print_strategy_stats(trades, label)
            corr = compute_correlation_with_chosen(trades, chosen_trades)
            if corr is not None:
                print(f"  Correlation with Chosen One: {corr:.3f}")
            else:
                print(f"  Correlation with Chosen One: insufficient overlap")
            gap_results[label] = {"trades": trades, "stats": stats, "corr": corr}

    # =========================================================================
    # STRATEGY 3: TUESDAY REVERSAL
    # =========================================================================
    print(f"\n{'='*90}")
    print("  STRATEGY 3: TUESDAY REVERSAL")
    print("  Fade Monday's direction on Tuesday (open -> close).")
    print(f"{'='*90}")

    tue_results = {}
    for vix_filter in [False, True]:
        vix_tag = " VIX>20" if vix_filter else ""
        label = f"Tuesday Reversal{vix_tag}"
        trades = build_tuesday_reversal_trades(es_post, vix_post, vix_filter=vix_filter)
        stats = print_strategy_stats(trades, label)
        corr = compute_correlation_with_chosen(trades, chosen_trades)
        if corr is not None:
            print(f"  Correlation with Chosen One: {corr:.3f}")
        else:
            print(f"  Correlation with Chosen One: insufficient overlap")
        tue_results[label] = {"trades": trades, "stats": stats, "corr": corr}

    # =========================================================================
    # PHIDIAS SIMULATION — best variant of each strategy type
    # =========================================================================
    print(f"\n{'='*90}")
    print("  PHIDIAS $50K FUNDAMENTAL SIMULATION")
    print(f"  Capital: ${PHIDIAS_CAPITAL:,} | Target: ${PHIDIAS_TARGET:,} | "
          f"EOD DD: ${PHIDIAS_MAX_DD:,} | OTP: ${PHIDIAS_OTP_COST}")
    print(f"{'='*90}")

    # Collect ALL variants for Phidias testing
    all_variants = {}
    all_variants.update(orb_results)
    all_variants.update(gap_results)
    all_variants.update(tue_results)

    phidias_results = []
    for label, data in all_variants.items():
        trades = data["trades"]
        if len(trades) == 0:
            continue
        pnl_list = trades.sort_values("entry_date")["pnl"].tolist()
        attempts = simulate_phidias(pnl_list, PHIDIAS_CAPITAL, PHIDIAS_TARGET, PHIDIAS_MAX_DD)
        s = phidias_summary(attempts, label)
        s["stats"] = data["stats"]
        s["corr"] = data["corr"]
        phidias_results.append(s)

    # Print attempt details for each variant
    for s in phidias_results:
        attempts = s["attempts"]
        label = s["label"]
        print(f"\n  {label}")
        print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'EOD DD':>10}")
        print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*10}")
        for a in attempts:
            print(f"  {a['attempt']:>3} {a['status']:>10} {a['trades_taken']:>8} "
                  f"${a['profit']:>11,.0f} ${a['eod_dd_at_end']:>9,.0f}")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print(f"\n{'='*90}")
    print("  PHIDIAS COMPARISON TABLE — All Intraday Candidates")
    print(f"{'='*90}")
    print(f"  {'Strategy':<30} {'Trades':>7} {'PF':>7} {'WR':>7} {'Pass%':>8} "
          f"{'P/F':>6} {'AvgTr':>7} {'ExpCost':>10} {'Corr':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*8} "
          f"{'-'*6} {'-'*7} {'-'*10} {'-'*6}")

    for s in sorted(phidias_results, key=lambda x: -x["pass_rate"]):
        st = s["stats"]
        pf_str = f"{st['pf']:.2f}" if st["pf"] < 100 else "inf"
        corr_str = f"{s['corr']:.2f}" if s["corr"] is not None else "N/A"
        if s["pass_rate"] > 0:
            print(f"  {s['label']:<30} {st['trades']:>7} {pf_str:>7} {st['wr']:>6.1%} "
                  f"{s['pass_rate']:>7.1f}% {s['passed']:>3}/{s['total']:<3} "
                  f"{s['avg_trades_pass']:>7.1f} ${s['exp_cost']:>9,.0f} {corr_str:>6}")
        else:
            print(f"  {s['label']:<30} {st['trades']:>7} {pf_str:>7} {st['wr']:>6.1%} "
                  f"{s['pass_rate']:>7.1f}% {s['passed']:>3}/{s['total']:<3} "
                  f"{'N/A':>7} {'N/A':>10} {corr_str:>6}")

    print(f"{'='*90}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*90}")
    print("  VERDICT")
    print(f"{'='*90}")

    viable = [s for s in phidias_results if s["pass_rate"] > 0 and s["stats"]["pf"] >= 1.2]
    marginal = [s for s in phidias_results if s["pass_rate"] > 0 and s["stats"]["pf"] < 1.2]
    dead = [s for s in phidias_results if s["stats"]["pf"] < 1.0]

    if viable:
        best = max(viable, key=lambda x: x["pass_rate"])
        print(f"\n  BEST VIABLE: {best['label']}")
        print(f"    PF {best['stats']['pf']:.3f}, Pass rate {best['pass_rate']:.1f}%, "
              f"Exp cost ${best['exp_cost']:,.0f}")
        if best["corr"] is not None and abs(best["corr"]) < 0.3:
            print(f"    Low correlation with Chosen One ({best['corr']:.2f}) — good diversifier")
        elif best["corr"] is not None:
            print(f"    Correlation with Chosen One: {best['corr']:.2f} — limited diversification")
    else:
        print(f"\n  NO VIABLE CANDIDATES (PF >= 1.2 with passes)")

    if dead:
        print(f"\n  DEAD ON ARRIVAL (PF < 1.0):")
        for s in dead:
            print(f"    {s['label']}: PF {s['stats']['pf']:.3f}, {s['stats']['trades']} trades")

    if marginal:
        print(f"\n  MARGINAL (PF 1.0-1.2, insufficient edge):")
        for s in marginal:
            print(f"    {s['label']}: PF {s['stats']['pf']:.3f}, {s['stats']['trades']} trades")

    # Reminder about data limitations
    print(f"\n  NOTE: ORB results use daily OHLC proxy. If promising, re-test")
    print(f"  with actual 30-minute intraday bars for accurate breakout timing")
    print(f"  and intraday P&L tracking.")
    print(f"\n  Fundamental account requires closing by 3:59 PM ET.")
    print(f"  All strategies here are intraday (open->close), so they qualify.")

    print(f"\n{'='*90}")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save best variant from each strategy type
    for name, results_dict in [("orb", orb_results), ("gap_fade", gap_results),
                                ("tuesday_reversal", tue_results)]:
        # Find best PF variant
        best_label = None
        best_pf = -1
        for label, data in results_dict.items():
            if data["stats"]["trades"] > 0 and data["stats"]["pf"] > best_pf:
                best_pf = data["stats"]["pf"]
                best_label = label
        if best_label is not None and len(results_dict[best_label]["trades"]) > 0:
            csv_path = os.path.join(RESULTS_DIR, f"intraday_{name}_trades.csv")
            results_dict[best_label]["trades"].to_csv(csv_path, index=False)
            print(f"Best {name} trades saved to {csv_path}")

    # Save comparison summary
    summary_rows = []
    for s in sorted(phidias_results, key=lambda x: -x["pass_rate"]):
        summary_rows.append({
            "strategy": s["label"],
            "trades": s["stats"]["trades"],
            "pf": round(s["stats"]["pf"], 3),
            "win_rate": round(s["stats"]["wr"], 3),
            "total_pnl": round(s["stats"]["total_pnl"], 0),
            "avg_pnl": round(s["stats"]["avg_pnl"], 0),
            "pass_rate": round(s["pass_rate"], 1),
            "passed": s["passed"],
            "failed": s["failed"],
            "exp_cost": round(s["exp_cost"], 0) if s["exp_cost"] < 1e6 else None,
            "corr_chosen_one": round(s["corr"], 3) if s["corr"] is not None else None,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "intraday_fundamental_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Comparison summary saved to {summary_path}")

    return phidias_results


if __name__ == "__main__":
    run()
