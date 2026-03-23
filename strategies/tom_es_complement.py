"""Turn-of-Month ES — Complement to the Chosen One (Week 1/4).

Tests whether adding TOM trades improves the Phidias pass rate and speed.

TOM rule: Buy 1 ES at close on trading day -2 of each month,
          sell at close on trading day +3 of next month (~5 day hold).
Driver: pension/salary flows around month-end (structurally independent
        from Week 1/4 ETF rebalancing driver).

Chosen One rule: Buy Monday open / sell Friday close, weeks 1 & 4 only,
                 skip when prior Friday VIX 15-20.

Costs: $30/trade ($5 comm + $25 slippage) for both strategies.
ES point value = $50.
Start: 2013-01-01 (same era as Chosen One validation).

Reports:
  1. TOM standalone stats
  2. Chosen One standalone stats (reproduced here for apples-to-apples)
  3. Return correlation between overlapping TOM and Chosen One trades
  4. Combined equity curve (both strategies on same account)
  5. Phidias $50K sim: Chosen One alone vs TOM alone vs Combined
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
INITIAL_CAPITAL = 100_000.0
START_DATE = "2013-01-01"

# Phidias $50K account
PHIDIAS_CAPITAL = 50_000
PHIDIAS_TARGET = 4_000
PHIDIAS_MAX_DD = 2_500
PHIDIAS_OTP_COST = 144.60


# =============================================================================
# TOM TRADE BUILDER
# =============================================================================

def build_tom_trades(es_data):
    """Build TOM trades: buy close on TD-2, sell close on TD+3.

    Returns DataFrame with entry_date, exit_date, pnl, etc.
    """
    close = es_data["Close"].astype(float)
    dates = close.index
    ym = dates.to_period("M")
    months = sorted(ym.unique())

    trades = []
    for k in range(len(months) - 1):
        cur_month = months[k]
        next_month = months[k + 1]

        cur_days = dates[ym == cur_month]
        next_days = dates[ym == next_month]

        if len(cur_days) < 2 or len(next_days) < 3:
            continue

        entry_date = cur_days[-2]   # Trading day -2
        exit_date = next_days[2]    # Trading day +3 (0-indexed: idx 2 = 3rd day)

        entry_price = close.loc[entry_date]
        exit_price = close.loc[exit_date]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "pnl": net_pnl,
            "strategy": "TOM",
        })

    return pd.DataFrame(trades)


# =============================================================================
# CHOSEN ONE TRADE BUILDER (Week 1/4 + VIX filter)
# =============================================================================

def get_week_of_month(date):
    return (date.day - 1) // 7 + 1


def build_chosen_one_trades(es_data, vix_data):
    """Build Chosen One trades: Week 1/4, Monday open -> Friday close, VIX filter.

    Uses the ISO-week grouping approach from the validated backtest
    (week14_final_checks.py) for exact reproducibility.
    """
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

        # VIX filter: prior Friday's close
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
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "pnl": net_pnl,
            "strategy": "ChosenOne",
            "week_num": week_num,
            "vix_friday": vix_val,
        })

    return pd.DataFrame(trades)


# =============================================================================
# COMBINED TRADE LIST (chronological merge, no double-counting overlaps)
# =============================================================================

def build_combined_trades(chosen_df, tom_df):
    """Merge both trade lists chronologically.

    If a TOM trade overlaps with a Chosen One trade (any date overlap),
    we still take BOTH — they're separate positions on the same instrument.
    The combined P&L is additive, and the risk is 2 contracts during overlap.

    Returns merged DataFrame sorted by entry_date.
    """
    combined = pd.concat([chosen_df, tom_df], ignore_index=True)
    combined = combined.sort_values("entry_date").reset_index(drop=True)
    return combined


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def analyze_overlap(chosen_df, tom_df):
    """Find TOM trades that overlap with Chosen One trades and compute correlation."""
    print(f"\n{'='*80}")
    print("  OVERLAP & CORRELATION ANALYSIS")
    print(f"{'='*80}")

    overlap_count = 0
    tom_overlapping_pnls = []
    chosen_overlapping_pnls = []

    for _, tom_trade in tom_df.iterrows():
        tom_start = tom_trade["entry_date"]
        tom_end = tom_trade["exit_date"]

        # Find any Chosen One trade that overlaps this TOM window
        for _, co_trade in chosen_df.iterrows():
            co_start = co_trade["entry_date"]
            co_end = co_trade["exit_date"]
            if co_start <= tom_end and co_end >= tom_start:
                overlap_count += 1
                tom_overlapping_pnls.append(tom_trade["pnl"])
                chosen_overlapping_pnls.append(co_trade["pnl"])
                break  # Only count first overlap per TOM trade

    print(f"  TOM trades:        {len(tom_df)}")
    print(f"  Chosen One trades: {len(chosen_df)}")
    print(f"  Overlapping pairs: {overlap_count} "
          f"({overlap_count/len(tom_df)*100:.0f}% of TOM trades)")

    if overlap_count >= 10:
        corr = np.corrcoef(tom_overlapping_pnls, chosen_overlapping_pnls)[0, 1]
        print(f"  Return correlation (overlapping trades): {corr:.3f}")
        if abs(corr) < 0.3:
            print(f"  -> Low correlation — strategies are largely independent")
        elif corr > 0.3:
            print(f"  -> Positive correlation — both win/lose together (less diversification)")
        else:
            print(f"  -> Negative correlation — natural hedge during overlaps")
    else:
        print(f"  -> Too few overlapping trades ({overlap_count}) for meaningful correlation")

    # How many days of the year is 2 contracts deployed?
    if overlap_count > 0:
        total_overlap_days = 0
        for _, tom_trade in tom_df.iterrows():
            tom_start = tom_trade["entry_date"]
            tom_end = tom_trade["exit_date"]
            for _, co_trade in chosen_df.iterrows():
                co_start = co_trade["entry_date"]
                co_end = co_trade["exit_date"]
                if co_start <= tom_end and co_end >= tom_start:
                    overlap_start = max(tom_start, co_start)
                    overlap_end = min(tom_end, co_end)
                    total_overlap_days += (overlap_end - overlap_start).days + 1
                    break
        years = (tom_df["exit_date"].max() - tom_df["entry_date"].min()).days / 365.25
        print(f"  Total overlap days: ~{total_overlap_days} ({total_overlap_days/years:.0f}/year)")
        print(f"  -> During overlaps, you hold 2 ES contracts (double margin/risk)")

    print(f"{'='*80}")


# =============================================================================
# PHIDIAS SIMULATION
# =============================================================================

def simulate_phidias(trade_pnls, capital, profit_target, max_dd):
    """Simulate Phidias eval attempts with EOD drawdown.

    Drawdown checked after each trade closes (EOD, not intraday).
    """
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


def phidias_summary(attempts, label):
    """Print and return summary stats for a set of Phidias attempts."""
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    total = len(passed) + len(failed)
    pass_rate = len(passed) / total * 100 if total > 0 else 0.0

    avg_trades_pass = np.mean([a["trades_taken"] for a in passed]) if passed else 0
    avg_trades_fail = np.mean([a["trades_taken"] for a in failed]) if failed else 0

    if pass_rate > 0:
        exp_attempts = 100.0 / pass_rate
        # Estimate trades/month based on actual trade frequency
        total_trades = sum(a["trades_taken"] for a in attempts if a["status"] in ("PASSED", "FAILED"))
        trades_per_month = total_trades / max(total, 1) / 2.0  # rough estimate
        if trades_per_month == 0:
            trades_per_month = 2.0
        exp_months = (exp_attempts - 1) * (avg_trades_fail / trades_per_month) + \
                     (avg_trades_pass / trades_per_month)
        exp_months = max(exp_months, avg_trades_pass / trades_per_month)
        exp_cost = exp_attempts * PHIDIAS_OTP_COST
    else:
        exp_attempts = float("inf")
        exp_months = float("inf")
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
        "exp_months": exp_months,
        "exp_cost": exp_cost,
        "attempts": attempts,
    }


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_trade_stats(df, label):
    """Print basic trade statistics."""
    pnls = pd.Series(df["pnl"].values, dtype=float)
    gross_w = pnls[pnls > 0].sum()
    gross_l = abs(pnls[pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = (pnls > 0).mean()
    years = (df["exit_date"].max() - df["entry_date"].min()).days / 365.25

    print(f"\n  {label}")
    print(f"  {'-'*50}")
    print(f"  Trades:          {len(pnls)}")
    print(f"  Trades/year:     {len(pnls)/years:.1f}")
    print(f"  Profit Factor:   {pf:.3f}")
    print(f"  Win Rate:        {wr:.1%}")
    print(f"  Total P&L:       ${pnls.sum():>12,.0f}")
    print(f"  Avg P&L/trade:   ${pnls.mean():>12,.0f}")
    print(f"  Best trade:      ${pnls.max():>12,.0f}")
    print(f"  Worst trade:     ${pnls.min():>12,.0f}")
    print(f"  Std dev:         ${pnls.std():>12,.0f}")

    # Year-by-year
    df_copy = df.copy()
    df_copy["year"] = df_copy["entry_date"].dt.year
    print(f"\n  Year-by-year:")
    for year in sorted(df_copy["year"].unique()):
        subset = df_copy[df_copy["year"] == year]
        yr_pnls = subset["pnl"]
        yr_gw = yr_pnls[yr_pnls > 0].sum()
        yr_gl = abs(yr_pnls[yr_pnls < 0].sum())
        yr_pf = yr_gw / yr_gl if yr_gl > 0 else float("inf")
        verdict = "+" if yr_pf > 1.0 else "-"
        print(f"    {year}: {len(subset):>3} trades, PF {yr_pf:>6.2f}, "
              f"WR {(yr_pnls>0).mean():>5.1%}, total ${yr_pnls.sum():>9,.0f} {verdict}")


def print_phidias_attempts(summary_dict):
    """Print detailed Phidias attempt log."""
    label = summary_dict["label"]
    attempts = summary_dict["attempts"]

    print(f"\n  {label} — Phidias $50K Eval")
    print(f"  Target: ${PHIDIAS_TARGET:,} | DD Limit: ${PHIDIAS_MAX_DD:,}")
    print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'Profit':>12} {'EOD DD':>10}")
    print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*12} {'-'*10}")
    for a in attempts:
        print(f"  {a['attempt']:>3} {a['status']:>10} {a['trades_taken']:>8} "
              f"${a['profit']:>11,.0f} ${a['eod_dd_at_end']:>9,.0f}")


# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 80)
    print("  TOM ES COMPLEMENT — Does Turn-of-Month improve the Chosen One?")
    print("  Phidias $50K: $4,000 target, $2,500 EOD DD limit")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    # Filter to post-2013
    es_post = es[es.index >= START_DATE]
    print(f"Using data from {START_DATE}: {len(es_post)} trading days")

    # =========================================================================
    # 1. BUILD TRADE LISTS
    # =========================================================================
    print(f"\n{'='*80}")
    print("  BUILDING TRADE LISTS")
    print(f"{'='*80}")

    tom_trades = build_tom_trades(es_post)
    chosen_trades = build_chosen_one_trades(es, vix)
    combined_trades = build_combined_trades(chosen_trades, tom_trades)

    print(f"  TOM trades:        {len(tom_trades)}")
    print(f"  Chosen One trades: {len(chosen_trades)}")
    print(f"  Combined trades:   {len(combined_trades)}")

    # =========================================================================
    # 2. STANDALONE STATS
    # =========================================================================
    print(f"\n{'='*80}")
    print("  STANDALONE STRATEGY STATS (2013-present)")
    print(f"{'='*80}")

    print_trade_stats(chosen_trades, "CHOSEN ONE (Week 1/4 + VIX filter)")
    print_trade_stats(tom_trades, "TURN OF MONTH (TD-2 to TD+3)")
    print_trade_stats(combined_trades, "COMBINED (both strategies)")

    # =========================================================================
    # 3. OVERLAP & CORRELATION
    # =========================================================================
    analyze_overlap(chosen_trades, tom_trades)

    # =========================================================================
    # 4. COMBINED EQUITY CURVE
    # =========================================================================
    print(f"\n{'='*80}")
    print("  COMBINED EQUITY CURVE")
    print(f"{'='*80}")

    close = es_post["Close"].astype(float)

    # Build equity curves for all three
    def build_equity(trade_df, label):
        """Build daily equity from a trade DataFrame."""
        eq = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
        cash = INITIAL_CAPITAL

        for _, t in trade_df.iterrows():
            entry_d = t["entry_date"]
            exit_d = t["exit_date"]

            # Find index positions
            entry_idx = close.index.get_loc(entry_d) if entry_d in close.index else None
            exit_idx = close.index.get_loc(exit_d) if exit_d in close.index else None

            if entry_idx is None or exit_idx is None:
                cash += t["pnl"]
                continue

            # Mark-to-market during hold
            for k in range(entry_idx, exit_idx + 1):
                mtm = (close.iloc[k] - t["entry_price"]) * ES_POINT_VALUE
                eq.iloc[k] = cash + mtm

            cash += t["pnl"]

        # Fill post-last-trade
        last_exit = trade_df["exit_date"].max() if len(trade_df) > 0 else close.index[0]
        if last_exit in close.index:
            last_idx = close.index.get_loc(last_exit)
            for k in range(last_idx + 1, len(close)):
                eq.iloc[k] = cash

        return eq

    # For combined: since both can be open simultaneously, equity = sum of both positions
    # Simpler approach: just sum the P&L chronologically
    chosen_eq = build_equity(chosen_trades, "Chosen One")
    tom_eq = build_equity(tom_trades, "TOM")

    # Combined: start at INITIAL_CAPITAL, add all trade P&Ls in order
    combined_sorted = combined_trades.sort_values("exit_date").reset_index(drop=True)
    combined_cumulative = INITIAL_CAPITAL + combined_sorted["pnl"].cumsum()

    # Drawdown analysis
    for label, pnls_series in [
        ("Chosen One", pd.Series(chosen_trades["pnl"].values)),
        ("TOM", pd.Series(tom_trades["pnl"].values)),
        ("Combined", pd.Series(combined_trades.sort_values("exit_date")["pnl"].values)),
    ]:
        cumulative = pnls_series.cumsum()
        peak = cumulative.cummax()
        dd = cumulative - peak
        max_dd = dd.min()
        print(f"  {label:.<30} Max trade-level drawdown: ${max_dd:>10,.0f}")

    # Save combined equity plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)

        # Plot all three equity curves
        chosen_cum = INITIAL_CAPITAL + pd.Series(
            chosen_trades.sort_values("exit_date")["pnl"].values).cumsum()
        tom_cum = INITIAL_CAPITAL + pd.Series(
            tom_trades.sort_values("exit_date")["pnl"].values).cumsum()
        combined_cum = INITIAL_CAPITAL + pd.Series(
            combined_sorted["pnl"].values).cumsum()

        chosen_dates = chosen_trades.sort_values("exit_date")["exit_date"].values
        tom_dates = tom_trades.sort_values("exit_date")["exit_date"].values
        combined_dates = combined_sorted["exit_date"].values

        axes[0].plot(chosen_dates, chosen_cum.values, label="Chosen One", linewidth=1.5, color="#2196F3")
        axes[0].plot(tom_dates, tom_cum.values, label="TOM", linewidth=1.5, color="#FF9800")
        axes[0].plot(combined_dates, combined_cum.values, label="Combined", linewidth=2.0, color="#4CAF50")
        axes[0].axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("Chosen One vs TOM vs Combined — Equity Curves", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].ticklabel_format(style="plain", axis="y")

        # Combined drawdown
        peak = combined_cum.cummax()
        dd = (combined_cum - peak)
        axes[1].fill_between(combined_dates, dd.values, 0, color="#F44336", alpha=0.4, label="Combined DD")
        axes[1].set_ylabel("Drawdown ($)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plot_path = os.path.join(RESULTS_DIR, "tom_es_complement_equity.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Equity curve saved to {plot_path}")
    except Exception as e:
        print(f"\n  Could not save plot: {e}")

    # =========================================================================
    # 5. PHIDIAS SIMULATION — ALL THREE CONFIGS
    # =========================================================================
    print(f"\n{'='*80}")
    print("  PHIDIAS $50K SIMULATION")
    print(f"  Capital: ${PHIDIAS_CAPITAL:,} | Target: ${PHIDIAS_TARGET:,} | "
          f"EOD DD: ${PHIDIAS_MAX_DD:,} | OTP: ${PHIDIAS_OTP_COST}")
    print(f"{'='*80}")

    configs = [
        ("Chosen One alone", chosen_trades.sort_values("exit_date")["pnl"].tolist()),
        ("TOM alone", tom_trades.sort_values("exit_date")["pnl"].tolist()),
        ("Combined (both)", combined_sorted["pnl"].tolist()),
    ]

    summaries = []
    for label, pnl_list in configs:
        attempts = simulate_phidias(pnl_list, PHIDIAS_CAPITAL, PHIDIAS_TARGET, PHIDIAS_MAX_DD)
        s = phidias_summary(attempts, label)
        summaries.append(s)
        print_phidias_attempts(s)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print(f"\n{'='*80}")
    print("  PHIDIAS COMPARISON — Chosen One vs TOM vs Combined")
    print(f"{'='*80}")
    print(f"  {'Strategy':<25} {'Pass%':>8} {'P/F':>6} {'AvgTr':>7} "
          f"{'ExpAtt':>8} {'ExpMo':>8} {'ExpCost':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")

    for s in summaries:
        if s["pass_rate"] > 0:
            print(f"  {s['label']:<25} {s['pass_rate']:>7.1f}% "
                  f"{s['passed']:>3}/{s['total']:<3} {s['avg_trades_pass']:>7.1f} "
                  f"{s['exp_attempts']:>8.1f} {s['exp_months']:>8.1f} "
                  f"${s['exp_cost']:>9,.0f}")
        else:
            print(f"  {s['label']:<25} {s['pass_rate']:>7.1f}% "
                  f"{s['passed']:>3}/{s['total']:<3} {'N/A':>7} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>10}")

    print(f"{'='*80}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")

    chosen_s = summaries[0]
    tom_s = summaries[1]
    combined_s = summaries[2]

    chosen_pnls = pd.Series(chosen_trades["pnl"].values)
    tom_pnls = pd.Series(tom_trades["pnl"].values)
    combined_pnls = pd.Series(combined_sorted["pnl"].values)

    chosen_pf = profit_factor(chosen_pnls)
    tom_pf = profit_factor(tom_pnls)
    combined_pf = profit_factor(combined_pnls)

    print(f"\n  Profit Factors:  Chosen One {chosen_pf:.3f} | TOM {tom_pf:.3f} | Combined {combined_pf:.3f}")

    if tom_pf < 1.0:
        print(f"\n  TOM is NEGATIVE expectancy (PF {tom_pf:.3f}). Do NOT add it.")
        print(f"  Adding losing trades to a winning strategy makes it worse.")
    elif tom_pf < 1.2:
        print(f"\n  TOM is marginal (PF {tom_pf:.3f} < 1.2). Probably not worth adding.")
    else:
        print(f"\n  TOM is independently profitable (PF {tom_pf:.3f}).")

    if combined_s["pass_rate"] > chosen_s["pass_rate"]:
        delta = combined_s["pass_rate"] - chosen_s["pass_rate"]
        print(f"  Combined IMPROVES pass rate by +{delta:.1f}pp "
              f"({chosen_s['pass_rate']:.1f}% -> {combined_s['pass_rate']:.1f}%)")
        print(f"  -> Adding TOM trades HELPS reach funded faster.")
    elif combined_s["pass_rate"] == chosen_s["pass_rate"]:
        print(f"  Combined has SAME pass rate as Chosen One ({combined_s['pass_rate']:.1f}%).")
        if combined_s["avg_trades_pass"] < chosen_s["avg_trades_pass"]:
            print(f"  -> But reaches target in fewer trades "
                  f"({combined_s['avg_trades_pass']:.1f} vs {chosen_s['avg_trades_pass']:.1f}).")
        else:
            print(f"  -> No benefit to adding TOM.")
    else:
        delta = chosen_s["pass_rate"] - combined_s["pass_rate"]
        print(f"  Combined HURTS pass rate by -{delta:.1f}pp "
              f"({chosen_s['pass_rate']:.1f}% -> {combined_s['pass_rate']:.1f}%)")
        print(f"  -> Adding TOM trades INCREASES RISK without enough return.")
        print(f"  -> Stick with the Chosen One alone.")

    print(f"\n{'='*80}")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tom_csv = os.path.join(RESULTS_DIR, "tom_es_complement_trades.csv")
    tom_trades.to_csv(tom_csv, index=False)
    print(f"\nTOM trades saved to {tom_csv}")

    combined_csv = os.path.join(RESULTS_DIR, "tom_es_complement_combined_trades.csv")
    combined_sorted.to_csv(combined_csv, index=False)
    print(f"Combined trades saved to {combined_csv}")

    return {
        "chosen_pf": chosen_pf,
        "tom_pf": tom_pf,
        "combined_pf": combined_pf,
        "chosen_pass_rate": chosen_s["pass_rate"],
        "tom_pass_rate": tom_s["pass_rate"],
        "combined_pass_rate": combined_s["pass_rate"],
    }


if __name__ == "__main__":
    run()
