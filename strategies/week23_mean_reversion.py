"""Week 2 & Week 3 ES Mean Reversion — Standalone Backtest.
THESIS: Equity indices are mean-reverting at weekly frequency. After a down week,
the following week tends to recover. This effect is documented in Moskowitz, Ooi,
Pedersen (2012) and is driven by institutional rebalancing flows.
This strategy ONLY trades during weeks 2 and 3 of each month — the weeks when
our core Week 1 & Week 4 calendar strategy is flat. This is designed as a
complementary filler, but we backtest it standalone first to validate the edge.
RULES:
1. Every Friday, check: is NEXT week a Week 2 or Week 3 of the month?
2. If yes: was THIS week a down week? (Friday close < prior Friday close)
3. Volume filter: 20-day volume MA must be rising
4. If ALL conditions met: buy 1 ES at THIS Friday's close
5. Exit: next Friday's close (hold exactly 1 week)
6. If conditions NOT met: stay flat
We also test variants:
  A) Base: down week + volume filter
  B) No volume filter: down week only
  C) Any week (not just 2/3): down week + volume filter on ALL weeks
  D) VIX filter added: same as A but skip when VIX 15-20 (match calendar strategy)
Costs: $5 RT commission + $12.50 slippage per side = $30 per trade (1 ES)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity
STRATEGY_NAME = "Week 2 & Week 3 Mean Reversion"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0
def get_week_of_month(date):
    """Week 1=days 1-7, Week 2=8-14, Week 3=15-21, Week 4=22-28, Week 5=29-31."""
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
def build_weekly_data(es_data, vix_data=None):
    """Build a weekly DataFrame with Friday closes, volume, and week-of-month info.
    Returns DataFrame indexed by Friday dates with columns:
      close, prior_close, down_week, vol_ma_rising, week_of_month_next,
      next_friday_close, vix_close
    """
    es_close = es_data["Close"].astype(float)
    es_volume = es_data["Volume"].astype(float)
    # Build daily df
    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "high": es_data["High"].astype(float),
        "low": es_data["Low"].astype(float),
        "volume": es_volume,
        "dow": es_close.index.dayofweek,
    }, index=es_close.index)
    if vix_data is not None:
        vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")
        df["vix"] = vix_close
    # 20-day volume MA
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_ma_rising"] = df["vol_ma_20"] > df["vol_ma_20"].shift(1)
    # Get Fridays only
    fridays = df[df["dow"] == 4].copy()
    # Prior Friday close
    fridays["prior_close"] = fridays["close"].shift(1)
    # Down week: this Friday close < prior Friday close
    fridays["down_week"] = fridays["close"] < fridays["prior_close"]
    # Next Friday close (for exit price)
    fridays["next_friday_close"] = fridays["close"].shift(-1)
    # What week of month is NEXT week?
    # Next Monday = this Friday + 3 days (or next trading day)
    fridays["next_monday"] = fridays.index + pd.Timedelta(days=3)
    fridays["week_of_month_next"] = [get_week_of_month(d) for d in fridays["next_monday"]]
    # VIX on this Friday (for filter)
    if "vix" in df.columns:
        fridays["vix_close"] = df.loc[fridays.index, "vix"]
    # Drop rows where we can't compute signals
    fridays = fridays.dropna(subset=["prior_close", "next_friday_close"])
    return fridays
def run_variant(fridays, variant_name, week_filter=None, require_down_week=True,
                require_vol_filter=True, vix_skip=False):
    """Run a single variant of the mean reversion strategy.
    Args:
        fridays: Weekly DataFrame from build_weekly_data()
        variant_name: Label for this variant
        week_filter: List of week numbers to trade (e.g., [2, 3]). None = all weeks.
        require_down_week: If True, only enter after a down week
        require_vol_filter: If True, require 20-day vol MA to be rising
        vix_skip: If True, skip when VIX is between 15.0 and 20.0
    """
    cost_per_trade = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
    # Build signal
    signal = pd.Series(True, index=fridays.index)
    if require_down_week:
        signal = signal & fridays["down_week"]
    if require_vol_filter:
        signal = signal & fridays["vol_ma_rising"]
    if week_filter is not None:
        signal = signal & fridays["week_of_month_next"].isin(week_filter)
    if vix_skip and "vix_close" in fridays.columns:
        vix_ok = ~((fridays["vix_close"] >= 15.0) & (fridays["vix_close"] <= 20.0))
        signal = signal & vix_ok
    # Generate trades
    trades = []
    for friday_date in fridays.index[signal]:
        row = fridays.loc[friday_date]
        entry_price = row["close"]  # Buy at this Friday's close
        exit_price = row["next_friday_close"]  # Sell at next Friday's close
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - cost_per_trade
        trade_dict = {
            "entry_date": friday_date,
            "exit_date": friday_date + pd.Timedelta(days=7),  # Approximate
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost_per_trade,
            "pnl": net_pnl,
            "week_of_month_next": row["week_of_month_next"],
            "down_week_pct": (row["close"] - row["prior_close"]) / row["prior_close"] * 100,
        }
        if "vix_close" in fridays.columns:
            trade_dict["vix_close"] = row["vix_close"]
        trades.append(trade_dict)
    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        print(f"\n  {variant_name}: NO TRADES GENERATED")
        return None, None
    trade_pnls = pd.Series(trade_df["pnl"].values, dtype=float)
    # Build equity curve
    equity_values = [INITIAL_CAPITAL]
    for pnl in trade_pnls:
        equity_values.append(equity_values[-1] + pnl)
    dates = [trade_df["entry_date"].iloc[0] - pd.Timedelta(days=7)]
    dates.extend(trade_df["entry_date"].tolist())
    equity = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
    position = pd.Series([0] + [1] * len(trade_pnls), index=pd.DatetimeIndex(dates))
    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )
    return stats, trade_df
def run():
    """Run all mean reversion variants."""
    print("=" * 80)
    print("WEEK 2 & WEEK 3 MEAN REVERSION — STANDALONE BACKTEST")
    print("ES Futures, Full History")
    print("=" * 80)
    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    fridays = build_weekly_data(es, vix)
    print(f"Total Fridays in dataset: {len(fridays)}")
    print(f"Down weeks: {fridays['down_week'].sum()} ({fridays['down_week'].mean():.1%})")
    # Define all variants
    variants = [
        {
            "name": "A) Week 2/3 + Down Week + Volume Filter",
            "week_filter": [2, 3],
            "require_down_week": True,
            "require_vol_filter": True,
            "vix_skip": False,
        },
        {
            "name": "B) Week 2/3 + Down Week Only (no vol filter)",
            "week_filter": [2, 3],
            "require_down_week": True,
            "require_vol_filter": False,
            "vix_skip": False,
        },
        {
            "name": "C) ALL Weeks + Down Week + Volume Filter",
            "week_filter": None,
            "require_down_week": True,
            "require_vol_filter": True,
            "vix_skip": False,
        },
        {
            "name": "D) Week 2/3 + Down Week + Vol Filter + VIX Skip 15-20",
            "week_filter": [2, 3],
            "require_down_week": True,
            "require_vol_filter": True,
            "vix_skip": True,
        },
        {
            "name": "E) Week 2/3 + Down Week + VIX Skip (no vol filter)",
            "week_filter": [2, 3],
            "require_down_week": True,
            "require_vol_filter": False,
            "vix_skip": True,
        },
        {
            "name": "F) ALL Weeks + Down Week Only (baseline)",
            "week_filter": None,
            "require_down_week": True,
            "require_vol_filter": False,
            "vix_skip": False,
        },
    ]
    all_results = {}
    for v in variants:
        print(f"\n{'='*60}")
        print(f"  {v['name']}")
        print(f"{'='*60}")
        stats, trade_df = run_variant(
            fridays,
            variant_name=v["name"],
            week_filter=v["week_filter"],
            require_down_week=v["require_down_week"],
            require_vol_filter=v["require_vol_filter"],
            vix_skip=v["vix_skip"],
        )
        if stats:
            print_summary(stats, v["name"])
            all_results[v["name"]] = {
                "stats": stats,
                "trades": trade_df,
            }
    # =============================================
    # COMPARISON TABLE
    # =============================================
    print("\n" + "=" * 100)
    print("VARIANT COMPARISON TABLE")
    print("=" * 100)
    if not all_results:
        print("  No variants produced results.")
        return
    col_w = 12
    print(f"\n  {'Variant':<55}", end="")
    print(f" {'PF':>{col_w}} {'WinRate':>{col_w}} {'Trades':>{col_w}} {'MaxDD':>{col_w}} "
          f"{'Avg$/tr':>{col_w}} {'Annual$':>{col_w}} {'Sharpe':>{col_w}}")
    print(f"  {'-'*55}", end="")
    print(f" {'-'*col_w} {'-'*col_w} {'-'*col_w} {'-'*col_w} "
          f"{'-'*col_w} {'-'*col_w} {'-'*col_w}")
    for name, r in all_results.items():
        s = r["stats"]
        t = r["trades"]
        avg_pnl = t["pnl"].mean() if len(t) > 0 else 0
        total_pnl = t["pnl"].sum() if len(t) > 0 else 0
        years = max(1, (t["entry_date"].max() - t["entry_date"].min()).days / 365.25) if len(t) > 0 else 1
        annual_pnl = total_pnl / years
        # Shorten name for display
        short_name = name[:55]
        print(f"  {short_name:<55}", end="")
        print(f" {s['profit_factor']:>{col_w}.3f}"
              f" {s['win_rate']:>{col_w}.1%}"
              f" {int(s['total_trades']):>{col_w}}"
              f" {s['max_drawdown']:>{col_w}.1%}"
              f" ${avg_pnl:>{col_w-1},.0f}"
              f" ${annual_pnl:>{col_w-1},.0f}"
              f" {s['sharpe_ratio']:>{col_w}.3f}")
    # =============================================
    # WEEK-BY-WEEK BREAKDOWN (for best variant)
    # =============================================
    # Find best variant by PF (among week 2/3 only variants)
    week23_variants = {k: v for k, v in all_results.items() if "Week 2/3" in k}
    if week23_variants:
        best_name = max(week23_variants.keys(), key=lambda k: week23_variants[k]["stats"]["profit_factor"])
        best = week23_variants[best_name]
        best_trades = best["trades"]
        print(f"\n{'='*60}")
        print(f"BREAKDOWN: {best_name}")
        print(f"{'='*60}")
        # By week of month
        print(f"\n  By target week:")
        for wk in sorted(best_trades["week_of_month_next"].unique()):
            subset = best_trades[best_trades["week_of_month_next"] == wk]
            wins = (subset["pnl"] > 0).sum()
            total = len(subset)
            wr = wins / total if total > 0 else 0
            avg = subset["pnl"].mean()
            print(f"    Week {wk}: {total} trades, {wr:.1%} win rate, ${avg:,.0f} avg")
        # By decade
        print(f"\n  By period:")
        best_trades_copy = best_trades.copy()
        best_trades_copy["year"] = best_trades_copy["entry_date"].dt.year
        periods = [
            ("2000-2005", 2000, 2005),
            ("2006-2010", 2006, 2010),
            ("2011-2015", 2011, 2015),
            ("2016-2020", 2016, 2020),
            ("2021-2026", 2021, 2026),
        ]
        for label, y1, y2 in periods:
            subset = best_trades_copy[(best_trades_copy["year"] >= y1) & (best_trades_copy["year"] <= y2)]
            if len(subset) == 0:
                continue
            wins = (subset["pnl"] > 0).sum()
            total = len(subset)
            wr = wins / total if total > 0 else 0
            avg = subset["pnl"].mean()
            total_pnl = subset["pnl"].sum()
            gross_wins = subset.loc[subset["pnl"] > 0, "pnl"].sum()
            gross_losses = abs(subset.loc[subset["pnl"] < 0, "pnl"].sum())
            pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
            print(f"    {label}: {total} trades, PF {pf:.2f}, {wr:.1%} WR, "
                  f"${avg:,.0f} avg, ${total_pnl:,.0f} total")
        # By VIX regime
        if "vix_close" in best_trades.columns:
            print(f"\n  By VIX regime:")
            vix_bins = [
                ("VIX < 15", 0, 15),
                ("VIX 15-20", 15, 20),
                ("VIX 20-25", 20, 25),
                ("VIX 25-30", 25, 30),
                ("VIX > 30", 30, 100),
            ]
            for label, lo, hi in vix_bins:
                subset = best_trades[(best_trades["vix_close"] >= lo) & (best_trades["vix_close"] < hi)]
                if len(subset) == 0:
                    continue
                wins = (subset["pnl"] > 0).sum()
                total = len(subset)
                wr = wins / total if total > 0 else 0
                avg = subset["pnl"].mean()
                print(f"    {label}: {total} trades, {wr:.1%} WR, ${avg:,.0f} avg")
    # =============================================
    # VERDICT
    # =============================================
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    if week23_variants:
        best_stats = week23_variants[best_name]["stats"]
        best_trades_final = week23_variants[best_name]["trades"]
        pf = best_stats["profit_factor"]
        trades_count = best_stats["total_trades"]
        wr = best_stats["win_rate"]
        print(f"\n  Best Week 2/3 variant: {best_name}")
        print(f"  PF: {pf:.3f} | Win Rate: {wr:.1%} | Trades: {trades_count}")
        print(f"  Avg trade: ${best_trades_final['pnl'].mean():,.0f}")
        if pf > 1.2 and trades_count >= 100:
            print(f"\n  ✅ TRADEABLE — Edge confirmed over {trades_count} trades")
            print(f"     Proceed to combined simulation with Week 1/4 calendar strategy")
        elif pf > 1.05 and trades_count >= 50:
            print(f"\n  ⚠️  MARGINAL EDGE — PF {pf:.3f} is weak but positive over {trades_count} trades")
            print(f"     The mean reversion adds trade count for consistency rule compliance")
            print(f"     but may not add much P&L. Test combined before committing.")
        else:
            print(f"\n  ❌ NO EDGE — PF {pf:.3f} over {trades_count} trades")
            print(f"     Do NOT add this to the portfolio. Find a different filler strategy")
            print(f"     or accept the consistency rule risk on the calendar strategy alone.")
    else:
        print("\n  ERROR: No Week 2/3 variants produced results.")
    print(f"\n{'='*80}")
    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if week23_variants:
        # Save best variant trades
        best_trades_save = week23_variants[best_name]["trades"]
        csv_path = os.path.join(RESULTS_DIR, "week23_mean_reversion_trades.csv")
        best_trades_save.to_csv(csv_path, index=False)
        print(f"Best variant trades saved to {csv_path}")
        # Save equity curve
        pnls = best_trades_save["pnl"].values
        eq_vals = [INITIAL_CAPITAL]
        for p in pnls:
            eq_vals.append(eq_vals[-1] + p)
        dates = [best_trades_save["entry_date"].iloc[0] - pd.Timedelta(days=7)]
        dates.extend(best_trades_save["entry_date"].tolist())
        eq = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
        plot_equity(eq, f"Week23 Mean Reversion Best Variant")
    # Save all variant trades
    for name, r in all_results.items():
        safe = name.split(")")[0].strip().lower() + "_trades.csv"
        safe = safe.replace(" ", "_").replace("/", "_").replace("+", "_")
        csv_path = os.path.join(RESULTS_DIR, f"week23_mr_{safe}")
        r["trades"].to_csv(csv_path, index=False)
if __name__ == "__main__":
    run()
