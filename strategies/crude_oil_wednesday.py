"""
Crude Oil Wednesday Strategy
=============================
Thesis: The weekly EIA Petroleum Status Report (released Wednesdays 10:30 AM ET)
creates a directional impulse in crude oil that persists through Friday.

Trade: Buy CL at Wednesday open, sell at Friday close.

Variants tested:
1. BASELINE: Every Wednesday-Friday
2. OVX/VIX FILTER: Skip when volatility is in a "dead zone"
3. DIRECTION FILTER: Use Monday-Tuesday direction for entry bias
4. SEASONAL MONTH FILTER: Only trade December through August

Contract specs:
- CL  (full):  $1,000/point, tick=0.01=$10
- QM  (e-mini): $500/point
- MCL (micro):  $100/point

Costs: $5 RT commission, 1 tick slippage per side = $20 total for CL,
       $2 for MCL ($10 slippage + $5 commission approximated to $5 total for micro).
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_data
from backtester import Trade, BacktestResult, phidias_simulation, correlation_by_week

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Contract specifications ────────────────────────────────────────────────
CONTRACTS = {
    "CL":  {"point_value": 1000, "commission": 5.0, "slippage": 20.0},
    "QM":  {"point_value": 500,  "commission": 5.0, "slippage": 10.0},
    "MCL": {"point_value": 100,  "commission": 5.0, "slippage": 2.0},
}


# ─── Data loading ───────────────────────────────────────────────────────────
def load_all_data(start: str = "2012-01-01") -> dict[str, pd.DataFrame]:
    """Load CL, OVX (with VIX fallback), and attempt MCL."""
    data = {}

    # CL futures
    data["CL"] = get_data("CL", start=start)
    if data["CL"].empty:
        raise RuntimeError("Failed to download CL data — cannot proceed.")

    # OVX (crude oil VIX) — fall back to VIX if unavailable
    ovx = get_data("OVX", start=start)
    if ovx.empty or len(ovx) < 100:
        print("OVX data unavailable or sparse — using VIX as proxy.")
        data["VOL"] = get_data("VIX", start=start)
        data["vol_source"] = "VIX"
    else:
        data["VOL"] = ovx
        data["vol_source"] = "OVX"

    # MCL micro crude — simulate from CL if unavailable
    mcl = get_data("MCL", start=start)
    if mcl.empty or len(mcl) < 100:
        print("MCL data unavailable — will simulate from CL (÷10 point value).")
        data["MCL_available"] = False
    else:
        data["MCL"] = mcl
        data["MCL_available"] = True

    return data


# ─── Week structure helpers ─────────────────────────────────────────────────
def build_weekly_trades(cl: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame of weekly trade opportunities.
    Each row = one calendar week with Mon open, Tue close, Wed open, Fri close.
    """
    cl = cl.copy()
    cl["weekday"] = cl.index.weekday  # Mon=0 ... Fri=4

    monday = cl[cl["weekday"] == 0][["Open"]].rename(columns={"Open": "mon_open"})
    tuesday = cl[cl["weekday"] == 1][["Close"]].rename(columns={"Close": "tue_close"})
    wednesday = cl[cl["weekday"] == 2][["Open"]].rename(columns={"Open": "wed_open"})
    friday = cl[cl["weekday"] == 4][["Close"]].rename(columns={"Close": "fri_close"})

    # Align by ISO week
    for df in [monday, tuesday, wednesday, friday]:
        df["year_week"] = df.index.isocalendar().year.astype(str) + "-" + \
                          df.index.isocalendar().week.astype(str).str.zfill(2)

    monday = monday.set_index("year_week")
    tuesday = tuesday.set_index("year_week")
    wednesday_df = wednesday.copy()
    wednesday_df["wed_date"] = wednesday_df.index
    wednesday_df = wednesday_df.set_index("year_week")
    friday_df = friday.copy()
    friday_df["fri_date"] = friday_df.index
    friday_df = friday_df.set_index("year_week")

    weeks = monday.join(tuesday, how="inner").join(wednesday_df, how="inner").join(friday_df, how="inner")
    weeks = weeks.dropna()

    # Monday-to-Tuesday move
    weeks["mon_tue_move"] = weeks["tue_close"] - weeks["mon_open"]
    weeks["mon_tue_pct"] = weeks["mon_tue_move"] / weeks["mon_open"]

    # Wednesday-to-Friday move (the trade)
    weeks["wed_fri_move"] = weeks["fri_close"] - weeks["wed_open"]

    # Month of Wednesday (for seasonal filter)
    weeks["month"] = weeks["wed_date"].dt.month

    return weeks


def add_volatility(weeks: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
    """Add Tuesday's closing volatility (OVX or VIX) to weekly DataFrame."""
    vol_df = vol_df.copy()
    vol_df["weekday"] = vol_df.index.weekday
    tue_vol = vol_df[vol_df["weekday"] == 1][["Close"]].rename(columns={"Close": "vol_close"})
    tue_vol["year_week"] = tue_vol.index.isocalendar().year.astype(str) + "-" + \
                           tue_vol.index.isocalendar().week.astype(str).str.zfill(2)
    tue_vol = tue_vol.set_index("year_week")
    return weeks.join(tue_vol, how="left")


# ─── Strategy variants ──────────────────────────────────────────────────────
def run_baseline(weeks: pd.DataFrame, contract: str = "CL", direction: int = 1) -> BacktestResult:
    """V1: Buy CL Wednesday open, sell Friday close, every week."""
    spec = CONTRACTS[contract]
    pv = spec["point_value"]
    # If simulating MCL from CL data, use MCL point value
    if contract == "MCL":
        effective_pv = CONTRACTS["MCL"]["point_value"]
    else:
        effective_pv = pv

    trades = []
    for _, row in weeks.iterrows():
        t = Trade(
            entry_date=pd.Timestamp(row["wed_date"]),
            exit_date=pd.Timestamp(row["fri_date"]),
            entry_price=row["wed_open"],
            exit_price=row["fri_close"],
            direction=direction,
            point_value=effective_pv,
            commission=spec["commission"],
            slippage=spec["slippage"] if contract != "MCL" else CONTRACTS["MCL"]["slippage"],
            label=f"baseline_{contract}",
        )
        trades.append(t)

    result = BacktestResult(trades=trades, strategy_name="CrudeOilWednesday",
                            variant_name=f"baseline_{contract}_{'long' if direction == 1 else 'short'}")
    return result


def run_vol_filter(weeks: pd.DataFrame, vol_source: str,
                   skip_low: float = None, skip_high: float = None,
                   min_vol: float = None, contract: str = "MCL") -> BacktestResult:
    """V2: Skip weeks when volatility is in dead zone or below threshold."""
    spec = CONTRACTS[contract]
    effective_pv = spec["point_value"]

    label_parts = []
    if skip_low is not None and skip_high is not None:
        label_parts.append(f"skip_{vol_source}_{skip_low}-{skip_high}")
    if min_vol is not None:
        label_parts.append(f"min_{vol_source}_{min_vol}")
    label = "_".join(label_parts) or "vol_filter"

    trades = []
    for _, row in weeks.iterrows():
        vol = row.get("vol_close")
        if pd.isna(vol):
            continue
        # Skip dead zone
        if skip_low is not None and skip_high is not None:
            if skip_low <= vol <= skip_high:
                continue
        # Minimum volatility threshold
        if min_vol is not None and vol < min_vol:
            continue

        t = Trade(
            entry_date=pd.Timestamp(row["wed_date"]),
            exit_date=pd.Timestamp(row["fri_date"]),
            entry_price=row["wed_open"],
            exit_price=row["fri_close"],
            direction=1,
            point_value=effective_pv,
            commission=spec["commission"],
            slippage=spec["slippage"],
            label=label,
        )
        trades.append(t)

    return BacktestResult(trades=trades, strategy_name="CrudeOilWednesday",
                          variant_name=f"vol_filter_{label}_{contract}")


def run_direction_filter(weeks: pd.DataFrame, mode: str = "momentum",
                         contract: str = "MCL") -> BacktestResult:
    """
    V3: Direction filter based on Monday-Tuesday move.
    mode='momentum': buy if Mon-Tue was up (continuation after report)
    mode='reversal': buy if Mon-Tue was down (mean reversion into report)
    mode='adaptive': buy if up, sell if down
    """
    spec = CONTRACTS[contract]
    effective_pv = spec["point_value"]

    trades = []
    for _, row in weeks.iterrows():
        move = row["mon_tue_move"]
        if pd.isna(move):
            continue

        if mode == "momentum":
            if move <= 0:
                continue
            direction = 1
        elif mode == "reversal":
            if move >= 0:
                continue
            direction = 1  # buy the dip
        elif mode == "adaptive":
            direction = 1 if move > 0 else -1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        t = Trade(
            entry_date=pd.Timestamp(row["wed_date"]),
            exit_date=pd.Timestamp(row["fri_date"]),
            entry_price=row["wed_open"],
            exit_price=row["fri_close"],
            direction=direction,
            point_value=effective_pv,
            commission=spec["commission"],
            slippage=spec["slippage"],
            label=f"direction_{mode}",
        )
        trades.append(t)

    return BacktestResult(trades=trades, strategy_name="CrudeOilWednesday",
                          variant_name=f"direction_{mode}_{contract}")


def run_seasonal_filter(weeks: pd.DataFrame, contract: str = "MCL",
                        active_months: tuple = (12, 1, 2, 3, 4, 5, 6, 7, 8)) -> BacktestResult:
    """V4: Only trade during strong seasonal months (Dec-Aug)."""
    spec = CONTRACTS[contract]
    effective_pv = spec["point_value"]

    trades = []
    for _, row in weeks.iterrows():
        if row["month"] not in active_months:
            continue
        t = Trade(
            entry_date=pd.Timestamp(row["wed_date"]),
            exit_date=pd.Timestamp(row["fri_date"]),
            entry_price=row["wed_open"],
            exit_price=row["fri_close"],
            direction=1,
            point_value=effective_pv,
            commission=spec["commission"],
            slippage=spec["slippage"],
            label="seasonal_dec_aug",
        )
        trades.append(t)

    return BacktestResult(trades=trades, strategy_name="CrudeOilWednesday",
                          variant_name=f"seasonal_dec_aug_{contract}")


# ─── Combined variant (best filters stacked) ────────────────────────────────
def run_combined(weeks: pd.DataFrame, vol_source: str, contract: str = "MCL",
                 min_vol: float = 30, active_months: tuple = (12, 1, 2, 3, 4, 5, 6, 7, 8),
                 direction_mode: str = None) -> BacktestResult:
    """Stack multiple filters: seasonal + vol + optional direction."""
    spec = CONTRACTS[contract]
    effective_pv = spec["point_value"]

    trades = []
    for _, row in weeks.iterrows():
        # Seasonal filter
        if row["month"] not in active_months:
            continue
        # Vol filter
        vol = row.get("vol_close")
        if pd.isna(vol) or vol < min_vol:
            continue
        # Direction filter
        direction = 1
        if direction_mode == "momentum":
            if row["mon_tue_move"] <= 0:
                continue
        elif direction_mode == "reversal":
            if row["mon_tue_move"] >= 0:
                continue
        elif direction_mode == "adaptive":
            direction = 1 if row["mon_tue_move"] > 0 else -1

        t = Trade(
            entry_date=pd.Timestamp(row["wed_date"]),
            exit_date=pd.Timestamp(row["fri_date"]),
            entry_price=row["wed_open"],
            exit_price=row["fri_close"],
            direction=direction,
            point_value=effective_pv,
            commission=spec["commission"],
            slippage=spec["slippage"],
            label="combined",
        )
        trades.append(t)

    label = f"combined_vol>{min_vol}_seasonal"
    if direction_mode:
        label += f"_{direction_mode}"
    return BacktestResult(trades=trades, strategy_name="CrudeOilWednesday",
                          variant_name=f"{label}_{contract}")


# ─── Reporting ──────────────────────────────────────────────────────────────
def print_stats(result: BacktestResult):
    """Pretty-print backtest statistics."""
    s = result.stats()
    print(f"\n{'=' * 70}")
    print(f"  {s.get('strategy', '')} — {s.get('variant', '')}")
    print(f"{'=' * 70}")
    if s["trades"] == 0:
        print("  No trades.")
        return s

    print(f"  Trades: {s['trades']}")
    print(f"  Win Rate: {s['win_rate']:.1%}")
    print(f"  Profit Factor: {s['profit_factor']:.2f}")
    print(f"  Total P&L: ${s['total_pnl']:,.2f}")
    print(f"  Avg P&L/trade: ${s['avg_pnl']:,.2f}")
    print(f"  Max Drawdown: ${s['max_drawdown']:,.2f}")
    print(f"  Avg Win: ${s['avg_win']:,.2f}  |  Avg Loss: ${s['avg_loss']:,.2f}")
    print(f"  Best Trade: ${s['best_trade']:,.2f}  |  Worst: ${s['worst_trade']:,.2f}")
    print(f"\n  Year-by-Year:")
    print(f"  {'Year':<6} {'Trades':>6} {'Total P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-' * 36}")
    for year, ys in sorted(s["yearly"].items()):
        print(f"  {year:<6} {ys['trades']:>6.0f} ${ys['total_pnl']:>11,.2f} ${ys['avg_pnl']:>9,.2f}")
    print()
    return s


def print_phidias(sim: dict, label: str = ""):
    """Pretty-print Phidias simulation results."""
    print(f"\n  Phidias Sim{' — ' + label if label else ''}")
    print(f"  Account: {sim['account_type']} | Balance: ${sim['starting_balance']:,} | "
          f"Target: ${sim['profit_target']:,} | Max DD: ${sim['max_drawdown']:,}")
    print(f"  Pass Rate: {sim['pass_rate']:.1%} ({sim['passes']}/{sim['n_simulations']})")
    if sim.get("avg_trades_to_pass"):
        print(f"  Avg Trades to Pass: {sim['avg_trades_to_pass']:.0f}")
    if sim.get("avg_trades_to_fail"):
        print(f"  Avg Trades to Fail: {sim['avg_trades_to_fail']:.0f}")
    print()


def print_correlation(corr: dict, label: str = ""):
    """Pretty-print correlation results."""
    print(f"\n  Correlation with Chosen One{' — ' + label if label else ''}")
    if corr["correlation"] is None:
        print(f"  {corr.get('note', 'N/A')} (common weeks: {corr['common_weeks']})")
    else:
        print(f"  Weekly P&L Correlation: {corr['correlation']:.4f}")
        print(f"  Common Weeks: {corr['common_weeks']}")
    print()


# ─── Main execution ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CRUDE OIL WEDNESDAY STRATEGY BACKTEST")
    print("  Buy CL at Wednesday open → Sell at Friday close")
    print("=" * 70)

    # Load data
    data = load_all_data(start="2012-01-01")
    cl = data["CL"]
    vol_df = data["VOL"]
    vol_source = data["vol_source"]
    print(f"\nCL data: {cl.index[0].date()} to {cl.index[-1].date()} ({len(cl)} bars)")
    print(f"Volatility source: {vol_source} ({len(vol_df)} bars)")

    # Build weekly structure
    weeks = build_weekly_trades(cl)
    weeks = add_volatility(weeks, vol_df)
    print(f"Complete trading weeks: {len(weeks)}")

    # Placeholder for Chosen One trades (empty if no file found)
    chosen_one_trades = []

    all_stats = []

    # ════════════════════════════════════════════════════════════════════════
    # VARIANT 1: BASELINE
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 70)
    print("  VARIANT 1: BASELINE — Buy every Wednesday, sell every Friday")
    print("━" * 70)

    for contract in ["CL", "QM", "MCL"]:
        result = run_baseline(weeks, contract=contract, direction=1)
        s = print_stats(result)
        all_stats.append(s)

        if contract == "MCL":
            # Phidias Swing sim
            sim_swing = phidias_simulation(
                result.trades, starting_balance=50000,
                daily_profit_target=4000, eod_max_drawdown=2500,
                can_hold_overnight=True
            )
            print_phidias(sim_swing, "MCL Baseline — Swing ($50K)")

            # Phidias Fundamental sim
            sim_fund = phidias_simulation(
                result.trades, starting_balance=50000,
                daily_profit_target=4000, eod_max_drawdown=2500,
                can_hold_overnight=False
            )
            print_phidias(sim_fund, "MCL Baseline — Fundamental ($50K)")

            # Correlation with Chosen One
            corr = correlation_by_week(result.trades, chosen_one_trades)
            print_correlation(corr, "MCL Baseline")

    # Also test short side for baseline
    result_short = run_baseline(weeks, contract="MCL", direction=-1)
    s = print_stats(result_short)
    all_stats.append(s)

    # ════════════════════════════════════════════════════════════════════════
    # VARIANT 2: OVX/VIX FILTER
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 70)
    print(f"  VARIANT 2: VOLATILITY FILTER (using {vol_source})")
    print("━" * 70)

    # Define filter ranges based on vol source
    if vol_source == "OVX":
        filter_configs = [
            {"skip_low": 20, "skip_high": 30, "label": "skip OVX 20-30"},
            {"skip_low": 25, "skip_high": 35, "label": "skip OVX 25-35"},
            {"min_vol": 30, "label": "only OVX > 30"},
            {"min_vol": 35, "label": "only OVX > 35"},
        ]
    else:  # VIX proxy
        filter_configs = [
            {"skip_low": 12, "skip_high": 18, "label": "skip VIX 12-18"},
            {"skip_low": 15, "skip_high": 22, "label": "skip VIX 15-22"},
            {"min_vol": 18, "label": "only VIX > 18"},
            {"min_vol": 22, "label": "only VIX > 22"},
        ]

    for cfg in filter_configs:
        result = run_vol_filter(
            weeks, vol_source=vol_source,
            skip_low=cfg.get("skip_low"), skip_high=cfg.get("skip_high"),
            min_vol=cfg.get("min_vol"), contract="MCL"
        )
        s = print_stats(result)
        all_stats.append(s)

        if result.trades:
            sim = phidias_simulation(
                result.trades, starting_balance=50000,
                daily_profit_target=4000, eod_max_drawdown=2500,
                can_hold_overnight=True
            )
            print_phidias(sim, f"MCL {cfg['label']} — Swing")

            corr = correlation_by_week(result.trades, chosen_one_trades)
            print_correlation(corr, cfg["label"])

    # ════════════════════════════════════════════════════════════════════════
    # VARIANT 3: DIRECTION FILTER
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 70)
    print("  VARIANT 3: DIRECTION FILTER (Monday-Tuesday move)")
    print("━" * 70)

    for mode in ["momentum", "reversal", "adaptive"]:
        result = run_direction_filter(weeks, mode=mode, contract="MCL")
        s = print_stats(result)
        all_stats.append(s)

        if result.trades:
            sim = phidias_simulation(
                result.trades, starting_balance=50000,
                daily_profit_target=4000, eod_max_drawdown=2500,
                can_hold_overnight=True
            )
            print_phidias(sim, f"MCL Direction {mode} — Swing")

            corr = correlation_by_week(result.trades, chosen_one_trades)
            print_correlation(corr, f"Direction {mode}")

    # ════════════════════════════════════════════════════════════════════════
    # VARIANT 4: SEASONAL MONTH FILTER
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 70)
    print("  VARIANT 4: SEASONAL FILTER (Dec-Aug only)")
    print("━" * 70)

    result = run_seasonal_filter(weeks, contract="MCL")
    s = print_stats(result)
    all_stats.append(s)

    if result.trades:
        sim_swing = phidias_simulation(
            result.trades, starting_balance=50000,
            daily_profit_target=4000, eod_max_drawdown=2500,
            can_hold_overnight=True
        )
        print_phidias(sim_swing, "MCL Seasonal — Swing")

        sim_fund = phidias_simulation(
            result.trades, starting_balance=50000,
            daily_profit_target=4000, eod_max_drawdown=2500,
            can_hold_overnight=False
        )
        print_phidias(sim_fund, "MCL Seasonal — Fundamental")

        corr = correlation_by_week(result.trades, chosen_one_trades)
        print_correlation(corr, "Seasonal")

    # ════════════════════════════════════════════════════════════════════════
    # COMBINED BEST FILTERS
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "━" * 70)
    print("  COMBINED FILTERS")
    print("━" * 70)

    min_vol_threshold = 30 if vol_source == "OVX" else 18
    for dir_mode in [None, "momentum", "reversal"]:
        result = run_combined(weeks, vol_source=vol_source, contract="MCL",
                              min_vol=min_vol_threshold, direction_mode=dir_mode)
        s = print_stats(result)
        all_stats.append(s)

        if result.trades:
            sim = phidias_simulation(
                result.trades, starting_balance=50000,
                daily_profit_target=4000, eod_max_drawdown=2500,
                can_hold_overnight=True
            )
            print_phidias(sim, f"Combined {'+ ' + dir_mode if dir_mode else '(no dir)'} — Swing")

            corr = correlation_by_week(result.trades, chosen_one_trades)
            print_correlation(corr, f"Combined {'+ ' + dir_mode if dir_mode else ''}")

    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Variant':<45} {'Trades':>6} {'WR':>6} {'PF':>7} {'Tot P&L':>12} {'Avg P&L':>10}")
    print(f"  {'-' * 88}")
    for s in all_stats:
        if s.get("trades", 0) == 0:
            continue
        print(f"  {s.get('variant', '?'):<45} {s['trades']:>6} "
              f"{s['win_rate']:>5.1%} {s['profit_factor']:>7.2f} "
              f"${s['total_pnl']:>11,.2f} ${s['avg_pnl']:>9,.2f}")

    # ════════════════════════════════════════════════════════════════════════
    # KEY QUESTION ANSWER
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  KEY QUESTION: PF > 1.5, 100+ trades, near-zero correlation?")
    print("=" * 70)
    candidates = [s for s in all_stats
                  if s.get("trades", 0) >= 100 and s.get("profit_factor", 0) > 1.5]
    if candidates:
        print(f"  Found {len(candidates)} variant(s) meeting PF > 1.5 with 100+ trades:")
        for c in candidates:
            print(f"    - {c['variant']}: PF={c['profit_factor']:.2f}, "
                  f"Trades={c['trades']}, Total=${c['total_pnl']:,.2f}")
    else:
        print("  No variants meet all criteria (PF > 1.5 + 100 trades).")
        # Show best candidates
        viable = [s for s in all_stats if s.get("trades", 0) >= 50]
        if viable:
            viable.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)
            print(f"\n  Closest candidates (50+ trades, sorted by PF):")
            for c in viable[:5]:
                print(f"    - {c['variant']}: PF={c['profit_factor']:.2f}, "
                      f"Trades={c['trades']}, WR={c['win_rate']:.1%}, "
                      f"Total=${c['total_pnl']:,.2f}")

    print("\n  Note: Correlation with Chosen One cannot be computed without")
    print("  Chosen One trade data. Add chosen_one_trades list to enable.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
