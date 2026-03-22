"""Futures Overnight + Turn-of-Month + VIX Overlay strategy.

Combined strategy on ES futures:
1. Base: buy ES at close, sell at next open (overnight hold) every trading day.
2. TOM overlay: 2 contracts during trading days -2 to +3 around month end.
3. VIX overlay: +1 contract if VIX closed above 25 yesterday.

Costs: $5 round trip commission per contract, $12.50 slippage per side per contract.
ES point value = $50, starting capital $100K.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity

STRATEGY_NAME = "ES Futures Overnight + TOM + VIX"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0       # $5 round trip per contract
SLIPPAGE_PER_SIDE = 12.50  # 1 tick = 0.25 pts * $50/pt = $12.50 per side per contract
INITIAL_CAPITAL = 100_000.0
VIX_THRESHOLD = 25.0


def build_tom_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Return boolean Series: True on trading days -2 to +3 around month end."""
    mask = pd.Series(False, index=dates)
    ym = dates.to_period("M")
    for m in ym.unique():
        month_days = dates[ym == m]
        n = len(month_days)
        if n < 3:
            continue
        # Last 2 trading days of month
        for j in range(max(0, n - 2), n):
            mask.loc[month_days[j]] = True
        # First 3 trading days of month
        for j in range(min(3, n)):
            mask.loc[month_days[j]] = True
    return mask


def compute_contracts(dates: pd.DatetimeIndex, vix_close: pd.Series,
                      mode: str = "full") -> pd.Series:
    """Compute number of contracts to hold overnight for each date.

    mode:
        'base'     — 1 contract every day
        'base+tom' — 1 base + 1 extra during TOM window
        'full'     — 1 base + 1 TOM overlay + 1 VIX overlay
    """
    contracts = pd.Series(1, index=dates, dtype=int)  # base: always 1

    if mode in ("base+tom", "full"):
        tom_mask = build_tom_mask(dates)
        contracts += tom_mask.astype(int)

    if mode == "full":
        # VIX overlay: if VIX closed above threshold YESTERDAY, add 1 contract
        vix_signal = (vix_close > VIX_THRESHOLD).shift(1).fillna(False)
        vix_signal = vix_signal.reindex(dates).fillna(False)
        contracts += vix_signal.astype(int)

    return contracts


def simulate_overnight(es: pd.DataFrame, contracts: pd.Series) -> dict:
    """Simulate overnight holds: buy at close, sell at next open.

    Each row i: buy contracts[i] at close[i], sell at open[i+1].
    Costs applied per contract: commission ($5 RT) + slippage ($12.50 * 2 sides).

    Returns dict with equity_curve, trade_pnls, trade_list, position.
    """
    close = es["Close"].astype(float)
    open_ = es["Open"].astype(float)

    contracts = contracts.reindex(close.index).fillna(0).astype(int)

    trades = []
    daily_pnl = pd.Series(0.0, index=close.index)
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL

    for i in range(len(close) - 1):
        n_contracts = contracts.iloc[i]
        if n_contracts <= 0:
            equity.iloc[i] = cash
            continue

        entry = close.iloc[i]
        exit_ = open_.iloc[i + 1]
        entry_date = close.index[i]
        exit_date = close.index[i + 1]

        # P&L per contract in points * $50/point
        points_pnl = (exit_ - entry) * ES_POINT_VALUE * n_contracts
        # Costs: commission + slippage on both entry and exit
        cost = n_contracts * (COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE)
        net_pnl = points_pnl - cost

        cash += net_pnl
        daily_pnl.iloc[i] = net_pnl
        equity.iloc[i] = cash

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry,
            "exit_price": exit_,
            "contracts": n_contracts,
            "gross_pnl": points_pnl,
            "costs": cost,
            "pnl": net_pnl,
        })

    # Last day: no overnight trade to close yet
    equity.iloc[-1] = cash

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)
    position = (contracts > 0).astype(float)

    return {
        "equity_curve": equity,
        "trade_pnls": trade_pnls,
        "trade_list": trade_list,
        "position": position,
    }


def run():
    """Run the combined overnight strategy with breakdowns."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    # Align on common dates
    common_idx = es.index.intersection(vix.index)
    es = es.loc[common_idx]
    vix_close = vix["Close"].reindex(common_idx).ffill()
    print(f"Common dates: {len(common_idx)} trading days\n")

    # --- Run three variants ---
    modes = [
        ("base", "Base Overnight Only"),
        ("base+tom", "Overnight + TOM Boost"),
        ("full", "Overnight + TOM + VIX (Full)"),
    ]

    results = {}
    for mode, label in modes:
        contracts = compute_contracts(common_idx, vix_close, mode=mode)
        result = simulate_overnight(es, contracts)
        stats = summary(
            trade_pnls=result["trade_pnls"],
            equity_curve=result["equity_curve"],
            position_series=result["position"],
        )
        results[mode] = (stats, result, label)
        print_summary(stats, label)

    # --- Comparison table ---
    print(f"{'='*75}")
    print(f"  Strategy Comparison")
    print(f"{'='*75}")
    print(f"  {'Variant':<30} {'PF':>7} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for mode, label in modes:
        s = results[mode][0]
        print(f"  {label:<30} {s['profit_factor']:>7.3f} {s['sharpe_ratio']:>8.3f} "
              f"{s['annualized_return']:>7.1%} {s['max_drawdown']:>7.1%} {s['total_trades']:>8}")
    print(f"{'='*75}\n")

    # --- Plot all three equity curves ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1], sharex=True)

    colors = {"base": "#9E9E9E", "base+tom": "#2196F3", "full": "#4CAF50"}
    for mode, label in modes:
        eq = results[mode][1]["equity_curve"]
        ax1.plot(eq.index, eq.values, label=label, linewidth=1.2, color=colors[mode])

    ax1.set_title(f"{STRATEGY_NAME} — Equity Curves", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style="plain", axis="y")

    # Drawdown of full strategy
    full_eq = results["full"][1]["equity_curve"]
    peak = full_eq.cummax()
    dd = (full_eq - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "futures_overnight_tom_equity.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {png_path}")

    # Save full trade list
    full_trades = results["full"][1]["trade_list"]
    if len(full_trades) > 0:
        csv_path = os.path.join(RESULTS_DIR, "futures_overnight_tom_trades.csv")
        full_trades.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return results


if __name__ == "__main__":
    run()
