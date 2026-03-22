"""Gold/Silver Ratio strategy.

Use GLD/SLV ratio. Z-score with 50-day rolling lookback.
Same entry/exit logic as NQ/ES pairs. Dollar-neutral exposure.

When Z > 2.0: GLD expensive vs SLV -> short GLD, long SLV.
When Z < -2.0: GLD cheap vs SLV -> long GLD, short SLV.
Exit at Z returning to +/- 0.5.

Default costs: 0.01% commission per side (ETF), 0.01% slippage per side.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.backtester import run_single
from core.metrics import summary, print_summary
from core.plotting import plot_equity

STRATEGY_NAME = "Gold/Silver Ratio"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

LOOKBACK = 50
ENTRY_Z = 2.0
EXIT_Z = 0.5


def generate_signals(gld: pd.DataFrame, slv: pd.DataFrame) -> tuple:
    """Generate pairs trading signals based on GLD/SLV ratio z-score.

    When Z > 2.0: GLD expensive relative to SLV -> short GLD, long SLV.
    When Z < -2.0: GLD cheap relative to SLV -> long GLD, short SLV.
    Exit when Z reverts to +/- 0.5.

    Returns:
        (signals_gld, signals_slv) — each a Series of 1, 0, -1.
    """
    idx = gld.index.intersection(slv.index)
    gld_close = gld["Close"].reindex(idx).astype(float)
    slv_close = slv["Close"].reindex(idx).astype(float)

    ratio = gld_close / slv_close
    ratio_mean = ratio.rolling(LOOKBACK).mean()
    ratio_std = ratio.rolling(LOOKBACK).std()
    zscore = (ratio - ratio_mean) / ratio_std

    signals_gld = pd.Series(0, index=idx, dtype=int)
    signals_slv = pd.Series(0, index=idx, dtype=int)

    in_trade = 0  # 0 = flat, 1 = long GLD/short SLV, -1 = short GLD/long SLV

    for i in range(LOOKBACK, len(idx)):
        z = zscore.iloc[i]
        if pd.isna(z):
            signals_gld.iloc[i] = signals_gld.iloc[i-1] if i > 0 else 0
            signals_slv.iloc[i] = signals_slv.iloc[i-1] if i > 0 else 0
            continue

        if in_trade == 0:
            if z > ENTRY_Z:
                # GLD expensive -> short GLD, long SLV
                in_trade = -1
            elif z < -ENTRY_Z:
                # GLD cheap -> long GLD, short SLV
                in_trade = 1
        elif in_trade == 1:
            # Long GLD, short SLV — exit when z rises back above -EXIT_Z
            if z > -EXIT_Z:
                in_trade = 0
        elif in_trade == -1:
            # Short GLD, long SLV — exit when z drops back below EXIT_Z
            if z < EXIT_Z:
                in_trade = 0

        signals_gld.iloc[i] = in_trade
        signals_slv.iloc[i] = -in_trade

    return signals_gld, signals_slv


def run():
    """Run Gold/Silver ratio backtest using dollar-neutral single-leg simulation."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    gld = get_data("GLD", start="2004-01-01")
    slv = get_data("SLV", start="2006-01-01")
    print(f"GLD range: {gld.index[0].date()} to {gld.index[-1].date()} ({len(gld)} days)")
    print(f"SLV range: {slv.index[0].date()} to {slv.index[-1].date()} ({len(slv)} days)")

    signals_gld, signals_slv = generate_signals(gld, slv)

    # Use common index
    idx = signals_gld.index

    # Simulate dollar-neutral: track combined P&L from both legs
    # Each leg uses half the capital
    gld_close = gld["Close"].reindex(idx).astype(float)
    slv_close = slv["Close"].reindex(idx).astype(float)

    initial_capital = 100_000.0
    equity = pd.Series(initial_capital, index=idx, dtype=float)
    cash = initial_capital
    gld_shares = 0.0
    slv_shares = 0.0
    entry_gld = 0.0
    entry_slv = 0.0
    entry_date = None
    trades = []

    cost_pct = 0.0001  # 0.01% commission per side
    slip_pct = 0.0001  # 0.01% slippage per side

    for i in range(len(idx)):
        date = idx[i]
        prev_sig = signals_gld.iloc[i-1] if i > 0 else 0
        cur_sig = signals_gld.iloc[i]

        if cur_sig != prev_sig:
            # Close existing
            if gld_shares != 0:
                exit_gld = gld_close.iloc[i]
                exit_slv = slv_close.iloc[i]
                pnl_gld = gld_shares * (exit_gld - entry_gld)
                pnl_slv = slv_shares * (exit_slv - entry_slv)
                close_cost = (abs(gld_shares * exit_gld) + abs(slv_shares * exit_slv)) * (cost_pct + slip_pct)
                total_pnl = pnl_gld + pnl_slv - close_cost
                cash += total_pnl
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "side_gld": "long" if gld_shares > 0 else "short",
                    "pnl_gld": pnl_gld, "pnl_slv": pnl_slv,
                    "pnl": total_pnl,
                })
                gld_shares = 0.0
                slv_shares = 0.0

            # Open new
            if cur_sig != 0:
                alloc = cash * 0.45  # Half capital per leg
                entry_gld = gld_close.iloc[i] * (1 + slip_pct * cur_sig)
                entry_slv = slv_close.iloc[i] * (1 - slip_pct * cur_sig)
                gld_shares = cur_sig * alloc / entry_gld
                slv_shares = -cur_sig * alloc / entry_slv
                open_cost = (abs(gld_shares * entry_gld) + abs(slv_shares * entry_slv)) * cost_pct
                cash -= open_cost
                entry_date = date

        # Mark to market
        mtm_gld = gld_shares * (gld_close.iloc[i] - entry_gld) if gld_shares != 0 else 0
        mtm_slv = slv_shares * (slv_close.iloc[i] - entry_slv) if slv_shares != 0 else 0
        equity.iloc[i] = cash + mtm_gld + mtm_slv

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)
    position = (signals_gld != 0).astype(float)

    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )

    print_summary(stats, STRATEGY_NAME)

    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "gold_silver_ratio_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
