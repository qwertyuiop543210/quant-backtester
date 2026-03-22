"""Overnight Returns study.

Compare: (A) buy SPY at close, sell next open vs (B) buy at open, sell at close.
Measurement study showing which session captures returns.

Default costs: 0.01% commission per side (ETF), 0.01% slippage per side.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import profit_factor, sharpe_ratio, max_drawdown, win_rate
from core.plotting import plot_equity

STRATEGY_NAME = "Overnight vs Intraday Returns"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def analyze_sessions(spy: pd.DataFrame) -> dict:
    """Break SPY returns into overnight and intraday sessions.

    Overnight: close-to-open (buy at close, sell next open).
    Intraday: open-to-close (buy at open, sell at close).

    Returns dict with both session DataFrames.
    """
    df = spy[["Open", "Close"]].copy()
    df = df.dropna()

    # Overnight return: previous close to current open
    df["overnight_return"] = df["Open"] / df["Close"].shift(1) - 1
    # Intraday return: current open to current close
    df["intraday_return"] = df["Close"] / df["Open"] - 1
    # Total daily return
    df["total_return"] = df["Close"] / df["Close"].shift(1) - 1

    df = df.dropna()

    # Apply costs: 0.01% commission + 0.01% slippage per side = 0.02% per trade each way
    cost_per_trade = 0.0002  # 0.02% round trip (commission + slippage)
    df["overnight_return_net"] = df["overnight_return"] - cost_per_trade
    df["intraday_return_net"] = df["intraday_return"] - cost_per_trade

    return df


def build_equity(returns: pd.Series, initial: float = 100_000.0) -> pd.Series:
    """Build equity curve from return series."""
    return initial * (1 + returns).cumprod()


def run():
    """Run overnight vs intraday analysis."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading SPY data...")

    spy = get_data("SPY", start="1993-01-01")
    print(f"Data range: {spy.index[0].date()} to {spy.index[-1].date()} ({len(spy)} days)")

    df = analyze_sessions(spy)

    # Build equity curves
    equity_overnight_gross = build_equity(df["overnight_return"])
    equity_intraday_gross = build_equity(df["intraday_return"])
    equity_overnight_net = build_equity(df["overnight_return_net"])
    equity_intraday_net = build_equity(df["intraday_return_net"])
    equity_buyhold = build_equity(df["total_return"])

    # Stats for each session
    years = (df.index[-1] - df.index[0]).days / 365.25

    print(f"\n{'='*65}")
    print(f"  {STRATEGY_NAME} — Results")
    print(f"{'='*65}")

    for label, ret_col, equity in [
        ("Overnight (gross)", "overnight_return", equity_overnight_gross),
        ("Overnight (net)",   "overnight_return_net", equity_overnight_net),
        ("Intraday (gross)",  "intraday_return", equity_intraday_gross),
        ("Intraday (net)",    "intraday_return_net", equity_intraday_net),
        ("Buy & Hold",        "total_return", equity_buyhold),
    ]:
        returns = df[ret_col]
        total_ret = equity.iloc[-1] / equity.iloc[0]
        cagr = total_ret ** (1 / years) - 1
        sharpe = sharpe_ratio(returns)
        maxdd = max_drawdown(equity)
        wr = win_rate(returns)
        n_trades = len(returns)

        # Profit factor from daily returns
        gross_wins = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        print(f"\n  {label}:")
        print(f"    CAGR:            {cagr:.1%}")
        print(f"    Sharpe Ratio:    {sharpe:.3f}")
        print(f"    Profit Factor:   {pf:.3f}")
        print(f"    Win Rate:        {wr:.1%}")
        print(f"    Max Drawdown:    {maxdd:.1%}")
        print(f"    Total Sessions:  {n_trades}")

    print(f"\n{'='*65}")

    # Verdict
    on_cagr = (equity_overnight_net.iloc[-1] / equity_overnight_net.iloc[0]) ** (1/years) - 1
    id_cagr = (equity_intraday_net.iloc[-1] / equity_intraday_net.iloc[0]) ** (1/years) - 1

    if on_cagr > id_cagr:
        print(f"  FINDING: Overnight session captures more returns ({on_cagr:.1%} vs {id_cagr:.1%} CAGR after costs)")
    else:
        print(f"  FINDING: Intraday session captures more returns ({id_cagr:.1%} vs {on_cagr:.1%} CAGR after costs)")
    print(f"{'='*65}\n")

    # Plot both equity curves
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(equity_overnight_net.index, equity_overnight_net.values,
            label="Overnight (net)", linewidth=1.2, color="#2196F3")
    ax.plot(equity_intraday_net.index, equity_intraday_net.values,
            label="Intraday (net)", linewidth=1.2, color="#F44336")
    ax.plot(equity_buyhold.index, equity_buyhold.values,
            label="Buy & Hold", linewidth=1.0, color="#4CAF50", alpha=0.7)
    ax.set_title("Overnight vs Intraday Returns (SPY, net of costs)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y")

    save_path = os.path.join(RESULTS_DIR, "overnight_returns_equity.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {save_path}")

    # Save session returns CSV
    csv_path = os.path.join(RESULTS_DIR, "overnight_returns_data.csv")
    df[["overnight_return", "intraday_return", "total_return",
        "overnight_return_net", "intraday_return_net"]].to_csv(csv_path)
    print(f"Session data saved to {csv_path}")

    return df


if __name__ == "__main__":
    run()
