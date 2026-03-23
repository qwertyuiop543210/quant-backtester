"""Plotting utilities for backtesting strategies."""

import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def plot_equity(equity_curve: pd.Series, strategy_name: str = "",
                save: bool = True) -> None:
    """Plot and optionally save an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity curve indexed by date.
    strategy_name : str
        Used for title and filename.
    save : bool
        If True, save PNG to results directory.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping equity plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_curve.index, equity_curve.values, linewidth=1.2)
    ax.set_title(f"Equity Curve — {strategy_name}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        safe_name = strategy_name.lower().replace(" ", "_").replace("—", "").replace("/", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        filepath = os.path.join(RESULTS_DIR, f"{safe_name}_equity.png")
        fig.savefig(filepath, dpi=150)
        print(f"Equity plot saved to {filepath}")

    plt.close(fig)
