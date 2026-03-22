"""Generate equity curve plots with drawdown overlay."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def plot_equity(equity_curve: pd.Series, strategy_name: str,
                save_path: str = None) -> str:
    """Plot equity curve with drawdown overlay. Save PNG to results/.

    Returns:
        Path to saved PNG file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if save_path is None:
        safe_name = strategy_name.lower().replace(" ", "_")
        save_path = os.path.join(RESULTS_DIR, f"{safe_name}_equity.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True)

    # Equity curve
    ax1.plot(equity_curve.index, equity_curve.values, linewidth=1.2, color="#2196F3")
    ax1.set_title(f"{strategy_name} — Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style="plain", axis="y")

    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to {save_path}")
    return save_path
