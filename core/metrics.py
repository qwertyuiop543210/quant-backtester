"""Performance metrics for backtesting strategies."""

import numpy as np
import pandas as pd


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    if len(trade_pnls) == 0:
        return 0.0
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = abs(trade_pnls[trade_pnls < 0].sum())
    if losses == 0:
        return float("inf")
    return gains / losses


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades with positive P&L."""
    if len(trade_pnls) == 0:
        return 0.0
    return (trade_pnls > 0).sum() / len(trade_pnls)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown in dollar terms."""
    running_max = equity_curve.cummax()
    drawdowns = equity_curve - running_max
    return float(drawdowns.min())


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from daily equity curve."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def summary(trade_pnls: pd.Series, equity_curve: pd.Series,
            position_series: pd.Series = None) -> dict:
    """Compute a standard set of performance statistics.

    Parameters
    ----------
    trade_pnls : pd.Series
        Net P&L per trade.
    equity_curve : pd.Series
        Daily equity curve (indexed by date).
    position_series : pd.Series, optional
        Daily position (1 = in trade, 0 = flat). Used for exposure %.

    Returns
    -------
    dict with keys: trades, win_rate, profit_factor, total_pnl, avg_pnl,
        max_drawdown, sharpe, best_trade, worst_trade, avg_win, avg_loss,
        exposure_pct.
    """
    n = len(trade_pnls)
    if n == 0:
        return {
            "trades": 0, "win_rate": 0, "profit_factor": 0,
            "total_pnl": 0, "avg_pnl": 0, "max_drawdown": 0,
            "sharpe": 0, "best_trade": 0, "worst_trade": 0,
            "avg_win": 0, "avg_loss": 0, "exposure_pct": 0,
        }

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]

    exposure = 0.0
    if position_series is not None and len(position_series) > 0:
        exposure = (position_series != 0).sum() / len(position_series)

    return {
        "trades": n,
        "win_rate": win_rate(trade_pnls),
        "profit_factor": profit_factor(trade_pnls),
        "total_pnl": float(trade_pnls.sum()),
        "avg_pnl": float(trade_pnls.mean()),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe": sharpe_ratio(equity_curve),
        "best_trade": float(trade_pnls.max()),
        "worst_trade": float(trade_pnls.min()),
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0,
        "exposure_pct": exposure,
    }


def print_summary(stats: dict, strategy_name: str = "") -> None:
    """Pretty-print a summary stats dict."""
    print(f"\n{'=' * 80}")
    print(f"  {strategy_name}")
    print(f"{'=' * 80}")
    if stats["trades"] == 0:
        print("  No trades.")
        return

    print(f"  Trades:        {stats['trades']}")
    print(f"  Win Rate:      {stats['win_rate']:.1%}")
    print(f"  Profit Factor: {stats['profit_factor']:.3f}")
    print(f"  Total P&L:     ${stats['total_pnl']:,.2f}")
    print(f"  Avg P&L:       ${stats['avg_pnl']:,.2f}")
    print(f"  Max Drawdown:  ${stats['max_drawdown']:,.2f}")
    print(f"  Sharpe Ratio:  {stats['sharpe']:.3f}")
    print(f"  Best Trade:    ${stats['best_trade']:,.2f}")
    print(f"  Worst Trade:   ${stats['worst_trade']:,.2f}")
    print(f"  Avg Win:       ${stats['avg_win']:,.2f}")
    print(f"  Avg Loss:      ${stats['avg_loss']:,.2f}")
    if stats.get("exposure_pct"):
        print(f"  Exposure:      {stats['exposure_pct']:.1%}")
    print(f"{'=' * 80}")
