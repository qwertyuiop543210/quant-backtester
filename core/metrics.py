"""Calculate strategy performance metrics."""

import numpy as np
import pandas as pd


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss. Returns inf if no losing trades."""
    gross_profit = trade_pnls[trade_pnls > 0].sum()
    gross_loss = abs(trade_pnls[trade_pnls < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = daily_returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(abs(dd.min()))


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades that are profitable."""
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / len(trade_pnls))


def annualized_return(equity_curve: pd.Series) -> float:
    """CAGR from equity curve."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days <= 0:
        return 0.0
    return float(total_return ** (365.25 / days) - 1)


def time_in_market(position_series: pd.Series) -> float:
    """Fraction of days with a non-zero position."""
    if len(position_series) == 0:
        return 0.0
    return float((position_series != 0).sum() / len(position_series))


def total_trades(trade_pnls: pd.Series) -> int:
    """Number of completed trades."""
    return len(trade_pnls)


def summary(trade_pnls: pd.Series, equity_curve: pd.Series,
            position_series: pd.Series) -> dict:
    """Compute all metrics and return as dict."""
    return {
        "profit_factor": profit_factor(trade_pnls),
        "sharpe_ratio": sharpe_ratio(equity_curve.pct_change().dropna()),
        "win_rate": win_rate(trade_pnls),
        "max_drawdown": max_drawdown(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "total_trades": total_trades(trade_pnls),
        "time_in_market": time_in_market(position_series),
    }


def print_summary(stats: dict, strategy_name: str = "Strategy") -> None:
    """Print formatted summary and verdict."""
    print(f"\n{'='*60}")
    print(f"  {strategy_name} — Backtest Results")
    print(f"{'='*60}")
    print(f"  Profit Factor:     {stats['profit_factor']:.3f}")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:.3f}")
    print(f"  Win Rate:          {stats['win_rate']:.1%}")
    print(f"  Total Trades:      {stats['total_trades']}")
    print(f"  Max Drawdown:      {stats['max_drawdown']:.1%}")
    print(f"  Annualized Return: {stats['annualized_return']:.1%}")
    print(f"  Time in Market:    {stats['time_in_market']:.1%}")
    print(f"{'='*60}")

    pf = stats["profit_factor"]
    trades = stats["total_trades"]
    if pf > 1.2 and trades > 100:
        print("  VERDICT: TRADEABLE")
    elif trades < 100:
        print(f"  VERDICT: NO EDGE — insufficient trades ({trades} < 100)")
    elif pf <= 1.05:
        print(f"  VERDICT: NO EDGE — profit factor {pf:.3f} indistinguishable from random")
    else:
        print(f"  VERDICT: NO EDGE — profit factor {pf:.3f} below 1.2 threshold")
    print(f"{'='*60}\n")
