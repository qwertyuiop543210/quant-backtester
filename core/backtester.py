"""
Core backtesting engine for event-driven strategies.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Represents a single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    point_value: float = 1.0
    commission: float = 0.0
    slippage: float = 0.0
    label: str = ""

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.direction * self.point_value

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission - self.slippage

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    """Container for backtest results and statistics."""
    trades: list[Trade] = field(default_factory=list)
    strategy_name: str = ""
    variant_name: str = ""

    @property
    def trade_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "direction": t.direction,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "holding_days": t.holding_days,
                "label": t.label,
            })
        return pd.DataFrame(records)

    def stats(self) -> dict:
        """Compute standard performance statistics."""
        if not self.trades:
            return {"trades": 0}

        df = self.trade_df
        pnls = df["net_pnl"].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        total_pnl = float(pnls.sum())
        avg_pnl = float(pnls.mean())
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown on cumulative P&L
        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Year-by-year breakdown
        df["year"] = df["entry_date"].dt.year
        yearly = df.groupby("year")["net_pnl"].agg(["sum", "count", "mean"]).rename(
            columns={"sum": "total_pnl", "count": "trades", "mean": "avg_pnl"}
        )

        return {
            "strategy": self.strategy_name,
            "variant": self.variant_name,
            "trades": len(pnls),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(float(wins.mean()), 2) if len(wins) > 0 else 0,
            "avg_loss": round(float(losses.mean()), 2) if len(losses) > 0 else 0,
            "best_trade": round(float(pnls.max()), 2),
            "worst_trade": round(float(pnls.min()), 2),
            "yearly": yearly.round(2).to_dict("index"),
        }


def phidias_simulation(
    trades: list[Trade],
    starting_balance: float = 50_000,
    daily_profit_target: float = 4_000,
    eod_max_drawdown: float = 2_500,
    max_loss_per_trade: float = None,
    can_hold_overnight: bool = True,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation of Phidias prop firm evaluation.

    For multi-day holds (like Wed-Fri), each trade is treated as a single
    event with its net P&L applied at exit.

    The eval passes if cumulative P&L reaches daily_profit_target before
    cumulative drawdown from peak hits eod_max_drawdown.

    Parameters
    ----------
    trades : list[Trade]
        Historical trades to sample from.
    starting_balance : float
        Starting account balance.
    daily_profit_target : float
        Cumulative profit target to pass eval.
    eod_max_drawdown : float
        Maximum end-of-day drawdown from peak equity before failing.
    max_loss_per_trade : float or None
        If set, cap individual trade loss at this amount.
    can_hold_overnight : bool
        If True, use Swing account rules. If False, Fundamental rules.
    n_simulations : int
        Number of Monte Carlo trials.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with pass_rate, avg_trades_to_pass, avg_trades_to_fail, etc.
    """
    if not trades:
        return {"pass_rate": 0, "n_simulations": n_simulations, "note": "no trades"}

    rng = np.random.default_rng(seed)
    pnls = np.array([t.net_pnl for t in trades])

    if max_loss_per_trade is not None:
        pnls = np.where(pnls < -max_loss_per_trade, -max_loss_per_trade, pnls)

    passes = 0
    trades_to_pass = []
    trades_to_fail = []
    max_trades_per_sim = 200  # cap iterations

    for _ in range(n_simulations):
        balance = starting_balance
        peak = balance
        passed = False
        failed = False

        for trade_num in range(1, max_trades_per_sim + 1):
            pnl = rng.choice(pnls)
            balance += pnl
            peak = max(peak, balance)
            dd = peak - balance

            if balance >= starting_balance + daily_profit_target:
                passes += 1
                trades_to_pass.append(trade_num)
                passed = True
                break

            if dd >= eod_max_drawdown:
                trades_to_fail.append(trade_num)
                failed = True
                break

        if not passed and not failed:
            trades_to_fail.append(max_trades_per_sim)

    pass_rate = passes / n_simulations

    return {
        "pass_rate": round(pass_rate, 4),
        "n_simulations": n_simulations,
        "passes": passes,
        "fails": n_simulations - passes,
        "avg_trades_to_pass": round(np.mean(trades_to_pass), 1) if trades_to_pass else None,
        "avg_trades_to_fail": round(np.mean(trades_to_fail), 1) if trades_to_fail else None,
        "account_type": "Swing" if can_hold_overnight else "Fundamental",
        "starting_balance": starting_balance,
        "profit_target": daily_profit_target,
        "max_drawdown": eod_max_drawdown,
    }


def correlation_by_week(trades_a: list[Trade], trades_b: list[Trade]) -> dict:
    """
    Compute weekly P&L correlation between two sets of trades.
    Matches trades by ISO calendar week.
    """
    def weekly_pnl(trades):
        if not trades:
            return pd.Series(dtype=float)
        data = [(t.exit_date, t.net_pnl) for t in trades]
        df = pd.DataFrame(data, columns=["date", "pnl"])
        df["week"] = df["date"].dt.isocalendar().year.astype(str) + "-W" + \
                     df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        return df.groupby("week")["pnl"].sum()

    wa = weekly_pnl(trades_a)
    wb = weekly_pnl(trades_b)

    common = wa.index.intersection(wb.index)
    if len(common) < 10:
        return {"correlation": None, "common_weeks": len(common),
                "note": "insufficient overlapping weeks"}

    corr = wa.loc[common].corr(wb.loc[common])
    return {
        "correlation": round(float(corr), 4),
        "common_weeks": len(common),
        "p_value": None,  # could add scipy stats test
    }
