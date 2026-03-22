"""Core backtester: takes signal series and price series, simulates execution."""

import pandas as pd
import numpy as np


def run_single(prices: pd.Series, signals: pd.Series,
               commission_per_trade: float = 0.0001,
               slippage_pct: float = 0.0001,
               initial_capital: float = 100_000.0) -> dict:
    """Backtest a single-instrument strategy.

    Args:
        prices: Close price series indexed by date.
        signals: Series of 1 (long), 0 (flat), -1 (short), same index as prices.
        commission_per_trade: Commission as fraction of trade value per side.
        slippage_pct: Slippage as fraction of price per side.
        initial_capital: Starting capital.

    Returns:
        dict with keys: equity_curve, position, trade_pnls, trade_list
    """
    prices, signals = prices.align(signals, join="inner")
    prices = prices.astype(float)
    signals = signals.astype(float)

    position = pd.Series(0.0, index=prices.index)
    equity = pd.Series(initial_capital, index=prices.index, dtype=float)
    cash = initial_capital
    shares = 0.0
    trades = []
    entry_price = 0.0
    entry_date = None

    for i in range(len(prices)):
        date = prices.index[i]
        price = prices.iloc[i]
        signal = signals.iloc[i]

        # Check for position change
        if i > 0:
            prev_signal = signals.iloc[i - 1]
        else:
            prev_signal = 0.0

        if signal != prev_signal:
            # Close existing position
            if shares != 0:
                exit_price = price * (1 - slippage_pct * np.sign(shares))
                proceeds = shares * exit_price
                cost = abs(shares * price) * commission_per_trade
                cash += proceeds - cost
                trade_pnl = shares * (exit_price - entry_price) - cost - abs(shares * entry_price) * commission_per_trade
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "side": "long" if shares > 0 else "short",
                    "pnl": trade_pnl,
                })
                shares = 0.0

            # Open new position
            if signal != 0:
                entry_price = price * (1 + slippage_pct * signal)
                # Size: use full capital
                shares = signal * (cash * 0.95) / entry_price
                cost = abs(shares * entry_price) * commission_per_trade
                cash -= shares * entry_price + cost
                entry_date = date

        # Mark to market
        position.iloc[i] = shares
        equity.iloc[i] = cash + shares * price

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)

    return {
        "equity_curve": equity,
        "position": (signals != 0).astype(float),
        "trade_pnls": trade_pnls,
        "trade_list": trade_list,
    }


def run_pairs(prices_a: pd.Series, prices_b: pd.Series,
              signals_a: pd.Series, signals_b: pd.Series,
              dollar_per_point_a: float = 1.0,
              dollar_per_point_b: float = 1.0,
              commission_per_trade: float = 5.0,
              slippage_ticks_a: float = 0.25,
              slippage_ticks_b: float = 0.25,
              initial_capital: float = 100_000.0) -> dict:
    """Backtest a pairs strategy with two legs.

    signals_a and signals_b should be opposite (one long, one short).
    Commission is a flat dollar amount per contract per side.

    Returns:
        dict with keys: equity_curve, position, trade_pnls, trade_list
    """
    idx = prices_a.index.intersection(prices_b.index)
    pa = prices_a.loc[idx].astype(float)
    pb = prices_b.loc[idx].astype(float)
    sa = signals_a.reindex(idx).fillna(0).astype(float)
    sb = signals_b.reindex(idx).fillna(0).astype(float)

    equity = pd.Series(initial_capital, index=idx, dtype=float)
    cash = initial_capital
    contracts_a = 0.0
    contracts_b = 0.0
    entry_a = 0.0
    entry_b = 0.0
    entry_date = None
    trades = []

    for i in range(len(idx)):
        date = idx[i]
        prev_sa = sa.iloc[i - 1] if i > 0 else 0.0
        cur_sa = sa.iloc[i]

        if cur_sa != prev_sa:
            # Close
            if contracts_a != 0:
                exit_a = pa.iloc[i]
                exit_b = pb.iloc[i]
                pnl_a = contracts_a * (exit_a - entry_a) * dollar_per_point_a
                pnl_b = contracts_b * (exit_b - entry_b) * dollar_per_point_b
                cost = 2 * commission_per_trade  # close both legs
                slippage_cost = (abs(contracts_a) * slippage_ticks_a * dollar_per_point_a +
                                 abs(contracts_b) * slippage_ticks_b * dollar_per_point_b)
                total_pnl = pnl_a + pnl_b - cost - slippage_cost
                cash += total_pnl
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "pnl_a": pnl_a, "pnl_b": pnl_b,
                    "pnl": total_pnl,
                })
                contracts_a = 0.0
                contracts_b = 0.0

            # Open
            if cur_sa != 0:
                # Dollar-neutral: allocate equal dollar exposure
                alloc = cash * 0.3
                contracts_a = cur_sa * alloc / (pa.iloc[i] * dollar_per_point_a)
                contracts_b = sb.iloc[i] * alloc / (pb.iloc[i] * dollar_per_point_b)
                entry_a = pa.iloc[i]
                entry_b = pb.iloc[i]
                entry_date = date
                cost = 2 * commission_per_trade
                slippage_cost = (abs(contracts_a) * slippage_ticks_a * dollar_per_point_a +
                                 abs(contracts_b) * slippage_ticks_b * dollar_per_point_b)
                cash -= cost + slippage_cost

        pnl_a = contracts_a * (pa.iloc[i] - entry_a) * dollar_per_point_a if contracts_a != 0 else 0
        pnl_b = contracts_b * (pb.iloc[i] - entry_b) * dollar_per_point_b if contracts_b != 0 else 0
        equity.iloc[i] = cash + pnl_a + pnl_b

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)
    position = (sa != 0).astype(float)

    return {
        "equity_curve": equity,
        "position": position,
        "trade_pnls": trade_pnls,
        "trade_list": trade_list,
    }
