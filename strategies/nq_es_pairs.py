"""NQ/ES Pairs strategy.

Calculate NQ/ES price ratio daily. Z-score with 50-day rolling lookback.
Entry at Z > 2.0 or Z < -2.0. Exit at Z returning to 0.5 / -0.5.
Track P&L on BOTH legs simultaneously.
Position sizing: NQ point = $20, ES point = $50. Dollar-neutral exposure.

Default costs: $5 round trip per futures contract, 1 tick slippage per side.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.backtester import run_pairs
from core.metrics import summary, print_summary
from core.plotting import plot_equity

STRATEGY_NAME = "NQ/ES Pairs"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

NQ_POINT_VALUE = 20.0
ES_POINT_VALUE = 50.0
LOOKBACK = 50
ENTRY_Z = 2.0
EXIT_Z = 0.5


def generate_signals(nq: pd.DataFrame, es: pd.DataFrame) -> tuple:
    """Generate pairs trading signals based on NQ/ES ratio z-score.

    When Z > 2.0: NQ is expensive relative to ES -> short NQ, long ES.
    When Z < -2.0: NQ is cheap relative to ES -> long NQ, short ES.
    Exit when Z reverts to +/- 0.5.

    Returns:
        (signals_nq, signals_es) — each a Series of 1, 0, -1.
    """
    idx = nq.index.intersection(es.index)
    nq_close = nq["Close"].reindex(idx).astype(float)
    es_close = es["Close"].reindex(idx).astype(float)

    ratio = nq_close / es_close
    ratio_mean = ratio.rolling(LOOKBACK).mean()
    ratio_std = ratio.rolling(LOOKBACK).std()
    zscore = (ratio - ratio_mean) / ratio_std

    signals_nq = pd.Series(0, index=idx, dtype=int)
    signals_es = pd.Series(0, index=idx, dtype=int)

    in_trade = 0  # 0 = flat, 1 = long NQ/short ES, -1 = short NQ/long ES

    for i in range(LOOKBACK, len(idx)):
        z = zscore.iloc[i]
        if pd.isna(z):
            signals_nq.iloc[i] = signals_nq.iloc[i-1] if i > 0 else 0
            signals_es.iloc[i] = signals_es.iloc[i-1] if i > 0 else 0
            continue

        if in_trade == 0:
            if z > ENTRY_Z:
                # NQ expensive -> short NQ, long ES
                in_trade = -1
            elif z < -ENTRY_Z:
                # NQ cheap -> long NQ, short ES
                in_trade = 1
        elif in_trade == 1:
            # Long NQ, short ES — exit when z rises back above -EXIT_Z
            if z > -EXIT_Z:
                in_trade = 0
        elif in_trade == -1:
            # Short NQ, long ES — exit when z drops back below EXIT_Z
            if z < EXIT_Z:
                in_trade = 0

        signals_nq.iloc[i] = in_trade
        signals_es.iloc[i] = -in_trade  # Opposite leg

    return signals_nq, signals_es


def run():
    """Run NQ/ES pairs backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading data...")

    nq = get_data("NQ", start="1999-01-01")
    es = get_data("ES", start="1999-01-01")
    print(f"NQ range: {nq.index[0].date()} to {nq.index[-1].date()} ({len(nq)} days)")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    signals_nq, signals_es = generate_signals(nq, es)

    result = run_pairs(
        prices_a=nq["Close"],
        prices_b=es["Close"],
        signals_a=signals_nq,
        signals_b=signals_es,
        dollar_per_point_a=NQ_POINT_VALUE,
        dollar_per_point_b=ES_POINT_VALUE,
        commission_per_trade=5.0,        # $5 round trip per contract
        slippage_ticks_a=0.25,           # 1 tick NQ = 0.25 points
        slippage_ticks_b=0.25,           # 1 tick ES = 0.25 points
        initial_capital=100_000.0,
    )

    stats = summary(
        trade_pnls=result["trade_pnls"],
        equity_curve=result["equity_curve"],
        position_series=result["position"],
    )

    print_summary(stats, STRATEGY_NAME)

    plot_equity(result["equity_curve"], STRATEGY_NAME)

    if len(result["trade_list"]) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "nq_es_pairs_trades.csv")
        result["trade_list"].to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats, result


if __name__ == "__main__":
    run()
