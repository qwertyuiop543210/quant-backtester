"""CPI Reversal strategy.

On CPI release days, measure the open-to-early-session move direction, then trade
the reversal. Intraday proxy using daily data:
- If open > prior close (gap up / bullish initial move): short at open, cover at close.
- If open < prior close (gap down / bearish initial move): buy at open, sell at close.

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

STRATEGY_NAME = "CPI Day Reversal (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0

# Hardcoded CPI release dates 2003-2026
# CPI is typically released on the second Tuesday or Wednesday of each month
CPI_DATES = [
    # 2003
    "2003-01-16", "2003-02-20", "2003-03-18", "2003-04-16",
    "2003-05-16", "2003-06-17", "2003-07-16", "2003-08-19",
    "2003-09-16", "2003-10-16", "2003-11-19", "2003-12-16",
    # 2004
    "2004-01-16", "2004-02-20", "2004-03-17", "2004-04-14",
    "2004-05-14", "2004-06-15", "2004-07-14", "2004-08-19",
    "2004-09-16", "2004-10-19", "2004-11-17", "2004-12-15",
    # 2005
    "2005-01-14", "2005-02-23", "2005-03-23", "2005-04-20",
    "2005-05-18", "2005-06-15", "2005-07-15", "2005-08-16",
    "2005-09-15", "2005-10-14", "2005-11-16", "2005-12-14",
    # 2006
    "2006-01-18", "2006-02-22", "2006-03-15", "2006-04-19",
    "2006-05-17", "2006-06-14", "2006-07-19", "2006-08-16",
    "2006-09-15", "2006-10-18", "2006-11-15", "2006-12-15",
    # 2007
    "2007-01-18", "2007-02-21", "2007-03-16", "2007-04-17",
    "2007-05-15", "2007-06-15", "2007-07-18", "2007-08-14",
    "2007-09-19", "2007-10-17", "2007-11-15", "2007-12-14",
    # 2008
    "2008-01-16", "2008-02-20", "2008-03-14", "2008-04-16",
    "2008-05-14", "2008-06-13", "2008-07-16", "2008-08-14",
    "2008-09-16", "2008-10-16", "2008-11-19", "2008-12-16",
    # 2009
    "2009-01-16", "2009-02-20", "2009-03-18", "2009-04-15",
    "2009-05-15", "2009-06-17", "2009-07-15", "2009-08-14",
    "2009-09-16", "2009-10-15", "2009-11-18", "2009-12-16",
    # 2010
    "2010-01-15", "2010-02-19", "2010-03-18", "2010-04-14",
    "2010-05-19", "2010-06-17", "2010-07-16", "2010-08-13",
    "2010-09-17", "2010-10-15", "2010-11-17", "2010-12-15",
    # 2011
    "2011-01-14", "2011-02-17", "2011-03-17", "2011-04-15",
    "2011-05-13", "2011-06-15", "2011-07-15", "2011-08-18",
    "2011-09-15", "2011-10-19", "2011-11-16", "2011-12-16",
    # 2012
    "2012-01-19", "2012-02-17", "2012-03-16", "2012-04-13",
    "2012-05-15", "2012-06-14", "2012-07-17", "2012-08-15",
    "2012-09-14", "2012-10-16", "2012-11-15", "2012-12-14",
    # 2013
    "2013-01-16", "2013-02-21", "2013-03-15", "2013-04-16",
    "2013-05-16", "2013-06-18", "2013-07-16", "2013-08-15",
    "2013-09-17", "2013-10-30", "2013-11-20", "2013-12-17",
    # 2014
    "2014-01-16", "2014-02-20", "2014-03-18", "2014-04-15",
    "2014-05-15", "2014-06-17", "2014-07-22", "2014-08-19",
    "2014-09-17", "2014-10-22", "2014-11-20", "2014-12-17",
    # 2015
    "2015-01-16", "2015-02-26", "2015-03-24", "2015-04-17",
    "2015-05-22", "2015-06-18", "2015-07-17", "2015-08-19",
    "2015-09-16", "2015-10-15", "2015-11-17", "2015-12-15",
    # 2016
    "2016-01-20", "2016-02-19", "2016-03-16", "2016-04-14",
    "2016-05-17", "2016-06-16", "2016-07-15", "2016-08-16",
    "2016-09-16", "2016-10-18", "2016-11-17", "2016-12-15",
    # 2017
    "2017-01-18", "2017-02-15", "2017-03-15", "2017-04-14",
    "2017-05-12", "2017-06-14", "2017-07-14", "2017-08-11",
    "2017-09-14", "2017-10-13", "2017-11-15", "2017-12-13",
    # 2018
    "2018-01-12", "2018-02-14", "2018-03-13", "2018-04-11",
    "2018-05-10", "2018-06-12", "2018-07-12", "2018-08-10",
    "2018-09-13", "2018-10-11", "2018-11-14", "2018-12-12",
    # 2019
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10",
    "2019-05-10", "2019-06-12", "2019-07-11", "2019-08-13",
    "2019-09-12", "2019-10-10", "2019-11-13", "2019-12-11",
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10",
    "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13",
    "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
    "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-16", "2026-10-13", "2026-11-12", "2026-12-09",
]


def run():
    """Run CPI reversal backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="2002-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    close = es["Close"].astype(float)
    open_ = es["Open"].astype(float)
    cpi_set = set(pd.to_datetime(CPI_DATES))

    trades = []
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)
    cash = INITIAL_CAPITAL
    position = pd.Series(0.0, index=close.index)

    for i in range(1, len(close)):
        date = close.index[i]
        equity.iloc[i] = cash

        if date not in cpi_set:
            continue

        prev_close = close.iloc[i - 1]
        today_open = open_.iloc[i]
        today_close = close.iloc[i]

        # Determine initial move direction from gap
        gap = today_open - prev_close

        if abs(gap) < 0.01:
            continue  # No meaningful gap, skip

        # Trade the reversal: if gap up, short; if gap down, long
        if gap > 0:
            # Gap up -> short reversal: short at open, cover at close
            direction = -1
            side = "short"
        else:
            # Gap down -> long reversal: buy at open, sell at close
            direction = 1
            side = "long"

        pnl_points = direction * (today_close - today_open)
        gross_pnl = pnl_points * ES_POINT_VALUE
        cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
        net_pnl = gross_pnl - cost
        cash += net_pnl
        equity.iloc[i] = cash
        position.iloc[i] = 1.0

        trades.append({
            "date": date,
            "gap_direction": "up" if gap > 0 else "down",
            "trade_side": side,
            "open": today_open,
            "close": today_close,
            "prev_close": prev_close,
            "gap_size": gap,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost,
            "pnl": net_pnl,
        })

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)

    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )

    print_summary(stats, STRATEGY_NAME)

    # Breakdown by gap direction
    if len(trade_list) > 0:
        print(f"{'='*60}")
        print(f"  Breakdown by Gap Direction")
        print(f"{'='*60}")
        for direction in ["up", "down"]:
            subset = trade_list[trade_list["gap_direction"] == direction]
            if len(subset) == 0:
                continue
            sub_pnls = subset["pnl"]
            wins = (sub_pnls > 0).sum()
            n = len(sub_pnls)
            avg_pnl = sub_pnls.mean()
            total = sub_pnls.sum()
            print(f"  Gap {direction}: {n} trades, {wins/n:.1%} win rate, "
                  f"avg ${avg_pnl:.0f}, total ${total:.0f}")
        print(f"{'='*60}\n")

    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "cpi_reversal_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
