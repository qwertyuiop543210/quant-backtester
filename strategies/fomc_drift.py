"""FOMC Drift strategy.

Buy ES at close the day before each FOMC announcement, sell at close on FOMC day.
Captures the well-documented pre-FOMC announcement drift.

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

STRATEGY_NAME = "FOMC Pre-Announcement Drift (ES)"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
INITIAL_CAPITAL = 100_000.0

# Hardcoded FOMC announcement dates 2000-2026
# Source: Federal Reserve Board historical and scheduled meeting dates
FOMC_DATES = [
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-03", "2001-01-31", "2001-03-20", "2001-05-15",
    "2001-06-27", "2001-08-21", "2001-10-02", "2001-11-06",
    "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25",
    "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-08-17", "2007-09-18", "2007-10-31",
    "2007-12-11",
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-11", "2008-03-18",
    "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16",
    "2008-10-08", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]


def run():
    """Run FOMC drift backtest."""
    print(f"\n--- {STRATEGY_NAME} ---")
    print("Loading ES data...")

    es = get_data("ES", start="1999-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")

    close = es["Close"].astype(float)
    fomc_set = set(pd.to_datetime(FOMC_DATES))

    trades = []
    equity_values = []
    cash = INITIAL_CAPITAL

    # Build daily equity for plotting
    equity = pd.Series(INITIAL_CAPITAL, index=close.index, dtype=float)

    for i in range(1, len(close)):
        date = close.index[i]
        prev_date = close.index[i - 1]

        # If today is an FOMC day, we entered yesterday at close
        if date in fomc_set:
            entry_price = close.iloc[i - 1]
            exit_price = close.iloc[i]
            pnl_points = exit_price - entry_price
            gross_pnl = pnl_points * ES_POINT_VALUE
            cost = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
            net_pnl = gross_pnl - cost
            cash += net_pnl
            trades.append({
                "entry_date": prev_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "gross_pnl": gross_pnl,
                "costs": cost,
                "pnl": net_pnl,
            })

        equity.iloc[i] = cash

    trade_list = pd.DataFrame(trades)
    trade_pnls = pd.Series(trade_list["pnl"].values, dtype=float) if len(trades) > 0 else pd.Series(dtype=float)
    # Position: 1 on FOMC days (for time-in-market calc)
    position = pd.Series(0, index=close.index, dtype=float)
    for d in fomc_set:
        if d in position.index:
            position.loc[d] = 1.0

    stats = summary(
        trade_pnls=trade_pnls,
        equity_curve=equity,
        position_series=position,
    )

    print_summary(stats, STRATEGY_NAME)

    # Save outputs
    plot_equity(equity, STRATEGY_NAME)

    if len(trade_list) > 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "fomc_drift_trades.csv")
        trade_list.to_csv(csv_path, index=False)
        print(f"Trade list saved to {csv_path}")

    return stats


if __name__ == "__main__":
    run()
