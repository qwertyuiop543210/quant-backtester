"""Combined Portfolio — Phidias $50K Swing evaluation simulator.

Runs BOTH the Chosen One (Week 1 & Week 4) and Dip Buyer strategies
on a single account, processing trades chronologically with daily
mark-to-market EOD drawdown checks.

Chosen One: Buy ES Monday open, sell Friday close, Week 1 & Week 4 only,
            skip VIX 15-20 at Friday close. $30 cost per trade.
Dip Buyer:  RSI mean reversion on ES, loaded from results/dip_buyer_trades.csv.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import profit_factor, win_rate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ES_POINT_VALUE = 50.0
COST_PER_TRADE = 30.0

# Phidias parameters
PHIDIAS_CAPITAL = 50_000.0
PHIDIAS_PROFIT_TARGET = 4_000.0
PHIDIAS_EOD_DRAWDOWN = 2_500.0

DATA_START = "2012-01-01"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_week_of_month(date: pd.Timestamp) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def generate_chosen_one_trades(es: pd.DataFrame, vix: pd.DataFrame):
    """Generate Chosen One trades inline: buy ES Monday open, sell Friday close,
    Week 1 & Week 4 only, skip if VIX 15-20 at prior Friday close.

    Returns (trades_df, skipped_by_vix_count).
    """
    vix_close = vix["Close"].reindex(es.index).ffill()
    es_open = es["Open"].astype(float)
    es_close = es["Close"].astype(float)

    trades = []
    skipped_vix = 0
    i = 0
    while i < len(es):
        date = es.index[i]
        # Look for Monday
        if date.dayofweek != 0:
            i += 1
            continue
        wom = get_week_of_month(date)
        if wom not in (1, 4):
            i += 1
            continue

        # Find Friday of this week
        fri_idx = None
        for j in range(i + 1, min(i + 6, len(es))):
            if es.index[j].dayofweek == 4 and es.index[j].isocalendar()[1] == date.isocalendar()[1]:
                fri_idx = j
                break
        if fri_idx is None:
            i += 1
            continue

        # VIX filter: skip if prior Friday close VIX is between 15-20
        prior_fri_vix = None
        for k in range(i - 1, max(i - 8, -1), -1):
            if es.index[k].dayofweek == 4:
                prior_fri_vix = vix_close.iloc[k]
                break
        if prior_fri_vix is not None and 15.0 <= prior_fri_vix <= 20.0:
            skipped_vix += 1
            i = fri_idx + 1
            continue

        entry_price = es_open.iloc[i]
        exit_price = es_close.iloc[fri_idx]
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": es.index[i],
            "exit_date": es.index[fri_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": COST_PER_TRADE,
            "pnl": net_pnl,
            "source": "CO",
        })

        i = fri_idx + 1

    return pd.DataFrame(trades), skipped_vix


def compute_daily_pnls(trade_row, es_close_series):
    """Compute daily P&L within a trade using ES close prices.
    Returns list of (date, daily_pnl) tuples.
    """
    entry_date = trade_row["entry_date"]
    exit_date = trade_row["exit_date"]
    entry_price = trade_row["entry_price"]
    cost = trade_row["costs"]

    # Get ES closes between entry and exit dates
    mask = (es_close_series.index >= entry_date) & (es_close_series.index <= exit_date)
    trade_closes = es_close_series[mask]

    if len(trade_closes) == 0:
        return [(exit_date, trade_row["pnl"])]

    daily_pnls = []
    for idx_pos, (date, close) in enumerate(trade_closes.items()):
        if idx_pos == 0:
            # Entry day: bought at entry_price (open), mark to close
            day_pnl = (close - entry_price) * ES_POINT_VALUE
            if len(trade_closes) == 1:
                day_pnl -= cost  # Single-day trade: apply costs
        elif idx_pos == len(trade_closes) - 1:
            # Exit day: close-to-close minus costs
            prev_close = trade_closes.iloc[idx_pos - 1]
            day_pnl = (close - prev_close) * ES_POINT_VALUE - cost
        else:
            # Middle day: close-to-close
            prev_close = trade_closes.iloc[idx_pos - 1]
            day_pnl = (close - prev_close) * ES_POINT_VALUE
        daily_pnls.append((date, day_pnl))

    return daily_pnls


# ---------------------------------------------------------------------------
# STEP 1: Load both trade lists
# ---------------------------------------------------------------------------

def load_trades(es, vix):
    """Load CO and DB trade lists, filter to 2012+."""

    # --- Chosen One (always generated inline with VIX 15-20 skip filter) ---
    print("Generating Chosen One trades inline (with VIX 15-20 skip)...")
    co_trades, skipped_vix = generate_chosen_one_trades(es, vix)

    # --- Dip Buyer ---
    db_csv = os.path.join(RESULTS_DIR, "dip_buyer_trades.csv")
    if not os.path.exists(db_csv):
        print("ERROR: Dip Buyer trades not found at results/dip_buyer_trades.csv")
        print("       Run dip_buyer.py first.")
        sys.exit(1)

    print("Loading Dip Buyer trades from CSV...")
    db_trades = pd.read_csv(db_csv, parse_dates=["entry_date", "exit_date"])
    db_trades["source"] = "DB"

    # Filter to 2012+
    co_trades = co_trades[co_trades["entry_date"] >= DATA_START].copy()
    db_trades = db_trades[db_trades["entry_date"] >= DATA_START].copy()

    print(f"  Chosen One trades (with VIX 15-20 skip): {len(co_trades)} trades (2012+)")
    print(f"  Chosen One trades skipped by VIX filter: {skipped_vix} trades (all years)")
    print(f"  Dip Buyer:  {len(db_trades)} trades (2012+)")

    return co_trades, db_trades


# ---------------------------------------------------------------------------
# STEP 2: Merge and validate
# ---------------------------------------------------------------------------

def merge_and_validate(co_trades, db_trades):
    """Merge both trade lists chronologically, check for overlaps."""

    # Standardize columns
    cols = ["entry_date", "exit_date", "entry_price", "exit_price",
            "pnl_points", "gross_pnl", "costs", "pnl", "source"]
    for col in cols:
        if col not in co_trades.columns:
            co_trades[col] = 0
        if col not in db_trades.columns:
            db_trades[col] = 0

    combined = pd.concat([co_trades[cols], db_trades[cols]], ignore_index=True)
    combined = combined.sort_values("entry_date").reset_index(drop=True)

    # Check overlaps
    overlap_count = 0
    for i in range(len(combined)):
        for j in range(i + 1, len(combined)):
            a_start = combined.iloc[i]["entry_date"]
            a_end = combined.iloc[i]["exit_date"]
            b_start = combined.iloc[j]["entry_date"]
            b_end = combined.iloc[j]["exit_date"]

            if b_start > a_end:
                break  # Sorted by entry_date, no more overlaps possible

            # Overlap if one starts before the other ends
            if a_start <= b_end and b_start <= a_end:
                overlap_count += 1

    print(f"\n  Merged trades: {len(combined)} total")
    print(f"  Overlap count: {overlap_count}")
    if overlap_count == 0:
        print("  (Zero overlaps — as expected from Dip Buyer's overlap filter)")

    return combined


# ---------------------------------------------------------------------------
# STEP 3: Combined Phidias simulation
# ---------------------------------------------------------------------------

def run_phidias_simulation(combined, es_close_series):
    """Run Phidias $50K Swing evaluation on combined chronological trades."""

    print(f"\n{'='*90}")
    print(f"  COMBINED PHIDIAS $50K SWING EVALUATION")
    print(f"  Rules: ${PHIDIAS_EOD_DRAWDOWN:,.0f} EOD drawdown from HWM, "
          f"${PHIDIAS_PROFIT_TARGET:,.0f} profit target")
    print(f"{'='*90}")
    print(f"  {'#':>3} {'Status':>10} {'Trades':>8} {'CO':>5} {'DB':>5} "
          f"{'Profit':>12} {'Peak Bal':>12} {'Final Bal':>12}")
    print(f"  {'-'*3} {'-'*10} {'-'*8} {'-'*5} {'-'*5} "
          f"{'-'*12} {'-'*12} {'-'*12}")

    attempts = []
    i = 0

    while i < len(combined):
        balance = PHIDIAS_CAPITAL
        high_water = balance
        start_trade = i
        status = "in_progress"
        co_count = 0
        db_count = 0

        while i < len(combined):
            trade = combined.iloc[i]

            # Compute daily P&Ls for this trade
            daily_pnls = compute_daily_pnls(trade, es_close_series)

            breached = False
            for date, day_pnl in daily_pnls:
                balance += day_pnl
                high_water = max(high_water, balance)
                if high_water - balance >= PHIDIAS_EOD_DRAWDOWN:
                    breached = True
                    break

            if trade["source"] == "CO":
                co_count += 1
            else:
                db_count += 1

            profit = balance - PHIDIAS_CAPITAL
            i += 1

            if breached:
                status = "FAILED"
                break
            if profit >= PHIDIAS_PROFIT_TARGET:
                status = "PASSED"
                break

        if status == "in_progress":
            status = "INCOMPLETE"

        attempt = {
            "attempt": len(attempts) + 1,
            "status": status,
            "trades_taken": i - start_trade,
            "co_trades": co_count,
            "db_trades": db_count,
            "profit": balance - PHIDIAS_CAPITAL,
            "peak_balance": high_water,
            "final_balance": balance,
        }
        attempts.append(attempt)

        print(f"  {attempt['attempt']:>3} {status:>10} {attempt['trades_taken']:>8} "
              f"{co_count:>5} {db_count:>5} "
              f"${attempt['profit']:>11,.0f} ${attempt['peak_balance']:>11,.0f} "
              f"${attempt['final_balance']:>11,.0f}")

        if status == "INCOMPLETE":
            break

    # Summary
    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    failed = sum(1 for a in attempts if a["status"] == "FAILED")
    incomplete = sum(1 for a in attempts if a["status"] == "INCOMPLETE")
    total = passed + failed
    pass_rate = passed / total * 100 if total > 0 else 0

    print(f"{'='*90}")
    print(f"  Total attempts: {total + incomplete} | "
          f"Passed: {passed} | Failed: {failed} | Incomplete: {incomplete}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    if passed > 0:
        avg_trades = np.mean([a["trades_taken"] for a in attempts if a["status"] == "PASSED"])
        print(f"  Avg trades to pass: {avg_trades:.1f}")
    if failed > 0:
        avg_trades_fail = np.mean([a["trades_taken"] for a in attempts if a["status"] == "FAILED"])
        print(f"  Avg trades to fail: {avg_trades_fail:.1f}")
    print(f"{'='*90}")

    return attempts


# ---------------------------------------------------------------------------
# STEP 3b: Single-strategy Phidias simulation (for comparison)
# ---------------------------------------------------------------------------

def run_single_phidias(trades_df, es_close_series, label):
    """Run Phidias sim on a single strategy's trades for comparison."""
    attempts = []
    i = 0

    while i < len(trades_df):
        balance = PHIDIAS_CAPITAL
        high_water = balance
        start_trade = i
        status = "in_progress"

        while i < len(trades_df):
            trade = trades_df.iloc[i]
            daily_pnls = compute_daily_pnls(trade, es_close_series)

            breached = False
            for date, day_pnl in daily_pnls:
                balance += day_pnl
                high_water = max(high_water, balance)
                if high_water - balance >= PHIDIAS_EOD_DRAWDOWN:
                    breached = True
                    break

            profit = balance - PHIDIAS_CAPITAL
            i += 1

            if breached:
                status = "FAILED"
                break
            if profit >= PHIDIAS_PROFIT_TARGET:
                status = "PASSED"
                break

        if status == "in_progress":
            status = "INCOMPLETE"
        attempts.append({
            "status": status,
            "trades_taken": i - start_trade,
            "profit": balance - PHIDIAS_CAPITAL,
        })
        if status == "INCOMPLETE":
            break

    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    failed = sum(1 for a in attempts if a["status"] == "FAILED")
    total = passed + failed
    pass_rate = passed / total * 100 if total > 0 else 0
    avg_trades_pass = np.mean([a["trades_taken"] for a in attempts if a["status"] == "PASSED"]) if passed > 0 else float("nan")
    avg_profit_pass = np.mean([a["profit"] for a in attempts if a["status"] == "PASSED"]) if passed > 0 else float("nan")

    return {
        "label": label,
        "total_attempts": len(attempts),
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "avg_trades_to_pass": avg_trades_pass,
        "avg_profit_when_passed": avg_profit_pass,
    }


# ---------------------------------------------------------------------------
# STEP 4: Comparison table
# ---------------------------------------------------------------------------

def print_comparison(co_stats, db_stats, comb_stats):
    """Print side-by-side comparison table."""

    print(f"\n{'='*90}")
    print(f"  COMPARISON TABLE: Phidias $50K Swing Evaluation")
    print(f"{'='*90}")
    print(f"  {'Metric':<30} {'Chosen One':>16} {'Dip Buyer':>16} {'Combined':>16}")
    print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*16}")

    def fmt_pct(v):
        return f"{v:.1f}%" if not np.isnan(v) else "n/a"

    def fmt_f(v):
        return f"{v:.1f}" if not np.isnan(v) else "n/a"

    def fmt_d(v):
        return f"${v:,.0f}" if not np.isnan(v) else "n/a"

    print(f"  {'Pass rate':<30} {fmt_pct(co_stats['pass_rate']):>16} "
          f"{fmt_pct(db_stats['pass_rate']):>16} {fmt_pct(comb_stats['pass_rate']):>16}")
    print(f"  {'Avg trades to pass':<30} {fmt_f(co_stats['avg_trades_to_pass']):>16} "
          f"{fmt_f(db_stats['avg_trades_to_pass']):>16} {fmt_f(comb_stats['avg_trades_to_pass']):>16}")
    print(f"  {'Total attempts':<30} {co_stats['total_attempts']:>16} "
          f"{db_stats['total_attempts']:>16} {comb_stats['total_attempts']:>16}")
    print(f"  {'Avg profit when passed':<30} {fmt_d(co_stats['avg_profit_when_passed']):>16} "
          f"{fmt_d(db_stats['avg_profit_when_passed']):>16} {fmt_d(comb_stats['avg_profit_when_passed']):>16}")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# STEP 5: Yearly combined P&L
# ---------------------------------------------------------------------------

def print_yearly_pnl(co_trades, db_trades):
    """Print year-by-year combined P&L table."""

    print(f"\n{'='*90}")
    print(f"  YEARLY COMBINED P&L")
    print(f"{'='*90}")

    co_yearly = co_trades.groupby(co_trades["entry_date"].dt.year).agg(
        CO_Trades=("pnl", "count"),
        CO_PnL=("pnl", "sum"),
    )
    db_yearly = db_trades.groupby(db_trades["entry_date"].dt.year).agg(
        DB_Trades=("pnl", "count"),
        DB_PnL=("pnl", "sum"),
    )

    all_years = sorted(set(co_yearly.index) | set(db_yearly.index))

    print(f"  {'Year':>6} {'CO_Tr':>7} {'CO_PnL':>12} {'DB_Tr':>7} {'DB_PnL':>12} "
          f"{'Combined':>12} {'Flag':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*12} {'-'*7} {'-'*12} {'-'*12} {'-'*8}")

    total_co_trades = 0
    total_db_trades = 0
    total_co_pnl = 0.0
    total_db_pnl = 0.0

    for year in all_years:
        co_tr = int(co_yearly.loc[year, "CO_Trades"]) if year in co_yearly.index else 0
        co_pnl = float(co_yearly.loc[year, "CO_PnL"]) if year in co_yearly.index else 0.0
        db_tr = int(db_yearly.loc[year, "DB_Trades"]) if year in db_yearly.index else 0
        db_pnl = float(db_yearly.loc[year, "DB_PnL"]) if year in db_yearly.index else 0.0
        combined = co_pnl + db_pnl
        flag = "<<< NEG" if combined < 0 else ""

        total_co_trades += co_tr
        total_db_trades += db_tr
        total_co_pnl += co_pnl
        total_db_pnl += db_pnl

        print(f"  {year:>6} {co_tr:>7} {co_pnl:>12,.0f} {db_tr:>7} {db_pnl:>12,.0f} "
              f"{combined:>12,.0f} {flag:>8}")

    total_combined = total_co_pnl + total_db_pnl
    print(f"  {'-'*6} {'-'*7} {'-'*12} {'-'*7} {'-'*12} {'-'*12}")
    print(f"  {'TOTAL':>6} {total_co_trades:>7} {total_co_pnl:>12,.0f} "
          f"{total_db_trades:>7} {total_db_pnl:>12,.0f} {total_combined:>12,.0f}")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# STEP 6: Statistics
# ---------------------------------------------------------------------------

def print_statistics(co_trades, db_trades, combined, es):
    """Print combined statistics."""

    print(f"\n{'='*90}")
    print(f"  COMBINED PORTFOLIO STATISTICS")
    print(f"{'='*90}")

    all_pnls = combined["pnl"].values
    co_pnls = co_trades["pnl"].values
    db_pnls = db_trades["pnl"].values

    # Combined total trades
    print(f"  Total trades:          {len(all_pnls)}")
    print(f"    Chosen One:          {len(co_pnls)}")
    print(f"    Dip Buyer:           {len(db_pnls)}")

    # Combined PF
    gains = all_pnls[all_pnls > 0].sum()
    losses = abs(all_pnls[all_pnls < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
    print(f"  Combined PF:           {pf_str}")

    # Combined win rate
    wr = (all_pnls > 0).sum() / len(all_pnls) if len(all_pnls) > 0 else 0
    print(f"  Combined win rate:     {wr:.1%}")

    # Monthly correlation between CO and DB returns
    def monthly_pnl(trades_df):
        df = trades_df[["entry_date", "pnl"]].copy()
        df["month"] = df["entry_date"].dt.to_period("M")
        return df.groupby("month")["pnl"].sum()

    co_monthly = monthly_pnl(co_trades)
    db_monthly = monthly_pnl(db_trades)
    common_months = co_monthly.index.intersection(db_monthly.index)

    if len(common_months) >= 3:
        corr = co_monthly.loc[common_months].corr(db_monthly.loc[common_months])
        print(f"  Monthly return corr:   {corr:.4f} ({len(common_months)} common months)")
    else:
        print(f"  Monthly return corr:   n/a (only {len(common_months)} common months)")

    # Max combined drawdown from equity peak
    cum_pnl = np.cumsum(all_pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0
    print(f"  Max drawdown:          ${max_dd:,.2f}")

    # Longest combined losing streak
    streak = 0
    max_streak = 0
    for pnl in all_pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Longest losing streak: {max_streak} trades")

    # Time in market
    trading_days = es.index[(es.index >= DATA_START)]
    total_days = len(trading_days)
    in_market = pd.Series(False, index=trading_days)

    for _, trade in combined.iterrows():
        mask = (trading_days >= trade["entry_date"]) & (trading_days <= trade["exit_date"])
        in_market[mask] = True

    pct_in_market = in_market.sum() / total_days * 100 if total_days > 0 else 0
    print(f"  Time in market:        {pct_in_market:.1f}% ({in_market.sum()} of {total_days} trading days)")

    # Additional stats
    print(f"\n  Avg trade P&L:         ${all_pnls.mean():,.2f}")
    print(f"  Best trade:            ${all_pnls.max():,.2f}")
    print(f"  Worst trade:           ${all_pnls.min():,.2f}")
    avg_win = all_pnls[all_pnls > 0].mean() if (all_pnls > 0).any() else 0
    avg_loss = all_pnls[all_pnls < 0].mean() if (all_pnls < 0).any() else 0
    print(f"  Avg win:               ${avg_win:,.2f}")
    print(f"  Avg loss:              ${avg_loss:,.2f}")
    print(f"  Total P&L:             ${all_pnls.sum():,.2f}")

    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"\n{'='*90}")
    print(f"  COMBINED PORTFOLIO: Chosen One + Dip Buyer")
    print(f"  Phidias $50K Swing Evaluation Simulator")
    print(f"{'='*90}")

    print("\nLoading market data...")
    es = get_data("ES", start="2000-01-01")
    vix = get_data("VIX", start="2000-01-01")
    es_close = es["Close"].astype(float)
    print(f"  ES:  {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"  VIX: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    # STEP 1: Load trades
    print(f"\n--- STEP 1: Load Trade Lists ---")
    co_trades, db_trades = load_trades(es, vix)

    # STEP 2: Merge and validate
    print(f"\n--- STEP 2: Merge and Validate ---")
    combined = merge_and_validate(co_trades, db_trades)

    # STEP 3: Combined Phidias simulation
    print(f"\n--- STEP 3: Combined Phidias Simulation ---")
    comb_attempts = run_phidias_simulation(combined, es_close)

    # Also run single-strategy simulations for comparison
    co_sorted = co_trades.sort_values("entry_date").reset_index(drop=True)
    db_sorted = db_trades.sort_values("entry_date").reset_index(drop=True)
    co_phidias = run_single_phidias(co_sorted, es_close, "Chosen One")
    db_phidias = run_single_phidias(db_sorted, es_close, "Dip Buyer")

    # Build combined stats for comparison
    comb_passed = [a for a in comb_attempts if a["status"] == "PASSED"]
    comb_failed = [a for a in comb_attempts if a["status"] == "FAILED"]
    comb_total = len(comb_passed) + len(comb_failed)
    comb_phidias = {
        "label": "Combined",
        "total_attempts": len(comb_attempts),
        "passed": len(comb_passed),
        "failed": len(comb_failed),
        "pass_rate": len(comb_passed) / comb_total * 100 if comb_total > 0 else 0,
        "avg_trades_to_pass": np.mean([a["trades_taken"] for a in comb_passed]) if comb_passed else float("nan"),
        "avg_profit_when_passed": np.mean([a["profit"] for a in comb_passed]) if comb_passed else float("nan"),
    }

    # STEP 4: Comparison table
    print(f"\n--- STEP 4: Comparison Table ---")
    print_comparison(co_phidias, db_phidias, comb_phidias)

    # STEP 5: Yearly P&L
    print(f"\n--- STEP 5: Yearly Combined P&L ---")
    print_yearly_pnl(co_trades, db_trades)

    # STEP 6: Statistics
    print(f"\n--- STEP 6: Combined Statistics ---")
    print_statistics(co_trades, db_trades, combined, es)


if __name__ == "__main__":
    run()
