"""Shared utilities for the three validation test scripts.

Provides constants, trade generators (Chosen One + Dip Buyer), metric helpers,
and data-loading convenience functions used by test1, test2, and test3.
"""

import numpy as np
import pandas as pd
from core.data_loader import get_data

# ── Constants ─────────────────────────────────────────────────────────────────

ES_POINT_VALUE = 50.0
COST_PER_TRADE = 30.0  # $5 RT commission + $12.50 slippage × 2 sides

ACTIVE_WEEKS = {1, 4}

VIX_SKIP_LO = 15.0
VIX_SKIP_HI = 20.0

RSI_ENTRY = 10
RSI_EXIT = 65
VIX_FLOOR = 20.0
VIX_CEILING = 35.0
TIME_STOP = 5


# ── Helper functions ──────────────────────────────────────────────────────────

def get_week_of_month(date) -> int:
    """Return which week of the month a date falls in (1-5)."""
    return (date.day - 1) // 7 + 1


def find_trading_weeks(dates: pd.DatetimeIndex) -> list[dict]:
    """Identify full trading weeks in *dates*.

    A trading week is anchored by a Monday and must reach at least Thursday
    (dayofweek >= 3).  Returns a list of dicts with keys:
        monday_idx, friday_idx, monday_date, friday_date, week_of_month
    where *friday_idx* points to the last trading day of the week (could be
    Thursday if Friday is a holiday).
    """
    weeks = []
    i = 0
    while i < len(dates):
        if dates[i].dayofweek == 0:  # Monday
            monday_idx = i
            monday_date = dates[i]
            iso_week = monday_date.isocalendar()[1]
            # Walk forward through the same ISO week
            j = i
            while j < len(dates) and dates[j].isocalendar()[1] == iso_week:
                j += 1
            last_idx = j - 1  # last trading day in this calendar week
            # Must reach at least Thursday (dayofweek >= 3)
            if dates[last_idx].dayofweek >= 3:
                weeks.append({
                    "monday_idx": monday_idx,
                    "friday_idx": last_idx,
                    "monday_date": monday_date,
                    "friday_date": dates[last_idx],
                    "week_of_month": get_week_of_month(monday_date),
                })
            i = j
        else:
            i += 1
    return weeks


def compute_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Compute Wilder's RSI using ewm (exponential weighted moving average)."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ── Chosen One trade generator ───────────────────────────────────────────────

def generate_co_trades(
    es: pd.DataFrame,
    vix: pd.DataFrame,
    vix_skip_lo: float = 15.0,
    vix_skip_hi: float = 20.0,
    point_value: float = 50.0,
    cost: float = 30.0,
    start_date: str = "2012-01-01",
) -> pd.DataFrame:
    """Generate Chosen One trades: buy Monday open, sell Friday close.

    Only Week 1 and Week 4.  VIX filter uses the last trading day before
    Monday (index mi - 1).  Skip when vix_skip_lo <= vix <= vix_skip_hi
    (inclusive both bounds).  When both *vix_skip_lo* and *vix_skip_hi* are
    None, no VIX filtering is applied.
    """
    es = es.loc[start_date:]
    vix_close = vix["Close"].reindex(es.index).ffill()

    weeks = find_trading_weeks(es.index)
    trades = []

    for w in weeks:
        if w["week_of_month"] not in ACTIVE_WEEKS:
            continue

        mi = w["monday_idx"]
        fi = w["friday_idx"]

        # VIX filter — use last trading day before Monday
        if vix_skip_lo is not None and vix_skip_hi is not None:
            if mi > 0:
                vix_val = float(vix_close.iloc[mi - 1])
            else:
                vix_val = float(vix_close.iloc[mi])
            if vix_skip_lo <= vix_val <= vix_skip_hi:
                continue

        entry_price = float(es["Open"].iloc[mi])
        exit_price = float(es["Close"].iloc[fi])
        pnl_points = exit_price - entry_price
        pnl = pnl_points * point_value - cost

        vix_at_entry = float(vix_close.iloc[mi - 1]) if mi > 0 else float(vix_close.iloc[mi])

        trades.append({
            "entry_date": es.index[mi],
            "exit_date": es.index[fi],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "pnl": pnl,
            "week_of_month": w["week_of_month"],
            "vix_at_entry": vix_at_entry,
            "tag": "CO",
        })

    if not trades:
        return pd.DataFrame(columns=[
            "entry_date", "exit_date", "entry_price", "exit_price",
            "pnl_points", "pnl", "week_of_month", "vix_at_entry", "tag",
        ])
    return pd.DataFrame(trades)


# ── Dip Buyer trade generator ────────────────────────────────────────────────

def generate_db_trades(
    es: pd.DataFrame,
    vix: pd.DataFrame,
    point_value: float = 50.0,
    cost: float = 30.0,
    start_date: str = "2012-01-01",
    rsi_entry: int = 10,
    rsi_exit: int = 65,
    vix_floor: float = 20.0,
    vix_ceiling: float = 35.0,
    time_stop: int = 5,
    overlap_filter: bool = True,
    lookahead_buffer: bool = True,
) -> pd.DataFrame:
    """Generate Dip Buyer trades.

    Signal on day i: 2-day RSI(close) < rsi_entry AND
                     vix_floor < VIX(close) < vix_ceiling.
    Entry: buy at day i+1 open.
    Exit: RSI(close) > rsi_exit OR hold_days >= time_stop.
    """
    es = es.loc[start_date:]
    vix_close = vix["Close"].reindex(es.index).ffill()
    es_close = es["Close"].astype(float)
    es_open = es["Open"].astype(float)
    rsi = compute_rsi(es_close, period=2)
    dates = es.index

    # Build overlap set (indices within W1/W4 weeks)
    overlap_indices = set()
    co_weeks = []
    if overlap_filter or lookahead_buffer:
        weeks = find_trading_weeks(dates)
        co_weeks = [w for w in weeks if w["week_of_month"] in ACTIVE_WEEKS]
        if overlap_filter:
            for w in co_weeks:
                for idx in range(w["monday_idx"], w["friday_idx"] + 1):
                    overlap_indices.add(idx)

    trades = []
    i = 0
    while i < len(es_close) - 1:  # need at least i+1 for entry
        # Signal check on day i
        rsi_val = rsi.iloc[i]
        vix_val = float(vix_close.iloc[i])
        if pd.isna(rsi_val) or rsi_val >= rsi_entry:
            i += 1
            continue
        if vix_val <= vix_floor or vix_val >= vix_ceiling:
            i += 1
            continue

        # Overlap filter: signal day must not fall in a CO week
        if overlap_filter and i in overlap_indices:
            i += 1
            continue

        entry_idx = i + 1

        # Lookahead buffer: need time_stop trading days before next CO Monday
        if lookahead_buffer and co_weeks:
            blocked = False
            for w in co_weeks:
                if w["monday_idx"] > entry_idx:
                    if w["monday_idx"] - entry_idx < time_stop:
                        blocked = True
                    break
            if blocked:
                i += 1
                continue

        # Edge case: not enough data for the trade
        if entry_idx + time_stop - 1 >= len(es_close):
            break

        entry_price = float(es_open.iloc[entry_idx])

        # Exit loop
        j = entry_idx
        hold_days = 1
        exit_reason = ""
        while j < len(es_close):
            rsi_j = rsi.iloc[j]
            if not pd.isna(rsi_j) and rsi_j > rsi_exit:
                exit_reason = "RSI"
                break
            if hold_days >= time_stop:
                exit_reason = "time_stop"
                break
            j += 1
            hold_days += 1

        if j >= len(es_close):
            break

        exit_price = float(es_close.iloc[j])
        pnl_points = exit_price - entry_price
        pnl = pnl_points * point_value - cost
        vix_at_entry = float(vix_close.iloc[i])

        trades.append({
            "entry_date": dates[entry_idx],
            "exit_date": dates[j],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "pnl": pnl,
            "week_of_month": get_week_of_month(dates[entry_idx]),
            "vix_at_entry": vix_at_entry,
            "tag": "DB",
            "hold_days": hold_days,
            "exit_reason": exit_reason,
        })

        # Resume scanning from day after exit
        i = j + 1
        continue

    if not trades:
        cols = [
            "entry_date", "exit_date", "entry_price", "exit_price",
            "pnl_points", "pnl", "week_of_month", "vix_at_entry", "tag",
            "hold_days", "exit_reason",
        ]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(trades)


# ── Combined trade builder ───────────────────────────────────────────────────

def build_combined_trades(
    es: pd.DataFrame,
    vix: pd.DataFrame,
    start_date: str = "2012-01-01",
    co_vix_skip_lo: float = 15.0,
    co_vix_skip_hi: float = 20.0,
    point_value: float = 50.0,
    cost: float = 30.0,
) -> pd.DataFrame:
    """Generate both CO and DB trades, concat, sort by entry_date."""
    co = generate_co_trades(es, vix, vix_skip_lo=co_vix_skip_lo,
                            vix_skip_hi=co_vix_skip_hi,
                            point_value=point_value, cost=cost,
                            start_date=start_date)
    db = generate_db_trades(es, vix, point_value=point_value, cost=cost,
                            start_date=start_date)
    combined = pd.concat([co, db], ignore_index=True)
    combined = combined.sort_values("entry_date").reset_index(drop=True)
    return combined


# ── Metric helpers ────────────────────────────────────────────────────────────

def pf_from_array(pnls: np.ndarray) -> float:
    """Profit factor from a numpy array of P&Ls."""
    if len(pnls) == 0:
        return 0.0
    wins = pnls[pnls > 0].sum()
    losses = np.abs(pnls[pnls < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def wr_from_array(pnls: np.ndarray) -> float:
    """Win rate from a numpy array of P&Ls."""
    if len(pnls) == 0:
        return 0.0
    return float((pnls > 0).sum() / len(pnls))


# ── ASCII histogram ──────────────────────────────────────────────────────────

def print_ascii_histogram(
    values,
    actual: float,
    bins: int = 20,
    width: int = 50,
    label: str = "PF",
) -> None:
    """Print a text histogram with an arrow marking *actual*'s position."""
    values = np.asarray(values, dtype=float)
    # Filter out inf and nan
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"  No finite values to plot for {label}.")
        return

    counts, edges = np.histogram(values, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    print(f"\n  {label} distribution ({len(values)} samples):")
    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * width)
        bar = "#" * bar_len
        lo, hi = edges[i], edges[i + 1]
        # Check if actual falls in this bin
        in_bin = (lo <= actual < hi) or (i == len(counts) - 1 and actual == hi)
        marker = " <-- actual" if in_bin else ""
        print(f"  {lo:8.3f} |{bar}{marker}")
    print(f"  {edges[-1]:8.3f} |")

    # Percentile rank
    rank = (values < actual).sum() / len(values) * 100
    print(f"  Actual {label}: {actual:.3f}  (percentile: {rank:.1f}%)")


# ── Data loader convenience ──────────────────────────────────────────────────

def load_data(start: str = "2012-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ES and VIX data, print date ranges, return (es, vix)."""
    es = get_data("ES", start=start)
    vix = get_data("VIX", start=start)
    print(f"ES  range: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")
    return es, vix
