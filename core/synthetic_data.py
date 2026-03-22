"""Generate synthetic market data for testing when downloads are unavailable."""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def generate_spy_like(start: str = "1993-01-29", end: str = "2025-12-31",
                      initial_price: float = 44.0, annual_return: float = 0.10,
                      annual_vol: float = 0.18, seed: int = 42) -> pd.DataFrame:
    """Generate realistic SPY-like daily OHLCV data.

    Uses geometric Brownian motion calibrated to historical SPY parameters.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, end=end, freq="B")  # Business days
    n = len(dates)

    daily_mu = annual_return / 252
    daily_sigma = annual_vol / np.sqrt(252)

    log_returns = np.random.normal(daily_mu, daily_sigma, n)

    # Add turn-of-month effect (subtle, ~2bp/day boost in last 2 + first 3 days)
    for i, d in enumerate(dates):
        ym_days = dates[dates.to_period("M") == d.to_period("M")]
        pos_in_month = list(ym_days).index(d)
        n_month = len(ym_days)
        if pos_in_month >= n_month - 2 or pos_in_month < 3:
            log_returns[i] += 0.0002  # Small TOM effect

    close = initial_price * np.exp(np.cumsum(log_returns))

    # Generate OHLV from close
    intraday_vol = daily_sigma * 0.5
    high = close * (1 + np.abs(np.random.normal(0, intraday_vol, n)))
    low = close * (1 - np.abs(np.random.normal(0, intraday_vol, n)))
    open_ = close * (1 + np.random.normal(0, intraday_vol * 0.3, n))
    volume = np.random.lognormal(mean=18, sigma=0.5, size=n).astype(int)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


def ensure_spy_data() -> pd.DataFrame:
    """Try to load cached SPY data; generate synthetic if unavailable."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "SPY.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if len(df) > 0:
            return df

    print("Network unavailable — generating synthetic SPY data for demonstration...")
    df = generate_spy_like()
    df.to_csv(cache_path)
    return df
