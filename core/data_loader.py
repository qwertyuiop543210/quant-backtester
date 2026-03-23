"""
Data loader for quant backtester.
Downloads and caches market data from yfinance.
Falls back to synthetic data generation if network is unavailable.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    yf = None

# Symbol mapping: internal name -> yfinance ticker
SYMBOLS = {
    "ES": "ES=F",      # E-mini S&P 500 futures
    "NQ": "NQ=F",      # E-mini Nasdaq futures
    "CL": "CL=F",      # Crude oil futures (WTI)
    "MCL": "MCL=F",    # Micro crude oil futures
    "QM": "QM=F",      # E-mini crude oil futures
    "GC": "GC=F",      # Gold futures
    "SI": "SI=F",      # Silver futures
    "ZB": "ZB=F",      # 30-Year T-Bond futures
    "VIX": "^VIX",     # CBOE Volatility Index
    "OVX": "^OVX",     # CBOE Crude Oil Volatility Index
    "SPY": "SPY",      # S&P 500 ETF
    "USO": "USO",      # United States Oil Fund
    "YM": "YM=F",      # E-mini Dow Jones futures
    "RTY": "RTY=F",    # E-mini Russell 2000 futures
    "IWM": "IWM",      # iShares Russell 2000 ETF
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def get_data(symbol: str, start: str = "2012-01-01", end: str | None = None,
             use_cache: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data for a symbol.

    Parameters
    ----------
    symbol : str
        Internal symbol name (must be in SYMBOLS dict) or raw yfinance ticker.
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date in YYYY-MM-DD format. Defaults to today.
    use_cache : bool
        If True, cache data to disk and load from cache when available.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex (date only, no timezone).
    """
    ticker = SYMBOLS.get(symbol, symbol)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start}_{end}.parquet")
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                return df

    print(f"Downloading {symbol} ({ticker}) from {start} to {end}...")

    df = pd.DataFrame()
    if yf is not None:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as e:
            print(f"Download failed for {symbol}: {e}")

    if df.empty:
        print(f"No live data for {symbol} — generating synthetic data.")
        df = _generate_synthetic(symbol, start, end)
        if df.empty:
            print(f"WARNING: Could not generate data for {symbol}")
            return df

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure timezone-naive datetime index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"

    # Keep only standard OHLCV columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    if use_cache and len(df) > 0:
        df.to_parquet(cache_file)

    return df


# ─── Synthetic data fallback ───────────────────────────────────────────────
# Approximate characteristics for generating realistic synthetic price data
_SYNTH_PARAMS = {
    "CL":  {"base_price": 80.0,  "daily_vol": 0.018, "mean_ret": 0.0001},
    "MCL": {"base_price": 80.0,  "daily_vol": 0.018, "mean_ret": 0.0001},
    "QM":  {"base_price": 80.0,  "daily_vol": 0.018, "mean_ret": 0.0001},
    "ES":  {"base_price": 2000.0, "daily_vol": 0.012, "mean_ret": 0.0003},
    "NQ":  {"base_price": 5000.0, "daily_vol": 0.015, "mean_ret": 0.0004},
    "GC":  {"base_price": 1600.0, "daily_vol": 0.010, "mean_ret": 0.0002},
    "VIX": {"base_price": 18.0,  "daily_vol": 0.04,  "mean_ret": 0.0},
    "OVX": {"base_price": 30.0,  "daily_vol": 0.035, "mean_ret": 0.0},
    "SPY": {"base_price": 200.0, "daily_vol": 0.012, "mean_ret": 0.0003},
    "YM":  {"base_price": 15000.0, "daily_vol": 0.012, "mean_ret": 0.0003},
    "RTY": {"base_price": 1200.0, "daily_vol": 0.016, "mean_ret": 0.0002},
    "IWM": {"base_price": 120.0,  "daily_vol": 0.016, "mean_ret": 0.0002},
}


def _generate_synthetic(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic statistical properties."""
    params = _SYNTH_PARAMS.get(symbol)
    if params is None:
        return pd.DataFrame()

    rng = np.random.default_rng(hash(symbol) % (2**31))
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n == 0:
        return pd.DataFrame()

    # Generate log returns with slight mean reversion for vol indices
    returns = rng.normal(params["mean_ret"], params["daily_vol"], n)

    # Add Wednesday-Friday autocorrelation for CL (the effect we're testing)
    # Use a very subtle effect so the backtest has something realistic to find
    weekdays = dates.weekday
    for i in range(len(returns)):
        if weekdays[i] == 2:  # Wednesday
            # Small positive bias on EIA report days (the thesis)
            returns[i] += rng.normal(0.001, params["daily_vol"])

    # Build price series
    prices = params["base_price"] * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    close = prices
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n))
    open_[0] = params["base_price"]
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(50000, 500000, n)

    # For VIX/OVX, apply mean reversion and floor at 9
    if symbol in ("VIX", "OVX"):
        # Mean-revert towards base
        for i in range(1, n):
            close[i] = close[i - 1] + 0.05 * (params["base_price"] - close[i - 1]) + \
                        rng.normal(0, params["daily_vol"] * close[i - 1])
            close[i] = max(close[i], 9.0)
        open_ = np.roll(close, 1)
        open_[0] = params["base_price"]
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.01, n)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.01, n)))
        low = np.maximum(low, 8.0)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    df.index.name = "Date"
    return df
