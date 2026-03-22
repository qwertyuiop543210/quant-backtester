"""Download and cache market data using yfinance."""

import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SYMBOLS = {
    "SPY": "SPY",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "VIX": "^VIX",
    "GLD": "GLD",
    "SLV": "SLV",
}


def get_data(symbol_key: str, start: str = "1993-01-01", end: str = None,
             force_download: bool = False) -> pd.DataFrame:
    """Download or load cached OHLCV data for a symbol.

    Args:
        symbol_key: Key from SYMBOLS dict (e.g. 'SPY', 'VIX').
        start: Start date string.
        end: End date string (defaults to today).
        force_download: Re-download even if cache exists.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
        Index is DatetimeIndex named 'Date'.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    ticker = SYMBOLS.get(symbol_key, symbol_key)
    cache_path = os.path.join(DATA_DIR, f"{symbol_key}.csv")

    if not force_download and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if len(df) > 0:
            return _clean(df)

    print(f"Downloading {ticker} ...")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if len(df) > 0:
            df.index.name = "Date"
            df.to_csv(cache_path)
            return _clean(df)
    except Exception as e:
        print(f"Download failed: {e}")

    # Fallback to synthetic data if download fails
    if symbol_key == "SPY":
        from core.synthetic_data import ensure_spy_data
        return _clean(ensure_spy_data())

    raise RuntimeError(f"Could not download {ticker} and no synthetic fallback available.")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then drop remaining NaN rows."""
    df = df.ffill()
    df = df.dropna()
    return df


def load_multiple(keys: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Load multiple symbols and return a dict."""
    return {k: get_data(k, **kwargs) for k in keys}
