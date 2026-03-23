"""Download and cache market data using yfinance, pandas_datareader, or direct URL."""

import io
import os
import urllib.request

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SYMBOLS = {
    "SPY": "SPY",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "RTY": "RTY=F",
    "ZB": "ZB=F",
    "GC": "GC=F",
    "VIX": "^VIX",
    "GLD": "GLD",
    "SLV": "SLV",
}

YAHOO_DIRECT_URL = (
    "https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    "?period1=0&period2=9999999999&interval=1d&events=history"
)


def get_data(symbol_key: str, start: str = "1993-01-01", end: str = None,
             force_download: bool = False) -> pd.DataFrame:
    """Download or load cached OHLCV data for a symbol.

    Tries in order: cache, yfinance, pandas_datareader, direct Yahoo URL.
    Raises RuntimeError if all methods fail.

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

    # Method 1: yfinance
    # auto_adjust=False for futures — auto_adjust corrupts continuous contract prices
    print(f"Downloading {ticker} via yfinance ...")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if len(df) > 0:
            # Use Close (not Adj Close) for futures; drop Adj Close if present
            if "Adj Close" in df.columns:
                df = df.drop(columns=["Adj Close"])
            df.index.name = "Date"
            df.to_csv(cache_path)
            return _clean(df)
    except Exception as e:
        print(f"  yfinance failed: {e}")

    # Method 2: pandas_datareader
    print(f"Downloading {ticker} via pandas_datareader ...")
    try:
        import pandas_datareader.data as web
        import datetime
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d") if end else datetime.datetime.now()
        df = web.DataReader(ticker, "yahoo", start_dt, end_dt)
        if len(df) > 0:
            df.index.name = "Date"
            df.to_csv(cache_path)
            return _clean(df)
    except Exception as e:
        print(f"  pandas_datareader failed: {e}")

    # Method 3: direct Yahoo Finance URL
    print(f"Downloading {ticker} via direct URL ...")
    try:
        url = YAHOO_DIRECT_URL.format(ticker=ticker)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), index_col="Date", parse_dates=True)
        if len(df) > 0:
            df.to_csv(cache_path)
            return _clean(df)
    except Exception as e:
        print(f"  Direct URL failed: {e}")

    raise RuntimeError(
        f"Could not download {ticker}. All methods failed (yfinance, pandas_datareader, direct URL). "
        f"Check network connectivity or place a CSV manually in {cache_path}."
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then drop remaining NaN rows."""
    df = df.ffill()
    df = df.dropna()
    return df


def load_multiple(keys: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Load multiple symbols and return a dict."""
    return {k: get_data(k, **kwargs) for k in keys}
