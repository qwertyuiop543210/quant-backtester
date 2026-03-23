"""VIX data fetching with staleness check.

ASSUMPTION (no backtest code to verify):
  - Uses yfinance ^VIX unadjusted Close (not Adj Close)
  - "Friday close" = the Close value on the most recent trading Friday
"""

from datetime import date, timedelta

import yfinance as yf


class StaleDataError(Exception):
    """Raised when the VIX data is too old or not from a Friday."""


def get_latest_friday_vix() -> tuple[float, date]:
    """Fetch the most recent Friday VIX closing price.

    Returns:
        (vix_close, as_of_date) where as_of_date is a Friday within the last
        7 calendar days.

    Raises:
        StaleDataError: if the data is not from a recent Friday.
        RuntimeError: if yfinance returns no data.
    """
    ticker = yf.Ticker("^VIX")
    # Fetch the last 10 calendar days to be safe around holidays
    hist = ticker.history(period="10d")

    if hist.empty:
        raise RuntimeError("yfinance returned no VIX data")

    # Get the last row
    last_date = hist.index[-1].date()
    last_close = float(hist["Close"].iloc[-1])

    # Find the most recent Friday in the data
    friday_rows = hist[hist.index.weekday == 4]  # 4 = Friday
    if friday_rows.empty:
        raise StaleDataError(
            f"No Friday data found in the last 10 days. "
            f"Most recent data is from {last_date} (not a Friday)."
        )

    friday_date = friday_rows.index[-1].date()
    friday_close = float(friday_rows["Close"].iloc[-1])

    # Staleness check: the Friday must be within the last 7 calendar days
    today = date.today()
    days_old = (today - friday_date).days
    if days_old > 7:
        raise StaleDataError(
            f"VIX data is from {friday_date}, which is {days_old} days old. "
            f"Maximum allowed age is 7 calendar days."
        )

    return friday_close, friday_date
