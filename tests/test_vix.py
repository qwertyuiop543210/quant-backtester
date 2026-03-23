"""Tests for src/vix.py — VIX fetcher with staleness check."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.vix import StaleDataError, get_latest_friday_vix


def _make_hist(dates: list[str], closes: list[float]) -> pd.DataFrame:
    """Helper to build a DataFrame mimicking yfinance history output."""
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"Close": closes}, index=idx)


class TestGetLatestFridayVix:
    @patch("src.vix.yf.Ticker")
    @patch("src.vix.date")
    def test_recent_friday_returns_correctly(self, mock_date, mock_ticker_cls):
        """A Friday within the last 7 days should pass."""
        # Suppose today is Monday 2025-09-08, last Friday was 2025-09-05
        mock_date.today.return_value = date(2025, 9, 8)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        hist = _make_hist(
            ["2025-09-03", "2025-09-04", "2025-09-05"],
            [18.0, 17.5, 16.3],
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_ticker_cls.return_value = mock_ticker

        vix_close, vix_date = get_latest_friday_vix()
        assert vix_close == 16.3
        assert vix_date == date(2025, 9, 5)

    @patch("src.vix.yf.Ticker")
    @patch("src.vix.date")
    def test_stale_date_raises(self, mock_date, mock_ticker_cls):
        """A Friday older than 7 days should raise StaleDataError."""
        mock_date.today.return_value = date(2025, 9, 15)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        hist = _make_hist(
            ["2025-09-03", "2025-09-04", "2025-09-05"],
            [18.0, 17.5, 16.3],
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(StaleDataError, match="days old"):
            get_latest_friday_vix()

    @patch("src.vix.yf.Ticker")
    def test_no_friday_in_data_raises(self, mock_ticker_cls):
        """If no Friday exists in the data, raise StaleDataError."""
        # Only Mon-Thu data
        hist = _make_hist(
            ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04"],
            [18.0, 17.5, 16.3, 15.0],
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(StaleDataError, match="No Friday data"):
            get_latest_friday_vix()

    @patch("src.vix.yf.Ticker")
    def test_empty_data_raises_runtime_error(self, mock_ticker_cls):
        """Empty DataFrame should raise RuntimeError."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(RuntimeError, match="no VIX data"):
            get_latest_friday_vix()
