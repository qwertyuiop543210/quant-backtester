"""Tests for src/alert_logic.py — should_trade boundary values and alert routing."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from src.alert_logic import should_trade, friday_alert, VIX_SKIP_LOW, VIX_SKIP_HIGH


class TestShouldTrade:
    """VIX skip zone: 15.0 <= vix <= 20.0 inclusive on both ends."""

    def test_below_skip_zone(self):
        assert should_trade(14.99) is True

    def test_lower_boundary_inclusive(self):
        assert should_trade(15.0) is False

    def test_middle_of_skip_zone(self):
        assert should_trade(17.5) is False

    def test_upper_boundary_inclusive(self):
        assert should_trade(20.0) is False

    def test_above_skip_zone(self):
        assert should_trade(20.01) is True

    def test_well_below(self):
        assert should_trade(10.0) is True

    def test_well_above(self):
        assert should_trade(35.0) is True

    def test_boundaries_match_constants(self):
        assert VIX_SKIP_LOW == 15.0
        assert VIX_SKIP_HIGH == 20.0


class TestFridayAlertWeekCheck:
    """Friday alert must check if NEXT Monday is in an active week, not this week's Monday."""

    @patch("src.alert_logic.send_message")
    @patch("src.alert_logic.already_sent", return_value=False)
    @patch("src.alert_logic.mark_sent")
    @patch("src.alert_logic.is_trading_day", return_value=True)
    def test_friday_before_week4_monday_fires(self, mock_td, mock_ms, mock_as, mock_send):
        # Friday 2025-09-19 (Week 3) → next Monday 2025-09-22 (day 22, Week 4 = active)
        friday_alert(date(2025, 9, 19))
        mock_send.assert_called_once()

    @patch("src.alert_logic.send_message")
    @patch("src.alert_logic.already_sent", return_value=False)
    def test_friday_of_week1_no_alert(self, mock_as, mock_send):
        # Friday 2025-09-05 (Week 1) → next Monday 2025-09-08 (day 8, Week 2 = not active)
        friday_alert(date(2025, 9, 5))
        mock_send.assert_not_called()

    @patch("src.alert_logic.send_message")
    @patch("src.alert_logic.already_sent", return_value=False)
    @patch("src.alert_logic.mark_sent")
    @patch("src.alert_logic.is_trading_day", return_value=True)
    def test_friday_before_week1_monday_fires(self, mock_td, mock_ms, mock_as, mock_send):
        # Friday 2025-08-29 (Week 5 of Aug) → next Monday 2025-09-01 (day 1, Week 1 = active)
        friday_alert(date(2025, 8, 29))
        mock_send.assert_called_once()
