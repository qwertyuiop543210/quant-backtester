"""Tests for src/alert_logic.py — should_trade boundary values."""

import pytest

from src.alert_logic import should_trade, VIX_SKIP_LOW, VIX_SKIP_HIGH


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
