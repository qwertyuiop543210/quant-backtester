"""Tests for src/week_filter.py."""

from datetime import date

import pytest

from src.week_filter import (
    get_this_weeks_monday,
    is_active_week,
    is_trading_day,
    week_of_month,
)


class TestWeekOfMonth:
    """Week number is based on the Monday's day-of-month: (day - 1) // 7 + 1."""

    def test_first_day_of_month(self):
        # Monday 2025-09-01 -> day 1 -> week 1
        assert week_of_month(date(2025, 9, 1)) == 1

    def test_last_day_of_week_1(self):
        # Monday 2025-09-07 is not possible (Sep 7 2025 is Sunday)
        # Monday with day=7 -> week 1
        assert week_of_month(date(2025, 7, 7)) == 1

    def test_first_day_of_week_2(self):
        # day=8 -> week 2
        assert week_of_month(date(2025, 9, 8)) == 2

    def test_first_day_of_week_4(self):
        # Monday 2025-09-22 -> day 22 -> week 4
        assert week_of_month(date(2025, 9, 22)) == 4

    def test_last_day_of_week_4(self):
        # day=28 -> week 4
        assert week_of_month(date(2025, 7, 28)) == 4

    def test_first_day_of_week_5(self):
        # day=29 -> week 5
        assert week_of_month(date(2025, 9, 29)) == 5


class TestIsActiveWeek:
    def test_week_1_is_active(self):
        # Monday 2025-09-01 -> week 1
        assert is_active_week(date(2025, 9, 1)) is True

    def test_week_2_is_not_active(self):
        # Monday 2025-09-08 -> week 2
        assert is_active_week(date(2025, 9, 8)) is False

    def test_week_3_is_not_active(self):
        # Monday 2025-09-15 -> week 3
        assert is_active_week(date(2025, 9, 15)) is False

    def test_week_4_is_active(self):
        # Monday 2025-09-22 -> week 4
        assert is_active_week(date(2025, 9, 22)) is True

    def test_week_5_is_not_active(self):
        # Monday 2025-09-29 -> week 5
        assert is_active_week(date(2025, 9, 29)) is False


class TestGetThisWeeksMonday:
    def test_monday_returns_itself(self):
        assert get_this_weeks_monday(date(2025, 9, 1)) == date(2025, 9, 1)

    def test_friday_returns_monday(self):
        # Friday 2025-09-05 -> Monday 2025-09-01
        assert get_this_weeks_monday(date(2025, 9, 5)) == date(2025, 9, 1)

    def test_wednesday_returns_monday(self):
        assert get_this_weeks_monday(date(2025, 9, 3)) == date(2025, 9, 1)


class TestIsTradingDay:
    def test_new_years_day_is_not_trading(self):
        # 2025-01-01 is a NYSE holiday (New Year's Day)
        assert is_trading_day(date(2025, 1, 1)) is False

    def test_normal_monday_is_trading(self):
        # 2025-01-06 is a regular Monday
        assert is_trading_day(date(2025, 1, 6)) is True

    def test_saturday_is_not_trading(self):
        assert is_trading_day(date(2025, 9, 6)) is False

    def test_sunday_is_not_trading(self):
        assert is_trading_day(date(2025, 9, 7)) is False
