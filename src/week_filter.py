"""Week-of-month logic and holiday detection.

ASSUMPTION (no backtest code to verify — taken from spec):
  - Week 1 = Monday's day-of-month falls on days 1–7
  - Week 4 = Monday's day-of-month falls on days 22–28
  - Active weeks: {1, 4}

ASSUMPTION (holiday handling not in backtest — Option A):
  - If Monday is a US market holiday, skip the entire week (no Tuesday roll).
"""

from datetime import date, timedelta

import pandas_market_calendars as mcal

ACTIVE_WEEKS = {1, 4}


def week_of_month(monday: date) -> int:
    """Return the week number (1-5) based on the Monday's day of month.

    Week 1: days 1–7
    Week 2: days 8–14
    Week 3: days 15–21
    Week 4: days 22–28
    Week 5: days 29–31
    """
    day = monday.day
    return (day - 1) // 7 + 1


def is_active_week(monday: date) -> bool:
    """Return True if the given Monday falls in an active trading week.

    The Monday's day-of-month determines the week number.
    Only weeks in ACTIVE_WEEKS ({1, 4}) are active.
    """
    return week_of_month(monday) in ACTIVE_WEEKS


def get_this_weeks_monday(today: date) -> date:
    """Return the Monday of the current ISO week."""
    return today - timedelta(days=today.weekday())


def is_trading_day(d: date) -> bool:
    """Return True if the given date is a NYSE trading day.

    Returns False for weekends, US market holidays, and any day the NYSE
    is closed.
    """
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=d.isoformat(), end_date=d.isoformat()
    )
    return len(schedule) > 0
