"""Friday and Monday alert decision logic.

ASSUMPTION (no backtest code to verify — taken from spec):
  - VIX skip zone: 15.0 <= vix_close <= 20.0 inclusive on both ends
  - "Friday close" = unadjusted Close from yfinance ^VIX
"""

import logging
from datetime import date, timedelta

from src.messages import (
    friday_holiday_message,
    friday_message,
    monday_skip_message,
    monday_trade_message,
)
from src.state import already_sent, mark_sent
from src.telegram import send_message
from src.vix import get_latest_friday_vix
from src.week_filter import (
    get_this_weeks_monday,
    is_active_week,
    is_trading_day,
    week_of_month,
)

logger = logging.getLogger(__name__)

VIX_SKIP_LOW = 15.0
VIX_SKIP_HIGH = 20.0


def should_trade(vix_close: float) -> bool:
    """Return True if the VIX close is outside the skip zone.

    Skip zone: 15.0 <= vix_close <= 20.0 inclusive.
    """
    return not (VIX_SKIP_LOW <= vix_close <= VIX_SKIP_HIGH)


def friday_alert(today: date) -> None:
    """Send a Friday heads-up alert if this is an active trading week.

    If the coming Monday is a market holiday, send a holiday warning instead.
    """
    if today.weekday() != 4:  # 4 = Friday
        logger.info("friday_alert called on a non-Friday (%s), no-op.", today)
        return

    monday = get_this_weeks_monday(today)
    if not is_active_week(monday):
        logger.info("Week of %s is not active, no-op.", monday)
        return

    wk = week_of_month(monday)
    next_monday = today + timedelta(days=3)
    alert_key = f"friday-{next_monday.isoformat()}"

    if already_sent(alert_key):
        logger.info("Alert %s already sent, skipping.", alert_key)
        return

    if not is_trading_day(next_monday):
        # ASSUMPTION: Monday is a holiday — skip the entire week (Option A).
        # The backtest does not handle holidays; this is a conservative choice.
        msg = friday_holiday_message(wk, next_monday.strftime("%Y-%m-%d"))
        send_message(msg)
        mark_sent(alert_key)
        logger.info("Sent Friday holiday alert for %s.", next_monday)
    else:
        msg = friday_message(wk, next_monday.strftime("%Y-%m-%d"))
        send_message(msg)
        mark_sent(alert_key)
        logger.info("Sent Friday heads-up alert for %s.", next_monday)


def monday_alert(today: date) -> None:
    """Send a Monday trade/skip confirmation based on Friday's VIX close."""
    if today.weekday() != 0:  # 0 = Monday
        logger.info("monday_alert called on a non-Monday (%s), no-op.", today)
        return

    if not is_active_week(today):
        logger.info("Week of %s is not active, no-op.", today)
        return

    alert_key = f"monday-{today.isoformat()}"

    if already_sent(alert_key):
        logger.info("Alert %s already sent, skipping.", alert_key)
        return

    vix_close, vix_date = get_latest_friday_vix()
    wk = week_of_month(today)
    monday_str = today.strftime("%Y-%m-%d")

    if should_trade(vix_close):
        msg = monday_trade_message(vix_close, monday_str, wk)
    else:
        msg = monday_skip_message(vix_close, monday_str, wk)

    send_message(msg)
    mark_sent(alert_key)
    logger.info(
        "Sent Monday alert for %s (VIX=%.2f, trade=%s).",
        today, vix_close, should_trade(vix_close),
    )
