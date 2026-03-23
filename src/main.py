"""Entrypoint for the ES alert system — called by Railway cron.

All datetime operations use America/New_York. Never naive UTC.
"""

import logging
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.alert_logic import friday_alert, monday_alert
from src.messages import error_message
from src.telegram import send_message
from src.week_filter import is_trading_day

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def main() -> None:
    today: date = datetime.now(tz=ET).date()
    weekday = today.weekday()  # 0=Mon, 4=Fri

    logger.info("ES alert system started. Today is %s (weekday=%d).", today, weekday)

    if not is_trading_day(today):
        logger.info("%s is not a NYSE trading day. Exiting cleanly.", today)
        return

    if weekday == 4:  # Friday
        friday_alert(today)
    elif weekday == 0:  # Monday
        monday_alert(today)
    else:
        logger.info("No alert scheduled today (%s). Exiting cleanly.", today)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Unhandled exception in alert system.")
        try:
            send_message(error_message(exc))
        except Exception:
            logger.exception("Failed to send error alert via Telegram.")
        sys.exit(1)
