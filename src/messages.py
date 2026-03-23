"""Human-readable Telegram message templates for ES alerts."""


def friday_message(week_number: int, monday_date: str) -> str:
    return (
        f"\u26a0\ufe0f ES Alert \u2014 Week {week_number} trade possible\n\n"
        f"Monday {monday_date}: buy ES at open, sell Friday close.\n"
        f"VIX check will run Monday morning."
    )


def friday_holiday_message(week_number: int, monday_date: str) -> str:
    return (
        f"\u26a0\ufe0f ES Alert \u2014 Week {week_number} \u2014 NO TRADE\n\n"
        f"Monday {monday_date} is a US market holiday.\n"
        f"ES is closed \u2014 skipping this week."
    )


def monday_trade_message(vix: float, monday_date: str, week_number: int) -> str:
    return (
        f"\u2705 ES \u2014 TRADE THIS WEEK\n\n"
        f"Week {week_number} | Monday {monday_date}\n"
        f"VIX Friday close: {vix:.2f} \u2014 outside skip zone (15.0\u201320.0)\n\n"
        f"Buy ES at open today. Sell Friday close."
    )


def monday_skip_message(vix: float, monday_date: str, week_number: int) -> str:
    return (
        f"\ud83d\udeab ES \u2014 SKIP THIS WEEK\n\n"
        f"Week {week_number} | Monday {monday_date}\n"
        f"VIX Friday close: {vix:.2f} \u2014 inside skip zone (15.0\u201320.0)\n\n"
        f"No trade this week."
    )


def error_message(exception: Exception) -> str:
    return f"\ud83d\udd34 ES Alert system error:\n\n{exception}"
