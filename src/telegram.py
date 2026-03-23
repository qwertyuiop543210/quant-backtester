"""Telegram message delivery."""

import os
import time

import requests


class TelegramError(Exception):
    """Raised when a Telegram message fails to send after retries."""


def send_message(text: str) -> None:
    """Send a message via Telegram Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment variables.
    Retries once on failure with a 5-second delay.

    Raises:
        TelegramError: if both attempts fail.
        ValueError: if env vars are not set.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID environment variable is not set")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    last_error = None
    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if not result.get("ok"):
                raise TelegramError(
                    f"Telegram API returned ok=false: {result}"
                )
            return
        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(5)

    raise TelegramError(
        f"Failed to send Telegram message after 2 attempts: {last_error}"
    )
