"""Duplicate-send protection using a local JSON file.

# NOTE: Railway's filesystem is ephemeral and resets on redeploy.
# This means dedup state is lost on each deploy. Acceptable for now.
# Future fix: mount a Railway volume or use an external KV store.
"""

import json
import os
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent / "state.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_state(data: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def already_sent(alert_key: str) -> bool:
    """Check if an alert with this key has already been sent.

    Alert key format: "friday-YYYY-MM-DD" or "monday-YYYY-MM-DD"
    where the date is the Monday of the trading week.
    """
    state = _load_state()
    return alert_key in state


def mark_sent(alert_key: str) -> None:
    """Record that an alert with this key has been sent."""
    state = _load_state()
    state[alert_key] = True
    _save_state(state)
