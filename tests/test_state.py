"""Tests for src/state.py — duplicate-send protection."""

import json
import os
from unittest.mock import patch

import pytest

from src.state import already_sent, mark_sent, STATE_FILE


@pytest.fixture(autouse=True)
def clean_state_file(tmp_path):
    """Use a temporary state file for each test."""
    test_state = tmp_path / "state.json"
    with patch("src.state.STATE_FILE", test_state):
        yield test_state


class TestState:
    def test_new_key_not_sent(self, clean_state_file):
        with patch("src.state.STATE_FILE", clean_state_file):
            assert already_sent("friday-2025-09-05") is False

    def test_sent_key_is_detected(self, clean_state_file):
        with patch("src.state.STATE_FILE", clean_state_file):
            mark_sent("friday-2025-09-05")
            assert already_sent("friday-2025-09-05") is True

    def test_different_keys_are_independent(self, clean_state_file):
        with patch("src.state.STATE_FILE", clean_state_file):
            mark_sent("friday-2025-09-05")
            assert already_sent("monday-2025-09-08") is False

    def test_state_persists_to_json(self, clean_state_file):
        with patch("src.state.STATE_FILE", clean_state_file):
            mark_sent("monday-2025-09-01")
            # Verify the JSON file was written
            with open(clean_state_file) as f:
                data = json.load(f)
            assert "monday-2025-09-01" in data
            assert data["monday-2025-09-01"] is True
