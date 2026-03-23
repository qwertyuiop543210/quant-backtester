"""Backward-compatible re-export — use core.backtester directly."""
from core.backtester import (  # noqa: F401
    Trade, BacktestResult, phidias_simulation, correlation_by_week,
)
