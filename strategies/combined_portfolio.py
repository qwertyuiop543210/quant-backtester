"""Combined Portfolio with Circuit Breaker Layers.

Adds two protective circuit breaker layers on top of the Dip Buyer strategy
and runs Phidias $50K Swing simulations to measure their impact.

LAYER 1 — VIX Regime Breaker:
    - Rolling 10-day window of VIX closes.
    - Activates when 3+ days have VIX close > 30.
    - Effect: Pause Chosen One entries; tighten Dip Buyer VIX ceiling 35 → 25.
    - Deactivates when VIX < 25 AND breaker active ≥ 20 trading days.

LAYER 2 — Account Drawdown Breaker:
    - Tracks cumulative closed P&L and high water mark within Phidias sim.
    - Activates when closed P&L drops $1,500 below HWM.
    - Effect: Pause ALL strategy entries for 20 trading days.
    - Deactivates after 20 trading days elapsed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import get_data
from core.metrics import profit_factor, win_rate
from strategies.dip_buyer import (
    compute_rsi, build_chosen_one_mask,
    ES_POINT_VALUE, COMMISSION_RT, SLIPPAGE_PER_SIDE, COST_PER_TRADE,
    RSI_PERIOD, RSI_ENTRY_THRESHOLD, RSI_EXIT_THRESHOLD, MAX_HOLD_DAYS,
    VIX_FLOOR, VIX_CEILING, CHOSEN_ONE_WEEKS,
    PHIDIAS_CAPITAL, PHIDIAS_PROFIT_TARGET, PHIDIAS_EOD_DRAWDOWN,
)


# ──────────────────────────────────────────────────────────────────────────────
# Generate ALL potential Dip Buyer trade signals (no circuit breaker filtering)
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_trades(es, vix_close, rsi, chosen_one_mask):
    """Generate every Dip Buyer trade that passes base RSI + VIX filters.

    Returns list of dicts, each containing:
        entry_date, exit_date, entry_idx, exit_idx, pnl, daily_pnls,
        vix_at_entry, chosen_one_overlap, entry_price, exit_price
    """
    es_open = es["Open"].astype(float)
    es_close = es["Close"].astype(float)

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None
    hold_days = 0

    for i in range(1, len(es_close)):
        if in_trade:
            hold_days += 1
            current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50.0
            exit_signal = current_rsi > RSI_EXIT_THRESHOLD or hold_days >= MAX_HOLD_DAYS

            if exit_signal:
                exit_price = es_close.iloc[i]
                pnl_points = exit_price - entry_price
                gross_pnl = pnl_points * ES_POINT_VALUE
                net_pnl = gross_pnl - COST_PER_TRADE

                # Daily P&Ls within trade for Phidias EOD drawdown checks
                trade_daily_pnls = []
                for d in range(entry_idx, i + 1):
                    if d == entry_idx:
                        day_pnl = (es_close.iloc[d] - entry_price) * ES_POINT_VALUE
                        if d == i:
                            day_pnl -= COST_PER_TRADE
                    elif d == i:
                        day_pnl = (es_close.iloc[d] - es_close.iloc[d - 1]) * ES_POINT_VALUE
                        day_pnl -= COST_PER_TRADE
                    else:
                        day_pnl = (es_close.iloc[d] - es_close.iloc[d - 1]) * ES_POINT_VALUE
                    trade_daily_pnls.append(day_pnl)

                vix_at_entry = vix_close.iloc[entry_idx - 1] if entry_idx > 0 else vix_close.iloc[entry_idx]
                trades.append({
                    "entry_date": es.index[entry_idx],
                    "exit_date": es.index[i],
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": net_pnl,
                    "daily_pnls": trade_daily_pnls,
                    "vix_at_entry": vix_at_entry,
                    "chosen_one_overlap": bool(chosen_one_mask.iloc[entry_idx]),
                })
                in_trade = False
                entry_idx = None
                entry_price = None
                hold_days = 0
        else:
            prev_rsi = rsi.iloc[i - 1] if not pd.isna(rsi.iloc[i - 1]) else 50.0
            prev_vix = vix_close.iloc[i - 1] if not pd.isna(vix_close.iloc[i - 1]) else 15.0
            signal = (
                prev_rsi < RSI_ENTRY_THRESHOLD
                and prev_vix > VIX_FLOOR
                and prev_vix < VIX_CEILING
            )
            if signal:
                entry_price = es_open.iloc[i]
                entry_idx = i
                hold_days = 0
                in_trade = True

    # Force close if still in trade
    if in_trade and entry_idx is not None:
        exit_price = es_close.iloc[-1]
        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE
        trade_daily_pnls = []
        for d in range(entry_idx, len(es_close)):
            if d == entry_idx:
                day_pnl = (es_close.iloc[d] - entry_price) * ES_POINT_VALUE
            else:
                day_pnl = (es_close.iloc[d] - es_close.iloc[d - 1]) * ES_POINT_VALUE
            trade_daily_pnls.append(day_pnl)
        trade_daily_pnls[-1] -= COST_PER_TRADE
        vix_at_entry = vix_close.iloc[entry_idx - 1] if entry_idx > 0 else vix_close.iloc[entry_idx]
        trades.append({
            "entry_date": es.index[entry_idx],
            "exit_date": es.index[-1],
            "entry_idx": entry_idx,
            "exit_idx": len(es_close) - 1,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": net_pnl,
            "daily_pnls": trade_daily_pnls,
            "vix_at_entry": vix_at_entry,
            "chosen_one_overlap": bool(chosen_one_mask.iloc[entry_idx]),
        })

    return trades


# ──────────────────────────────────────────────────────────────────────────────
# VIX Regime Breaker (Layer 1)
# ──────────────────────────────────────────────────────────────────────────────

class VIXRegimeBreaker:
    """Track rolling VIX window and gate entries when regime is stressed.

    Parameters
    ----------
    vix_close : pd.Series
        VIX close prices indexed by date.
    window : int
        Rolling window in trading days (default 10).
    trigger_days : int
        Number of days in window with VIX > trigger_level to activate (default 3).
    trigger_level : float
        VIX threshold for counting stressed days (default 30).
    deactivation_vix : float
        VIX must close below this to deactivate (default 25).
    min_active_days : int
        Minimum trading days breaker must stay active (default 20).
    tightened_ceiling : float
        Dip Buyer VIX ceiling while breaker is active (default 25).
    """

    def __init__(self, vix_close, window=10, trigger_days=3, trigger_level=30.0,
                 deactivation_vix=25.0, min_active_days=20, tightened_ceiling=25.0):
        self.vix_close = vix_close
        self.window = window
        self.trigger_days = trigger_days
        self.trigger_level = trigger_level
        self.deactivation_vix = deactivation_vix
        self.min_active_days = min_active_days
        self.tightened_ceiling = tightened_ceiling

        # State
        self.active = False
        self.activation_day_count = 0  # days since activation
        self.activation_dates = []
        self.deactivation_dates = []
        self.trades_skipped = 0
        self.pnl_avoided = 0.0

    def _count_stressed_days(self, date):
        """Count days with VIX > trigger_level in trailing window ending at date."""
        loc = self.vix_close.index.get_loc(date)
        start = max(0, loc - self.window + 1)
        window_vals = self.vix_close.iloc[start:loc + 1]
        return int((window_vals > self.trigger_level).sum())

    def check_activation(self, date):
        """Check if breaker should activate on this date."""
        if self.active:
            return
        stressed = self._count_stressed_days(date)
        if stressed >= self.trigger_days:
            self.active = True
            self.activation_day_count = 0
            self.activation_dates.append(date)

    def check_deactivation(self, date):
        """Check if breaker should deactivate on this date."""
        if not self.active:
            return
        self.activation_day_count += 1
        current_vix = self.vix_close.loc[date] if date in self.vix_close.index else 30.0
        if current_vix < self.deactivation_vix and self.activation_day_count >= self.min_active_days:
            self.active = False
            self.deactivation_dates.append(date)

    def should_skip_trade(self, trade, original_ceiling=VIX_CEILING):
        """Decide whether to skip a trade based on breaker state.

        When active:
        - Skip Chosen One trades entirely.
        - Tighten Dip Buyer VIX ceiling.

        Returns (skip: bool, reason: str)
        """
        if not self.active:
            return False, ""

        # Skip Chosen One entries
        if trade.get("chosen_one_overlap", False):
            return True, "VIX breaker: Chosen One paused"

        # Tighten VIX ceiling for Dip Buyer
        if trade["vix_at_entry"] >= self.tightened_ceiling:
            return True, f"VIX breaker: VIX {trade['vix_at_entry']:.1f} >= tightened ceiling {self.tightened_ceiling}"

        return False, ""

    def record_skip(self, trade):
        self.trades_skipped += 1
        self.pnl_avoided += trade["pnl"]

    def reset(self):
        """Reset state for a new simulation."""
        self.active = False
        self.activation_day_count = 0
        self.activation_dates = []
        self.deactivation_dates = []
        self.trades_skipped = 0
        self.pnl_avoided = 0.0

    def summary(self):
        return {
            "activations": len(self.activation_dates),
            "deactivations": len(self.deactivation_dates),
            "trades_skipped": self.trades_skipped,
            "pnl_avoided": self.pnl_avoided,
            "activation_dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                                 for d in self.activation_dates],
            "deactivation_dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                                   for d in self.deactivation_dates],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Account Drawdown Breaker (Layer 2)
# ──────────────────────────────────────────────────────────────────────────────

class AccountDrawdownBreaker:
    """Pause all entries when closed P&L drops too far below HWM.

    Parameters
    ----------
    drawdown_threshold : float
        Dollar amount below HWM to trigger pause (default $1,500).
    cooling_period : int
        Trading days to pause after activation (default 20).
    """

    def __init__(self, drawdown_threshold=1500.0, cooling_period=20):
        self.drawdown_threshold = drawdown_threshold
        self.cooling_period = cooling_period

        # State
        self.closed_pnl = 0.0
        self.hwm = 0.0
        self.active = False
        self.days_since_activation = 0
        self.activation_count = 0
        self.total_days_paused = 0
        self.trades_skipped = 0
        self.activation_dates = []
        self.deactivation_dates = []

    def update_pnl(self, trade_pnl):
        """Update closed P&L after a trade completes."""
        self.closed_pnl += trade_pnl
        self.hwm = max(self.hwm, self.closed_pnl)

    def check_activation(self):
        """Check if breaker should activate based on current P&L vs HWM."""
        if self.active:
            return
        if self.hwm - self.closed_pnl >= self.drawdown_threshold:
            self.active = True
            self.days_since_activation = 0
            self.activation_count += 1

    def advance_day(self, date=None):
        """Advance one trading day. Deactivate if cooling period elapsed."""
        if self.active:
            self.days_since_activation += 1
            self.total_days_paused += 1
            if self.days_since_activation >= self.cooling_period:
                self.active = False
                if date is not None:
                    self.deactivation_dates.append(date)

    def should_skip(self):
        """Returns True if all entries should be paused."""
        return self.active

    def record_skip(self):
        self.trades_skipped += 1

    def reset(self):
        """Reset state for a new simulation."""
        self.closed_pnl = 0.0
        self.hwm = 0.0
        self.active = False
        self.days_since_activation = 0
        self.activation_count = 0
        self.total_days_paused = 0
        self.trades_skipped = 0
        self.activation_dates = []
        self.deactivation_dates = []

    def summary(self):
        return {
            "activation_count": self.activation_count,
            "total_days_paused": self.total_days_paused,
            "trades_skipped": self.trades_skipped,
            "activation_dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                                 for d in self.activation_dates],
            "deactivation_dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                                   for d in self.deactivation_dates],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Phidias simulation with circuit breakers
# ──────────────────────────────────────────────────────────────────────────────

def simulate_phidias_with_breakers(
    all_trades, es_dates, vix_close,
    use_vix_breaker=False, use_account_breaker=False,
    # VIX breaker params
    vix_window=10, vix_trigger_days=3, vix_trigger_level=30.0,
    vix_deactivation=25.0, vix_min_active=20, vix_tightened_ceiling=25.0,
    # Account breaker params
    acct_drawdown_threshold=1500.0, acct_cooling_period=20,
):
    """Run sequential Phidias simulation with optional circuit breakers.

    Processes trades in chronological order (no Monte Carlo — deterministic).
    VIX breaker: market-regime filter, persists across all attempts.
    Account breaker: tracks cumulative closed P&L across all attempts
        (portfolio-level risk), pauses entries for cooling_period days.

    Returns dict with simulation results and breaker statistics.
    """
    # Initialize breakers (persist across entire simulation)
    vix_breaker = None
    if use_vix_breaker:
        vix_breaker = VIXRegimeBreaker(
            vix_close, window=vix_window, trigger_days=vix_trigger_days,
            trigger_level=vix_trigger_level, deactivation_vix=vix_deactivation,
            min_active_days=vix_min_active, tightened_ceiling=vix_tightened_ceiling,
        )

    acct_breaker = None
    if use_account_breaker:
        acct_breaker = AccountDrawdownBreaker(
            drawdown_threshold=acct_drawdown_threshold,
            cooling_period=acct_cooling_period,
        )

    # Track last processed date for advancing account breaker between trades
    last_processed_date = None

    # Run Phidias attempts
    attempts = []
    trades_taken_list = []
    trades_skipped_list = []
    vix_breaker_timeline = []  # (date, event)
    acct_breaker_timeline = []

    # Pre-compute VIX breaker state for every trading day (for timeline)
    vix_was_active = False
    if vix_breaker is not None:
        for idx in range(len(es_dates)):
            d = es_dates[idx]
            vix_breaker.check_activation(d)
            if vix_breaker.active and not vix_was_active:
                vix_breaker_timeline.append((d, "ON"))
            vix_breaker.check_deactivation(d)
            if not vix_breaker.active and vix_was_active:
                vix_breaker_timeline.append((d, "OFF"))
            vix_was_active = vix_breaker.active
        if vix_breaker.active:
            vix_breaker_timeline.append((es_dates[-1], "STILL ACTIVE"))
        # Reset for actual simulation
        vix_breaker.reset()

    i = 0
    while i < len(all_trades):
        balance = PHIDIAS_CAPITAL
        high_water = balance
        start_trade_idx = i
        status = "in_progress"
        attempt_trades_taken = 0
        attempt_trades_skipped = 0

        while i < len(all_trades):
            trade = all_trades[i]
            trade_date = trade["entry_date"]

            # --- Advance account breaker days between trades ---
            if acct_breaker is not None and last_processed_date is not None:
                # Advance through all trading days from last processed to this entry
                mask = (es_dates > last_processed_date) & (es_dates <= trade_date)
                gap_dates = es_dates[mask]
                for gd in gap_dates:
                    was_active = acct_breaker.active
                    acct_breaker.advance_day(gd)
                    if was_active and not acct_breaker.active:
                        acct_breaker_timeline.append((gd, "OFF"))

            # --- VIX breaker checks (persists across attempts) ---
            if vix_breaker is not None:
                vix_breaker.check_activation(trade_date)
                vix_breaker.check_deactivation(trade_date)

            # --- Decide whether to skip ---
            if acct_breaker is not None and acct_breaker.should_skip():
                acct_breaker.record_skip()
                attempt_trades_skipped += 1
                trades_skipped_list.append(trade)
                last_processed_date = trade["entry_date"]
                i += 1
                continue

            if vix_breaker is not None:
                vix_skip, reason = vix_breaker.should_skip_trade(trade)
                if vix_skip:
                    vix_breaker.record_skip(trade)
                    attempt_trades_skipped += 1
                    trades_skipped_list.append(trade)
                    last_processed_date = trade["entry_date"]
                    i += 1
                    continue

            # --- Take the trade ---
            breached = False
            for daily_pnl in trade["daily_pnls"]:
                balance += daily_pnl
                high_water = max(high_water, balance)
                if high_water - balance >= PHIDIAS_EOD_DRAWDOWN:
                    breached = True
                    break

            attempt_trades_taken += 1
            trades_taken_list.append(trade)

            # Update account breaker with closed P&L (persists across attempts)
            if acct_breaker is not None:
                acct_breaker.update_pnl(trade["pnl"])
                acct_breaker.check_activation()
                if acct_breaker.active and acct_breaker.days_since_activation == 0:
                    acct_breaker.activation_dates.append(trade["exit_date"])
                    acct_breaker_timeline.append((trade["exit_date"], "ON"))

            last_processed_date = trade["exit_date"]
            profit = balance - PHIDIAS_CAPITAL
            i += 1

            if breached:
                status = "FAILED"
                break
            if profit >= PHIDIAS_PROFIT_TARGET:
                status = "PASSED"
                break

        attempts.append({
            "attempt": len(attempts) + 1,
            "start_trade": start_trade_idx + 1,
            "end_trade": i,
            "trades_taken": attempt_trades_taken,
            "trades_skipped": attempt_trades_skipped,
            "final_balance": balance,
            "peak_balance": high_water,
            "profit": balance - PHIDIAS_CAPITAL,
            "status": status,
        })
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break

    # Compute results
    passed = sum(1 for a in attempts if a["status"] == "PASSED")
    failed = sum(1 for a in attempts if a["status"] == "FAILED")
    total_attempts = passed + failed
    pass_rate = passed / total_attempts * 100 if total_attempts > 0 else 0

    total_trades_taken = sum(a["trades_taken"] for a in attempts)
    total_trades_skipped = sum(a["trades_skipped"] for a in attempts)
    all_taken_pnls = [t["pnl"] for t in trades_taken_list]
    total_pnl = sum(all_taken_pnls) if all_taken_pnls else 0
    worst_loss = min(all_taken_pnls) if all_taken_pnls else 0

    pf = 0.0
    if all_taken_pnls:
        gains = sum(p for p in all_taken_pnls if p > 0)
        losses = abs(sum(p for p in all_taken_pnls if p < 0))
        pf = gains / losses if losses > 0 else float("inf")

    return {
        "attempts": attempts,
        "pass_rate": pass_rate,
        "passed": passed,
        "failed": failed,
        "total_attempts": total_attempts,
        "trades_taken": total_trades_taken,
        "trades_skipped": total_trades_skipped,
        "total_pnl": total_pnl,
        "profit_factor": pf,
        "worst_loss": worst_loss,
        "vix_breaker_summary": vix_breaker.summary() if vix_breaker else None,
        "acct_breaker_summary": acct_breaker.summary() if acct_breaker else None,
        "vix_breaker_timeline": vix_breaker_timeline,
        "acct_breaker_timeline": acct_breaker_timeline,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_config_comparison(results_dict):
    """Print comparison table for multiple configurations."""
    print(f"\n{'='*100}")
    print(f"  Phidias Simulation — Circuit Breaker Comparison")
    print(f"{'='*100}")
    header = (f"  {'Configuration':<30} {'Pass%':>7} {'Passed':>7} {'Failed':>7} "
              f"{'Taken':>7} {'Skipped':>8} {'Total P&L':>12} {'PF':>8} {'Worst':>10}")
    print(header)
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")

    for name, r in results_dict.items():
        pf_str = f"{r['profit_factor']:.3f}" if r['profit_factor'] != float("inf") else "inf"
        print(f"  {name:<30} {r['pass_rate']:>6.1f}% {r['passed']:>7} {r['failed']:>7} "
              f"{r['trades_taken']:>7} {r['trades_skipped']:>8} "
              f"${r['total_pnl']:>11,.0f} {pf_str:>8} ${r['worst_loss']:>9,.0f}")
    print(f"{'='*100}")


def print_sensitivity_table(title, results_list, param_name):
    """Print sensitivity analysis table."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    header = (f"  {param_name:<25} {'Pass%':>7} {'Passed':>7} {'Failed':>7} "
              f"{'Taken':>7} {'Skipped':>8} {'Total P&L':>12} {'PF':>8} {'Worst':>10}")
    print(header)
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")

    for label, r in results_list:
        pf_str = f"{r['profit_factor']:.3f}" if r['profit_factor'] != float("inf") else "inf"
        print(f"  {label:<25} {r['pass_rate']:>6.1f}% {r['passed']:>7} {r['failed']:>7} "
              f"{r['trades_taken']:>7} {r['trades_skipped']:>8} "
              f"${r['total_pnl']:>11,.0f} {pf_str:>8} ${r['worst_loss']:>9,.0f}")
    print(f"{'='*100}")


def print_attempts_detail(results, label):
    """Print per-attempt detail for a configuration."""
    print(f"\n  {label} — Attempt Detail:")
    print(f"  {'#':>3} {'Status':>10} {'Taken':>7} {'Skipped':>8} {'Profit':>12} "
          f"{'Peak Bal':>12} {'Final Bal':>12}")
    print(f"  {'-'*3} {'-'*10} {'-'*7} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for a in results["attempts"]:
        print(f"  {a['attempt']:>3} {a['status']:>10} {a['trades_taken']:>7} "
              f"{a['trades_skipped']:>8} ${a['profit']:>11,.0f} "
              f"${a['peak_balance']:>11,.0f} ${a['final_balance']:>11,.0f}")


def print_breaker_timeline(vix_timeline, acct_timeline):
    """Print activation/deactivation timeline for both breakers."""
    print(f"\n{'='*100}")
    print(f"  Circuit Breaker Activation Timeline (2012-2026)")
    print(f"{'='*100}")

    all_events = []
    for date, event in vix_timeline:
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        all_events.append((date, f"VIX Regime Breaker {event}", date_str))
    for date, event in acct_timeline:
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        all_events.append((date, f"Account Drawdown Breaker {event}", date_str))

    all_events.sort(key=lambda x: x[0])

    if not all_events:
        print("  No breaker activations.")
    else:
        print(f"  {'Date':<14} {'Event':<40}")
        print(f"  {'-'*14} {'-'*40}")
        for _, event, date_str in all_events:
            print(f"  {date_str:<14} {event}")
    print(f"{'='*100}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'#'*100}")
    print(f"  COMBINED PORTFOLIO — Circuit Breaker Analysis")
    print(f"  Phidias $50K Swing | EOD DD ${PHIDIAS_EOD_DRAWDOWN:,.0f} | "
          f"Target ${PHIDIAS_PROFIT_TARGET:,.0f}")
    print(f"{'#'*100}")

    # --- Load data ---
    print("\nLoading data...")
    es = get_data("ES", start="2012-01-01")
    vix = get_data("VIX", start="2012-01-01")
    print(f"ES: {es.index[0].date()} to {es.index[-1].date()} ({len(es)} days)")
    print(f"VIX: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

    vix_close = vix["Close"].reindex(es.index).ffill()
    rsi = compute_rsi(es["Close"].astype(float), period=RSI_PERIOD)
    chosen_one_mask = build_chosen_one_mask(es.index)

    # --- Generate all potential trades ---
    print("\nGenerating all Dip Buyer trades (base filters: RSI<10, VIX 20-35)...")
    all_trades = generate_all_trades(es, vix_close, rsi, chosen_one_mask)
    print(f"Total potential trades: {len(all_trades)}")

    trade_pnls = [t["pnl"] for t in all_trades]
    if trade_pnls:
        print(f"  Total P&L: ${sum(trade_pnls):,.0f}")
        print(f"  Win rate: {sum(1 for p in trade_pnls if p > 0)/len(trade_pnls):.1%}")
        gains = sum(p for p in trade_pnls if p > 0)
        losses_val = abs(sum(p for p in trade_pnls if p < 0))
        pf = gains / losses_val if losses_val > 0 else float("inf")
        print(f"  Profit factor: {pf:.3f}")

    es_dates = es.index

    # ======================================================================
    # PART 1: Four configurations comparison
    # ======================================================================
    print(f"\n{'#'*100}")
    print(f"  PART 1: Four Configuration Comparison")
    print(f"{'#'*100}")

    configs = {
        "1. No breakers (baseline)": dict(
            use_vix_breaker=False, use_account_breaker=False),
        "2. VIX breaker only": dict(
            use_vix_breaker=True, use_account_breaker=False),
        "3. Account breaker only": dict(
            use_vix_breaker=False, use_account_breaker=True),
        "4. Both breakers": dict(
            use_vix_breaker=True, use_account_breaker=True),
    }

    config_results = {}
    for name, kwargs in configs.items():
        r = simulate_phidias_with_breakers(
            all_trades, es_dates, vix_close, **kwargs)
        config_results[name] = r
        print_attempts_detail(r, name)

        # Print breaker summaries
        if r["vix_breaker_summary"]:
            vs = r["vix_breaker_summary"]
            print(f"    VIX Breaker: {vs['activations']} activations, "
                  f"{vs['trades_skipped']} skipped, "
                  f"P&L avoided: ${vs['pnl_avoided']:,.0f}")
        if r["acct_breaker_summary"]:
            acs = r["acct_breaker_summary"]
            print(f"    Account Breaker: {acs['activation_count']} activations, "
                  f"{acs['total_days_paused']} days paused, "
                  f"{acs['trades_skipped']} skipped")

    print_config_comparison(config_results)

    # ======================================================================
    # PART 2: Layer 1 (VIX Breaker) Sensitivity
    # ======================================================================
    print(f"\n{'#'*100}")
    print(f"  PART 2: Layer 1 (VIX Regime Breaker) Sensitivity Analysis")
    print(f"{'#'*100}")

    # --- Trigger days sensitivity ---
    trigger_days_results = []
    for td in [2, 3, 4]:
        label = f"{td} of 10 days > VIX 30"
        if td == 3:
            label += " (base)"
        r = simulate_phidias_with_breakers(
            all_trades, es_dates, vix_close,
            use_vix_breaker=True, use_account_breaker=False,
            vix_trigger_days=td, vix_trigger_level=30.0)
        trigger_days_results.append((label, r))

    # --- Trigger level sensitivity ---
    trigger_level_results = []
    for tl in [25.0, 30.0, 35.0]:
        label = f"3 of 10 days > VIX {int(tl)}"
        if tl == 30.0:
            label += " (base)"
        r = simulate_phidias_with_breakers(
            all_trades, es_dates, vix_close,
            use_vix_breaker=True, use_account_breaker=False,
            vix_trigger_days=3, vix_trigger_level=tl)
        trigger_level_results.append((label, r))

    print_sensitivity_table(
        "Layer 1 Sensitivity: Trigger Days (VIX > 30 threshold)",
        trigger_days_results, "Trigger Condition")

    print_sensitivity_table(
        "Layer 1 Sensitivity: Trigger Level (3 of 10 days)",
        trigger_level_results, "Trigger Condition")

    # ======================================================================
    # PART 3: Layer 2 (Account Drawdown Breaker) Sensitivity
    # ======================================================================
    print(f"\n{'#'*100}")
    print(f"  PART 3: Layer 2 (Account Drawdown Breaker) Sensitivity Analysis")
    print(f"{'#'*100}")

    # --- Drawdown threshold sensitivity ---
    dd_threshold_results = []
    for dd in [1000.0, 1500.0, 2000.0]:
        label = f"Pause at -${int(dd):,} from HWM"
        if dd == 1500.0:
            label += " (base)"
        r = simulate_phidias_with_breakers(
            all_trades, es_dates, vix_close,
            use_vix_breaker=False, use_account_breaker=True,
            acct_drawdown_threshold=dd)
        dd_threshold_results.append((label, r))

    # --- Cooling period sensitivity ---
    cooling_results = []
    for cp in [10, 20, 30]:
        label = f"Cooling {cp} days"
        if cp == 20:
            label += " (base)"
        r = simulate_phidias_with_breakers(
            all_trades, es_dates, vix_close,
            use_vix_breaker=False, use_account_breaker=True,
            acct_cooling_period=cp)
        cooling_results.append((label, r))

    print_sensitivity_table(
        "Layer 2 Sensitivity: Drawdown Threshold",
        dd_threshold_results, "Threshold")

    print_sensitivity_table(
        "Layer 2 Sensitivity: Cooling Period",
        cooling_results, "Period")

    # ======================================================================
    # PART 4: Activation Timeline
    # ======================================================================
    print(f"\n{'#'*100}")
    print(f"  PART 4: Breaker Activation Timeline (Both Breakers Active)")
    print(f"{'#'*100}")

    both_result = config_results["4. Both breakers"]
    print_breaker_timeline(
        both_result["vix_breaker_timeline"],
        both_result["acct_breaker_timeline"])

    # Also show individual timelines
    vix_only = config_results["2. VIX breaker only"]
    acct_only = config_results["3. Account breaker only"]

    print(f"\n  VIX Breaker Only — Activation Timeline:")
    if vix_only["vix_breaker_summary"]:
        vs = vix_only["vix_breaker_summary"]
        for i, (on, off) in enumerate(zip(vs["activation_dates"],
                                          vs["deactivation_dates"])):
            print(f"    Period {i+1}: {on} → {off}")
        # Any still-active at end
        if len(vs["activation_dates"]) > len(vs["deactivation_dates"]):
            print(f"    Period {len(vs['activation_dates'])}: "
                  f"{vs['activation_dates'][-1]} → (still active at end of data)")

    print(f"\n  Account Breaker Only — Activation Timeline:")
    if acct_only["acct_breaker_summary"]:
        acs = acct_only["acct_breaker_summary"]
        for i, ad in enumerate(acs["activation_dates"]):
            deact = acs["deactivation_dates"][i] if i < len(acs["deactivation_dates"]) else "(still active)"
            print(f"    Activation {i+1}: {ad} → {deact}")

    print(f"\n{'#'*100}")
    print(f"  END OF CIRCUIT BREAKER ANALYSIS")
    print(f"{'#'*100}\n")


if __name__ == "__main__":
    run()
