"""Week 1 & Week 4 Strategy — Four Final Stress Tests.

TEST 1: TRADE CONCENTRATION
Are a small number of outsized trades driving the PF 2.65?
If top 10 trades = 80%+ of total P&L, the edge is fragile.

TEST 2: BOOTSTRAP PF CONFIDENCE INTERVAL
Resample the 193 post-2013 trades 10,000 times with replacement.
Compute PF for each sample. Report 95% confidence interval.
If the lower bound is below 1.2, the edge may not be real.

TEST 3: CALENDAR WINDOW REDEFINE
Replace "Week 1 (days 1-7) and Week 4 (days 22-28)" with
"Last 2 trading days of month through first 3 trading days of next month."
This is more directly aligned with the institutional rebalancing thesis.

TEST 4: VIX PERCENTILE SKIP ZONE
Replace "skip when VIX is 15.0-20.0" with
"skip when VIX is in the 30th-50th percentile of trailing 252-day distribution."
This normalizes for changing VIX regimes over time.

All tests use post-2013 data only (the relevant regime).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

from core.data_loader import get_data
from core.metrics import summary, print_summary
from core.plotting import plot_equity

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

ES_POINT_VALUE = 50.0
COMMISSION_RT = 5.0
SLIPPAGE_PER_SIDE = 12.50
COST_PER_TRADE = COMMISSION_RT + 2 * SLIPPAGE_PER_SIDE
INITIAL_CAPITAL = 100_000.0

# Phidias 50K sim
PHIDIAS_CAPITAL = 50_000.0
PHIDIAS_TARGET = 4_000.0
PHIDIAS_DD = 2_500.0

# Only test post-2013 regime
REGIME_START = "2013-01-01"


def get_week_of_month(date):
    day = date.day
    if day <= 7:
        return 1
    elif day <= 14:
        return 2
    elif day <= 21:
        return 3
    elif day <= 28:
        return 4
    else:
        return 5


def simulate_phidias(trade_pnls):
    """Simulate Phidias 50K Swing eval attempts."""
    attempts = []
    i = 0
    while i < len(trade_pnls):
        balance = PHIDIAS_CAPITAL
        high_water = PHIDIAS_CAPITAL
        start = i
        status = "in_progress"
        while i < len(trade_pnls):
            balance += trade_pnls.iloc[i]
            high_water = max(high_water, balance)
            trailing = high_water - balance
            profit = balance - PHIDIAS_CAPITAL
            i += 1
            if trailing >= PHIDIAS_DD:
                status = "FAILED"
                break
            if profit >= PHIDIAS_TARGET:
                status = "PASSED"
                break
        attempts.append({"trades_taken": i - start, "profit": balance - PHIDIAS_CAPITAL, "status": status})
        if status == "in_progress":
            attempts[-1]["status"] = "INCOMPLETE"
            break
    return attempts


def build_original_trades(es_data, vix_data):
    """Build the original Week 1/4 + VIX 15-20 skip trades (post-2013 only)."""
    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")

    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
        "week_of_month": [get_week_of_month(d) for d in es_close.index],
    }, index=es_close.index)

    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

    trades = []
    for week_key, group in df.groupby("week_key", sort=True):
        if len(group) < 3:
            continue
        if group.index[0] < pd.Timestamp(REGIME_START):
            continue

        monday_candidates = group[group["dow"] == 0]
        first_day = monday_candidates.index[0] if len(monday_candidates) > 0 else group.index[0]
        week_num = get_week_of_month(first_day)

        if week_num not in [1, 4]:
            continue

        prior_days = df.index[df.index < group.index[0]]
        if len(prior_days) == 0:
            continue
        prior_friday = prior_days[-1]
        vix_val = df.loc[prior_friday, "vix"]

        if pd.isna(vix_val) or 15.0 <= vix_val <= 20.0:
            continue

        entry_day = group.index[0]
        exit_day = group.index[-1]
        entry_price = df.loc[entry_day, "open"]
        exit_price = df.loc[exit_day, "close"]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "week_num": week_num,
            "vix_friday": vix_val,
            "year": entry_day.year,
        })

    return pd.DataFrame(trades)


# =============================================================================
# TEST 1: TRADE CONCENTRATION
# =============================================================================
def test_1_concentration(trades):
    print("\n" + "=" * 80)
    print("TEST 1: TRADE CONCENTRATION ANALYSIS")
    print("Are a few outsized trades driving the headline PF?")
    print("=" * 80)

    pnls = trades["pnl"].copy()
    total_pnl = pnls.sum()
    total_trades = len(pnls)

    print(f"\n  Total trades: {total_trades}")
    print(f"  Total P&L: ${total_pnl:,.0f}")

    # Sort by absolute contribution
    sorted_pnls = pnls.sort_values(ascending=False)

    # Top N trades as % of total P&L
    print(f"\n  Contribution of top trades to total P&L:")
    for n in [1, 3, 5, 10, 20]:
        if n > total_trades:
            continue
        top_n_sum = sorted_pnls.head(n).sum()
        pct = top_n_sum / total_pnl * 100 if total_pnl != 0 else 0
        print(f"    Top {n:>2} trades: ${top_n_sum:>10,.0f} ({pct:>5.1f}% of total)")

    # Bottom N (worst trades)
    print(f"\n  Worst trades:")
    worst = sorted_pnls.tail(5)
    for i, (idx, val) in enumerate(worst.items()):
        date = trades.loc[idx, "entry_date"]
        print(f"    #{i+1}: ${val:>8,.0f} (week of {date.date()})")

    # Best trades
    print(f"\n  Best trades:")
    best = sorted_pnls.head(5)
    for i, (idx, val) in enumerate(best.items()):
        date = trades.loc[idx, "entry_date"]
        print(f"    #{i+1}: ${val:>8,.0f} (week of {date.date()})")

    # What happens if we remove top 5 trades?
    without_top5 = pnls.drop(sorted_pnls.head(5).index)
    gross_w = without_top5[without_top5 > 0].sum()
    gross_l = abs(without_top5[without_top5 < 0].sum())
    pf_without = gross_w / gross_l if gross_l > 0 else float("inf")

    print(f"\n  PF with ALL trades: {pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()):.3f}")
    print(f"  PF WITHOUT top 5 trades: {pf_without:.3f}")

    # What happens if we remove top 10 trades?
    without_top10 = pnls.drop(sorted_pnls.head(10).index)
    gross_w10 = without_top10[without_top10 > 0].sum()
    gross_l10 = abs(without_top10[without_top10 < 0].sum())
    pf_without10 = gross_w10 / gross_l10 if gross_l10 > 0 else float("inf")
    print(f"  PF WITHOUT top 10 trades: {pf_without10:.3f}")

    # Verdict
    top5_pct = sorted_pnls.head(5).sum() / total_pnl * 100 if total_pnl != 0 else 0
    top10_pct = sorted_pnls.head(10).sum() / total_pnl * 100 if total_pnl != 0 else 0

    print(f"\n  VERDICT:")
    if top10_pct > 80:
        print(f"  ❌ CONCENTRATED — Top 10 trades = {top10_pct:.0f}% of P&L. Edge is fragile.")
    elif top10_pct > 50:
        print(f"  ⚠️  MODERATELY CONCENTRATED — Top 10 = {top10_pct:.0f}%. Some tail dependence.")
    else:
        print(f"  ✅ WELL DISTRIBUTED — Top 10 = {top10_pct:.0f}%. Edge is broad-based.")

    if pf_without10 > 1.2:
        print(f"  ✅ PF still {pf_without10:.2f} without top 10 trades — edge survives removal")
    else:
        print(f"  ❌ PF drops to {pf_without10:.2f} without top 10 — fragile")


# =============================================================================
# TEST 2: BOOTSTRAP PF CONFIDENCE INTERVAL
# =============================================================================
def test_2_bootstrap(trades):
    print("\n" + "=" * 80)
    print("TEST 2: BOOTSTRAP PROFIT FACTOR CONFIDENCE INTERVAL")
    print("Resampling 10,000 times to estimate true PF range")
    print("=" * 80)

    pnls = trades["pnl"].values
    n_trades = len(pnls)
    n_bootstrap = 10_000

    np.random.seed(42)

    bootstrap_pfs = []
    bootstrap_wrs = []
    bootstrap_totals = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(pnls, size=n_trades, replace=True)
        gross_w = sample[sample > 0].sum()
        gross_l = abs(sample[sample < 0].sum())
        pf = gross_w / gross_l if gross_l > 0 else 10.0  # Cap at 10
        wr = (sample > 0).mean()

        bootstrap_pfs.append(min(pf, 10.0))
        bootstrap_wrs.append(wr)
        bootstrap_totals.append(sample.sum())

    bootstrap_pfs = np.array(bootstrap_pfs)
    bootstrap_wrs = np.array(bootstrap_wrs)
    bootstrap_totals = np.array(bootstrap_totals)

    # Confidence intervals
    pf_ci_lower = np.percentile(bootstrap_pfs, 2.5)
    pf_ci_upper = np.percentile(bootstrap_pfs, 97.5)
    pf_median = np.percentile(bootstrap_pfs, 50)

    wr_ci_lower = np.percentile(bootstrap_wrs, 2.5)
    wr_ci_upper = np.percentile(bootstrap_wrs, 97.5)

    total_ci_lower = np.percentile(bootstrap_totals, 2.5)
    total_ci_upper = np.percentile(bootstrap_totals, 97.5)

    # Point estimates
    actual_gross_w = pnls[pnls > 0].sum()
    actual_gross_l = abs(pnls[pnls < 0].sum())
    actual_pf = actual_gross_w / actual_gross_l if actual_gross_l > 0 else float("inf")
    actual_wr = (pnls > 0).mean()

    print(f"\n  Observed PF: {actual_pf:.3f}")
    print(f"  Bootstrap median PF: {pf_median:.3f}")
    print(f"  95% CI for PF: [{pf_ci_lower:.3f}, {pf_ci_upper:.3f}]")

    print(f"\n  Observed Win Rate: {actual_wr:.1%}")
    print(f"  95% CI for Win Rate: [{wr_ci_lower:.1%}, {wr_ci_upper:.1%}]")

    print(f"\n  Observed Total P&L: ${pnls.sum():,.0f}")
    print(f"  95% CI for Total P&L: [${total_ci_lower:,.0f}, ${total_ci_upper:,.0f}]")

    # Probability PF > various thresholds
    print(f"\n  Probability estimates:")
    for threshold in [1.0, 1.2, 1.5, 2.0, 2.5]:
        prob = (bootstrap_pfs > threshold).mean() * 100
        print(f"    P(PF > {threshold:.1f}) = {prob:.1f}%")

    # Probability of losing money
    prob_loss = (bootstrap_totals < 0).mean() * 100
    print(f"    P(total P&L < 0) = {prob_loss:.1f}%")

    # Worst-case scenarios
    print(f"\n  Worst-case bootstrap outcomes (bottom 5%):")
    print(f"    5th percentile PF: {np.percentile(bootstrap_pfs, 5):.3f}")
    print(f"    5th percentile total P&L: ${np.percentile(bootstrap_totals, 5):,.0f}")

    # Verdict
    print(f"\n  VERDICT:")
    if pf_ci_lower > 1.5:
        print(f"  ✅ STRONG — Even at 95% CI lower bound, PF is {pf_ci_lower:.2f}")
    elif pf_ci_lower > 1.2:
        print(f"  ✅ VIABLE — 95% CI lower bound PF is {pf_ci_lower:.2f} (above 1.2 threshold)")
    elif pf_ci_lower > 1.0:
        print(f"  ⚠️  MARGINAL — 95% CI lower bound PF is {pf_ci_lower:.2f} (barely above breakeven)")
    else:
        print(f"  ❌ UNCERTAIN — 95% CI includes PF < 1.0. Edge may not be real.")

    print(f"  ChatGPT suggested haircutting to PF 1.3-1.8. Bootstrap says: [{pf_ci_lower:.2f}, {pf_ci_upper:.2f}]")


# =============================================================================
# TEST 3: CALENDAR WINDOW REDEFINE
# =============================================================================
def test_3_calendar_redefine(es_data, vix_data):
    print("\n" + "=" * 80)
    print("TEST 3: CALENDAR WINDOW — MONTH-END/MONTH-START vs WEEK 1/4")
    print("'Last 2 trading days + first 3 trading days' vs 'Week 1 and Week 4'")
    print("=" * 80)

    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")

    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "vix": vix_close,
        "dow": es_close.index.dayofweek,
    }, index=es_close.index)

    # Filter to post-2013
    df = df[df.index >= REGIME_START]

    # Group by year-month to find month boundaries
    df["year_month"] = df.index.to_period("M")
    months = df["year_month"].unique()

    # Build turn-of-month windows: last 2 trading days of month + first 3 of next month
    tom_windows = []
    for i in range(len(months) - 1):
        current_month = months[i]
        next_month = months[i + 1]

        current_days = df[df["year_month"] == current_month].index
        next_days = df[df["year_month"] == next_month].index

        if len(current_days) < 2 or len(next_days) < 3:
            continue

        # Last 2 trading days of current month
        entry_candidates = current_days[-2:]
        # First 3 trading days of next month
        exit_candidates = next_days[:3]

        # Entry: open of first day in window (2nd to last td of month)
        entry_day = entry_candidates[0]
        # Exit: close of last day in window (3rd td of next month)
        exit_day = exit_candidates[-1]

        # VIX check: use the Friday before entry (or the day before entry)
        prior_days_to_entry = df.index[df.index < entry_day]
        if len(prior_days_to_entry) == 0:
            continue
        vix_check_day = prior_days_to_entry[-1]
        vix_val = df.loc[vix_check_day, "vix"]

        tom_windows.append({
            "entry_day": entry_day,
            "exit_day": exit_day,
            "vix_check_day": vix_check_day,
            "vix_val": vix_val,
        })

    # Run with original VIX filter (skip 15-20)
    trades_tom = []
    for w in tom_windows:
        if pd.isna(w["vix_val"]) or 15.0 <= w["vix_val"] <= 20.0:
            continue

        entry_price = df.loc[w["entry_day"], "open"]
        exit_price = df.loc[w["exit_day"], "close"]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl_points = exit_price - entry_price
        gross_pnl = pnl_points * ES_POINT_VALUE
        net_pnl = gross_pnl - COST_PER_TRADE

        trades_tom.append({
            "entry_date": w["entry_day"],
            "exit_date": w["exit_day"],
            "pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "vix_friday": w["vix_val"],
            "year": w["entry_day"].year,
        })

    trades_tom_df = pd.DataFrame(trades_tom)

    if len(trades_tom_df) > 0:
        pnls = trades_tom_df["pnl"]
        gross_w = pnls[pnls > 0].sum()
        gross_l = abs(pnls[pnls < 0].sum())
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        wr = (pnls > 0).mean()

        print(f"\n  Turn-of-Month Window (last 2 + first 3 trading days):")
        print(f"    Trades: {len(pnls)}")
        print(f"    PF: {pf:.3f}")
        print(f"    Win Rate: {wr:.1%}")
        print(f"    Avg Trade: ${pnls.mean():,.0f}")
        print(f"    Total P&L: ${pnls.sum():,.0f}")

        # Phidias sim
        attempts = simulate_phidias(pd.Series(pnls.values, dtype=float))
        passed = sum(1 for a in attempts if a["status"] == "PASSED")
        total_att = sum(1 for a in attempts if a["status"] in ["PASSED", "FAILED"])
        pass_rate = passed / total_att * 100 if total_att > 0 else 0
        print(f"    Phidias Pass Rate: {pass_rate:.1f}% ({passed}/{total_att})")

        # Period breakdown
        print(f"\n    Period breakdown:")
        for y1, y2 in [(2013, 2016), (2017, 2019), (2020, 2022), (2023, 2026)]:
            sub = trades_tom_df[(trades_tom_df["year"] >= y1) & (trades_tom_df["year"] <= y2)]
            if len(sub) == 0:
                continue
            sp = sub["pnl"]
            sgw = sp[sp > 0].sum()
            sgl = abs(sp[sp < 0].sum())
            spf = sgw / sgl if sgl > 0 else float("inf")
            print(f"      {y1}-{y2}: {len(sub)} trades, PF {spf:.2f}, WR {(sp>0).mean():.1%}, "
                  f"avg ${sp.mean():,.0f}")
    else:
        print("\n  ERROR: No TOM trades generated")

    return trades_tom_df


# =============================================================================
# TEST 4: VIX PERCENTILE SKIP ZONE
# =============================================================================
def test_4_vix_percentile(es_data, vix_data):
    print("\n" + "=" * 80)
    print("TEST 4: VIX PERCENTILE SKIP ZONE vs FIXED 15-20")
    print("Skip when VIX is in 30th-50th percentile of trailing 252 days")
    print("=" * 80)

    es_close = es_data["Close"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(es_close.index, method="ffill")

    # Compute rolling VIX percentile
    vix_pctile = vix_close.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 252 else np.nan
    )

    df = pd.DataFrame({
        "close": es_close,
        "open": es_data["Open"].astype(float),
        "vix": vix_close,
        "vix_pctile": vix_pctile,
        "dow": es_close.index.dayofweek,
        "week_of_month": [get_week_of_month(d) for d in es_close.index],
    }, index=es_close.index)

    df = df[df.index >= REGIME_START]
    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

    # Test multiple percentile skip zones
    percentile_zones = [
        ("Skip 30-50 pctile", 0.30, 0.50),
        ("Skip 25-50 pctile", 0.25, 0.50),
        ("Skip 30-60 pctile", 0.30, 0.60),
        ("Skip 35-55 pctile", 0.35, 0.55),
        ("Skip 40-60 pctile", 0.40, 0.60),
        ("No VIX filter", None, None),
        ("Fixed 15-20 (original)", "fixed", "fixed"),
    ]

    results = {}

    for zone_label, lo, hi in percentile_zones:
        trades = []

        for week_key, group in df.groupby("week_key", sort=True):
            if len(group) < 3:
                continue

            monday_candidates = group[group["dow"] == 0]
            first_day = monday_candidates.index[0] if len(monday_candidates) > 0 else group.index[0]
            week_num = get_week_of_month(first_day)

            if week_num not in [1, 4]:
                continue

            prior_days = df.index[df.index < group.index[0]]
            if len(prior_days) == 0:
                continue
            prior_friday = prior_days[-1]
            vix_val = df.loc[prior_friday, "vix"]
            vix_pct = df.loc[prior_friday, "vix_pctile"]

            if pd.isna(vix_val):
                continue

            # Apply filter
            skip = False
            if lo == "fixed":
                skip = 15.0 <= vix_val <= 20.0
            elif lo is not None:
                if not pd.isna(vix_pct) and lo <= vix_pct <= hi:
                    skip = True
            # else: no filter

            if skip:
                continue

            entry_day = group.index[0]
            exit_day = group.index[-1]
            entry_price = df.loc[entry_day, "open"]
            exit_price = df.loc[exit_day, "close"]

            if pd.isna(entry_price) or pd.isna(exit_price):
                continue

            pnl_points = exit_price - entry_price
            gross_pnl = pnl_points * ES_POINT_VALUE
            net_pnl = gross_pnl - COST_PER_TRADE

            trades.append({"pnl": net_pnl, "year": entry_day.year})

        trade_df = pd.DataFrame(trades)
        if len(trade_df) == 0:
            continue

        pnls = trade_df["pnl"]
        gross_w = pnls[pnls > 0].sum()
        gross_l = abs(pnls[pnls < 0].sum())
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        wr = (pnls > 0).mean()

        # Phidias sim
        attempts = simulate_phidias(pd.Series(pnls.values, dtype=float))
        passed = sum(1 for a in attempts if a["status"] == "PASSED")
        total_att = sum(1 for a in attempts if a["status"] in ["PASSED", "FAILED"])
        pass_rate = passed / total_att * 100 if total_att > 0 else 0

        results[zone_label] = {
            "trades": len(pnls),
            "pf": pf,
            "wr": wr,
            "avg": pnls.mean(),
            "total": pnls.sum(),
            "pass_rate": pass_rate,
        }

    # Print comparison table
    print(f"\n  {'Filter':<30} {'Trades':>7} {'PF':>8} {'WR':>7} {'Avg$':>9} {'Total$':>11} {'Pass%':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*11} {'-'*7}")
    for label, r in results.items():
        is_original = " ◄" if "original" in label else ""
        print(f"  {label:<30} {r['trades']:>7} {r['pf']:>8.3f} {r['wr']:>6.1%} "
              f"${r['avg']:>8,.0f} ${r['total']:>10,.0f} {r['pass_rate']:>6.1f}%{is_original}")

    # Verdict
    original = results.get("Fixed 15-20 (original)", {})
    best_pctile = None
    best_pf = 0
    for label, r in results.items():
        if "pctile" in label and r["pf"] > best_pf and r["trades"] >= 100:
            best_pf = r["pf"]
            best_pctile = label

    print(f"\n  VERDICT:")
    if best_pctile and results[best_pctile]["pf"] > original.get("pf", 0) * 1.05:
        print(f"  📊 {best_pctile} improves PF from {original.get('pf', 0):.3f} to {results[best_pctile]['pf']:.3f}")
        print(f"     Consider adopting percentile-based filter")
    else:
        print(f"  ✅ Fixed 15-20 filter holds up. No percentile variant meaningfully improves it.")
        print(f"     Keep the original filter — it's simpler and works.")


# =============================================================================
# MAIN
# =============================================================================
def run():
    print("=" * 80)
    print("FOUR FINAL STRESS TESTS — WEEK 1/4 + VIX FILTER STRATEGY")
    print("Post-2013 regime only")
    print("=" * 80)

    print("\nLoading data...")
    es = get_data("ES", start="1997-01-01")
    vix = get_data("VIX", start="1997-01-01")
    print(f"ES range: {es.index[0].date()} to {es.index[-1].date()}")

    # Build original trades (post-2013)
    original_trades = build_original_trades(es, vix)
    print(f"Original strategy trades (post-2013): {len(original_trades)}")

    # Run all four tests
    test_1_concentration(original_trades)
    test_2_bootstrap(original_trades)
    test_3_calendar_redefine(es, vix)
    test_4_vix_percentile(es, vix)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL FOUR TESTS")
    print("=" * 80)
    print("""
  TEST 1 (Concentration): Check above — is the edge broad or driven by outliers?
  TEST 2 (Bootstrap CI): Check above — what's the realistic PF range?
  TEST 3 (Calendar redefine): Check above — does month-end/start beat Week 1/4?
  TEST 4 (VIX percentile): Check above — does percentile beat fixed 15-20?

  DECISION RULES:
  - If concentration is fine AND bootstrap lower CI > 1.2: edge is real, deploy as-is
  - If calendar redefine beats original by >10% PF: adopt new calendar window
  - If VIX percentile beats original by >10% PF: adopt percentile filter
  - If neither improvement helps: keep the original strategy unchanged
    """)
    print("=" * 80)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    original_trades.to_csv(os.path.join(RESULTS_DIR, "week14_stress_test_trades.csv"), index=False)
    print(f"Trades saved to results/week14_stress_test_trades.csv")


if __name__ == "__main__":
    run()
