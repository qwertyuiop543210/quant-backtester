"""Week 1 & Week 4 + VIX Filter — Multi-Instrument Thesis Confirmation.
Tests the exact same strategy rules on:
  - ES (S&P 500) — the original, for comparison
  - NQ (Nasdaq 100) — QQQ receives massive passive flows
  - RTY (Russell 2000) — IWM small cap, potentially stronger flow effect
Same rules on all:
  1. Week 1 (days 1-7) or Week 4 (days 22-28) of month
  2. Skip when prior Friday VIX close is 15.0-20.0
  3. Buy Monday open, sell Friday close
  4. No stop loss
Point values:
  ES = $50/point
  NQ = $20/point
  RTY = $50/point (actually $10/point for micro, $50 for mini)
Tests post-2013 only (the relevant regime).
ACCEPTANCE GATES (same as ES):
  1. PF > 1.5
  2. 100+ trades
  3. All sub-periods profitable
  4. Phidias 50K pass rate > 45%
If both NQ and RTY pass, we have 3 independent Phidias accounts.
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
REGIME_START = "2013-01-01"
INSTRUMENTS = {
    "ES": {
        "point_value": 50.0,
        "commission_rt": 5.0,
        "slippage_per_side": 12.50,
        "description": "S&P 500 E-mini (original)",
    },
    "NQ": {
        "point_value": 20.0,
        "commission_rt": 5.0,
        "slippage_per_side": 5.00,  # NQ tick = 0.25 pts * $20 = $5
        "description": "Nasdaq 100 E-mini",
    },
    "RTY": {
        "point_value": 50.0,
        "commission_rt": 5.0,
        "slippage_per_side": 5.00,  # RTY tick = 0.10 pts * $50 = $5
        "description": "Russell 2000 E-mini",
    },
}
PHIDIAS_CAPITAL = 50_000.0
PHIDIAS_TARGET = 4_000.0
PHIDIAS_DD = 2_500.0
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
def build_trades(instrument_data, vix_data, point_value, commission_rt, slippage_per_side):
    """Build Week 1/4 + VIX filter trades for any instrument."""
    close = instrument_data["Close"].astype(float)
    open_price = instrument_data["Open"].astype(float)
    vix_close = vix_data["Close"].astype(float).reindex(close.index, method="ffill")
    cost_per_trade = commission_rt + 2 * slippage_per_side
    df = pd.DataFrame({
        "close": close,
        "open": open_price,
        "vix": vix_close,
        "dow": close.index.dayofweek,
        "week_of_month": [get_week_of_month(d) for d in close.index],
    }, index=close.index)
    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values
    df["week_key"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)
    trades = []
    for week_key, group in df.groupby("week_key", sort=True):
        if len(group) < 3 or group.index[0] < pd.Timestamp(REGIME_START):
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
        gross_pnl = pnl_points * point_value
        net_pnl = gross_pnl - cost_per_trade
        trades.append({
            "entry_date": entry_day,
            "exit_date": exit_day,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_points": pnl_points,
            "gross_pnl": gross_pnl,
            "costs": cost_per_trade,
            "pnl": net_pnl,
            "week_num": week_num,
            "vix_friday": vix_val,
            "year": entry_day.year,
        })
    return pd.DataFrame(trades)
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
def analyze_instrument(name, params, vix_data):
    """Run full analysis for one instrument."""
    print(f"\n{'='*70}")
    print(f"  {name}: {params['description']}")
    print(f"  Point value: ${params['point_value']}/pt | Commission: ${params['commission_rt']} RT")
    print(f"{'='*70}")
    try:
        data = get_data(name, start="1997-01-01")
        print(f"  Data range: {data.index[0].date()} to {data.index[-1].date()} ({len(data)} days)")
    except Exception as e:
        print(f"  ❌ FAILED TO LOAD DATA: {e}")
        return None
    trades = build_trades(data, vix_data, params["point_value"],
                          params["commission_rt"], params["slippage_per_side"])
    if len(trades) == 0:
        print(f"  ❌ No trades generated")
        return None
    trade_pnls = pd.Series(trades["pnl"].values, dtype=float)
    # Basic stats
    wins = (trade_pnls > 0).sum()
    losses = (trade_pnls <= 0).sum()
    gross_w = trade_pnls[trade_pnls > 0].sum()
    gross_l = abs(trade_pnls[trade_pnls < 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = wins / len(trade_pnls)
    print(f"\n  Overall Stats (post-2013):")
    print(f"    Trades: {len(trades)}")
    print(f"    Win Rate: {wr:.1%}")
    print(f"    Profit Factor: {pf:.3f}")
    print(f"    Avg Trade: ${trade_pnls.mean():,.0f}")
    print(f"    Total P&L: ${trade_pnls.sum():,.0f}")
    # Week 1 vs Week 4
    print(f"\n  Week 1 vs Week 4:")
    for wk in [1, 4]:
        subset = trades[trades["week_num"] == wk]
        if len(subset) == 0:
            continue
        sp = subset["pnl"]
        sgw = sp[sp > 0].sum()
        sgl = abs(sp[sp < 0].sum())
        spf = sgw / sgl if sgl > 0 else float("inf")
        print(f"    Week {wk}: {len(subset)} trades, PF {spf:.2f}, WR {(sp>0).mean():.1%}, "
              f"avg ${sp.mean():,.0f}")
    # Sub-period breakdown
    print(f"\n  Sub-period Breakdown:")
    print(f"  {'Period':<12} {'Trades':>7} {'PF':>8} {'WR':>7} {'Avg$':>9} {'Total$':>11}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*7} {'-'*9} {'-'*11}")
    periods = [
        ("2013-2016", 2013, 2016),
        ("2017-2019", 2017, 2019),
        ("2020-2022", 2020, 2022),
        ("2023-2026", 2023, 2026),
    ]
    period_results = []
    for label, y1, y2 in periods:
        subset = trades[(trades["year"] >= y1) & (trades["year"] <= y2)]
        if len(subset) == 0:
            period_results.append({"label": label, "profitable": False, "trades": 0})
            print(f"  {label:<12} {'0':>7}")
            continue
        sp = subset["pnl"]
        sgw = sp[sp > 0].sum()
        sgl = abs(sp[sp < 0].sum())
        spf = sgw / sgl if sgl > 0 else float("inf")
        swr = (sp > 0).mean()
        profitable = spf > 1.0
        period_results.append({"label": label, "profitable": profitable, "trades": len(subset)})
        verdict = "✅" if profitable else "❌"
        print(f"  {label:<12} {len(subset):>7} {spf:>8.2f} {swr:>6.1%} "
              f"${sp.mean():>8,.0f} ${sp.sum():>10,.0f} {verdict}")
    # Year by year
    print(f"\n  Year-by-Year:")
    for year in sorted(trades["year"].unique()):
        subset = trades[trades["year"] == year]
        sp = subset["pnl"]
        sgw = sp[sp > 0].sum()
        sgl = abs(sp[sp < 0].sum())
        spf = sgw / sgl if sgl > 0 else float("inf")
        verdict = "✅" if spf > 1.0 else "❌"
        print(f"    {year}: {len(subset):>3} trades, PF {spf:>6.2f}, "
              f"WR {(sp>0).mean():>5.1%}, total ${sp.sum():>9,.0f} {verdict}")
    # Phidias sim
    attempts = simulate_phidias(trade_pnls)
    passed = [a for a in attempts if a["status"] == "PASSED"]
    failed = [a for a in attempts if a["status"] == "FAILED"]
    total_att = len(passed) + len(failed)
    pass_rate = len(passed) / total_att * 100 if total_att > 0 else 0
    print(f"\n  Phidias 50K Swing Simulation:")
    print(f"    Pass Rate: {pass_rate:.1f}% ({len(passed)}/{total_att})")
    if passed:
        print(f"    Avg trades to pass: {np.mean([a['trades_taken'] for a in passed]):.1f}")
    # Bootstrap PF (quick version, 5000 samples)
    np.random.seed(42)
    bootstrap_pfs = []
    for _ in range(5000):
        sample = np.random.choice(trade_pnls.values, size=len(trade_pnls), replace=True)
        gw = sample[sample > 0].sum()
        gl = abs(sample[sample < 0].sum())
        bpf = gw / gl if gl > 0 else 10.0
        bootstrap_pfs.append(min(bpf, 10.0))
    pf_ci_lower = np.percentile(bootstrap_pfs, 2.5)
    pf_ci_upper = np.percentile(bootstrap_pfs, 97.5)
    print(f"\n  Bootstrap PF 95% CI: [{pf_ci_lower:.3f}, {pf_ci_upper:.3f}]")
    # Acceptance gates
    profitable_periods = sum(1 for p in period_results if p.get("profitable", False) and p.get("trades", 0) > 0)
    total_periods = sum(1 for p in period_results if p.get("trades", 0) > 0)
    gate_1 = pf > 1.5
    gate_2 = len(trades) >= 100
    gate_3 = profitable_periods == total_periods
    gate_4 = pass_rate > 45
    print(f"\n  ACCEPTANCE GATES:")
    print(f"    Gate 1: PF > 1.50         → {pf:.3f} → {'✅' if gate_1 else '❌'}")
    print(f"    Gate 2: 100+ trades       → {len(trades)} → {'✅' if gate_2 else '❌'}")
    print(f"    Gate 3: All periods +     → {profitable_periods}/{total_periods} → {'✅' if gate_3 else '❌'}")
    print(f"    Gate 4: Pass rate > 45%   → {pass_rate:.1f}% → {'✅' if gate_4 else '❌'}")
    all_passed = gate_1 and gate_2 and gate_3 and gate_4
    print(f"\n  VERDICT: {'✅ PASSED — Deploy as Phidias account' if all_passed else '❌ FAILED'}")
    if not all_passed:
        failed_gates = []
        if not gate_1: failed_gates.append(f"PF {pf:.3f}")
        if not gate_2: failed_gates.append(f"{len(trades)} trades")
        if not gate_3: failed_gates.append(f"{profitable_periods}/{total_periods} periods")
        if not gate_4: failed_gates.append(f"{pass_rate:.1f}% pass rate")
        print(f"  Failed: {', '.join(failed_gates)}")
    # Save trades
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"week14_{name.lower()}_trades.csv")
    trades.to_csv(csv_path, index=False)
    # Save equity curve
    eq_vals = [100_000.0]
    for p in trade_pnls:
        eq_vals.append(eq_vals[-1] + p)
    dates = [trades["entry_date"].iloc[0] - pd.Timedelta(days=7)]
    dates.extend(trades["entry_date"].tolist())
    eq = pd.Series(eq_vals, index=pd.DatetimeIndex(dates))
    plot_equity(eq, f"Week14 {name} (post-2013)")
    return {
        "name": name,
        "trades": len(trades),
        "pf": pf,
        "wr": wr,
        "avg_trade": trade_pnls.mean(),
        "total_pnl": trade_pnls.sum(),
        "pass_rate": pass_rate,
        "pf_ci_lower": pf_ci_lower,
        "pf_ci_upper": pf_ci_upper,
        "all_gates_passed": all_passed,
    }
def run():
    """Run thesis confirmation across all instruments."""
    print("=" * 80)
    print("WEEK 1/4 + VIX FILTER — MULTI-INSTRUMENT THESIS CONFIRMATION")
    print("Same rules on ES, NQ, RTY. Post-2013 only.")
    print("=" * 80)
    print("\nLoading VIX data...")
    vix = get_data("VIX", start="1997-01-01")
    print(f"VIX range: {vix.index[0].date()} to {vix.index[-1].date()}")
    results = {}
    for name, params in INSTRUMENTS.items():
        result = analyze_instrument(name, params, vix)
        if result:
            results[name] = result
    # =============================================
    # COMPARISON TABLE
    # =============================================
    print("\n" + "=" * 80)
    print("CROSS-INSTRUMENT COMPARISON")
    print("=" * 80)
    if not results:
        print("  No results to compare")
        return
    col_w = 18
    print(f"\n  {'Metric':<25}", end="")
    for name in results:
        print(f" {name:>{col_w}}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in results:
        print(f" {'-'*col_w}", end="")
    print()
    rows = [
        ("Trades", lambda r: f"{r['trades']}"),
        ("Profit Factor", lambda r: f"{r['pf']:.3f}"),
        ("Win Rate", lambda r: f"{r['wr']:.1%}"),
        ("Avg Trade ($)", lambda r: f"${r['avg_trade']:,.0f}"),
        ("Total P&L ($)", lambda r: f"${r['total_pnl']:,.0f}"),
        ("Bootstrap CI", lambda r: f"[{r['pf_ci_lower']:.2f}-{r['pf_ci_upper']:.2f}]"),
        ("Phidias Pass Rate", lambda r: f"{r['pass_rate']:.1f}%"),
        ("All Gates Passed", lambda r: "✅ YES" if r["all_gates_passed"] else "❌ NO"),
    ]
    for label, fmt_fn in rows:
        print(f"  {label:<25}", end="")
        for name in results:
            val = fmt_fn(results[name])
            print(f" {val:>{col_w}}", end="")
        print()
    # =============================================
    # THESIS VERDICT
    # =============================================
    print("\n" + "=" * 80)
    print("THESIS CONFIRMATION VERDICT")
    print("=" * 80)
    passed_instruments = [name for name, r in results.items() if r["all_gates_passed"]]
    failed_instruments = [name for name, r in results.items() if not r["all_gates_passed"]]
    print(f"\n  Instruments that passed all gates: {', '.join(passed_instruments) if passed_instruments else 'None'}")
    print(f"  Instruments that failed: {', '.join(failed_instruments) if failed_instruments else 'None'}")
    if len(passed_instruments) >= 2:
        print(f"\n  ✅ THESIS CONFIRMED — Institutional flow effect present in {len(passed_instruments)} instruments")
        print(f"     The calendar effect is not ES-specific. It's a structural market phenomenon.")
        print(f"     Deploy {len(passed_instruments)} independent Phidias accounts:")
        for name in passed_instruments:
            r = results[name]
            print(f"       {name}: PF {r['pf']:.2f}, {r['pass_rate']:.0f}% pass rate")
        total_cost = len(passed_instruments) * 116
        print(f"     Total eval cost: ${total_cost} ({len(passed_instruments)} × $116)")
    elif len(passed_instruments) == 1:
        print(f"\n  ⚠️  PARTIALLY CONFIRMED — Only {passed_instruments[0]} passed all gates")
        print(f"     The effect may be concentrated in that instrument specifically.")
        print(f"     Stick with {passed_instruments[0]} only.")
    else:
        print(f"\n  ❌ THESIS NOT CONFIRMED across instruments")
        print(f"     ES may still work standalone — check its individual results above.")
    # Correlation analysis between instruments
    if len(passed_instruments) >= 2:
        print(f"\n  CORRELATION CHECK (monthly P&L between passed instruments):")
        for i, name_a in enumerate(passed_instruments):
            for name_b in passed_instruments[i+1:]:
                trades_a = pd.read_csv(os.path.join(RESULTS_DIR, f"week14_{name_a.lower()}_trades.csv"),
                                       parse_dates=["entry_date"])
                trades_b = pd.read_csv(os.path.join(RESULTS_DIR, f"week14_{name_b.lower()}_trades.csv"),
                                       parse_dates=["entry_date"])
                monthly_a = trades_a.set_index("entry_date")["pnl"].resample("ME").sum()
                monthly_b = trades_b.set_index("entry_date")["pnl"].resample("ME").sum()
                common = monthly_a.index.intersection(monthly_b.index)
                if len(common) >= 12:
                    corr = monthly_a.loc[common].corr(monthly_b.loc[common])
                    print(f"    {name_a} vs {name_b}: {corr:.3f} monthly correlation")
                    if corr > 0.7:
                        print(f"    ⚠️  High correlation — running both adds limited diversification")
                    else:
                        print(f"    ✅ Moderate/low correlation — running both adds diversification")
    print("\n" + "=" * 80)
if __name__ == "__main__":
    run()
