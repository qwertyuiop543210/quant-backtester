# Combined Alert System — Chosen One + Dip Buyer

A TradingView Pine Script v5 indicator that generates daily Telegram alerts for two ES futures trading strategies via a single webhook.

## Architecture

**Single alert, daily heartbeat.** One `alert()` call fires on every bar close with a dynamically constructed message. Silent = broken system, not "no action needed."

- Action messages (🟢🔴🔵✅) tell you exactly what to do
- Heartbeat messages (💤) confirm the system is alive
- Warning messages (⚠️) mean something needs manual attention

---

## Setup Guide

### Step 1: Load the Indicator

1. Open TradingView and navigate to the **ES1!** (E-mini S&P 500 continuous futures) chart
2. Set the timeframe to **1D** (Daily)
3. Click the **Indicators** button (fx icon) in the top toolbar
4. Go to **My Scripts** tab (if you've saved it) or use the **Pine Editor**
5. If using Pine Editor:
   - Click **Open** → **New indicator**
   - Delete the default code
   - Paste the entire contents of `combined_alerts.pine`
   - Click **Save** (name it "CO+DB Alerts")
   - Click **Add to Chart**
6. You should see:
   - A separate indicator pane below the chart with RSI(2) and VIX plotted
   - Background colors on the main chart (green = CO active, blue = DB active, yellow = DB blocked)
   - An info table in the top-right corner of the main chart

### Step 2: Create the Alert

1. Right-click on the chart → **Add Alert**, or press `Alt+A`
2. Configure the alert:
   - **Condition**: Select "Combined Alert System — Chosen One + Dip Buyer"
   - **Trigger**: Select **"Any alert() function call"**
   - **Frequency**: Once Per Bar Close
   - **Expiration**: Set to maximum available (or "Open-ended" on Premium/higher plans)
3. Under **Notifications**:
   - Enable **Webhook URL**
   - Enter your Cloudflare Worker URL: `https://your-worker.workers.dev`
4. **Alert name**: "CO+DB Daily Alert" (optional, for your reference)
5. Click **Create**

> **Important:** The alert message field can be left as default. The indicator dynamically sets the message content via `alert()`.

### Step 3: Cloudflare Worker Setup

Your Cloudflare Worker acts as a relay between TradingView webhooks and Telegram.

1. **Get a Telegram Bot Token**: Message [@BotFather](https://t.me/BotFather) on Telegram, create a bot, save the token
2. **Get your Chat ID**: Message [@userinfobot](https://t.me/userinfobot) to get your numeric chat ID
3. **Deploy the Worker** with these environment variables:
   - `TELEGRAM_BOT_TOKEN` — your bot token
   - `TELEGRAM_CHAT_ID` — your chat ID
4. The Worker receives the raw alert text as the request body and forwards it to Telegram via the Bot API
5. Test by manually triggering a webhook to your Worker URL

---

## Message Types

### Action Messages

These require you to do something. They arrive at market close (4:00 PM ET) on the daily bar.

#### 🟢 Chosen One — Trade Monday (Friday heads-up)
```
🟢 CHOSEN ONE — TRADE MONDAY
Week 1 | VIX: 22.45
Buy 1 ES at 9:30 AM open Monday
---
RSI(2): 45.32 | Dip Buyer: HUNTING
```
**What to do:** Place a market order for 1 ES at Monday's open (9:30 AM ET).

#### 🔴 Chosen One — Skip Monday (Friday heads-up)
```
🔴 CHOSEN ONE — SKIP MONDAY
Week 4 | VIX: 17.50 (in 15-20 dead zone)
No trade this week
---
RSI(2): 52.10 | Dip Buyer: BLOCKED (CO week)
```
**What to do:** Nothing. VIX is in the 15-20 dead zone. No trade this week.

#### ✅ Chosen One — Position Open (Monday confirmation)
```
✅ CHOSEN ONE — POSITION OPEN
Week 1 | Entry at Monday open
Hold until Friday close
---
RSI(2): 48.20 | VIX: 23.10
```
**What to do:** Confirms you should be long 1 ES. Hold until Friday.

#### 🔴 Chosen One — Close Position (Friday exit)
```
🔴 CHOSEN ONE — CLOSE POSITION
Week 1 | Friday close
Sell 1 ES before 4:00 PM today
---
RSI(2): 55.30 | VIX: 21.80
```
**What to do:** Close your ES position before 4:00 PM ET today.

#### 🔵 Dip Buyer — Buy Tomorrow (entry signal)
```
🔵 DIP BUYER — BUY TOMORROW
RSI(2): 7.85 | VIX: 25.30
Buy 1 ES at 9:30 AM open tomorrow
---
Chosen One: INACTIVE (Week 2)
```
**What to do:** Place a market order for 1 ES at tomorrow's open (9:30 AM ET).

If the signal fires on Friday:
```
🔵 DIP BUYER — BUY TOMORROW
RSI(2): 6.20 | VIX: 28.10
Buy 1 ES at Monday 9:30 AM open
---
Chosen One: INACTIVE (Week 3)
```

#### 🔵 Dip Buyer — Close (RSI exit)
```
🔵 DIP BUYER — CLOSE POSITION
RSI(2): 68.42 crossed above 65 at today's close
Sell position in after-hours or at tomorrow's open
Day 3 of 5
---
Chosen One: INACTIVE (Week 2)
```
**What to do:** The RSI exit fired at today's close. You cannot sell at a close that already happened. Sell in after-hours or at tomorrow's open. This ~1 day slippage vs backtest is a known limitation.

#### 🔵 Dip Buyer — Close (time stop)
```
🔵 DIP BUYER — CLOSE POSITION (TIME STOP)
RSI(2): 42.15 | Day 5 of 5
Sell 1 ES before 4:00 PM today
---
Chosen One: INACTIVE (Week 3)
```
**What to do:** Close your ES position before 4:00 PM ET today. The 5-day max hold has been reached.

#### ⚠️ System Warning
```
⚠️ SYSTEM WARNING
VIX data unavailable — check manually before trading
RSI(2): 35.20 | Week: 2
```
**What to do:** Check VIX manually. Do not enter any new trades until VIX data is confirmed.

### Heartbeat Messages

No action needed. Confirms the system is alive and monitoring.

```
💤 DAILY STATUS
Week 2 | Day: Wed
VIX: 18.50 | RSI(2): 52.30
Chosen One: INACTIVE (Week 2)
Dip Buyer: HUNTING
Next action: Monitoring for signals
```

During active positions:
```
💤 DAILY STATUS
Week 1 | Day: Wed
VIX: 22.10 | RSI(2): 45.80
Chosen One: ACTIVE Day 3/5
Dip Buyer: BLOCKED (CO week)
Next action: CO holding, exit Friday close
```

Day 4 early warning for Dip Buyer:
```
💤 DAILY STATUS
Week 3 | Day: Thu
VIX: 26.40 | RSI(2): 48.90
Chosen One: INACTIVE (Week 3)
Dip Buyer: ACTIVE Day 4/5
Next action: DB Day 4/5 — time stop tomorrow if RSI < 65
```

---

## Weekly Workflow Examples

### Example 1: Active Chosen One Week (Trade)

| Day | Message | Action |
|-----|---------|--------|
| Fri (prior week) | 🟢 CHOSEN ONE — TRADE MONDAY (Week 1, VIX: 22.45) | Prepare to buy Monday |
| Mon | ✅ CHOSEN ONE — POSITION OPEN (Week 1) | Confirm long 1 ES |
| Tue | 💤 DAILY STATUS (CO: ACTIVE Day 2/5) | Hold |
| Wed | 💤 DAILY STATUS (CO: ACTIVE Day 3/5) | Hold |
| Thu | 💤 DAILY STATUS (CO: ACTIVE Day 4/5) | Hold |
| Fri | 🔴 CHOSEN ONE — CLOSE POSITION | Sell before 4 PM |

### Example 2: Active Chosen One Week (VIX Skip)

| Day | Message | Action |
|-----|---------|--------|
| Fri (prior week) | 🔴 CHOSEN ONE — SKIP MONDAY (VIX: 17.50, dead zone) | No trade |
| Mon | 💤 DAILY STATUS (CO: SKIP VIX 15-20) | Nothing |
| Tue-Fri | 💤 DAILY STATUS | Normal monitoring |

### Example 3: Dip Buyer Signal During Off-Week

| Day | Message | Action |
|-----|---------|--------|
| Mon | 💤 DAILY STATUS (Week 2, DB: HUNTING) | Monitoring |
| Tue | 🔵 DIP BUYER — BUY TOMORROW (RSI: 8.20, VIX: 27.50) | Buy Wed open |
| Wed | 💤 DAILY STATUS (DB: ACTIVE Day 1/5) | Holding |
| Thu | 💤 DAILY STATUS (DB: ACTIVE Day 2/5) | Holding |
| Fri | 🔵 DIP BUYER — CLOSE POSITION (RSI: 68.42 > 65) | Sell after-hours/next open |

### Example 4: Quiet Week (Heartbeat Only)

| Day | Message | Action |
|-----|---------|--------|
| Mon | 💤 DAILY STATUS (Week 2, VIX: 14.20, RSI: 55) | Nothing |
| Tue | 💤 DAILY STATUS | Nothing |
| Wed | 💤 DAILY STATUS | Nothing |
| Thu | 💤 DAILY STATUS | Nothing |
| Fri | 💤 DAILY STATUS | Nothing |

### Example 5: Both Strategies in Same Month

| Week | What Happens |
|------|-------------|
| Week 1 (CO) | 🟢 Friday heads-up → ✅ Monday entry → 🔴 Friday exit |
| Week 2 | 💤 Heartbeat. Tuesday: 🔵 DB signal → Wed entry → Thu RSI exit |
| Week 3 | 💤 Heartbeat only. DB hunting but RSI never drops below 10 |
| Week 4 (CO) | 🟢 Friday heads-up → ✅ Monday entry → 🔴 Friday exit |

---

## Strategy Rules Quick Reference

| | Chosen One | Dip Buyer |
|---|-----------|-----------|
| **When** | Week 1 & Week 4 (Mon open → Fri close) | Any non-CO week when conditions met |
| **Entry signal** | Calendar (automatic) | RSI(2) < 10 AND VIX > 20 AND VIX < 35 |
| **Entry execution** | Monday 9:30 AM open | Next day's 9:30 AM open |
| **Exit** | Friday close | RSI(2) > 65 (at close) OR 5 trading days |
| **VIX filter** | Skip if Friday VIX 15.0-20.0 | Must be > 20 AND < 35 |
| **Position size** | 1 ES contract | 1 ES contract |
| **Stop loss** | None | None |
| **Overlap rule** | Priority — always trades first | Blocked during CO weeks + 5-day buffer |

### Week of Month Calculation

```
week_of_month = floor((monday_day_of_month - 1) / 7) + 1
```

- Always based on the **Monday** that starts the trading week
- Week 1: Monday falls on days 1-7
- Week 2: Monday falls on days 8-14
- Week 3: Monday falls on days 15-21
- Week 4: Monday falls on days 22-28
- Week 5: Monday falls on days 29-31 (rare, no CO trade)

### Lookahead Buffer

Dip Buyer cannot enter if there are fewer than 5 **weekdays** (Mon-Fri) between today and the next Chosen One Monday. This prevents the hold period from overlapping with a CO week.

---

## Troubleshooting

### "I stopped getting messages"
- **Most likely:** Your TradingView alert expired. Free plans have short expiration limits.
- **Fix:** Open TradingView → Alerts panel → check if the alert is active. Re-create if expired.
- **Also check:** Is the ES1! chart still loaded with the indicator? Did TradingView log you out?

### "VIX shows as N/A"
- **Cause:** CBOE VIX data feed issue in TradingView.
- **Impact:** System fires ⚠️ warning and defaults to SKIP for both strategies.
- **Fix:** Check if CBOE:VIX chart loads separately. Usually resolves within hours. Check VIX manually before trading.

### "RSI exit came after market close"
- **This is by design.** The RSI > 65 condition is evaluated on the bar's close. By the time you receive the Telegram message, the market has closed.
- **What to do:** Sell in after-hours (if your broker supports ES after-hours) or at the next day's open.
- **Impact:** ~1 day slippage vs the backtest. The backtest assumes exit at close, but in practice you exit at the next opportunity.

### "Dip Buyer blocked but RSI < 10"
- **This is by design.** The overlap filter or lookahead buffer is active.
- **Check the message:** It will say "BLOCKED (CO week)" or "BLOCKED (buffer <5d)".
- **Why:** Only one position at a time. Chosen One has priority. The buffer prevents Dip Buyer from entering within 5 trading days of a CO week.

### "No Monday confirmation after Friday heads-up"
- **Possible cause:** Monday was a market holiday (e.g., MLK Day, Presidents Day). No bar = no alert.
- **What to do:** If Friday said TRADE MONDAY and there's no Monday message, check if the market was closed. If it was open, check that the alert is still active.

### "Two signals on the same day"
- **This can happen** when CO exits on Friday and a heads-up fires for the next CO week in the same message.
- **Both actions appear** in a single Telegram message separated by a blank line.
- **Read the entire message** before acting.

---

## Phidias-Specific Notes

This system is designed for the **Phidias $50K Swing evaluation**:

- **Position size:** Always 1 ES mini contract. Never more than 1 at a time.
- **Drawdown:** Checked on closed P&L only (EOD drawdown). No intraday drawdown rule.
- **Overnight holds:** Both strategies hold overnight. **Swing account required** (not Day Trading account).
- **Over-weekend holds:** Chosen One holds Mon-Fri (over 4 nights). Dip Buyer may hold over weekends if the position spans Friday → Monday.
- **Commission:** Budget $5 round-trip + ~$25 slippage per trade.
- **Profit target:** $4,000 cumulative. With average trade P&L from backtests, expect to need ~10-20 trades.
- **Account rules:**
  - 1 mini ES contract maximum overnight
  - No scaling in/out — always flat or 1 contract
  - Track P&L manually alongside Phidias dashboard

### Risk Management

- **No stop losses** are built into either strategy. The backtest shows this is optimal for these specific setups, but be aware of tail risk.
- **Max Chosen One drawdown:** Depends on weekly ES moves. A 5% weekly drop on ES at ~5000 = ~$12,500 loss on 1 contract (250 points × $50).
- **Max Dip Buyer drawdown:** Limited by the 5-day time stop. A sustained decline after RSI < 10 entry could result in significant loss.
- **Combined risk:** Since only one position is active at a time, max exposure is always 1 ES contract.

---

## Chart Visuals Reference

| Color | Meaning |
|-------|---------|
| **Green background** | Chosen One position active (main chart) |
| **Blue background** | Dip Buyer position active (main chart) |
| **Yellow background** | Dip Buyer blocked by overlap filter or buffer (main chart) |
| **Green RSI line** | RSI(2) below 10 (entry zone) |
| **Red RSI line** | RSI(2) above 65 (exit zone) |
| **White RSI line** | RSI(2) in neutral zone |
| **Orange line** | VIX close (indicator pane) |
| **Red shading** | VIX 15-20 dead zone (indicator pane) |
| **Blue shading** | VIX 20-35 Dip Buyer zone (indicator pane) |

### Info Table (top-right of main chart)

| Field | Values |
|-------|--------|
| Week / Day | W1 Mon, W2 Tue, etc. |
| Chosen One | ACTIVE Day X/5, SKIP (VIX 15-20), INACTIVE (WX) |
| Dip Buyer | HUNTING, ACTIVE Day X/5, BLOCKED (CO week), BLOCKED (buffer <5d), INACTIVE (VIX), ENTRY PENDING |
| RSI(2) | Numeric value, green if < 10, red if > 65 |
| VIX | Numeric value, red if 15-20, green if 20-35 |
| Days to CO | Trading days until next Chosen One week |
| Next Action | Brief description of upcoming event |
