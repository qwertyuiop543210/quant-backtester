# ES Combined Alert System — Chosen One + Dip Buyer

A single TradingView Pine Script indicator that sends daily Telegram alerts for two ES futures strategies via webhook.

## Quick Start

### 1. Add the Indicator to TradingView

1. Open TradingView and load the **ES1!** (ES Futures Continuous) chart
2. Set the timeframe to **Daily (1D)**
3. Open Pine Editor (bottom panel) → click "Open" → paste the contents of `combined_alerts.pine`
4. Click "Add to chart"
5. Verify: you should see green/blue background highlights on historical data and a status table in the top-right corner

### 2. Create the Single Alert

1. Click the **Alert** button (clock icon) on the right toolbar
2. Settings:
   - **Condition**: select "ES Combined Alerts — Chosen One + Dip Buyer"
   - **Sub-condition**: "Any alert() function call"
   - **Trigger**: "Once Per Bar Close"
   - **Expiration**: set to maximum available (or "Open-ended" on Premium+)
3. **Notifications** tab:
   - Check **Webhook URL**
   - Enter your Cloudflare Worker URL: `https://your-worker.workers.dev`
4. **Message**: leave as-is (the indicator generates dynamic messages)
5. Click **Create**

> **Important**: You only create ONE alert. The indicator fires it every single day with a different message depending on what's happening.

### 3. Cloudflare Worker Setup (Telegram Relay)

Your Cloudflare Worker receives the TradingView webhook and forwards it to Telegram.

Required environment variables (set as Worker secrets):
- `TELEGRAM_BOT_TOKEN` — from @BotFather on Telegram
- `TELEGRAM_CHAT_ID` — your personal chat or group ID

The Worker receives a POST request with the alert message as the body and forwards it to the Telegram Bot API:
```
POST https://api.telegram.org/bot{TOKEN}/sendMessage
{
  "chat_id": "{CHAT_ID}",
  "text": "{webhook body}",
  "parse_mode": "HTML"
}
```

## Message Types

### Action Messages (do something)

| Emoji | Message | Action Required |
|-------|---------|-----------------|
| `🟢` | CHOSEN ONE — TRADE MONDAY | Buy 1 ES at Monday 9:30 AM open |
| `🔴` | CHOSEN ONE — SKIP MONDAY | No trade — VIX in 15-20 dead zone |
| `✅` | CHOSEN ONE — POSITION OPEN | Confirmation: position entered at Monday open |
| `🔴` | CHOSEN ONE — CLOSE POSITION | Sell 1 ES before 4:00 PM Friday |
| `🔵` | DIP BUYER — BUY TOMORROW | Buy 1 ES at 9:30 AM open next trading day |
| `🔵` | DIP BUYER — CLOSE POSITION | Sell: RSI crossed above 65 (sell after-hours/next open) |
| `🔵` | DIP BUYER — CLOSE POSITION (TIME STOP) | Sell 1 ES before 4:00 PM today (day 5) |

### Heartbeat Messages (system alive, no action)

| Emoji | Message | Meaning |
|-------|---------|---------|
| `💤` | DAILY STATUS | No action needed — system is running normally |
| `⚠️` | SYSTEM WARNING | VIX data unavailable — check manually |

### If Messages Stop Arriving

The system is designed so that **silence = broken**. Every trading day should produce a message. If you stop receiving messages:

1. Check the TradingView alert — it may have expired
2. Check your Cloudflare Worker logs for errors
3. Check Telegram bot status
4. Re-create the alert if needed

## Example Weekly Workflows

### Active Chosen One Week (Traded)

```
Friday (week before):
🟢 CHOSEN ONE — TRADE MONDAY
Week 1 | VIX: 22.50
Buy 1 ES at 9:30 AM open Monday
---
RSI(2): 45.23 | VIX: 22.50
CO: INACTIVE (Week 5) | DB: HUNTING

Monday:
✅ CHOSEN ONE — POSITION OPEN
Week 1 | Entry at Monday open
Hold until Friday close
---
RSI(2): 43.10 | VIX: 22.80
CO: ACTIVE Day 1/5 | DB: BLOCKED (CO week)

Tuesday–Thursday:
💤 DAILY STATUS
Week 1 | Day: Tue
VIX: 21.90 | RSI(2): 51.30
Chosen One: ACTIVE Day 2/5
Dip Buyer: BLOCKED (CO week)
Next: CO close Friday (Day 2/5)

Friday:
🔴 CHOSEN ONE — CLOSE POSITION
Week 1 | Friday close
Sell 1 ES before 4:00 PM today
---
RSI(2): 55.40 | VIX: 21.50
CO: ACTIVE Day 5/5 | DB: BLOCKED (CO week)
```

### Active Chosen One Week (VIX Skip)

```
Friday (week before):
🔴 CHOSEN ONE — SKIP MONDAY
Week 4 | VIX: 17.30 (in 15-20 dead zone)
No trade this week
---
RSI(2): 60.10 | VIX: 17.30
CO: INACTIVE (Week 3) | DB: HUNTING

Monday:
💤 DAILY STATUS
Week 4 | Day: Mon
VIX: 17.50 | RSI(2): 58.20
Chosen One: SKIP (VIX 15-20)
Dip Buyer: BLOCKED (CO week)
Next: Watching for signals
```

### Dip Buyer Signal During Off-Week

```
Wednesday:
🔵 DIP BUYER — BUY TOMORROW
RSI(2): 8.45 | VIX: 24.30
Buy 1 ES at 9:30 AM open tomorrow
---
RSI(2): 8.45 | VIX: 24.30
CO: INACTIVE (Week 2) | DB: ENTRY PENDING

Thursday (entry day):
💤 DAILY STATUS
Week 2 | Day: Thu
VIX: 25.10 | RSI(2): 15.20
Chosen One: INACTIVE (Week 2)
Dip Buyer: ACTIVE Day 1/5
Next: DB watching RSI > 65 or Day 5 stop

Friday:
🔵 DIP BUYER — CLOSE POSITION
RSI(2): 68.30 crossed above 65 at today's close
Sell position in after-hours or at tomorrow's open
Day 2 of 5
---
RSI(2): 68.30 | VIX: 23.80
CO: INACTIVE (Week 2) | DB: ACTIVE Day 2/5
```

### Quiet Week (Heartbeat Only)

```
Monday–Friday:
💤 DAILY STATUS
Week 3 | Day: Mon
VIX: 14.20 | RSI(2): 52.00
Chosen One: INACTIVE (Week 3)
Dip Buyer: INACTIVE (VIX 14.20)
Next: Watching for signals
```

### Both Strategies Active in Same Month

```
Week 1: Chosen One trades (Mon open → Fri close)
Week 2: Dip Buyer can hunt (RSI < 10 + VIX 20-35 required)
Week 3: Dip Buyer can hunt (with lookahead buffer: blocked last ~5 days)
Week 4: Chosen One trades (Mon open → Fri close)
Week 5: Dip Buyer can hunt (if month has 5 weeks)
```

The overlap filter ensures only one position at a time. Chosen One has priority — if a CO week is approaching, the Dip Buyer entry is blocked by the lookahead buffer.

## Troubleshooting

### "I stopped getting messages"
- **Most likely**: the TradingView alert expired. TradingView free/Plus plans have alert expiration limits. Re-create the alert.
- Check your Cloudflare Worker logs (`wrangler tail`) for POST failures.
- Verify the Telegram bot hasn't been blocked — send it a test message.

### "VIX shows as N/A"
- The CBOE:VIX data feed on TradingView occasionally has delays or gaps.
- The indicator will fire a `⚠️ SYSTEM WARNING` alert.
- Check VIX manually on the CBOE website or another source before making any trade decisions.

### "RSI exit came after market close"
- **This is expected behavior.** The RSI exit signal fires when the daily bar closes (4:00 PM ET). By definition, you can't sell at a close that already happened.
- The alert message says: "Sell position in after-hours or at tomorrow's open."
- This creates ~1 day of slippage vs the backtest. The backtest assumes exit at the close where RSI > 65, but in live trading you'll exit the following session.
- This is a known limitation documented in the indicator comments.

### "Dip Buyer blocked but RSI < 10"
- The overlap filter or lookahead buffer is active. This is **by design**.
- Check the status table for the specific block reason:
  - **BLOCKED (CO week)**: current week is Week 1 or Week 4
  - **BLOCKED (buffer Xd)**: fewer than 5 trading days until the next CO week starts
- The Dip Buyer will resume hunting once the block condition clears.

### "Week number seems wrong"
- Week of month is calculated as `(dayofmonth - 1) / 7 + 1`:
  - Days 1–7 = Week 1
  - Days 8–14 = Week 2
  - Days 15–21 = Week 3
  - Days 22–28 = Week 4
  - Days 29–31 = Week 5
- This is a fixed calendar formula, not ISO week numbers.

## Strategy Rules Quick Reference

| Parameter | Chosen One | Dip Buyer |
|-----------|-----------|-----------|
| **Instrument** | ES (1 contract) | ES (1 contract) |
| **Direction** | Long only | Long only |
| **Entry trigger** | Calendar (Week 1 & 4 Monday) | RSI(2) < 10 + VIX 20-35 |
| **Entry execution** | Monday 9:30 AM open | Next day 9:30 AM open |
| **Exit trigger** | Calendar (Friday close) | RSI(2) > 65 or 5-day time stop |
| **Exit execution** | Friday before 4:00 PM | At close (RSI) or before 4 PM (time stop) |
| **VIX filter** | Skip if VIX 15.0-20.0 | Require VIX > 20 and < 35 |
| **Stop loss** | None | None |
| **Max hold** | 5 trading days (Mon-Fri) | 5 trading days |
| **Priority** | Higher (blocks Dip Buyer) | Lower (defers to Chosen One) |

## Phidias $50K Swing Account Notes

- **Position size**: 1 ES contract at a time (never more)
- **Account type**: Swing account required (positions held overnight and over weekends)
- **Drawdown check**: end-of-day closed P&L only (no intraday margin calls in the backtest)
- **Profit target**: $4,000 cumulative
- **EOD drawdown limit**: $2,500 from starting balance
- **Costs**: $5 round-trip commission + $25 slippage ($12.50 per side) = $30 per trade
- **Overnight**: both strategies hold overnight by design — Chosen One holds Mon→Fri, Dip Buyer holds 1-5 days
- **One position at a time**: the overlap filter guarantees you never have two ES positions simultaneously
