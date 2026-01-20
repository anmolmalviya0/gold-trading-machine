# THE ULTIMATE MACHINE

## Quick Start

```bash
# 1. Install dependencies
cd /Users/anmol/Desktop/gold/trading_intelligence
pip3 install -r requirements.txt

# 2. Train ML models (first time only)
python3 demo_train.py

# 3. Run THE ULTIMATE MACHINE
python3 ultimate_machine.py
```

## What It Does

🎯 **Real-time intelligence for BTC & PAXG across 4 timeframes**

- Connects to Binance WebSocket (live data)
- Builds 5m/15m/30m/1h candles
- Engineers 60+ technical features
- 3-engine ML ensemble predictions
- Multi-timeframe consensus voting
- Calculates exact entry/SL/TP levels
- Logs all signals to SQLite database
- Sends Telegram alerts (high-confidence only)
- Self-monitoring watchdog

## Dashboard

```bash
streamlit run ui/streamlit_app.py
```

Visual interface with trade plan cards.

## Telegram Setup (Optional)

```bash
python3 -c "from alerts.telegram import setup_telegram; setup_telegram()"
```

Follow prompts to configure Telegram notifications.

## System Architecture

```
Binance → Candles → Features → ML Ensemble → Signals → Alerts
                                               ↓
                                         Performance DB
                                               ↓
                                           Dashboard
```

## Output Format

Every strong signal includes:

- **Signal**: BUY/SELL
- **Confidence**: 0-100%
- **Bias Strength**: STRONG/MODERATE/WEAK
- **Regime**: TREND/CHOP/MEANREV
- **Entry Zone**: Price range
- **Stop Loss**: Exact price
- **Take Profits**: TP1/TP2/TP3
- **Pip Targets**: Points/pips for each TP
- **Risk:Reward**: Ratio

## Logs

- Console: Human-readable
- File: JSON structured (`logs/trading_system_YYYYMMDD.log`)

## Database

SQLite: `data/performance.db`

Query signals:
```python
from data.performance_db import PerformanceDB

with PerformanceDB() as db:
    signals = db.get_signals(asset='BTC', days=7)
    summary = db.get_summary()
    print(summary)
```

## Monitoring

Watchdog checks every 30s:
- Memory usage (alert if > 2GB)
- CPU usage (alert if > 90%)
- Uptime tracking

## Production Deployment

### systemd Service (Linux)

```ini
[Unit]
Description=Trading Intelligence System
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/trading_intelligence
ExecStart=/usr/bin/python3 ultimate_machine.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### macOS LaunchAgent

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading.intelligence</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/anmol/Desktop/gold/trading_intelligence/ultimate_machine.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

## Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="your_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

## Important Notes

⚠️ **This is a decision-support system**
- Does NOT auto-execute trades
- Provides intelligence for manual trading
- User makes final execution decision

✅ **Safety First**
- All signals are recommendations
- Always verify before executing
- Use proper position sizing
- Never risk more than you can afford to lose

## Support

Logs are your friend. Check `logs/` directory for detailed JSON logs of all system activity.

---

**THE ULTIMATE MACHINE** - Institutional-grade trading intelligence, locally on your Mac.
