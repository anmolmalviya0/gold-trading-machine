# TERMINAL - Institutional Quantitative Trading System ("The Switchblade")

> **"The goal is not action. The goal is Alpha."**

TERMINAL is a high-frequency, multi-timeframe quantitative trading engine designed for the cryptocurrency and commodities markets. It leverages an ensemble of LightGBM models ("Switchblade Protocol") to detect high-probability setups with institutional-grade risk management.

![System Architecture](https://img.shields.io/badge/Architecture-Event--Driven-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-LIVE_PAPER_TRADING-green?style=for-the-badge&animate=pulse)
![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)

---

## 🏗️ Architecture

The system is composed of four autonomous subsystems operating in a distributed environment:

| **Dashboard** | Real-time Visualization | Next.js / React |
| **Intel Scout (`intel_scout.py`)** | Sovereign Reasoning (Gemini/Perplexity) | Python / LLM |
| **Eternal Guard (`eternal_guard.sh`)** | 24/7 Lid-Close Station Hardening | Bash / Caffeinate |

---

## 🚀 Installation & Setup

### Prerequisites
- macOS (tested on Sonoma/Sequoia)
- Anaconda (Python 3.10+)
- `talib` (Technical Analysis Library)

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd aegis_alpha
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Ensure talib is installed via conda if pip fails:
   # conda install -c conda-forge ta-lib
   ```

3. **Sovereign Ignition (Option 1 - FAST & FREE)**
   ```bash
   ./eternal_guard.sh
   ```
   *This keeps the station alive 24/7, even with the lid closed. Required for iPhone awareness.*

4. **Integrate Intelligence (Phase 4)**
   - Copy `.env.example` to `.env`
   - Inject your Binance, Gemini, and Perplexity keys into the Vault.
   - Run `python3 src/intel_scout.py` to ignite the Sovereign Mind.

---

## 🕹️ Operations

### The Command Center
Interact with the bot using the daemon controller:

```bash
# Check Status (The Heartbeat)
./daemon.sh status

# Monitor Logs (The Mind)
./daemon.sh logs

# Emergency Stop (The Kill Switch)
./daemon.sh stop

# Restart (The Defibrillator)
./daemon.sh restart
```

### Safety & Risk Management ("The Hard Deck")
- **Risk per Trade**: Max 3% of Equity (Hard Coded)
- **Stop Loss**: Dynamic -1.5 ATR
- **Take Profit**: Dynamic +2.0 ATR
- **Confidence Threshold**: Signals ignored below 65% confidence.

---

## 📡 Signal Logic (Switchblade Protocol)

The system avoids "over-trading" by filtering for high-conviction setups.

- **HOLD (Neutral)**: Confidence < 0.35 OR Confidence < 0.65 but no strong directional bias.
- **BUY (Long)**: Confidence > 0.65 + Multi-Timeframe Confirmation.
- **SELL (Short)**: Confidence > 0.65 + Multi-Timeframe Confirmation.

*Current Status: 14-Day Paper Trading Trial (Day 1).*

---

## 🔧 Troubleshooting

**"Operation not permitted" Error:**
If macOS blocks execution, run the Defibrillator:
```bash
xattr -cr src/
chmod +x src/*.py
./daemon.sh restart
```

---

*© 2026 AEGIS Quantitative Research. Proprietary Code.*
