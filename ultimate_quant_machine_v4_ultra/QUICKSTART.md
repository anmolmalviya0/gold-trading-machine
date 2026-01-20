# V4 PRO MAX ULTRA - QUICK REFERENCE

## One-Line Commands

```bash
# Setup & install
./setup.sh

# Download all historical data (2021-2026)
python main.py download

# Backtest 15m PAXG
python main.py backtest --symbol PAXGUSDT --tf 15m

# Walk-forward validation (CRITICAL)
python main.py backtest --symbol PAXGUSDT --tf 15m --walkforward

# Live paper trading (manual execution)
python main.py live --symbol PAXGUSDT
```

## Key Files (What They Do)

| File | Purpose |
|------|---------|
| `main.py` | Entry point (backtest/live/download) |
| `config.yaml` | All thresholds (spread, RSI, ATR, scoring, risk) |
| `indicators.py` | Wilder RSI/ATR, pivots, divergence, sweeps |
| `strategy_engine.py` | 0-100 score, trade card generation |
| `conviction_monitor.py` | USD gates (spread, depth, slippage, OBI) |
| `backtester.py` | Strict fees, slippage, TP/SL sequencing |
| `walkforward.py` | 2021/2024/2025 split validation |
| `live_ws.py` | Binance WebSocket (candle close only) |
| `db.py` | SQLite WAL, dedupe, persistence |
| `alerts.py` | Telegram + local logging |
| `download_history.py` | Fetch Binance historical data |

## What Each Phase Does

| Phase | What | Files |
|-------|------|-------|
| 0 | Rules of reality (50-60% global) | None (conceptual) |
| 1 | Data (WS + REST) | `live_ws.py` |
| 2 | Features (indicators) | `indicators.py` |
| 3 | Signals (score 0-100) | `strategy_engine.py` |
| 4 | Conviction (USD gates) | `conviction_monitor.py` |
| 5 | Risk (TP/SL ratios) | `strategy_engine.py` |
| 6 | Backtest (honest) | `backtester.py` |
| 7 | Live (trade cards) | `live_ws.py`, `alerts.py` |
| 8 | Safety (dedupe, WAL) | `db.py` |

## Important Thresholds (in config.yaml)

```yaml
# Scoring (0-100, trade if ≥ 80)
divergence_confirmed: 30    # Pivot-to-pivot
trend_alignment: 20         # EMA50 > EMA200
liquidity_sweep: 15         # Wick beyond pivot
regime_good: 15            # ADX > 25 or BB Wide
conviction_pass: 20        # All USD gates pass

# Conviction gates (ALL must pass)
spread_filter.PAXG: 0.0008  # < 0.08%
spread_filter.BTC: 0.0004   # < 0.04%
depth_usd_min: 50000        # $50k minimum
slippage_impact_max: 0.001  # < 0.1%
obi_usd.buy_min: 0.80       # Bid/Ask ratio

# Risk (pick ONE mode)
win_rate_mode.tp_ratio: 1.4   # For 65% win rate
expectancy_mode.tp_ratio: 2.0 # For higher expectancy
```

## Typical Workflow

### Step 1: Download Data
```bash
python main.py download
# Fetches PAXGUSDT + BTCUSDT in 5m/15m/30m/1h
# Saves to: data/PAXGUSDT_15m.csv, etc.
```

### Step 2: Backtest Single Year
```bash
python main.py backtest --symbol PAXGUSDT --tf 15m
# Output shows: win rate, profit factor, PnL, max drawdown
```

### Step 3: Run Walk-Forward (Critical!)
```bash
python main.py backtest --symbol PAXGUSDT --tf 15m --walkforward
# Tests on 2021-23 (train) → 2024 (val) → 2025 (test)
# Check if overfitting: Is std < 5%? Is degradation < 5%?
```

**If walk-forward is STABLE + VALID:**
→ System is real, not overfit, ready for live

**If degradation > 10% or std > 10%:**
→ Overfit or unstable. Re-tune thresholds.

### Step 4: Paper Trade (2-4 weeks)
```bash
python main.py live --symbol PAXGUSDT
# Connects to WebSocket, outputs trade cards
# You execute manually (full control, no algo spam)
# Compare results vs backtest
```

### Step 5: Go Live (Only If Step 4 Matches Step 3)
- Walk-forward: VALID
- Live paper: ±5% of backtest
- Max drawdown: Acceptable
- Ready? Deploy.

## Tuning Checklist

If results are poor:

1. **Too few trades?** → Lower `trade_threshold` (90 → 75)
2. **Too many losses?** → Increase RSI oversold/overbought
3. **Inconsistent?** → Increase `depth_usd_min` or spread filter
4. **Collapsing live?** → Overfit; run walk-forward
5. **Spread gate blocking?** → Adjust `spread_filter` per asset

## Red Flags

🚩 Backtest shows 65% but walk-forward shows 51% → **OVERFIT**  
🚩 Results vary > 10% across years → **UNSTABLE**  
🚩 Live performance ≠ backtest → **Slippage/fees off** or **overfit**  
🚩 Only PAXG works, not BTC → **Not generalizing** (re-tune)  
🚩 Depth gate always rejecting → **Illiquid times** (adjust or skip those hours)  

## Database

```python
# Signals stored with auto-dedupe
db.store_signal(symbol, tf, direction, score, dedupe_key, ...)
# Same signal within 1 candle? Skipped.

# Trades auto-persisted
db.store_trade(signal_id, entry_price, entry_time)
db.close_trade(trade_id, exit_price, "TP_HIT")

# Survives restarts (SQLite WAL)
```

## Expected Performance

| Scenario | Win Rate | Profit Factor | Trade Frequency |
|----------|----------|---------------|-----------------|
| Global multi-asset | 50-60% | 0.9-1.3 | 5-15/week |
| Best pair (PAXG 15m) | 55-67% | 1.1-1.8 | 2-5/week |
| Overfit (lucky) | 65%+ | 2.0+ | High variance |
| Real (stable) | 52-62% | 1.0-1.3 | Consistent |

## Support

Error: "No bid/ask data"  
→ Order book fetch failed. Check Binance API or network.

Error: "Insufficient liquidity"  
→ Size too large or depth too low. Reduce size_usd.

Error: "Signal already exists"  
→ Dedupe working. Same signal sent twice (OK, safe).

Error: "Database locked"  
→ WSL journal crash. Delete `*.db-wal`, restart.

---

**Remember: Valid ≠ Profitable. But overfit is definitely not valid.**

Run walk-forward. Trust the math.
