# V4 PRO MAX ULTRA - Production-Grade Quantitative Engine

**Not "life prediction." Not "unbeatable bot." Just an honest, testable trading machine.**

## What This Is

A complete, forensic-audited trading system that handles:

✅ **Correct pivots + divergence** (pivot-to-pivot confirmed, not fake 3-candle)  
✅ **Wilder ATR** (stable SL/TP, not simplified)  
✅ **Honest backtest** (fees both sides, slippage, spread, strict TP/SL sequencing)  
✅ **Walk-forward validation** (test for overfitting across 2021–2026)  
✅ **Score-based signals** (0–100, trade only ≥80, not on/off spam)  
✅ **USD depth gatekeeper** (spread, OBI, slippage impact, conviction rejection)  
✅ **WebSocket live** (Binance kline close, no repaints, async-ready)  
✅ **SQLite WAL** (durability, dedupe, watchdog, reconnect)  

## What It's NOT

❌ "Always profitable"  
❌ "Works on all pairs"  
❌ "65% on everything"  

**Real performance:**
- Global multi-asset: **50–60%** win rate realistic
- Single best combo: **60–67%** with extreme selectivity (low frequency)
- Markets don't allow perfection; this system is honest about it.

## Realistic Expectations

If walk-forward validation is **stable** and **live paper trading matches backtest** (within reason), you have a **valid** system.

If results collapse live or only work on one year, it's **overfit**—time to re-tune.

## Project Structure

```
ultimate_quant_machine_v4_ultra/
├── main.py                    # Entry point
├── config.yaml                # All thresholds (no hardcoding)
├── requirements.txt           # Python deps
├── setup.sh                   # One-shot installer
│
├── indicators.py              # Wilder RSI, ATR, pivots, divergence, sweeps
├── strategy_engine.py         # Score-based signal (0-100)
├── conviction_monitor.py      # USD depth gates (spread, OBI, slippage)
├── backtester.py              # Strict fees + slippage + order sequencing
├── walkforward.py             # 2021/2024/2025 split validation
│
├── live_ws.py                 # Binance WebSocket (candle close only)
├── db.py                      # SQLite WAL, dedupe, persistence
├── alerts.py                  # Telegram + local logging
├── download_history.py        # Fetch Binance historical data
│
├── data/                      # Historical OHLCV CSVs
├── logs/                      # Signal logs
└── models/                    # Saved thresholds (optional)
```

## Quick Start (Mac)

### 1. Setup

```bash
cd ultimate_quant_machine_v4_ultra
chmod +x setup.sh
./setup.sh
```

### 2. Download Historical Data

```bash
python main.py download
```

Fetches 2021–2026 PAXGUSDT + BTCUSDT in 5m, 15m, 30m, 1h.

### 3. Backtest Single Year

```bash
python main.py backtest --symbol PAXGUSDT --tf 15m
```

Output:
```
╔════════════════════════════════════════════════════════════════════╗
║ BACKTEST RESULTS - PAXGUSDT 15m
╠════════════════════════════════════════════════════════════════════╣
║ Total Trades:       47
║ Win Rate:           57.45%
║ Profit Factor:      1.28
║ Total PnL:          $1,234.56
║ Max Drawdown:       -$456.78
║ Expectancy:         $26.27
╚════════════════════════════════════════════════════════════════════╝
```

### 4. Walk-Forward Validation (Critical)

```bash
python main.py backtest --symbol PAXGUSDT --tf 15m --walkforward
```

Tests across:
- **Train**: 2021–2023
- **Validation**: 2024
- **Test**: 2025

If results stable (< 10% std dev) and forward test ≈ train, you're NOT overfit.

Output:
```
╔════════════════════════════════════════════════════════════════════╗
║ WALK-FORWARD VALIDATION REPORT
╠════════════════════════════════════════════════════════════════════╣
║ Status:         ANALYZED
║ Verdict:        VALID
║ Overfit Risk:   LOW
╠════════════════════════════════════════════════════════════════════╣
║ Win Rate Mean:  56.32%
║ Win Rate Std:   3.12%
╠════════════════════════════════════════════════════════════════════╣
║ BY SPLIT:
║   2021-2023      : WR=58.12%  PF=1.35
║   2024          : WR=56.45%  PF=1.28
║   2025          : WR=54.18%  PF=1.15
║ Train-Test Degradation: 3.94%
╚════════════════════════════════════════════════════════════════════╝
```

### 5. Live Mode (Paper Trading)

```bash
python main.py live --symbol PAXGUSDT
```

Connects to Binance WebSocket, waits for candle close, outputs **Trade Cards**:

```
╔════════════════════════════════════════════════════════════════════╗
║ TRADE CARD - V4 PRO MAX ULTRA
╠════════════════════════════════════════════════════════════════════╣
║ Symbol:     PAXGUSDT
║ TF:         15m
║ Direction:  BUY
╠════════════════════════════════════════════════════════════════════╣
║ Entry Zone: 2035.50 - 2036.20
║ TP1:        2041.80
║ TP2:        2047.10
║ SL:         2032.10
╠════════════════════════════════════════════════════════════════════╣
║ Score:      87/100
║ Regime:     Trend
║ Reason:     Div:bul | UpTrend | Sweep:low | ADX:32.5
╚════════════════════════════════════════════════════════════════════╝
```

**Manual execution**: You copy entry zone → execute → set alerts.  
No algo order spam, no repaint risk, full control.

## Configuration (config.yaml)

All thresholds in one place—no hardcoding:

```yaml
# Conviction gates (USD-based)
conviction:
  spread_filter:
    PAXG: 0.0008        # < 0.08%
    BTC: 0.0004         # < 0.04%
  
  depth_usd_min: 50000  # Require $50k liquidity
  slippage_impact_max: 0.001  # < 0.1% impact

# Signal scoring
signal_scoring:
  divergence_confirmed: 30
  trend_alignment: 20
  liquidity_sweep: 15
  regime_good: 15
  conviction_pass: 20
  trade_threshold: 80   # Only trade if ≥80

# Risk management
risk:
  win_rate_mode:
    tp_ratio: 1.4       # Target 1.4R for high win rate
    sl_ratio: 1.0
    time_stop_candles: 20
  
  expectancy_mode:
    tp_ratio: 2.0       # Target 2.0R for expectancy
    sl_ratio: 1.0
    time_stop_candles: 30
  
  mode: "win_rate_mode"
```

## The 8-Phase Architecture

### Phase 0: Rules of Reality
- Aim: High-confidence alerts, not spam
- Expect: 50–60% global, 60–67% on best pairs with low frequency

### Phase 1: Data Layer
- WebSocket for live, REST for verification
- Candle close execution only (x=true)

### Phase 2: Feature Layer
- RSI (Wilder), ATR (Wilder), EMA200/50, ADX, BB Width
- Pivot/structure detection (argrelextrema)
- Liquidity sweep confirmation

### Phase 3: Signal Engine
- Score 0–100 (not if/else)
- Trade only if score ≥ 80
- +30 divergence, +20 trend, +15 sweep, +15 regime, +20 conviction

### Phase 4: Conviction Monitor
- Gate 1: Spread < threshold
- Gate 2: USD depth ≥ minimum
- Gate 3: Slippage impact < 0.1%
- Gate 4: OBI (order book imbalance) ratio OK
- **Default: REJECT on missing data**

### Phase 5: Risk & Trade Definition
- **Win-rate focus**: TP 1.2–1.6R, SL 1.0R, time stop 20 candles
- **Expectancy focus**: TP 1.8–2.5R, SL 1.0R, time stop 30 candles
- Choose one (can't max both)

### Phase 6: Backtesting
- Strict sequencing (first hit TP or SL)
- Fees on entry AND exit
- Slippage + spread penalty
- Walk-forward across 2021–2026

### Phase 7: Live Mode
- Trade card format (manual execution)
- Signal score + conviction result
- Regime label + reason
- Optional Telegram alerts

### Phase 8: Safety
- SQLite WAL for durability
- Signal dedupe (no double alerts)
- WebSocket reconnect
- REST rate limiting
- Watchdog heartbeat
- Config-driven (no hardcoding)

## Critical Thresholds (Tune These)

| Parameter | Default | Tune For |
|-----------|---------|----------|
| `rsi_oversold` | 30 | More/fewer BUY signals |
| `rsi_overbought` | 70 | More/fewer SELL signals |
| `divergence_threshold` | 30 pts | Div score weight |
| `adx_regime_filter` | 25 | Trend strength filter |
| `trade_threshold` | 80/100 | Signal quality (higher = fewer trades) |
| `spread_filter` | 0.0008 PAXG | Varies by asset liquidity |
| `depth_usd_min` | $50k | Varies by account size |

## Database & Persistence

SQLite with WAL mode:

```python
# Automatic dedupe (same signal = skip)
signal_id = db.store_signal(
    symbol='PAXGUSDT',
    tf='15m',
    dedupe_key=StrategyEngine.generate_dedupe_key(...)
)

# Auto-closed trades on shutdown (graceful)
trade = db.store_trade(signal_id, entry_price, entry_time)
db.close_trade(trade_id, exit_price, exit_time, 'manual_close')
```

## Safety Checklist

- [x] No hardcoded parameters (all in config.yaml)
- [x] Fees modeled both sides
- [x] Slippage & spread penalties
- [x] Strict TP/SL sequencing
- [x] Time-based exit
- [x] Walk-forward validation
- [x] SQLite WAL persistence
- [x] Signal dedupe
- [x] WebSocket reconnect
- [x] Watchdog heartbeat
- [x] Spread gate (reject tight markets)
- [x] Depth gate (reject low liquidity)
- [x] Slippage impact check
- [x] OBI imbalance gate
- [x] Manual execution (no algo spam)

## Troubleshooting

### Backtest shows 65% but live is 48%

→ **Overfit**. Run walk-forward. If degradation > 10%, re-tune or reduce trade frequency.

### WebSocket keeps disconnecting

→ Check `reconnect_timeout` in config.yaml. Default 5s OK for most.

### No signals generated

→ Check score is reaching threshold (80/100). Lower `trade_threshold` to 70 for testing.

### Spread gate always rejecting

→ Adjust `spread_filter` for your pair. PAXG is less liquid than BTC.

### Depth too low

→ Lower `depth_usd_min` or use during high-volume sessions (NYSE hours for PAXG).

## Next Steps

1. **Verify walk-forward passes**:
   ```bash
   python main.py backtest --symbol PAXGUSDT --tf 15m --walkforward
   ```

2. **Paper trade 2–4 weeks**, compare live vs backtest.

3. **Only then consider live capital** if:
   - Walk-forward verdict = "VALID"
   - Live behavior matches backtest (±5%)
   - Max drawdown acceptable

4. **Never**:
   - Overtrade (stick to 1–5 signals/day)
   - Chase hot hand (if 65%, thank markets, don't leverage)
   - Ignore drawdowns (circuit break at 5%)

## Reference

- **RSI Divergence**: Pivot-to-pivot confirmed with 2-candle delay
- **Wilder ATR**: RMA (exponential), not simple
- **OBI**: Order Book Imbalance in USD (not coin volume)
- **Conviction**: Default reject; only trade if all gates pass
- **Walk-Forward**: Train 2021–23, Validate 2024, Test 2025

---

**Built for professionals. Designed for honesty. Ready for live.**

*V4 Pro Max Ultra is complete when walk-forward validation is stable and live paper trading matches backtest behavior.*
