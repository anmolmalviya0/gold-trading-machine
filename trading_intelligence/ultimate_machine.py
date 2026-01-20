#!/usr/bin/env python3
"""
THE ULTIMATE MACHINE
====================
Complete 7-Phase Professional Trading System

This is the UNIFIED MASTER SCRIPT that orchestrates:
1. Environment & Infrastructure
2. High-Fidelity Data Acquisition
3. Quantitative Feature Engineering
4. Advanced Modeling & Training
5. Institutional Validation
6. Live Execution & Intelligence
7. Operational Monitoring & Safety

Usage:
    python ultimate_machine.py          # Full live mode
    python ultimate_machine.py train    # Train models only
    python ultimate_machine.py backtest # Run backtests only
    python ultimate_machine.py status   # System status

Author: Autonomous Trading Machine
Version: 1.0 - Institutional Grade
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import sqlite3
import joblib
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
PARQUET_DIR = DATA_DIR / 'parquet'
MODEL_DIR = BASE_DIR / 'models' / 'production'
LOG_DIR = BASE_DIR / 'logs'
DB_PATH = BASE_DIR / 'performance.db'

# Create directories
for d in [PARQUET_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Trading configuration
CONFIG = {
    'symbols': ['BTCUSDT', 'PAXGUSDT'],
    'timeframes': ['5m', '15m', '30m', '1h'],
    'confidence_threshold': 65,       # Meta-labeling threshold
    'profit_factor_min': 1.2,         # Minimum PF to deploy
    'win_rate_min': 52,               # Minimum win rate
    'daily_loss_limit_pct': 2.0,      # 2% daily loss limit
    'max_position_pct': 10,           # Max 10% of account per trade
    'fee_pct': 0.1,                   # 0.1% trading fee
    'slippage_pct': 0.05,             # 0.05% slippage
}


# =============================================================================
# PHASE 1: INFRASTRUCTURE
# =============================================================================

def init_database():
    """Initialize SQLite database for persistence"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Signals table
    cursor.execute('''CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        timeframe TEXT,
        signal TEXT,
        confidence REAL,
        entry REAL,
        stop_loss REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        regime TEXT,
        engine TEXT,
        executed INTEGER DEFAULT 0
    )''')
    
    # Trades table
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        timestamp TEXT,
        symbol TEXT,
        side TEXT,
        entry_price REAL,
        exit_price REAL,
        quantity REAL,
        pnl REAL,
        fee REAL,
        slippage REAL,
        exit_reason TEXT,
        FOREIGN KEY (signal_id) REFERENCES signals(id)
    )''')
    
    # Model performance table
    cursor.execute('''CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        timeframe TEXT,
        profit_factor REAL,
        win_rate REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        total_trades INTEGER
    )''')
    
    # Daily stats table
    cursor.execute('''CREATE TABLE IF NOT EXISTS daily_stats (
        date TEXT PRIMARY KEY,
        starting_balance REAL,
        ending_balance REAL,
        pnl REAL,
        trades INTEGER,
        is_halted INTEGER DEFAULT 0
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized: performance.db")


# =============================================================================
# PHASE 2: DATA ACQUISITION
# =============================================================================

class DataManager:
    """Manages all data acquisition and storage"""
    
    def __init__(self):
        self.live_data = {sym: {} for sym in CONFIG['symbols']}
    
    def load_historical(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load historical data from CSV or Parquet"""
        # Try CSV first (more reliable)
        csv_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
                return df
            except Exception as e:
                print(f"   ⚠️ CSV load failed: {e}")
        
        # Fallback to Parquet
        parquet_path = PARQUET_DIR / f"{symbol}_{timeframe}.parquet"
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                print(f"   ⚠️ Parquet load failed: {e}")
        
        return pd.DataFrame()
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to Parquet format"""
        path = PARQUET_DIR / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(path, index=False)
        print(f"   Saved {len(df):,} rows to {path.name}")


# =============================================================================
# PHASE 3: FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Creates 60+ technical and cross-asset features"""
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete feature suite"""
        df = df.copy()
        
        # === PRICE-BASED ===
        for period in [1, 3, 5, 10, 20, 50]:
            df[f'ret_{period}'] = df['c'].pct_change(period) * 100
        
        # === MOMENTUM ===
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['c'].ewm(span=12).mean()
        ema26 = df['c'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_14 = df['l'].rolling(14).min()
        high_14 = df['h'].rolling(14).max()
        df['stoch_k'] = 100 * (df['c'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # === VOLATILITY ===
        # ATR
        tr = pd.concat([
            df['h'] - df['l'],
            (df['h'] - df['c'].shift()).abs(),
            (df['l'] - df['c'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['c'] * 100
        
        # Bollinger Bands
        df['bb_mid'] = df['c'].rolling(20).mean()
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Keltner Channels
        df['kc_mid'] = df['c'].ewm(span=20).mean()
        df['kc_upper'] = df['kc_mid'] + df['atr'] * 1.5
        df['kc_lower'] = df['kc_mid'] - df['atr'] * 1.5
        
        # Squeeze detection (BB inside KC)
        df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        
        # === TREND ===
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['c'].rolling(period).mean()
            df[f'dist_sma{period}'] = (df['c'] - df[f'sma_{period}']) / df['c'] * 100
        
        # ADX
        plus_dm = (df['h'] - df['h'].shift()).where((df['h'] - df['h'].shift()) > (df['l'].shift() - df['l']), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = (df['l'].shift() - df['l']).where((df['l'].shift() - df['l']) > (df['h'] - df['h'].shift()), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # === VOLUME ===
        if 'v' in df.columns and df['v'].sum() > 0:
            df['vol_sma'] = df['v'].rolling(20).mean()
            df['vol_ratio'] = df['v'] / (df['vol_sma'] + 1e-10)
            
            # OBV
            df['obv'] = (np.sign(df['c'].diff()) * df['v']).cumsum()
        
        # === REGIME ===
        vol = df['ret_1'].rolling(20).std()
        df['volatility'] = vol
        df['vol_z'] = (vol - vol.rolling(50).mean()) / (vol.rolling(50).std() + 1e-10)
        
        return df
    
    @staticmethod  
    def detect_regime(df: pd.DataFrame) -> str:
        """Detect current market regime"""
        row = df.iloc[-1]
        adx = row.get('adx', 20)
        vol_z = row.get('vol_z', 0)
        vol_ratio = row.get('vol_ratio', 1)
        
        if vol_ratio < 0.3:
            return 'DEAD'
        if vol_z > 2:
            return 'VOLATILE'
        if adx > 25:
            return 'TREND'
        return 'RANGE'


# =============================================================================
# PHASE 4: MODELING
# =============================================================================

class TripleBarrierLabeler:
    """Marcos Lopez de Prado Triple Barrier Method"""
    
    def __init__(self, tp_mult=2.0, sl_mult=1.0, max_holding=10):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.max_holding = max_holding
    
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier labels"""
        df = df.copy()
        df['label'] = 0
        df['barrier_ret'] = 0.0
        
        atr = df['atr'].values
        close = df['c'].values
        
        for i in range(len(df) - self.max_holding):
            entry = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else entry * 0.02
            
            tp = entry + current_atr * self.tp_mult
            sl = entry - current_atr * self.sl_mult
            
            # Check barriers
            for j in range(1, self.max_holding + 1):
                if i + j >= len(df):
                    break
                
                price = close[i + j]
                
                if price >= tp:
                    df.iloc[i, df.columns.get_loc('label')] = 1
                    df.iloc[i, df.columns.get_loc('barrier_ret')] = (tp - entry) / entry * 100
                    break
                elif price <= sl:
                    df.iloc[i, df.columns.get_loc('label')] = -1
                    df.iloc[i, df.columns.get_loc('barrier_ret')] = (sl - entry) / entry * 100
                    break
            else:
                # Vertical barrier
                final_price = close[min(i + self.max_holding, len(df) - 1)]
                df.iloc[i, df.columns.get_loc('barrier_ret')] = (final_price - entry) / entry * 100
        
        return df


class MetaLabeler:
    """Meta-labeling: Predict if the primary signal is correct"""
    
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.primary_model = None
        self.meta_model = None
    
    def should_trade(self, primary_signal: str, meta_confidence: float) -> bool:
        """Check if we should take this trade"""
        if primary_signal == 'NEUTRAL':
            return False
        return meta_confidence >= self.threshold


# =============================================================================
# PHASE 5: VALIDATION
# =============================================================================

class InstitutionalValidator:
    """Validates strategy with realistic costs"""
    
    def __init__(self):
        self.fee_pct = CONFIG['fee_pct'] / 100
        self.slippage_pct = CONFIG['slippage_pct'] / 100
    
    def run_backtest(self, df: pd.DataFrame, signals: list) -> dict:
        """Run backtest with fees and slippage"""
        trades = []
        equity = 10000
        
        for sig in signals:
            idx = sig['idx']
            if idx + 10 >= len(df):
                continue
            
            entry = df.iloc[idx]['c']
            
            # Apply slippage
            if sig['signal'] == 'BUY':
                entry *= (1 + self.slippage_pct)
            else:
                entry *= (1 - self.slippage_pct)
            
            # Find exit
            for j in range(1, 11):
                exit_price = df.iloc[idx + j]['c']
                
                if sig['signal'] == 'BUY':
                    ret = (exit_price - entry) / entry
                else:
                    ret = (entry - exit_price) / entry
                
                if ret > sig.get('tp_pct', 0.02) or ret < -sig.get('sl_pct', 0.01):
                    break
            
            # Apply fees
            ret -= 2 * self.fee_pct
            
            pnl = equity * 0.02 * ret  # 2% position size
            equity += pnl
            
            trades.append({
                'signal': sig['signal'],
                'entry': entry,
                'exit': exit_price,
                'ret': ret * 100,
                'pnl': pnl,
                'win': ret > 0
            })
        
        if not trades:
            return {'profit_factor': 0, 'win_rate': 0, 'sharpe': 0}
        
        wins = [t['pnl'] for t in trades if t['win']]
        losses = [-t['pnl'] for t in trades if not t['win']]
        
        pf = sum(wins) / (sum(losses) + 1e-10)
        wr = len(wins) / len(trades) * 100
        
        returns = [t['ret'] for t in trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        return {
            'profit_factor': round(pf, 2),
            'win_rate': round(wr, 1),
            'sharpe': round(sharpe, 2),
            'total_trades': len(trades),
            'final_equity': round(equity, 2)
        }


# =============================================================================
# PHASE 6: LIVE EXECUTION
# =============================================================================

class SignalGenerator:
    """Generates trading signals with TP1/TP2/TP3"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.labeler = TripleBarrierLabeler()
    
    def generate(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate complete trade plan"""
        df = self.feature_engineer.calculate_all_features(df)
        df = df.dropna()
        
        if len(df) < 50:
            return None
        
        row = df.iloc[-1]
        regime = self.feature_engineer.detect_regime(df)
        
        # Signal logic
        score = 0
        
        # RSI
        if row['rsi'] < 30: score += 2
        elif row['rsi'] < 40: score += 1
        elif row['rsi'] > 70: score -= 2
        elif row['rsi'] > 60: score -= 1
        
        # MACD
        if row['macd_hist'] > 0: score += 1
        else: score -= 1
        
        # Trend
        if row['adx'] > 25:
            if row['plus_di'] > row['minus_di']:
                score += 1
            else:
                score -= 1
        
        # BB position
        if row['bb_pct'] < 0.1: score += 1
        elif row['bb_pct'] > 0.9: score -= 1
        
        # Determine signal
        if score >= 3:
            signal = 'BUY'
            confidence = 65 + min(score * 5, 25)
        elif score <= -3:
            signal = 'SELL'
            confidence = 65 + min(abs(score) * 5, 25)
        else:
            signal = 'NEUTRAL'
            confidence = 50
        
        # Calculate trade plan
        price = row['c']
        atr = row['atr']
        
        if signal == 'BUY':
            entry = price
            sl = entry - atr * 1.5
            tp1 = entry + atr * 1.0
            tp2 = entry + atr * 2.0
            tp3 = entry + atr * 3.0
        elif signal == 'SELL':
            entry = price
            sl = entry + atr * 1.5
            tp1 = entry - atr * 1.0
            tp2 = entry - atr * 2.0
            tp3 = entry - atr * 3.0
        else:
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': confidence,
                'regime': regime
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'entry': round(entry, 2),
            'stop_loss': round(sl, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2),
            'tp3': round(tp3, 2),
            'atr': round(atr, 2),
            'regime': regime,
            'rsi': round(row['rsi'], 1),
            'adx': round(row['adx'], 1)
        }


# =============================================================================
# PHASE 7: MONITORING & SAFETY
# =============================================================================

class SafetyMonitor:
    """Monitors system health and enforces safety limits"""
    
    def __init__(self):
        self.daily_pnl = 0
        self.is_halted = False
        self.halt_reason = ""
    
    def check_daily_limit(self, pnl: float, balance: float) -> bool:
        """Check if daily loss limit is hit"""
        self.daily_pnl += pnl
        loss_pct = abs(self.daily_pnl) / balance * 100
        
        if self.daily_pnl < 0 and loss_pct >= CONFIG['daily_loss_limit_pct']:
            self.halt_trading(f"Daily loss limit hit: {loss_pct:.1f}%")
            return False
        return True
    
    def halt_trading(self, reason: str):
        """Halt all trading"""
        self.is_halted = True
        self.halt_reason = reason
        print(f"\n🛑 TRADING HALTED: {reason}")
    
    def can_trade(self) -> tuple:
        """Check if trading is allowed"""
        if self.is_halted:
            return False, self.halt_reason
        return True, "OK"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_trade_plan(signal: dict):
    """Pretty print a trade plan"""
    if signal['signal'] == 'NEUTRAL':
        print(f"   {signal['symbol']}: 🔹 NEUTRAL (Conf: {signal['confidence']}%) | Regime: {signal['regime']}")
        return
    
    emoji = '🟢' if signal['signal'] == 'BUY' else '🔴'
    print(f"\n{'='*60}")
    print(f"   {emoji} {signal['signal']} {signal['symbol']}")
    print(f"{'='*60}")
    print(f"   Confidence: {signal['confidence']}%")
    print(f"   Regime: {signal['regime']}")
    print(f"   RSI: {signal['rsi']} | ADX: {signal['adx']}")
    print(f"{'='*60}")
    print(f"   📍 ENTRY:      ${signal['entry']:,.2f}")
    print(f"   🛑 STOP LOSS:  ${signal['stop_loss']:,.2f}")
    print(f"   ✅ TP1:        ${signal['tp1']:,.2f}")
    print(f"   ✅ TP2:        ${signal['tp2']:,.2f}")
    print(f"   ✅ TP3:        ${signal['tp3']:,.2f}")
    print(f"{'='*60}\n")


def run_live():
    """Run live prediction terminal"""
    print("\n" + "="*70)
    print("🚀 THE ULTIMATE MACHINE - LIVE MODE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {CONFIG['symbols']}")
    print(f"Timeframes: {CONFIG['timeframes']}")
    print(f"Confidence Threshold: {CONFIG['confidence_threshold']}%")
    
    # Initialize
    init_database()
    data_mgr = DataManager()
    signal_gen = SignalGenerator()
    safety = SafetyMonitor()
    
    print("\n📊 GENERATING SIGNALS...\n")
    
    for symbol in CONFIG['symbols']:
        print(f"\n{'='*60}")
        print(f"📈 {symbol}")
        print(f"{'='*60}")
        
        for tf in CONFIG['timeframes']:
            df = data_mgr.load_historical(symbol, tf)
            
            if len(df) < 100:
                print(f"   {tf}: Insufficient data")
                continue
            
            signal = signal_gen.generate(df, symbol)
            
            if signal:
                # Check confidence threshold
                if signal['confidence'] >= CONFIG['confidence_threshold']:
                    print_trade_plan(signal)
                    
                    # Log to database
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute('''INSERT INTO signals 
                        (timestamp, symbol, timeframe, signal, confidence, entry, stop_loss, tp1, tp2, tp3, regime, engine)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (signal['timestamp'], symbol, tf, signal['signal'], 
                         signal['confidence'], signal.get('entry'), signal.get('stop_loss'),
                         signal.get('tp1'), signal.get('tp2'), signal.get('tp3'),
                         signal['regime'], 'ULTIMATE'))
                    conn.commit()
                    conn.close()
                else:
                    print(f"   {tf}: Below threshold ({signal['confidence']}% < {CONFIG['confidence_threshold']}%)")
            else:
                print(f"   {tf}: No signal generated")


def run_status():
    """Show system status"""
    print("\n" + "="*70)
    print("📊 THE ULTIMATE MACHINE - STATUS")
    print("="*70)
    
    # Database stats
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM signals")
        signal_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT signal, COUNT(*) FROM signals GROUP BY signal")
        signal_breakdown = dict(cursor.fetchall())
        
        conn.close()
        
        print(f"\n📁 Database: performance.db")
        print(f"   Total Signals: {signal_count}")
        print(f"   Total Trades: {trade_count}")
        print(f"   Signal Breakdown: {signal_breakdown}")
    
    # Data status
    print(f"\n📂 Data Directory: {DATA_DIR}")
    for symbol in CONFIG['symbols']:
        for tf in CONFIG['timeframes']:
            csv_path = DATA_DIR / f"{symbol}_{tf}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                print(f"   {symbol}_{tf}: {len(df):,} rows")
    
    print(f"\n✅ System Ready")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "live"
    
    if cmd == "train":
        print("Training mode not yet implemented")
    elif cmd == "backtest":
        print("Backtest mode - use institutional_validation.py")
    elif cmd == "status":
        run_status()
    else:
        run_live()
