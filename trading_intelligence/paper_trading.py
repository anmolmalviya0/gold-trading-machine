"""
PHASE 3: PAPER TRADING ENGINE
==============================
Simulates live trading with persistent tracking.

Features:
- Real-time signal execution simulation
- SQLite trade logging
- Performance tracking
- Kill switch / circuit breakers
- Telegram alerts

Usage:
    python paper_trading.py
"""
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import json
import pandas as pd
import numpy as np
import joblib
import aiohttp
import os

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models' / 'ensemble'
DATA_DIR = BASE_DIR.parent / 'market_data'
DB_PATH = BASE_DIR / 'paper_trading.db'

CONFIG = {
    'symbols': ['BTCUSDT', 'PAXGUSDT'],
    'timeframe': '1h',
    'position_size': 1000,  # USD per trade
    'max_concurrent': 2,     # Max open positions
    'max_daily_trades': 10,
    'max_daily_loss': 100,   # USD - circuit breaker
    'min_confidence': 55,    # Only trade above this
    'cooldown_minutes': 60,  # Min time between trades per symbol
    
    # Telegram (set in .env or environment)
    'telegram_token': os.environ.get('TELEGRAM_TOKEN'),
    'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID'),
}

# Feature columns (must match training)
FEATURE_COLS = [
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
    'rsi', 'macd_hist',
    'dist_sma10', 'dist_sma20', 'dist_sma50', 'dist_sma100',
    'trend_10_20', 'trend_20_50', 'trend_50_100',
    'atr_pct', 'vol_20', 'vol_rank',
    'bb_width', 'bb_position',
    'vol_ratio', 'vol_zscore',
    'roc_5', 'roc_10', 'roc_20',
]


# === DATABASE ===

def init_database():
    """Initialize paper trading database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            size REAL NOT NULL,
            confidence REAL,
            reason_codes TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            pnl REAL,
            status TEXT DEFAULT 'OPEN',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Daily stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            pnl REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0
        )
    ''')
    
    # Signals log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL,
            entry REAL,
            sl REAL,
            tp REAL,
            reason_codes TEXT,
            action_taken TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DB_PATH}")


# === DATA CLASSES ===

@dataclass
class Signal:
    symbol: str
    side: str  # BUY or SELL
    confidence: float
    entry: float
    sl: float
    tp: float
    atr: float
    reason_codes: List[str]
    timestamp: datetime


@dataclass
class OpenPosition:
    id: int
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    sl: float
    tp: float
    size: float
    confidence: float


# === PAPER TRADING ENGINE ===

class PaperTradingEngine:
    """
    Paper trading engine with:
    - Signal generation from ensemble models
    - Position management
    - Circuit breakers
    - Performance tracking
    """
    
    def __init__(self):
        self.models = {}
        self.open_positions: Dict[str, OpenPosition] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.running = False
        self.kill_switch = False
        
        init_database()
        self._load_models()
        self._load_open_positions()
    
    def _load_models(self):
        """Load trained ensemble models"""
        print("📥 Loading models...")
        for symbol in CONFIG['symbols']:
            try:
                model_path = MODEL_DIR / f"{symbol}_ensemble.pkl"
                if model_path.exists():
                    data = joblib.load(model_path)
                    self.models[symbol] = {
                        'scaler': data['scaler'],
                        'lgbm': data['lgbm_model'],
                        'rf': data['rf_model'],
                        'lgbm_weight': data['lgbm_weight'],
                        'rf_weight': data['rf_weight'],
                        'feature_cols': data['feature_cols']
                    }
                    print(f"   ✅ {symbol}")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
    
    def _load_open_positions(self):
        """Load open positions from database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM paper_trades WHERE status = 'OPEN'")
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            pos = OpenPosition(
                id=row[0],
                symbol=row[1],
                side=row[2],
                entry_price=row[3],
                entry_time=datetime.fromisoformat(row[4]),
                sl=row[5],
                tp=row[6],
                size=row[7],
                confidence=row[8]
            )
            self.open_positions[pos.symbol] = pos
        
        print(f"   📂 Loaded {len(self.open_positions)} open positions")
    
    # === SIGNAL GENERATION ===
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for prediction"""
        df = df.copy()
        
        # Returns
        for p in [1, 3, 5, 10, 20]:
            df[f'ret_{p}'] = df['c'].pct_change(p) * 100
        
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['c'].ewm(span=12).mean()
        ema26 = df['c'].ewm(span=26).mean()
        df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        
        # MAs
        for p in [10, 20, 50, 100]:
            df[f'sma{p}'] = df['c'].rolling(p).mean()
            df[f'dist_sma{p}'] = (df['c'] - df[f'sma{p}']) / df['c'] * 100
        
        # Trends
        df['trend_10_20'] = (df['sma10'] - df['sma20']) / df['c'] * 100
        df['trend_20_50'] = (df['sma20'] - df['sma50']) / df['c'] * 100
        df['trend_50_100'] = (df['sma50'] - df['sma100']) / df['c'] * 100
        
        # ATR
        tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), 
                       (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['c'] * 100
        
        # Volatility
        df['log_ret'] = np.log(df['c'] / df['c'].shift(1)) * 100
        df['vol_20'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
        df['vol_rank'] = df['vol_20'].rolling(100).rank(pct=True)
        
        # Bollinger
        df['bb_mid'] = df['sma20']
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_width'] = (4 * df['bb_std']) / df['bb_mid'] * 100
        df['bb_position'] = (df['c'] - (df['bb_mid'] - 2*df['bb_std'])) / (4*df['bb_std'] + 1e-10)
        
        # Volume
        df['vol_sma'] = df['v'].rolling(20).mean()
        df['vol_ratio'] = df['v'] / (df['vol_sma'] + 1e-10)
        df['vol_zscore'] = (df['v'] - df['vol_sma']) / (df['v'].rolling(20).std() + 1e-10)
        
        # ROC
        for p in [5, 10, 20]:
            df[f'roc_{p}'] = (df['c'] / df['c'].shift(p) - 1) * 100
        
        return df
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate signal for symbol"""
        if symbol not in self.models:
            return None
        
        # Calculate features
        df = self.calculate_features(df)
        
        if df.empty or len(df) < 100:
            return None
        
        # Get latest row
        row = df.iloc[-1]
        model = self.models[symbol]
        
        # Prepare features
        feature_cols = [c for c in model['feature_cols'] if c in df.columns]
        features = row[feature_cols].values.reshape(1, -1)
        
        # Handle NaN
        if np.isnan(features).any():
            return None
        
        # Scale
        X_scaled = model['scaler'].transform(features)
        
        # Predict
        lgbm_prob = model['lgbm'].predict_proba(X_scaled)[0][1]
        rf_prob = model['rf'].predict_proba(X_scaled)[0][1]
        
        ensemble_prob = (
            model['lgbm_weight'] * lgbm_prob +
            model['rf_weight'] * rf_prob
        )
        
        # Determine signal
        side = 'BUY' if ensemble_prob >= 0.5 else 'SELL'
        confidence = ensemble_prob * 100 if side == 'BUY' else (1 - ensemble_prob) * 100
        
        # SL/TP
        atr = row['atr']
        price = row['c']
        
        if side == 'BUY':
            sl = price - atr * 1.0
            tp = price + atr * 2.0
        else:
            sl = price + atr * 1.0
            tp = price - atr * 2.0
        
        # Reason codes (top features)
        reason_codes = ['rsi', 'macd_hist', 'trend_20_50']  # Simplified
        
        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            entry=price,
            sl=sl,
            tp=tp,
            atr=atr,
            reason_codes=reason_codes,
            timestamp=datetime.now()
        )
    
    # === POSITION MANAGEMENT ===
    
    def can_open_position(self, signal: Signal) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        # Kill switch
        if self.kill_switch:
            return False, "Kill switch active"
        
        # Max positions
        if len(self.open_positions) >= CONFIG['max_concurrent']:
            return False, "Max positions reached"
        
        # Already has position for symbol
        if signal.symbol in self.open_positions:
            return False, "Position already open"
        
        # Daily trade limit
        if self.daily_trades >= CONFIG['max_daily_trades']:
            return False, "Daily trade limit reached"
        
        # Daily loss limit (circuit breaker)
        if self.daily_pnl <= -CONFIG['max_daily_loss']:
            return False, "Daily loss limit reached"
        
        # Confidence threshold
        if signal.confidence < CONFIG['min_confidence']:
            return False, f"Confidence {signal.confidence:.1f}% < {CONFIG['min_confidence']}%"
        
        # Cooldown
        if signal.symbol in self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time[signal.symbol]).total_seconds() / 60
            if elapsed < CONFIG['cooldown_minutes']:
                return False, f"Cooldown: {CONFIG['cooldown_minutes'] - elapsed:.0f}m remaining"
        
        return True, "OK"
    
    def open_position(self, signal: Signal) -> bool:
        """Open a paper trade"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO paper_trades 
            (symbol, side, entry_price, entry_time, sl, tp, size, confidence, reason_codes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        ''', (
            signal.symbol,
            signal.side,
            signal.entry,
            signal.timestamp.isoformat(),
            signal.sl,
            signal.tp,
            CONFIG['position_size'],
            signal.confidence,
            ','.join(signal.reason_codes)
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update state
        self.open_positions[signal.symbol] = OpenPosition(
            id=trade_id,
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry,
            entry_time=signal.timestamp,
            sl=signal.sl,
            tp=signal.tp,
            size=CONFIG['position_size'],
            confidence=signal.confidence
        )
        
        self.last_trade_time[signal.symbol] = signal.timestamp
        self.daily_trades += 1
        
        print(f"   📈 OPENED: {signal.side} {signal.symbol} @ ${signal.entry:.2f}")
        print(f"      SL: ${signal.sl:.2f} | TP: ${signal.tp:.2f} | Conf: {signal.confidence:.1f}%")
        
        return True
    
    def check_exits(self, symbol: str, high: float, low: float, close: float):
        """Check if position should exit"""
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        exit_reason = None
        exit_price = None
        
        if pos.side == 'BUY':
            if high >= pos.tp:
                exit_reason = 'TP'
                exit_price = pos.tp
            elif low <= pos.sl:
                exit_reason = 'SL'
                exit_price = pos.sl
        else:  # SELL
            if low <= pos.tp:
                exit_reason = 'TP'
                exit_price = pos.tp
            elif high >= pos.sl:
                exit_reason = 'SL'
                exit_price = pos.sl
        
        if exit_reason:
            self.close_position(pos, exit_price, exit_reason)
    
    def close_position(self, pos: OpenPosition, exit_price: float, exit_reason: str):
        """Close a paper trade"""
        # Calculate PnL
        if pos.side == 'BUY':
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size
        
        # Apply fees
        pnl -= pos.size * 0.001 * 2  # 0.1% entry + exit
        
        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE paper_trades 
            SET exit_price = ?, exit_time = ?, exit_reason = ?, pnl = ?, status = 'CLOSED'
            WHERE id = ?
        ''', (exit_price, datetime.now().isoformat(), exit_reason, pnl, pos.id))
        conn.commit()
        conn.close()
        
        # Update state
        del self.open_positions[pos.symbol]
        self.daily_pnl += pnl
        
        icon = "✅" if pnl > 0 else "❌"
        print(f"   {icon} CLOSED: {pos.symbol} via {exit_reason} | PnL: ${pnl:.2f}")
    
    # === STATS ===
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl) FROM paper_trades WHERE status = 'CLOSED'")
        row = cursor.fetchone()
        
        total = row[0] or 0
        wins = row[1] or 0
        total_pnl = row[2] or 0
        
        conn.close()
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'total_pnl': total_pnl,
            'open_positions': len(self.open_positions),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'kill_switch': self.kill_switch
        }
    
    def print_stats(self):
        """Print current stats"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 PAPER TRADING STATS")
        print("="*50)
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Total PnL: ${stats['total_pnl']:.2f}")
        print(f"   Open Positions: {stats['open_positions']}")
        print(f"   Today Trades: {stats['daily_trades']}")
        print(f"   Today PnL: ${stats['daily_pnl']:.2f}")
        print("="*50)


# Typing helper
from typing import Tuple


# === MAIN ===

if __name__ == "__main__":
    print("="*70)
    print("📈 PHASE 3: PAPER TRADING ENGINE")
    print("="*70)
    
    engine = PaperTradingEngine()
    
    # Load sample data and test
    print("\n🧪 Testing signal generation...")
    
    for symbol in CONFIG['symbols']:
        try:
            parquet_path = DATA_DIR / 'parquet' / f"{symbol}_1h.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            else:
                csv_path = DATA_DIR / f"{symbol}_1h.csv"
                df = pd.read_csv(csv_path)
                df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
            
            df = df.tail(500)
            
            signal = engine.generate_signal(symbol, df)
            
            if signal:
                print(f"\n   {symbol}:")
                print(f"      Signal: {signal.side}")
                print(f"      Confidence: {signal.confidence:.1f}%")
                print(f"      Entry: ${signal.entry:.2f}")
                print(f"      SL: ${signal.sl:.2f} | TP: ${signal.tp:.2f}")
                
                # Test opening position
                can_open, reason = engine.can_open_position(signal)
                print(f"      Can Open: {can_open} ({reason})")
                
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
    
    engine.print_stats()
    
    print("\n✅ Paper trading engine ready!")
    print("   Database: paper_trading.db")
