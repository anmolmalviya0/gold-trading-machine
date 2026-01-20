"""
Performance Database - SQLite trade logging
Track all signals and performance metrics
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


class PerformanceDB:
    """Trade and signal logging database"""
    
    def __init__(self, db_path: str = "trading_intelligence/data/performance.db"):
        self.db_path = db_path
        
        # Ensure directory
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
        logger.info(f"Performance DB initialized: {db_path}")
    
    def _create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                bias_strength TEXT,
                regime TEXT,
                entry_min REAL,
                entry_max REAL,
                stop_loss REAL,
                tp1 REAL,
                tp2 REAL,
                tp3 REAL,
                risk_reward REAL,
                reason_codes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_signals INTEGER NOT NULL,
                buy_signals INTEGER NOT NULL,
                sell_signals INTEGER NOT NULL,
                high_conf_signals INTEGER NOT NULL,
                avg_confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def log_signal(self, asset: str, timeframe: str, signal_data: Dict):
        """Log a signal"""
        cursor = self.conn.cursor()
        
        levels = signal_data.get('levels', {})
        entry_zone = levels.get('entry_zone', {})
        take_profits = levels.get('take_profits', {})
        
        cursor.execute("""
            INSERT INTO signals (
                timestamp, asset, timeframe, signal, confidence, 
                bias_strength, regime, entry_min, entry_max, stop_loss,
                tp1, tp2, tp3, risk_reward, reason_codes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            asset,
            timeframe,
            signal_data['signal'],
            signal_data['confidence'],
            signal_data.get('bias_strength'),
            signal_data.get('regime'),
            entry_zone.get('min'),
            entry_zone.get('max'),
            levels.get('stop_loss'),
            take_profits.get('tp1'),
            take_profits.get('tp2'),
            take_profits.get('tp3'),
            levels.get('risk_reward'),
            ','.join(signal_data.get('reason_codes', []))
        ))
        
        self.conn.commit()
        logger.debug(f"Signal logged: {asset} {timeframe} {signal_data['signal']}")
    
    def update_daily_stats(self, date: str = None):
        """Update daily statistics"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        cursor = self.conn.cursor()
        
        # Calculate stats for date
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN signal = 'SELL' THEN 1 ELSE 0 END) as sells,
                SUM(CASE WHEN confidence >= 80 THEN 1 ELSE 0 END) as high_conf,
                AVG(confidence) as avg_conf
            FROM signals
            WHERE DATE(timestamp) = ?
        """, (date,))
        
        row = cursor.fetchone()
        
        if row and row[0] > 0:
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats 
                (date, total_signals, buy_signals, sell_signals, high_conf_signals, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (date, row[0], row[1], row[2], row[3], row[4]))
            
            self.conn.commit()
    
    def get_signals(self, asset: str = None, days: int = 7) -> pd.DataFrame:
        """Get recent signals"""
        query = """
            SELECT * FROM signals 
            WHERE DATE(timestamp) >= DATE('now', '-' || ? || ' days')
        """
        params = [days]
        
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        
        query += " ORDER BY timestamp DESC LIMIT 100"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_daily_stats(self, days: int = 30) -> pd.DataFrame:
        """Get daily statistics"""
        query = """
            SELECT * FROM daily_stats
            ORDER BY date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=[days])
        return df
    
    def get_summary(self) -> Dict:
        """Get overall summary"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_signals,
                SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END) as total_buys,
                SUM(CASE WHEN signal = 'SELL' THEN 1 ELSE 0 END) as total_sells,
                AVG(confidence) as avg_confidence,
                MAX(confidence) as max_confidence
            FROM signals
        """)
        
        row = cursor.fetchone()
        
        return {
            'total_signals': row[0] or 0,
            'buy_count': row[1] or 0,
            'sell_count': row[2] or 0,
            'avg_confidence': round(row[3], 1) if row[3] else 0,
            'max_confidence': row[4] or 0
        }
    
    def close(self):
        """Close database"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Demo
if __name__ == "__main__":
    with PerformanceDB() as db:
        # Log test signal
        signal = {
            'signal': 'BUY',
            'confidence': 85,
            'bias_strength': 'STRONG',
            'regime': 'TREND',
            'levels': {
                'entry_zone': {'min': 94850, 'max': 95150},
                'stop_loss': 94200,
                'take_profits': {'tp1': 95800, 'tp2': 96500, 'tp3': 97500},
                'risk_reward': 2.0
            },
            'reason_codes': ['trend_aligned', '3_timeframes_bullish']
        }
        
        db.log_signal('BTC', '15m', signal)
        db.update_daily_stats()
        
        # Get summary
        summary = db.get_summary()
        print(f"📊 Summary: {summary}")
        
        # Recent signals
        signals = db.get_signals(days=1)
        print(f"\n📋 Recent signals: {len(signals)}")
