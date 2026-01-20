"""
Database Layer - SQLite WAL with dedupe & persistence
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class SignalDatabase:
    """SQLite database for signals, trades, and candles"""
    
    def __init__(self, db_path: str = "./data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """Initialize database with WAL mode"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # Candles table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                is_complete INTEGER DEFAULT 0,
                UNIQUE(symbol, timeframe, open_time)
            )
        """)
        
        # Signals table (with dedupe key)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                dedupe_key TEXT UNIQUE NOT NULL,
                direction TEXT NOT NULL,
                score REAL NOT NULL,
                divergence_type TEXT,
                entry_zone_low REAL,
                entry_zone_high REAL,
                tp1 REAL,
                tp2 REAL,
                sl REAL,
                conviction_pass INTEGER DEFAULT 0,
                conviction_reason TEXT,
                regime TEXT,
                reason TEXT,
                raw_data TEXT
            )
        """)
        
        # Trades table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                signal_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_time INTEGER NOT NULL,
                exit_price REAL,
                exit_time INTEGER,
                exit_reason TEXT,
                tp_hit INTEGER DEFAULT 0,
                sl_hit INTEGER DEFAULT 0,
                time_stop_hit INTEGER DEFAULT 0,
                pnl_pct REAL,
                pnl_usd REAL,
                status TEXT DEFAULT 'open',
                FOREIGN KEY(signal_id) REFERENCES signals(id)
            )
        """)
        
        self.conn.commit()
        logger.info("Database initialized with WAL mode")
    
    def store_candle(self, symbol: str, tf: str, o: float, h: float, l: float, c: float, 
                     v: float, open_time: int, close_time: int, is_complete: bool = False):
        """Store candle (idempotent)"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, open_time, close_time, open, high, low, close, volume, is_complete)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, tf, open_time, close_time, o, h, l, c, v, int(is_complete)))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to store candle: {e}")
    
    def store_signal(self, symbol: str, tf: str, direction: str, score: float, 
                     dedupe_key: str, entry_zone: tuple, tp1: float, tp2: float, sl: float,
                     conviction_pass: bool, conviction_reason: str, regime: str, reason: str,
                     divergence_type: Optional[str] = None, raw_data: Optional[Dict] = None) -> Optional[int]:
        """
        Store signal with automatic dedupe.
        Returns signal_id if stored, None if duplicate.
        """
        try:
            timestamp = int(datetime.utcnow().timestamp() * 1000)
            raw_json = json.dumps(raw_data) if raw_data else None
            
            cursor = self.conn.execute("""
                INSERT INTO signals 
                (symbol, timeframe, timestamp, dedupe_key, direction, score, divergence_type,
                 entry_zone_low, entry_zone_high, tp1, tp2, sl, conviction_pass, conviction_reason,
                 regime, reason, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, tf, timestamp, dedupe_key, direction, score, divergence_type,
                  entry_zone[0], entry_zone[1], tp1, tp2, sl, int(conviction_pass), 
                  conviction_reason, regime, reason, raw_json))
            
            self.conn.commit()
            logger.info(f"Signal stored: {dedupe_key} (id={cursor.lastrowid})")
            return cursor.lastrowid
            
        except sqlite3.IntegrityError:
            logger.info(f"Signal already exists (dedupe): {dedupe_key}")
            return None
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            return None
    
    def get_signal(self, signal_id: int) -> Optional[Dict]:
        """Retrieve signal by ID"""
        cursor = self.conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
        row = cursor.fetchone()
        if row:
            return self._signal_row_to_dict(row)
        return None
    
    def _signal_row_to_dict(self, row) -> Dict:
        """Convert DB row to dict"""
        return {
            'id': row[0],
            'symbol': row[1],
            'timeframe': row[2],
            'timestamp': row[3],
            'direction': row[5],
            'score': row[6],
            'divergence_type': row[7],
            'entry_zone': (row[8], row[9]),
            'tp1': row[10],
            'tp2': row[11],
            'sl': row[12],
            'conviction_pass': bool(row[13]),
            'conviction_reason': row[14],
            'regime': row[15],
            'reason': row[16],
        }
    
    def store_trade(self, signal_id: int, symbol: str, direction: str, entry_price: float, 
                    entry_time: int, tp_ratio: float, sl_ratio: float) -> int:
        """Create trade entry"""
        cursor = self.conn.execute("""
            INSERT INTO trades 
            (signal_id, symbol, direction, entry_price, entry_time, status)
            VALUES (?, ?, ?, ?, ?, 'open')
        """, (signal_id, symbol, direction, entry_price, entry_time))
        self.conn.commit()
        return cursor.lastrowid
    
    def close_trade(self, trade_id: int, exit_price: float, exit_time: int, 
                   exit_reason: str, tp_hit: bool = False, sl_hit: bool = False, 
                   time_stop_hit: bool = False):
        """Close a trade"""
        cursor = self.conn.execute("SELECT entry_price FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        if row:
            entry_price = row[0]
            pnl_pct = ((exit_price - entry_price) / entry_price)
            
            self.conn.execute("""
                UPDATE trades 
                SET exit_price = ?, exit_time = ?, exit_reason = ?, status = 'closed',
                    tp_hit = ?, sl_hit = ?, time_stop_hit = ?, pnl_pct = ?
                WHERE id = ?
            """, (exit_price, exit_time, exit_reason, int(tp_hit), int(sl_hit), 
                  int(time_stop_hit), pnl_pct, trade_id))
            self.conn.commit()
    
    def get_recent_signals(self, symbol: str, tf: str, limit: int = 100) -> List[Dict]:
        """Get recent signals"""
        cursor = self.conn.execute("""
            SELECT * FROM signals 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (symbol, tf, limit))
        return [self._signal_row_to_dict(row) for row in cursor.fetchall()]
    
    def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get trades"""
        cursor = self.conn.execute("""
            SELECT * FROM trades 
            WHERE symbol = ?
            ORDER BY entry_time DESC
            LIMIT ?
        """, (symbol, limit))
        return cursor.fetchall()
    
    def close(self):
        """Close database"""
        if self.conn:
            self.conn.close()
            logger.info("Database closed")
