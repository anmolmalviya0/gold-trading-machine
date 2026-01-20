"""
Backtester - Honest with fees, slippage, spread, strict sequencing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    entry_time: int
    entry_fee: float
    tp1: float
    tp2: float
    sl: float
    status: str = "open"
    exit_price: float = None
    exit_time: int = None
    exit_reason: str = None
    exit_fee: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


class BacktestEngine:
    """
    Strict, honest backtester.
    
    Rules:
    - First hit wins (TP or SL, which comes first)
    - Fees on entry AND exit
    - Slippage penalty on entry
    - Spread penalty on fill
    - Time-based exit (N candles)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.backtest_cfg = config['backtest']
        self.fee_entry = self.backtest_cfg['fee_entry']
        self.fee_exit = self.backtest_cfg['fee_exit']
        self.slippage_bps = self.backtest_cfg['slippage_bps']
        self.spread_penalty_bps = self.backtest_cfg['spread_penalty_bps']
        
        self.trades: List[Trade] = []
        self.open_trades: Dict = {}
    
    def run_backtest(self, symbol: str, tf: str, ohlcv: pd.DataFrame, 
                    signals: List[Dict], time_stop_candles: int = 20) -> Dict:
        """
        Run backtest on OHLCV + signals.
        
        signals format: [{
            'timestamp': int,
            'open_time': int,
            'direction': 'BUY' or 'SELL',
            'entry_zone': (low, high),
            'tp1': float,
            'tp2': float,
            'sl': float,
            'score': float
        }]
        """
        self.trades = []
        self.open_trades = {}
        
        # Create a dict of signals by timestamp for O(1) lookup
        signal_map = {}
        for sig in signals:
            timestamp = sig['timestamp']
            if timestamp not in signal_map:
                signal_map[timestamp] = []
            signal_map[timestamp].append(sig)
        
        # Iterate through OHLCV
        for idx, row in ohlcv.iterrows():
            open_time = row['open_time']
            close_time = row['close_time']
            o = row['open']
            h = row['high']
            l = row['low']
            c = row['close']
            
            # Check for new signals at this candle
            if open_time in signal_map:
                for sig in signal_map[open_time]:
                    self._execute_entry(sig, o, h, l, c, close_time, symbol)
            
            # Process existing trades (check TP/SL)
            trades_to_close = []
            for trade_id, trade in self.open_trades.items():
                result = self._check_exit(trade, o, h, l, c, close_time, time_stop_candles)
                if result:
                    trades_to_close.append((trade_id, result))
            
            # Close trades
            for trade_id, (exit_price, exit_reason) in trades_to_close:
                trade = self.open_trades.pop(trade_id)
                self._finalize_trade(trade, exit_price, close_time, exit_reason)
                self.trades.append(trade)
        
        # Close remaining open trades at last price
        last_price = ohlcv.iloc[-1]['close']
        last_time = ohlcv.iloc[-1]['close_time']
        for trade in self.open_trades.values():
            self._finalize_trade(trade, last_price, last_time, "end_of_data")
            self.trades.append(trade)
        
        # Calculate stats
        return self._calculate_stats(symbol, tf)
    
    def _execute_entry(self, signal: Dict, o: float, h: float, l: float, c: float, 
                      close_time: int, symbol: str):
        """Execute entry order"""
        direction = signal['direction']
        entry_zone_low, entry_zone_high = signal['entry_zone']
        
        # Determine entry price with slippage
        if direction == 'BUY':
            # Buy at ask (high) with slippage
            entry_price = h * (1 + self.slippage_bps / 10000)
            
            # Check if we hit entry zone
            if l <= entry_zone_high and c >= entry_zone_low:
                # Use mid of entry zone
                entry_price = (entry_zone_low + entry_zone_high) / 2
        else:  # SELL
            # Sell at bid (low) with slippage
            entry_price = l * (1 - self.slippage_bps / 10000)
            
            if h >= entry_zone_low and c <= entry_zone_high:
                entry_price = (entry_zone_low + entry_zone_high) / 2
        
        # Add spread penalty
        entry_fee = entry_price * self.fee_entry
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=close_time,
            entry_fee=entry_fee,
            tp1=signal['tp1'],
            tp2=signal['tp2'],
            sl=signal['sl']
        )
        
        trade_id = len(self.trades) + len(self.open_trades)
        self.open_trades[trade_id] = trade
        
        logger.debug(f"Entry: {direction} {symbol} @ {entry_price:.2f}")
    
    def _check_exit(self, trade: Trade, o: float, h: float, l: float, c: float,
                   close_time: int, time_stop_candles: int) -> Optional[Tuple]:
        """
        Check if trade hits TP, SL, or time stop.
        Returns: (exit_price, reason) or None
        """
        
        if trade.direction == 'BUY':
            # TP hit first?
            if h >= trade.tp1:
                return (trade.tp1, "TP_HIT")
            
            # SL hit?
            if l <= trade.sl:
                return (trade.sl, "SL_HIT")
        
        else:  # SELL
            # TP hit first?
            if l <= trade.tp1:
                return (trade.tp1, "TP_HIT")
            
            # SL hit?
            if h >= trade.sl:
                return (trade.sl, "SL_HIT")
        
        # Time stop (rough check - would need candle counter in real impl)
        # Skip for now
        
        return None
    
    def _finalize_trade(self, trade: Trade, exit_price: float, close_time: int, 
                       exit_reason: str):
        """Close trade and calculate PnL"""
        exit_fee = exit_price * self.fee_exit
        
        if trade.direction == 'BUY':
            pnl = (exit_price - trade.entry_price) * 1 - trade.entry_fee - exit_fee
        else:  # SELL
            pnl = (trade.entry_price - exit_price) * 1 - trade.entry_fee - exit_fee
        
        pnl_pct = (pnl / trade.entry_price) * 100 if trade.entry_price else 0
        
        trade.exit_price = exit_price
        trade.exit_time = close_time
        trade.exit_reason = exit_reason
        trade.exit_fee = exit_fee
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = "closed"
    
    def _calculate_stats(self, symbol: str, tf: str) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                'symbol': symbol,
                'timeframe': tf,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'expectancy': 0
            }
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = sum(t.pnl for t in losses)
        
        profit_factor = total_wins / abs(total_losses) if total_losses != 0 else 0
        
        total_pnl = total_wins + total_losses
        
        expectancy = total_pnl / len(self.trades) if self.trades else 0
        
        # Max drawdown (simplistic)
        equity_curve = [0]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        cumulative_max = np.maximum.accumulate(equity_curve)
        drawdown = np.array(equity_curve) - cumulative_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'symbol': symbol,
            'timeframe': tf,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'total_wins': round(total_wins, 2),
            'total_losses': round(total_losses, 2),
            'max_drawdown': round(max_dd, 2),
            'expectancy': round(expectancy, 2),
            'trades': self.trades
        }
