"""
FORGE TRADING SYSTEM - FORENSIC-GRADE BACKTESTER
=================================================
- Strict TP/SL sequencing (first hit wins)
- Fees on entry and exit
- Slippage modeling
- Walk-forward validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml
from dataclasses import dataclass

from .features import add_features
from .signals import SignalEngine
from .conviction import ConvictionFilter


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class Trade:
    """Single trade record"""
    symbol: str
    timeframe: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str = None
    exit_price: float = None
    exit_reason: str = None
    sl: float = 0
    tp: float = 0
    score: int = 0
    pnl: float = 0
    pnl_pct: float = 0
    r_multiple: float = 0


class Backtester:
    """Forensic-grade backtesting engine"""
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.risk = self.config.get('risk', {})
        self.fee_rate = self.risk.get('fee_rate', 0.001)
        self.slippage_rate = self.risk.get('slippage_rate', 0.0005)
        self.max_holding_bars = self.risk.get('max_holding_bars', 20)
        
        self.signal_engine = SignalEngine(self.config)
        self.conviction_filter = ConvictionFilter(self.config)
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        
        # Kelly Criterion tracking
        self.use_kelly = self.risk.get('use_kelly', False) # If false, uses fixed size
        self.kelly_fraction = self.risk.get('kelly_fraction', 0.5) # Half-Kelly is safer
        self.max_position_size = self.risk.get('max_position_size', 0.50) # Cap max size (e.g. 50% of equity)
        
    def _get_position_size(self) -> float:
        """
        Calculate position size (fraction of equity) using Kelly Criterion
        f = (p(b+1) - 1) / b
        p = win rate
        b = avg_win / avg_loss (payoff ratio)
        """
        if not self.use_kelly or len(self.trades) < 10:
            return 1.0 # Default to 100% allocation (base behavior) or fixed size
            
        # Calculate rolling metrics (last 50 trades)
        recent_trades = self.trades[-50:]
        wins = [t for t in recent_trades if t.pnl > 0]
        losses = [t for t in recent_trades if t.pnl <= 0]
        
        if not losses: return self.max_position_size
        
        p = len(wins) / len(recent_trades)
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl_pct for t in losses])) if losses else 0
        
        if avg_loss == 0: return self.max_position_size
        
        b = avg_win / avg_loss
        
        # Kelly formula
        f = (p * (b + 1) - 1) / b
        
        # Apply fractional Kelly and limits
        size = f * self.kelly_fraction
        
        # Clamp between 0.05 and max_position_size
        return max(0.05, min(size, self.max_position_size))
    
    def _apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """Apply slippage to price"""
        slip = price * self.slippage_rate
        
        if is_entry:
            # Entry: buy higher, sell lower
            return price + slip if direction == 'BUY' else price - slip
        else:
            # Exit: buy exits lower, sell exits higher
            return price - slip if direction == 'BUY' else price + slip
    
    def _apply_fees(self, price: float, qty: float = 1.0) -> float:
        """Calculate fee for transaction"""
        return price * qty * self.fee_rate
    
    def _check_exit(self, candle: pd.Series, trade: Trade) -> Tuple[bool, float, str]:
        """
        Check if trade should exit on this candle.
        Uses strict TP/SL sequencing (first hit wins).
        """
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        if trade.direction == 'BUY':
            # Check SL first (lower priority but check order matters)
            if low <= trade.sl:
                return True, trade.sl, 'SL'
            # Then TP
            if high >= trade.tp:
                return True, trade.tp, 'TP'
        else:
            # SELL
            if high >= trade.sl:
                return True, trade.sl, 'SL'
            if low <= trade.tp:
                return True, trade.tp, 'TP'
        
        return False, 0, ''
    
    def run(self, df: pd.DataFrame, symbol: str, timeframe: str, 
            start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest on historical data.
        """
        self.trades = []
        self.equity_curve = [1.0]  # Start with 1.0 (100%)
        
        # Add features
        df = add_features(df, self.config)
        df = df.reset_index(drop=True)
        
        # Filter by date if provided
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
        
        if len(df) < 100:
            return self._empty_result()
        
        current_trade: Optional[Trade] = None
        bars_held = 0
        
        for i in range(50, len(df)):
            candle = df.iloc[i]
            history = df.iloc[:i+1]
            
            # If in trade, check exit
            if current_trade:
                bars_held += 1
                
                # Time-based exit
                if bars_held >= self.max_holding_bars:
                    exit_price = self._apply_slippage(candle['close'], current_trade.direction, False)
                    fee = self._apply_fees(exit_price)
                    
                    current_trade.exit_time = str(candle.get('timestamp', i))
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = 'TIME'
                    
                    self._calculate_pnl(current_trade, fee)
                    self._calculate_pnl(current_trade, fee)
                    self.trades.append(current_trade)
                    
                    # Apply position size to equity curve
                    allocation = self._get_position_size()
                    pnl_impact = current_trade.pnl_pct * allocation
                    self.equity_curve.append(self.equity_curve[-1] * (1 + pnl_impact))
                    
                    current_trade = None
                    bars_held = 0
                    continue
                
                # Check TP/SL
                should_exit, exit_price, reason = self._check_exit(candle, current_trade)
                
                if should_exit:
                    exit_price = self._apply_slippage(exit_price, current_trade.direction, False)
                    fee = self._apply_fees(exit_price)
                    
                    current_trade.exit_time = str(candle.get('timestamp', i))
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = reason
                    
                    self._calculate_pnl(current_trade, fee)
                    self._calculate_pnl(current_trade, fee)
                    self.trades.append(current_trade)
                    
                    # Apply position size to equity curve
                    allocation = self._get_position_size()
                    pnl_impact = current_trade.pnl_pct * allocation
                    self.equity_curve.append(self.equity_curve[-1] * (1 + pnl_impact))
                    
                    current_trade = None
                    bars_held = 0
                    continue
            
            else:
                # Check for new signal
                signal = self.signal_engine.generate_signal(history, symbol, timeframe)
                
                if signal:
                    # Apply conviction filter
                    conviction = self.conviction_filter.evaluate(signal, history)
                    
                    if conviction['passed']:
                        # Enter trade
                        entry_price = self._apply_slippage(signal['entry'], signal['direction'], True)
                        fee = self._apply_fees(entry_price)
                        
                        current_trade = Trade(
                            symbol=symbol,
                            timeframe=timeframe,
                            direction=signal['direction'],
                            entry_time=str(candle.get('timestamp', i)),
                            entry_price=entry_price,
                            sl=signal['sl'],
                            tp=signal['tp'],
                            score=signal['score']
                        )
                        bars_held = 0
        
        # Close any open trade at end
        if current_trade:
            exit_price = self._apply_slippage(df.iloc[-1]['close'], current_trade.direction, False)
            fee = self._apply_fees(exit_price)
            
            current_trade.exit_time = str(df.iloc[-1].get('timestamp', len(df)))
            current_trade.exit_price = exit_price
            current_trade.exit_reason = 'END'
            
            self._calculate_pnl(current_trade, fee)
            self.trades.append(current_trade)
        
        return self._calculate_metrics()
    
    def _calculate_pnl(self, trade: Trade, exit_fee: float):
        """Calculate PnL for a trade"""
        entry_fee = self._apply_fees(trade.entry_price)
        total_fees = entry_fee + exit_fee
        
        if trade.direction == 'BUY':
            raw_pnl = trade.exit_price - trade.entry_price
        else:
            raw_pnl = trade.entry_price - trade.exit_price
        
        trade.pnl = raw_pnl - total_fees
        trade.pnl_pct = trade.pnl / trade.entry_price
        
        # Calculate R-multiple
        risk = abs(trade.entry_price - trade.sl)
        if risk > 0:
            trade.r_multiple = trade.pnl / risk
        else:
            trade.r_multiple = 0
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics"""
        if not self.trades:
            return self._empty_result()
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades if total_trades else 0
        
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_r = np.mean([t.r_multiple for t in self.trades])
        
        # Max drawdown
        equity = self.equity_curve
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Net profit
        net_profit = sum(t.pnl for t in self.trades)
        
        return {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_r': avg_r,
            'max_drawdown': max_dd,
            'net_profit': net_profit,
            'net_profit_pct': (self.equity_curve[-1] - 1) * 100 if self.equity_curve else 0,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _empty_result(self) -> Dict:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_r': 0,
            'max_drawdown': 0,
            'net_profit': 0,
            'net_profit_pct': 0,
            'trades': [],
            'equity_curve': [1.0]
        }


class WalkForwardValidator:
    """Walk-forward validation system"""
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.wf = self.config.get('walkforward', {})
    
    def validate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Run walk-forward validation.
        Returns results for each period with warnings if failed.
        """
        results = {}
        periods = [
            ('train', self.wf.get('train_start'), self.wf.get('train_end')),
            ('validate', self.wf.get('validate_start'), self.wf.get('validate_end')),
            ('test', self.wf.get('test_start'), self.wf.get('test_end')),
            ('forward', self.wf.get('forward_start'), self.wf.get('forward_end'))
        ]
        
        bt = Backtester(self.config)
        
        for period_name, start, end in periods:
            result = bt.run(df, symbol, timeframe, start, end)
            
            # Add warnings
            warnings = []
            if result['total_trades'] < 10:
                warnings.append(f"Low trade count: {result['total_trades']}")
            if result['win_rate'] < 0.4:
                warnings.append(f"Low win rate: {result['win_rate']:.1%}")
            if result['profit_factor'] < 1.0:
                warnings.append(f"Unprofitable: PF={result['profit_factor']:.2f}")
            if result['max_drawdown'] > 0.2:
                warnings.append(f"High drawdown: {result['max_drawdown']:.1%}")
            
            results[period_name] = {
                **result,
                'period': period_name,
                'start': start,
                'end': end,
                'warnings': warnings,
                'passed': len(warnings) == 0
            }
        
        # Overall assessment
        passed_count = sum(1 for r in results.values() if r['passed'])
        
        return {
            'periods': results,
            'passed_count': passed_count,
            'total_periods': len(periods),
            'valid': passed_count >= 2  # At least 2 periods must pass
        }
