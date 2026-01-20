"""
REALISTIC BACKTEST ENGINE
=========================
Simulates TP/SL + Fees + Slippage

This is NOT just accuracy testing.
This simulates REAL TRADING with:
- TP hit detection
- SL hit detection
- Fees (0.1%)
- Slippage (0.05%)
- Expectancy calculation
- Profit factor
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json

class Trade:
    """Single trade record"""
    def __init__(self, entry_time, entry_price, side, sl, tp, size=1000):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.side = side  # 'BUY' or 'SELL'
        self.sl = sl
        self.tp = tp
        self.size = size
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.fees = 0.0
        self.slippage = 0.0
        self.net_pnl = 0.0
        
    def to_dict(self):
        return {
            'entry_time': str(self.entry_time),
            'entry_price': self.entry_price,
            'side': self.side,
            'sl': self.sl,
            'tp': self.tp,
            'exit_time': str(self.exit_time) if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'gross_pnl': self.pnl,
            'fees': self.fees,
            'slippage': self.slippage,
            'net_pnl': self.net_pnl,
            'pnl_pct': self.pnl_pct
        }


class RealisticBacktester:
    """
    Backtest with REAL simulation:
    - TP/SL detection using highs/lows
    - Entry/exit fees
    - Slippage
    - Position sizing
    """
    
    def __init__(self, 
                 fee_pct: float = 0.001,      # 0.1%
                 slippage_pct: float = 0.0005, # 0.05%
                 max_holding: int = 100):
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.max_holding = max_holding
        
        self.trades: List[Trade] = []
        self.active_trade: Trade = None
    
    def open_trade(self, time, price, side, sl, tp, size=1000):
        """Open a new trade"""
        if self.active_trade:
            return False  # Already have active trade
        
        # Apply entry slippage
        if side == 'BUY':
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)
        
        self.active_trade = Trade(time, entry_price, side, sl, tp, size)
        return True
    
    def check_exit(self, time, high, low, close):
        """
        Check if TP or SL is hit.
        Returns (hit, reason) where reason is 'TP', 'SL', or 'TIMEOUT'
        """
        if not self.active_trade:
            return False, None
        
        trade = self.active_trade
        
        if trade.side == 'BUY':
            # Check TP (high >= tp)
            if high >= trade.tp:
                return True, 'TP'
            # Check SL (low <= sl)
            if low <= trade.sl:
                return True, 'SL'
        else:  # SELL
            # Check TP (low <= tp, because in short TP is lower)
            if low <= trade.tp:
                return True, 'TP'
            # Check SL (high >= sl)
            if high >= trade.sl:
                return True, 'SL'
        
        # Check timeout
        if (time - trade.entry_time).total_seconds() / 60 > self.max_holding:
            return True, 'TIMEOUT'
        
        return False, None
    
    def close_trade(self, time, exit_price, reason):
        """Close active trade"""
        if not self.active_trade:
            return
        
        trade = self.active_trade
        trade.exit_time = time
        trade.exit_reason = reason
        
        # Apply exit slippage
        if trade.side == 'BUY':
            exit_price_adj = exit_price * (1 - self.slippage_pct)
        else:
            exit_price_adj = exit_price * (1 + self.slippage_pct)
        
        trade.exit_price = exit_price_adj
        
        # Calculate gross PnL
        if trade.side == 'BUY':
            trade.pnl = (exit_price_adj - trade.entry_price) * (trade.size / trade.entry_price)
        else:
            trade.pnl = (trade.entry_price - exit_price_adj) * (trade.size /  trade.entry_price)
        
        # Fees
        trade.fees = trade.size * self.fee_pct * 2  # Entry + exit
        
        # Slippage cost
        slippage_entry = abs(trade.entry_price - (exit_price if trade.side == 'BUY' else exit_price)) * self.slippage_pct
        slippage_exit = abs(exit_price - exit_price_adj)
        trade.slippage = (slippage_entry + slippage_exit) * (trade.size / trade.entry_price)
        
        # Net PnL
        trade.net_pnl = trade.pnl - trade.fees
        trade.pnl_pct = (trade.net_pnl / trade.size) * 100
        
        self.trades.append(trade)
        self.active_trade = None
    
    def run(self, signals_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
        """
        Run backtest.
        
        Parameters:
        -----------
        signals_df : DataFrame with columns ['time', 'signal', 'confidence', 'entry', 'sl', 'tp']
        price_df : DataFrame with columns ['time', 'o', 'h', 'l', 'c', 'v']
        
        Returns:
        --------
        Dict with backtest results
        """
        self.trades = []
        self.active_trade = None
        
        # Merge signals and prices
        price_df = price_df.copy()
        price_df['time'] = pd.to_datetime(price_df['time'])
        signals_df['time'] = pd.to_datetime(signals_df['time'])
        
        for idx, row in price_df.iterrows():
            time = row['time']
            high = row['h']
            low = row['l']
            close = row['c']
            
            # Check if we need to exit active trade
            if self.active_trade:
                hit, reason = self.check_exit(time, high, low, close)
                
                if hit:
                    # Determine exit price
                    if reason == 'TP':
                        exit_price = self.active_trade.tp
                    elif reason == 'SL':
                        exit_price = self.active_trade.sl
                    else:  # TIMEOUT
                        exit_price = close
                    
                    self.close_trade(time, exit_price, reason)
            
            # Check for new signal
            if not self.active_trade:
                signal_row = signals_df[signals_df['time'] == time]
                
                if not signal_row.empty:
                    sig = signal_row.iloc[0]
                    
                    if sig['signal'] in ['BUY', 'SELL']:
                        self.open_trade(
                            time=time,
                            price=sig['entry'],
                            side=sig['signal'],
                            sl=sig['sl'],
                            tp=sig['tp'],
                            size=1000
                        )
        
        # Close any remaining active trade
        if self.active_trade:
            last_row = price_df.iloc[-1]
            self.close_trade(last_row['time'], last_row['c'], 'END')
        
        # Calculate statistics
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'total_pnl': 0,
                'total_fees': 0,
                'sharpe': 0
            }
        
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]
        
        total_pnl = sum(t.net_pnl for t in self.trades)
        total_fees = sum(t.fees for t in self.trades)
        
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
        
        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
        
        expectancy = avg_win * (len(wins)/len(self.trades)) + avg_loss * (len(losses)/len(self.trades)) if self.trades else 0
        
        # Sharpe (simplified)
        returns = [t.pnl_pct for t in self.trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'sharpe': sharpe,
            'exit_reasons': {
                'TP': sum(1 for t in self.trades if t.exit_reason == 'TP'),
                'SL': sum(1 for t in self.trades if t.exit_reason == 'SL'),
                'TIMEOUT': sum(1 for t in self.trades if t.exit_reason == 'TIMEOUT'),
            }
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def print_summary(self):
        """Print backtest summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("📊 REALISTIC BACKTEST RESULTS")
        print("="*70)
        
        print(f"\n💼 Trade Statistics:")
        print(f"   Total Trades:  {stats['total_trades']}")
        print(f"   Wins:          {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"   Losses:        {stats['losses']}")
        
        print(f"\n💰 Performance:")
        print(f"   Total PnL:     ${stats['total_pnl']:.2f}")
        print(f"   Total Fees:    ${stats['total_fees']:.2f}")
        print(f"   Avg Win:       ${stats['avg_win']:.2f}")
        print(f"   Avg Loss:      ${stats['avg_loss']:.2f}")
        print(f"   Expectancy:    ${stats['expectancy']:.2f}")
        
        print(f"\n📈 Metrics:")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        print(f"   Sharpe Ratio:  {stats['sharpe']:.2f}")
        
        print(f"\n🚪 Exit Reasons:")
        print(f"   Take Profit:   {stats['exit_reasons']['TP']}")
        print(f"   Stop Loss:     {stats['exit_reasons']['SL']}")
        print(f"   Timeout:       {stats['exit_reasons']['TIMEOUT']}")
        
        verdict = "✅ PROFITABLE" if stats['total_pnl'] > 0 and stats['profit_factor'] > 1.0 else "❌ NOT PROFITABLE"
        print(f"\n{verdict}")
        print("="*70)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("="*70)
    print("🧪 REALISTIC BACKTEST ENGINE - Test")
    print("="*70)
    
    # Load some test data
    price_df = pd.read_csv('/Users/anmol/Desktop/gold/market_data/BTCUSDT_1h.csv')
    price_df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    price_df = price_df.tail(1000)  # Last 1000 candles
    
    # Create dummy signals (every 20 candles)
    signals = []
    for i in range(0, len(price_df), 20):
        row = price_df.iloc[i]
        atr = (row['h'] - row['l']) * 1.5  # Approx ATR
        
        signals.append({
            'time': row['time'],
            'signal': 'BUY' if i % 40 == 0 else 'SELL',
            'confidence': 60,
            'entry': row['c'],
            'sl': row['c'] - atr if i % 40 == 0 else row['c'] + atr,
            'tp': row['c'] + atr * 2 if i % 40 == 0 else row['c'] - atr * 2
        })
    
    signals_df = pd.DataFrame(signals)
    
    # Run backtest
    backtester = RealisticBacktester(
        fee_pct=0.001,
        slippage_pct=0.0005,
        max_holding=50
    )
    
    stats = backtester.run(signals_df, price_df)
    backtester.print_summary()
    
    # Save trades
    trades_df = backtester.get_trades_df()
    print(f"\n📄 Sample trades:")
    print(trades_df.head(10))
