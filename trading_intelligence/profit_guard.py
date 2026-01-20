"""
PROFIT GUARD - Money Management Layer
======================================
Implements institutional-grade capital protection:

1. Kelly Criterion - Optimal position sizing
2. Trailing ATR Stop - Dynamic profit locking
3. 2% Daily Circuit Breaker - Loss limit

Usage:
    from profit_guard import ProfitGuard
    
    guard = ProfitGuard(account_balance=10000)
    size = guard.calculate_position_size(signal)
    guard.update_trailing_stop(position, current_price)
"""
import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional
from dataclasses import dataclass, asdict


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Position:
    """Current position representation"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0.0
    entry_time: datetime = None
    highest_price: float = 0.0
    lowest_price: float = 0.0


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    starting_balance: float
    current_balance: float
    pnl: float
    pnl_pct: float
    trades: int
    wins: int
    losses: int
    is_halted: bool = False


# =============================================================================
# KELLY CRITERION POSITION SIZING
# =============================================================================

class KellyCriterion:
    """
    Optimal position sizing based on Kelly Criterion.
    
    Kelly = (Win_Rate * Avg_Win - Loss_Rate * Avg_Loss) / Avg_Win
    
    We use "Half-Kelly" (0.5x Kelly) for conservative sizing.
    """
    
    def __init__(self, 
                 max_position_pct: float = 0.10,  # Max 10% of account
                 kelly_fraction: float = 0.5):    # Half-Kelly
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
    
    def calculate(self, 
                  win_rate: float, 
                  avg_win: float, 
                  avg_loss: float,
                  account_balance: float,
                  confidence: float = 0.5) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Parameters:
        -----------
        win_rate : Historical win rate (0-1)
        avg_win : Average winning trade (absolute)
        avg_loss : Average losing trade (absolute)
        account_balance : Current account balance
        confidence : Signal confidence (0-1) - scales the size
        
        Returns:
        --------
        Position size in account currency
        """
        if avg_win <= 0 or avg_loss <= 0 or win_rate <= 0:
            return 0.0
        
        # Kelly formula
        loss_rate = 1 - win_rate
        kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # Negative Kelly = don't trade
        if kelly <= 0:
            return 0.0
        
        # Apply fraction (Half-Kelly by default)
        kelly = kelly * self.kelly_fraction
        
        # Scale by confidence
        kelly = kelly * confidence
        
        # Cap at max position
        kelly = min(kelly, self.max_position_pct)
        
        # Calculate size
        position_size = account_balance * kelly
        
        return round(position_size, 2)
    
    def calculate_from_stats(self, 
                             stats: dict,
                             account_balance: float,
                             confidence: float = 0.5) -> float:
        """Calculate from trading stats dict"""
        total = stats.get('total_trades', 0)
        wins = stats.get('wins', 0)
        
        if total < 10:  # Not enough data
            # Use conservative default
            return account_balance * 0.02 * confidence
        
        win_rate = wins / total
        avg_win = stats.get('avg_win', 20)
        avg_loss = stats.get('avg_loss', 20)
        
        return self.calculate(win_rate, avg_win, avg_loss, 
                              account_balance, confidence)


# =============================================================================
# TRAILING ATR STOP
# =============================================================================

class TrailingATRStop:
    """
    Dynamic trailing stop based on ATR.
    
    Locks in profits as the trade moves in your favor.
    Stop moves up (for longs) / down (for shorts) but never backwards.
    """
    
    def __init__(self, 
                 initial_mult: float = 1.5,
                 trailing_mult: float = 2.0):
        self.initial_mult = initial_mult
        self.trailing_mult = trailing_mult
    
    def calculate_initial_stop(self, 
                                entry: float, 
                                atr: float, 
                                side: str) -> float:
        """Calculate initial stop loss"""
        if side == 'BUY':
            return round(entry - atr * self.initial_mult, 2)
        else:
            return round(entry + atr * self.initial_mult, 2)
    
    def update_trailing_stop(self,
                              position: Position,
                              current_price: float,
                              atr: float) -> float:
        """
        Update trailing stop based on current price.
        
        Parameters:
        -----------
        position : Current position
        current_price : Current market price
        atr : Current ATR value
        
        Returns:
        --------
        New trailing stop level
        """
        if position.side == 'BUY':
            # Track highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
            
            # Calculate new stop (trail behind highest)
            new_stop = position.highest_price - atr * self.trailing_mult
            
            # Only move stop up, never down
            if new_stop > position.trailing_stop:
                position.trailing_stop = round(new_stop, 2)
        
        else:  # SELL
            # Track lowest price
            if position.lowest_price == 0 or current_price < position.lowest_price:
                position.lowest_price = current_price
            
            # Calculate new stop (trail above lowest)
            new_stop = position.lowest_price + atr * self.trailing_mult
            
            # Only move stop down, never up
            if position.trailing_stop == 0 or new_stop < position.trailing_stop:
                position.trailing_stop = round(new_stop, 2)
        
        return position.trailing_stop
    
    def check_stop_hit(self, position: Position, current_price: float) -> bool:
        """Check if trailing stop was hit"""
        if position.trailing_stop == 0:
            return False
        
        if position.side == 'BUY':
            return current_price <= position.trailing_stop
        else:
            return current_price >= position.trailing_stop


# =============================================================================
# DAILY CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Halts trading if daily loss exceeds threshold.
    
    Default: 2% daily loss limit
    """
    
    def __init__(self, 
                 daily_loss_limit: float = 0.02,  # 2%
                 cooldown_hours: int = 24):
        self.daily_loss_limit = daily_loss_limit
        self.cooldown_hours = cooldown_hours
        self.state_file = Path(__file__).parent / 'logs' / 'circuit_breaker.json'
        self.state_file.parent.mkdir(exist_ok=True)
        
        self.is_halted = False
        self.halt_time: Optional[datetime] = None
        self.halt_reason: str = ""
        self.daily_pnl: float = 0.0
        self.starting_balance: float = 0.0
        
        self._load_state()
    
    def _load_state(self):
        """Load persisted state"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                
                # Check if same day
                if data.get('date') == str(date.today()):
                    self.is_halted = data.get('is_halted', False)
                    self.daily_pnl = data.get('daily_pnl', 0.0)
                    self.starting_balance = data.get('starting_balance', 0.0)
                    if data.get('halt_time'):
                        self.halt_time = datetime.fromisoformat(data['halt_time'])
            except:
                pass
    
    def _save_state(self):
        """Persist state to disk"""
        data = {
            'date': str(date.today()),
            'is_halted': self.is_halted,
            'halt_time': self.halt_time.isoformat() if self.halt_time else None,
            'halt_reason': self.halt_reason,
            'daily_pnl': self.daily_pnl,
            'starting_balance': self.starting_balance
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def set_starting_balance(self, balance: float):
        """Set starting balance for the day"""
        self.starting_balance = balance
        self._save_state()
    
    def update_pnl(self, trade_pnl: float) -> bool:
        """
        Update daily P&L and check circuit breaker.
        
        Returns:
        --------
        True if trading should continue, False if halted
        """
        self.daily_pnl += trade_pnl
        
        if self.starting_balance > 0:
            pnl_pct = self.daily_pnl / self.starting_balance
            
            if pnl_pct <= -self.daily_loss_limit:
                self.trigger(f"Daily loss limit hit: {pnl_pct:.2%}")
                return False
        
        self._save_state()
        return True
    
    def trigger(self, reason: str):
        """Trigger the circuit breaker"""
        self.is_halted = True
        self.halt_time = datetime.now()
        self.halt_reason = reason
        self._save_state()
        
        print(f"🛑 CIRCUIT BREAKER TRIGGERED: {reason}")
    
    def check_can_trade(self) -> tuple:
        """
        Check if trading is allowed.
        
        Returns:
        --------
        Tuple of (can_trade: bool, reason: str)
        """
        if not self.is_halted:
            return True, "OK"
        
        # Check if cooldown has expired
        if self.halt_time:
            elapsed = datetime.now() - self.halt_time
            if elapsed.total_seconds() > self.cooldown_hours * 3600:
                self.reset()
                return True, "Cooldown expired"
        
        remaining = self.cooldown_hours * 3600 - (datetime.now() - self.halt_time).total_seconds()
        return False, f"Trading halted. Remaining: {remaining/3600:.1f} hours"
    
    def reset(self):
        """Reset the circuit breaker"""
        self.is_halted = False
        self.halt_time = None
        self.halt_reason = ""
        self.daily_pnl = 0.0
        self._save_state()
        print("✅ Circuit breaker reset")


# =============================================================================
# PROFIT GUARD (Combined Manager)
# =============================================================================

class ProfitGuard:
    """
    Combined money management system.
    
    Integrates:
    - Kelly Criterion sizing
    - Trailing ATR stops
    - Daily circuit breaker
    """
    
    def __init__(self, 
                 account_balance: float = 10000,
                 daily_loss_limit: float = 0.02):
        
        self.account_balance = account_balance
        self.kelly = KellyCriterion()
        self.trailing = TrailingATRStop()
        self.circuit = CircuitBreaker(daily_loss_limit)
        
        # Set starting balance for circuit breaker
        self.circuit.set_starting_balance(account_balance)
        
        # Trading stats (would normally come from DB)
        self.stats = {
            'total_trades': 100,
            'wins': 55,
            'losses': 45,
            'avg_win': 50,
            'avg_loss': 40
        }
        
        # Active positions
        self.positions: Dict[str, Position] = {}
    
    def can_trade(self) -> tuple:
        """Check if trading is allowed"""
        return self.circuit.check_can_trade()
    
    def calculate_position_size(self, 
                                 signal: dict,
                                 risk_per_trade: float = 0.01) -> dict:
        """
        Calculate position size for a signal.
        
        Parameters:
        -----------
        signal : Signal dict with 'confidence', 'stop_mult', etc.
        risk_per_trade : Max risk per trade as % of account
        
        Returns:
        --------
        dict with 'size', 'risk', 'kelly_pct'
        """
        confidence = signal.get('confidence', 50) / 100
        
        # Kelly-based size
        kelly_size = self.kelly.calculate_from_stats(
            self.stats, 
            self.account_balance, 
            confidence
        )
        
        # Size multiplier from signal
        size_mult = signal.get('size_mult', 1.0)
        kelly_size *= size_mult
        
        # Risk-based cap
        # If stop is 2% away, max position = risk_per_trade / 0.02
        atr_pct = signal.get('atr_pct', 1.5)
        stop_mult = signal.get('stop_mult', 1.5)
        stop_distance_pct = (atr_pct * stop_mult) / 100
        
        if stop_distance_pct > 0:
            risk_based_size = (self.account_balance * risk_per_trade) / stop_distance_pct
        else:
            risk_based_size = kelly_size
        
        # Take smaller of Kelly and risk-based
        final_size = min(kelly_size, risk_based_size)
        
        return {
            'size': round(final_size, 2),
            'kelly_pct': round(kelly_size / self.account_balance * 100, 2),
            'risk_pct': round(risk_per_trade * 100, 2),
            'stop_distance_pct': round(stop_distance_pct * 100, 2)
        }
    
    def open_position(self, symbol: str, signal: dict, 
                       entry_price: float, atr: float) -> Position:
        """Open a new position with proper sizing and stops"""
        
        # Calculate size
        sizing = self.calculate_position_size(signal)
        quantity = sizing['size'] / entry_price
        
        # Calculate stops
        side = signal['signal']
        stop_mult = signal.get('stop_mult', 1.5)
        tp_mult = signal.get('tp_mult', 2.0)
        
        if side == 'BUY':
            stop_loss = entry_price - atr * stop_mult
            take_profit = entry_price + atr * tp_mult
        else:
            stop_loss = entry_price + atr * stop_mult
            take_profit = entry_price - atr * tp_mult
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            trailing_stop=round(stop_loss, 2),
            entry_time=datetime.now(),
            highest_price=entry_price if side == 'BUY' else 0,
            lowest_price=entry_price if side == 'SELL' else 0
        )
        
        self.positions[symbol] = position
        return position
    
    def update_position(self, symbol: str, current_price: float, atr: float) -> dict:
        """Update position trailing stop and check for exit"""
        if symbol not in self.positions:
            return {'action': 'NONE', 'reason': 'No position'}
        
        position = self.positions[symbol]
        
        # Update trailing stop
        new_stop = self.trailing.update_trailing_stop(position, current_price, atr)
        
        # Check if stop was hit
        if self.trailing.check_stop_hit(position, current_price):
            # Calculate P&L
            if position.side == 'BUY':
                pnl = (current_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - current_price) * position.quantity
            
            # Update circuit breaker
            self.circuit.update_pnl(pnl)
            
            # Remove position
            del self.positions[symbol]
            
            return {
                'action': 'CLOSE',
                'reason': 'Trailing stop hit',
                'exit_price': current_price,
                'pnl': round(pnl, 2),
                'trailing_stop': new_stop
            }
        
        # Check take profit
        if position.side == 'BUY' and current_price >= position.take_profit:
            pnl = (current_price - position.entry_price) * position.quantity
            self.circuit.update_pnl(pnl)
            del self.positions[symbol]
            return {
                'action': 'CLOSE',
                'reason': 'Take profit hit',
                'exit_price': current_price,
                'pnl': round(pnl, 2)
            }
        elif position.side == 'SELL' and current_price <= position.take_profit:
            pnl = (position.entry_price - current_price) * position.quantity
            self.circuit.update_pnl(pnl)
            del self.positions[symbol]
            return {
                'action': 'CLOSE',
                'reason': 'Take profit hit',
                'exit_price': current_price,
                'pnl': round(pnl, 2)
            }
        
        return {
            'action': 'HOLD',
            'trailing_stop': new_stop,
            'unrealized_pnl': round(
                (current_price - position.entry_price) * position.quantity 
                if position.side == 'BUY' 
                else (position.entry_price - current_price) * position.quantity,
                2
            )
        }
    
    def get_status(self) -> dict:
        """Get current profit guard status"""
        can_trade, reason = self.can_trade()
        return {
            'account_balance': self.account_balance,
            'can_trade': can_trade,
            'trade_status': reason,
            'daily_pnl': self.circuit.daily_pnl,
            'is_halted': self.circuit.is_halted,
            'positions': len(self.positions),
            'stats': self.stats
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("💰 PROFIT GUARD - Money Management Test")
    print("="*70)
    
    # Initialize
    guard = ProfitGuard(account_balance=10000)
    
    # Test Kelly sizing
    signal = {
        'signal': 'BUY',
        'confidence': 75,
        'stop_mult': 1.5,
        'tp_mult': 2.0,
        'atr_pct': 1.5
    }
    
    sizing = guard.calculate_position_size(signal)
    print(f"\n📊 Position Sizing:")
    print(f"  Size: ${sizing['size']}")
    print(f"  Kelly %: {sizing['kelly_pct']}%")
    print(f"  Risk %: {sizing['risk_pct']}%")
    print(f"  Stop Distance: {sizing['stop_distance_pct']}%")
    
    # Test position open
    position = guard.open_position('BTCUSDT', signal, 100000, 1500)
    print(f"\n📈 Position Opened:")
    print(f"  Entry: ${position.entry_price}")
    print(f"  Quantity: {position.quantity:.4f}")
    print(f"  Stop Loss: ${position.stop_loss}")
    print(f"  Take Profit: ${position.take_profit}")
    
    # Test trailing stop update
    result = guard.update_position('BTCUSDT', 101500, 1500)
    print(f"\n🔄 After Price Rise to $101,500:")
    print(f"  Action: {result['action']}")
    print(f"  Trailing Stop: ${result.get('trailing_stop', 'N/A')}")
    print(f"  Unrealized P&L: ${result.get('unrealized_pnl', 'N/A')}")
    
    # Test circuit breaker
    print(f"\n🛡️ Circuit Breaker Status:")
    status = guard.get_status()
    print(f"  Can Trade: {status['can_trade']}")
    print(f"  Daily P&L: ${status['daily_pnl']}")
    print(f"  Is Halted: {status['is_halted']}")
