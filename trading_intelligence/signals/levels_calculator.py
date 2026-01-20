"""
Levels Calculator - Convert ML signals to actionable trade plans
Entry zones, ATR-based stops, R-multiple targets
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LevelsCalculator:
    """Calculate entry, stop loss, and take profit levels"""
    
    def __init__(self, atr_stop_multiplier: float = 2.0, r_multiples: list = [1.0, 2.0, 3.0]):
        """
        Args:
            atr_stop_multiplier: ATR multiple for stop distance
            r_multiples: R-multiples for TP1, TP2, TP3
        """
        self.atr_stop_multiplier = atr_stop_multiplier
        self.r_multiples = r_multiples
    
    def calculate_levels(self, df: pd.DataFrame, signal: str, confidence: float) -> Optional[Dict]:
        """
        Calculate trade levels
        
        Args:
            df: DataFrame with latest price data and indicators
            signal: 'BUY' or 'SELL'
            confidence: 0-100
            
        Returns:
            {
                'entry_zone': {'min': float, 'max': float},
                'stop_loss': float,
                'take_profits': {'tp1': float, 'tp2': float, 'tp3': float},
                'risk_reward': float,
                'valid': bool
            }
        """
        if signal == 'NO-TRADE':
            return None
        
        # Get latest bar
        latest = df.iloc[-1]
        current_price = latest['close']
        atr = latest.get('atr_14', current_price * 0.02)  # Fallback to 2% if no ATR
        
        # Entry zone (tight around current price for aggressive entry)
        entry_spread = atr * 0.3  # 30% of ATR
        
        if signal == 'BUY':
            entry_min = current_price - entry_spread
            entry_max = current_price + entry_spread
            entry_mid = current_price
            
            # Stop loss below recent structure
            stop_loss = current_price - (self.atr_stop_multiplier * atr)
            
            # Calculate risk
            risk_distance = entry_mid - stop_loss
            
            # Take profits
            tp1 = entry_mid + (risk_distance * self.r_multiples[0])
            tp2 = entry_mid + (risk_distance * self.r_multiples[1])
            tp3 = entry_mid + (risk_distance * self.r_multiples[2])
        
        elif signal == 'SELL':
            entry_min = current_price - entry_spread
            entry_max = current_price + entry_spread
            entry_mid = current_price
            
            # Stop loss above recent structure
            stop_loss = current_price + (self.atr_stop_multiplier * atr)
            
            # Calculate risk
            risk_distance = stop_loss - entry_mid
            
            # Take profits
            tp1 = entry_mid - (risk_distance * self.r_multiples[0])
            tp2 = entry_mid - (risk_distance * self.r_multiples[1])
            tp3 = entry_mid - (risk_distance * self.r_multiples[2])
        
        else:
            return None
        
        # Risk:Reward validation
        risk_reward = (tp1 - entry_mid) / risk_distance if signal == 'BUY' else (entry_mid - tp1) / risk_distance
        
        # Minimum R:R threshold
        valid = risk_reward >= 1.5 and confidence >= 60
        
        return {
            'entry_zone': {
                'min': round(entry_min, 2),
                'max': round(entry_max, 2),
                'mid': round(entry_mid, 2)
            },
            'stop_loss': round(stop_loss, 2),
            'take_profits': {
                'tp1': round(tp1, 2),
                'tp2': round(tp2, 2),
                'tp3': round(tp3, 2)
            },
            'risk_reward': round(risk_reward, 2),
            'risk_distance': round(risk_distance, 2),
            'valid': valid
        }
    
    def calculate_pips(self, asset: str, price_diff: float) -> float:
        """
        Calculate pip/point value based on asset
        
        Args:
            asset: 'BTC', 'PAXG', 'XAU'
            price_diff: Price difference
            
        Returns:
            Pips/points
        """
        if asset == 'BTC':
            # BTC: 1 point = $1
            return round(price_diff, 0)
        
        elif asset in ['PAXG', 'XAU']:
            # Gold: 1 pip = $0.01
            return round(price_diff * 100, 1)
        
        return round(price_diff, 2)


# Demo
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'close': [95000],
        'atr_14': [500]
    })
    
    calc = LevelsCalculator()
    
    # BUY signal
    levels = calc.calculate_levels(df, 'BUY', confidence=75)
    
    print("📊 BUY Signal Levels:")
    print(f"   Entry Zone: ${levels['entry_zone']['min']} - ${levels['entry_zone']['max']}")
    print(f"   Stop Loss: ${levels['stop_loss']}")
    print(f"   TP1: ${levels['take_profits']['tp1']}")
    print(f"   TP2: ${levels['take_profits']['tp2']}")
    print(f"   TP3: ${levels['take_profits']['tp3']}")
    print(f"   Risk:Reward: 1:{levels['risk_reward']}")
    print(f"   Valid: {levels['valid']}")
    
    # Calculate pips
    points = calc.calculate_pips('BTC', levels['take_profits']['tp1'] - levels['entry_zone']['mid'])
    print(f"\n   TP1 Target: {points} points")
