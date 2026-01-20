"""
Indicators Module - Institutional Grade
(Wilder RSI, Wilder ATR, Pivot detection, etc.)
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)


class Indicators:
    """Core indicator calculations"""
    
    @staticmethod
    def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI using RMA (Exponential Moving Average)"""
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Wilder's RMA
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's ATR using RMA"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's RMA (different from standard EMA)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0):
        """Bollinger Bands"""
        sma = close.rolling(period).mean()
        std_dev = close.rolling(period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = upper - lower
        
        return upper, sma, lower, width
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX (Average Directional Index)"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, 1e-10)
        
        dx = 100 * (di_diff / di_sum)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        
        return adx
    
    @staticmethod
    def detect_pivots(high: pd.Series, low: pd.Series, left_bars: int = 5, 
                      right_bars: int = 5, min_height: float = 0.0001) -> tuple:
        """
        Detect pivot highs and lows using argrelextrema
        Returns: (pivot_high_idx, pivot_low_idx)
        """
        # Pivot highs
        ph_idx = argrelextrema(high.values, np.greater, order=left_bars)[0]
        
        # Pivot lows
        pl_idx = argrelextrema(low.values, np.less, order=left_bars)[0]
        
        return ph_idx, pl_idx
    
    @staticmethod
    def detect_divergence_rsi_pivot(close: pd.Series, high: pd.Series, low: pd.Series, 
                                    rsi: pd.Series, ph_idx: np.ndarray, pl_idx: np.ndarray,
                                    lookback: int = 50, confirm_delay: int = 2) -> list:
        """
        Detect RSI pivot-to-pivot divergence
        
        Bullish div: Lower low in price, higher low in RSI
        Bearish div: Higher high in price, lower high in RSI
        
        Returns list of divergences: {
            'type': 'bullish' or 'bearish',
            'bar_index': current bar,
            'price_level': pivot price,
            'rsi_level': pivot RSI,
            'confirmed': bool
        }
        """
        divergences = []
        current_idx = len(close) - 1
        
        if current_idx < lookback:
            return divergences
        
        recent_high = high.iloc[current_idx]
        recent_low = low.iloc[current_idx]
        recent_rsi = rsi.iloc[current_idx]
        
        # Look for BULLISH divergence (lower low in price, higher low in RSI)
        valid_pl = pl_idx[pl_idx < current_idx]
        if len(valid_pl) >= 2:
            last_pl = valid_pl[-1]
            prev_pl = valid_pl[-2]
            
            if (recent_low < low.iloc[last_pl] and 
                recent_rsi > rsi.iloc[last_pl]):
                
                # Check if we have confirmation delay
                is_confirmed = (current_idx - last_pl) >= confirm_delay
                
                divergences.append({
                    'type': 'bullish',
                    'bar_index': current_idx,
                    'price_level': recent_low,
                    'rsi_level': recent_rsi,
                    'confirmed': is_confirmed,
                    'pivot_bar': last_pl
                })
        
        # Look for BEARISH divergence (higher high in price, lower high in RSI)
        valid_ph = ph_idx[ph_idx < current_idx]
        if len(valid_ph) >= 2:
            last_ph = valid_ph[-1]
            prev_ph = valid_ph[-2]
            
            if (recent_high > high.iloc[last_ph] and 
                recent_rsi < rsi.iloc[last_ph]):
                
                is_confirmed = (current_idx - last_ph) >= confirm_delay
                
                divergences.append({
                    'type': 'bearish',
                    'bar_index': current_idx,
                    'price_level': recent_high,
                    'rsi_level': recent_rsi,
                    'confirmed': is_confirmed,
                    'pivot_bar': last_ph
                })
        
        return divergences
    
    @staticmethod
    def detect_liquidity_sweep(high: pd.Series, low: pd.Series, close: pd.Series,
                               ph_idx: np.ndarray, pl_idx: np.ndarray,
                               wick_beyond_pct: float = 0.995,
                               close_back_pct: float = 0.50) -> list:
        """
        Detect liquidity sweeps:
        - Wick beyond pivot (e.g., 99.5%)
        - Close back inside range (50% retracement)
        
        Returns: [{'type': 'sweep_low' or 'sweep_high', 'bar_index': idx, ...}]
        """
        sweeps = []
        current_idx = len(close) - 1
        
        if current_idx < 5:
            return sweeps
        
        # Check if current low swept below recent pivot low
        if len(pl_idx) > 0:
            last_pl = pl_idx[-1]
            pl_price = low.iloc[last_pl]
            
            current_low = low.iloc[current_idx]
            current_close = close.iloc[current_idx]
            current_high = high.iloc[current_idx]
            
            # Wick below pivot
            if current_low < (pl_price * wick_beyond_pct):
                # Close back inside
                close_range = current_high - current_low
                if current_close > (current_low + close_range * close_back_pct):
                    sweeps.append({
                        'type': 'sweep_low',
                        'bar_index': current_idx,
                        'pivot_bar': last_pl,
                        'pivot_price': pl_price,
                        'sweep_price': current_low
                    })
        
        # Check if current high swept above recent pivot high
        if len(ph_idx) > 0:
            last_ph = ph_idx[-1]
            ph_price = high.iloc[last_ph]
            
            current_high = high.iloc[current_idx]
            current_close = close.iloc[current_idx]
            current_low = low.iloc[current_idx]
            
            # Wick above pivot
            if current_high > (ph_price / wick_beyond_pct):
                # Close back inside
                close_range = current_high - current_low
                if current_close < (current_high - close_range * close_back_pct):
                    sweeps.append({
                        'type': 'sweep_high',
                        'bar_index': current_idx,
                        'pivot_bar': last_ph,
                        'pivot_price': ph_price,
                        'sweep_price': current_high
                    })
        
        return sweeps
