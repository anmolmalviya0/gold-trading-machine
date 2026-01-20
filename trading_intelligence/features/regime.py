"""
Regime Detection - Market state classification
Identifies TREND / CHOP / MEANREV states
"""
import pandas as pd
import numpy as np
from typing import Tuple


class RegimeDetector:
    """Classify market regime"""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime classification to dataframe
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with regime columns
        """
        df = df.copy()
        
        # Calculate regime features
        df = RegimeDetector._add_regime_features(df)
        
        # Classify regime
        df['regime'] = df.apply(RegimeDetector._classify_row, axis=1)
        df['regime_numeric'] = df['regime'].map({'TREND': 1, 'MEANREV': 0, 'CHOP': -1})
        
        return df
    
    @staticmethod
    def _add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection"""
        
        # ADX for trend strength (should already exist from technical indicators)
        if 'adx' not in df.columns:
            df['adx'] = 25  # Default neutral
        
        # Bollinger Band width (squeeze detection)
        if 'bb_width' not in df.columns:
            close = df['close'].values
            sma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            df['bb_width'] = (std20 * 4) / sma20
        
        # Historical volatility trend
        if 'hist_vol_20' in df.columns:
            df['vol_trend'] = df['hist_vol_20'].pct_change(10)
        else:
            df['vol_trend'] = 0
        
        # Price trend consistency
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            df['ema_alignment'] = (df['ema_20'] > df['ema_50']).astype(int)
            df['price_above_ema20'] = (df['close'] > df['ema_20']).astype(int)
            df['price_above_ema50'] = (df['close'] > df['ema_50']).astype(int)
        
        # Range behavior
        df['close_20h'] = df['close'].rolling(20).max()
        df['close_20l'] = df['close'].rolling(20).min()
        df['range_position'] = (df['close'] - df['close_20l']) / (df['close_20h'] - df['close_20l'] + 1e-10)
        
        return df
    
    @staticmethod
    def _classify_row(row) -> str:
        """
        Classify single row
        
        TREND: Strong directional movement
        CHOP: Low volatility, random walk
        MEANREV: High volatility, mean-reverting
        """
        adx = row.get('adx', 25)
        bb_width = row.get('bb_width', 0.1)
        vol_trend = row.get('vol_trend', 0)
        
        # CHOP: Low ADX + Low BB Width (squeeze)
        if adx < 20 and bb_width < 0.08:
            return 'CHOP'
        
        # TREND: High ADX + Expanding volatility
        elif adx > 30:
            return 'TREND'
        
        # MEANREV: Moderate ADX + High volatility + contracting
        elif adx >= 20 and bb_width > 0.12:
            return 'MEANREV'
        
        # Default to CHOP for uncertain states
        else:
            return 'CHOP'
    
    @staticmethod
    def get_regime_stats(df: pd.DataFrame) -> dict:
        """Get distribution of regimes"""
        if 'regime' not in df.columns:
            return {}
        
        total = len(df)
        counts = df['regime'].value_counts()
        
        return {
            'trend_pct': counts.get('TREND', 0) / total,
            'meanrev_pct': counts.get('MEANREV', 0) / total,
            'chop_pct': counts.get('CHOP', 0) / total,
            'current_regime': df['regime'].iloc[-1]
        }


# Demo
if __name__ == "__main__":
    # Generate sample data with different regimes
    np.random.seed(42)
    
    # Trending period
    trend_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 2 + 0.5),  # Uptrend
        'adx': np.random.uniform(35, 45, 100),
        'bb_width': np.random.uniform(0.10, 0.15, 100)
    })
    
    # Chopping period
    chop_data = pd.DataFrame({
        'close': 100 + np.random.randn(100) * 0.5,  # Range-bound
        'adx': np.random.uniform(10, 18, 100),
        'bb_width': np.random.uniform(0.04, 0.07, 100)
    })
    
    # Mean-reversion period
    meanrev_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 3),  # High vol
        'adx': np.random.uniform(22, 28, 100),
        'bb_width': np.random.uniform(0.13, 0.20, 100)
    })
    
    # Combine
    df = pd.concat([trend_data, chop_data, meanrev_data], ignore_index=True)
    
    # Detect regimes
    df = RegimeDetector.detect_regime(df)
    
    # Stats
    stats = RegimeDetector.get_regime_stats(df)
    
    print("📊 Regime Distribution:")
    print(f"   TREND: {stats['trend_pct']:.1%}")
    print(f"   MEANREV: {stats['meanrev_pct']:.1%}")
    print(f"   CHOP: {stats['chop_pct']:.1%}")
    print(f"\n Current Regime: {stats['current_regime']}")
