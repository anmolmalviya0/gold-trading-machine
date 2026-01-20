"""
REGIME SWITCHER - Adaptive Engine Selection
============================================
Implements 3 specialized trading engines + a master switcher:

1. TREND ENGINE - Optimized for high-volatility breakouts
2. MEAN-REVERSION ENGINE - Optimized for sideways "chop"
3. VOLATILE ENGINE - Defensive mode for extreme conditions

The Regime Switcher identifies the current market state and
selects the optimal engine automatically.

Usage:
    from regime_switcher import RegimeSwitcher
    
    switcher = RegimeSwitcher()
    engine, signal = switcher.get_signal(df, symbol)
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Identifies the current market regime based on volatility and trend metrics.
    
    Regimes:
    - TREND: Strong directional movement (ADX > 25)
    - RANGE: Sideways consolidation (ADX < 20, low volatility)
    - VOLATILE: Extreme volatility (ATR > 2x normal)
    - DEAD: No movement (volume collapse)
    """
    
    def __init__(self):
        self.lookback = 50
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate regime identification metrics"""
        if len(df) < self.lookback:
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        recent = df.tail(self.lookback)
        
        # ADX calculation
        high = recent['h']
        low = recent['l']
        close = recent['c']
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Directional Movement
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean().iloc[-1]
        
        # Volatility metrics
        returns = close.pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        avg_volatility = returns.rolling(50).std().mean() * np.sqrt(252)
        vol_ratio = volatility / (avg_volatility + 1e-10)
        
        # Volume analysis
        if 'v' in df.columns:
            volume = recent['v']
            avg_volume = volume.rolling(20).mean().iloc[-1]
            recent_volume = volume.tail(5).mean()
            vol_collapse = recent_volume < avg_volume * 0.3
        else:
            vol_collapse = False
        
        return {
            'adx': adx,
            'volatility': volatility,
            'vol_ratio': vol_ratio,
            'atr_pct': atr / close.iloc[-1] * 100,
            'vol_collapse': vol_collapse
        }
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect the current market regime.
        
        Returns:
        --------
        Tuple of (regime_name, confidence)
        """
        metrics = self.calculate_metrics(df)
        
        if 'regime' in metrics:
            return metrics['regime'], 0
        
        adx = metrics['adx']
        vol_ratio = metrics['vol_ratio']
        vol_collapse = metrics['vol_collapse']
        
        # Decision tree
        if vol_collapse:
            return 'DEAD', 0.8
        
        if vol_ratio > 2.0:
            return 'VOLATILE', 0.9
        
        if adx > 25:
            return 'TREND', min(0.5 + (adx - 25) / 50, 0.95)
        
        if adx < 20 and vol_ratio < 1.2:
            return 'RANGE', 0.7
        
        # Default to mixed/unclear
        return 'RANGE', 0.5


# =============================================================================
# TREND ENGINE
# =============================================================================

class TrendEngine:
    """
    Optimized for high-volatility breakouts.
    
    Features:
    - Momentum-based entries
    - Wide stops (2.5x ATR)
    - Pyramid into winners
    """
    
    NAME = "TREND"
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_cols = [
            'ret_5', 'ret_10', 'ret_20',
            'rsi', 'macd_hist',
            'adx', 'plus_di', 'minus_di',
            'atr_pct', 'vol_ratio'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-specific features"""
        df = df.copy()
        
        # Returns
        for period in [5, 10, 20]:
            df[f'ret_{period}'] = df['c'].pct_change(period) * 100
        
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['c'].ewm(span=12).mean()
        ema26 = df['c'].ewm(span=26).mean()
        df['macd_hist'] = ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()
        
        # ATR
        tr = pd.concat([
            df['h'] - df['l'],
            (df['h'] - df['c'].shift()).abs(),
            (df['l'] - df['c'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['c'] * 100
        
        # ADX
        high = df['h']
        low = df['l']
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        
        dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        # Volatility ratio
        vol = df['c'].pct_change().rolling(20).std()
        df['vol_ratio'] = vol / vol.rolling(50).mean()
        
        return df
    
    def get_signal(self, df: pd.DataFrame) -> dict:
        """Generate trend-following signal"""
        df = self.prepare_features(df)
        row = df.iloc[-1]
        
        score = 0
        
        # Strong momentum
        if row['ret_10'] > 2:
            score += 2
        elif row['ret_10'] > 1:
            score += 1
        elif row['ret_10'] < -2:
            score -= 2
        elif row['ret_10'] < -1:
            score -= 1
        
        # RSI extreme (follow momentum)
        if row['rsi'] > 60:
            score += 1
        elif row['rsi'] < 40:
            score -= 1
        
        # ADX trend strength
        if row['adx'] > 25:
            # Amplify signal in strong trend
            score = int(score * 1.5)
        
        # DI crossover
        if row['plus_di'] > row['minus_di']:
            score += 1
        else:
            score -= 1
        
        # Signal
        if score >= 3:
            signal, conf = 'BUY', 70 + min(score * 5, 25)
        elif score <= -3:
            signal, conf = 'SELL', 70 + min(abs(score) * 5, 25)
        elif score > 0:
            signal, conf = 'BUY', 55 + score * 5
        elif score < 0:
            signal, conf = 'SELL', 55 + abs(score) * 5
        else:
            signal, conf = 'NEUTRAL', 50
        
        return {
            'signal': signal,
            'confidence': conf,
            'engine': self.NAME,
            'score': score,
            'stop_mult': 2.5,  # Wider stops for trends
            'tp_mult': 4.0     # Larger targets
        }


# =============================================================================
# MEAN REVERSION ENGINE
# =============================================================================

class MeanReversionEngine:
    """
    Optimized for sideways "chop" / range-bound markets.
    
    Features:
    - Fade extremes (RSI, Bollinger Bands)
    - Tight stops (1.0x ATR)
    - Quick profit taking
    """
    
    NAME = "MEAN_REV"
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean-reversion features"""
        df = df.copy()
        
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # Bollinger Bands
        df['bb_mid'] = df['c'].rolling(20).mean()
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Distance from SMA
        df['dist_sma20'] = (df['c'] - df['bb_mid']) / df['c'] * 100
        
        # ATR
        tr = pd.concat([
            df['h'] - df['l'],
            (df['h'] - df['c'].shift()).abs(),
            (df['l'] - df['c'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def get_signal(self, df: pd.DataFrame) -> dict:
        """Generate mean-reversion signal"""
        df = self.prepare_features(df)
        row = df.iloc[-1]
        
        score = 0
        
        # RSI extreme (fade it)
        if row['rsi'] > 70:
            score -= 2  # Overbought → SELL
        elif row['rsi'] > 60:
            score -= 1
        elif row['rsi'] < 30:
            score += 2  # Oversold → BUY
        elif row['rsi'] < 40:
            score += 1
        
        # Bollinger Band position
        if row['bb_pct'] > 0.95:
            score -= 2  # At upper band → SELL
        elif row['bb_pct'] > 0.8:
            score -= 1
        elif row['bb_pct'] < 0.05:
            score += 2  # At lower band → BUY
        elif row['bb_pct'] < 0.2:
            score += 1
        
        # Distance from mean
        if row['dist_sma20'] > 3:
            score -= 1  # Too far above → expect pullback
        elif row['dist_sma20'] < -3:
            score += 1  # Too far below → expect bounce
        
        # Signal
        if score >= 3:
            signal, conf = 'BUY', 65 + min(score * 5, 20)
        elif score <= -3:
            signal, conf = 'SELL', 65 + min(abs(score) * 5, 20)
        elif score > 0:
            signal, conf = 'BUY', 52 + score * 4
        elif score < 0:
            signal, conf = 'SELL', 52 + abs(score) * 4
        else:
            signal, conf = 'NEUTRAL', 50
        
        return {
            'signal': signal,
            'confidence': conf,
            'engine': self.NAME,
            'score': score,
            'stop_mult': 1.0,  # Tighter stops for ranges
            'tp_mult': 1.5     # Quick profit taking
        }


# =============================================================================
# VOLATILE ENGINE
# =============================================================================

class VolatileEngine:
    """
    Defensive mode for extreme volatility conditions.
    
    Strategy:
    - Reduce position sizes
    - Wide stops (3x ATR)
    - Only take high-conviction setups
    """
    
    NAME = "VOLATILE"
    
    def get_signal(self, df: pd.DataFrame) -> dict:
        """Generate defensive signal"""
        # In volatile markets, stay mostly flat
        # Only signal on extreme conditions
        
        row = df.iloc[-1]
        
        # Calculate short-term momentum
        ret_5 = df['c'].pct_change(5).iloc[-1] * 100
        
        if ret_5 > 5:
            # Massive surge - might continue
            return {
                'signal': 'BUY',
                'confidence': 55,
                'engine': self.NAME,
                'score': 1,
                'stop_mult': 3.0,
                'tp_mult': 2.0,
                'size_mult': 0.5  # Half position
            }
        elif ret_5 < -5:
            return {
                'signal': 'SELL',
                'confidence': 55,
                'engine': self.NAME,
                'score': -1,
                'stop_mult': 3.0,
                'tp_mult': 2.0,
                'size_mult': 0.5
            }
        else:
            # Stay flat in volatile chop
            return {
                'signal': 'NO_TRADE',
                'confidence': 75,
                'engine': self.NAME,
                'score': 0,
                'stop_mult': 3.0,
                'tp_mult': 2.0,
                'size_mult': 0.0
            }


# =============================================================================
# REGIME SWITCHER (Master Controller)
# =============================================================================

class RegimeSwitcher:
    """
    Master controller that selects the optimal engine based on market regime.
    
    Flow:
    1. Detect current regime (TREND/RANGE/VOLATILE/DEAD)
    2. Select appropriate engine
    3. Generate signal using selected engine
    4. Apply meta-labeling filter if confidence < threshold
    """
    
    def __init__(self, meta_threshold: float = 0.65):
        self.detector = RegimeDetector()
        self.engines = {
            'TREND': TrendEngine(),
            'RANGE': MeanReversionEngine(),
            'VOLATILE': VolatileEngine(),
            'DEAD': VolatileEngine(),  # Use defensive in dead markets
            'UNKNOWN': MeanReversionEngine()  # Default
        }
        self.meta_threshold = meta_threshold
        self.current_regime = 'UNKNOWN'
        self.current_engine = None
    
    def get_signal(self, df: pd.DataFrame, symbol: str = 'BTCUSDT') -> dict:
        """
        Generate signal using regime-adaptive engine selection.
        
        Parameters:
        -----------
        df : DataFrame with OHLCV data
        symbol : Trading symbol
        
        Returns:
        --------
        dict with signal, confidence, regime, engine, and parameters
        """
        # Step 1: Detect regime
        regime, regime_conf = self.detector.detect_regime(df)
        self.current_regime = regime
        
        # Step 2: Select engine
        engine = self.engines[regime]
        self.current_engine = engine.NAME
        
        # Step 3: Generate signal
        signal_data = engine.get_signal(df)
        
        # Step 4: Apply meta-labeling filter
        if signal_data['confidence'] / 100 < self.meta_threshold:
            signal_data['signal'] = 'NO_TRADE'
            signal_data['meta_filtered'] = True
        else:
            signal_data['meta_filtered'] = False
        
        # Add regime info
        signal_data['regime'] = regime
        signal_data['regime_confidence'] = regime_conf
        signal_data['symbol'] = symbol
        
        return signal_data
    
    def get_status(self) -> dict:
        """Get current switcher status"""
        return {
            'regime': self.current_regime,
            'engine': self.current_engine,
            'meta_threshold': self.meta_threshold
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🔄 REGIME SWITCHER - Test")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    price = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df = pd.DataFrame({
        'time': dates,
        'o': price,
        'h': price * 1.01,
        'l': price * 0.99,
        'c': price,
        'v': np.random.randint(1000, 10000, 200)
    })
    
    # Test regime detection
    detector = RegimeDetector()
    regime, conf = detector.detect_regime(df)
    print(f"\nDetected Regime: {regime} ({conf:.0%})")
    
    # Test switcher
    switcher = RegimeSwitcher()
    signal = switcher.get_signal(df, 'BTCUSDT')
    
    print(f"\nSignal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']}%")
    print(f"Engine Used: {signal['engine']}")
    print(f"Meta-Filtered: {signal['meta_filtered']}")
    print(f"Stop Mult: {signal['stop_mult']}x ATR")
    print(f"TP Mult: {signal['tp_mult']}x ATR")
