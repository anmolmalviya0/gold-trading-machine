"""
Feature Store - Unified feature pipeline
Combines all feature engineering components
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .technical import TechnicalIndicators
from .regime import RegimeDetector

logger = logging.getLogger(__name__)


class FeatureStore:
    """Unified feature engineering pipeline"""
    
    def __init__(self):
        self.feature_names = None
        self.scaler = None  # For normalization if needed
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with all features
        """
        if len(df) < 200:
            logger.warning(f"Insufficient data for features: {len(df)} bars")
            return df
        
        # Calculate technical indicators
        df = TechnicalIndicators.calculate_all(df)
        logger.debug(f"Calculated {len(TechnicalIndicators.get_feature_names())} technical features")
        
        # Detect market regime
        df = RegimeDetector.detect_regime(df)
        logger.debug("Regime classification complete")
        
        # Add  derived features
        df = self._add_derived_features(df)
        
        # Store feature names
        self.feature_names = self._get_model_features()
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/interaction features
        """
        # Momentum x Volatility interaction
        if 'rsi_14' in df.columns and 'atr_pct' in df.columns:
            df['rsi_atr_interaction'] = df['rsi_14'] * df['atr_pct']
        
        # Trend strength x Volume
        if 'adx' in df.columns and 'volume_ratio' in df.columns:
            df['adx_volume_interaction'] = df['adx'] * df['volume_ratio']
        
        # BB position x Regime
        if 'bb_position' in df.columns and 'regime_numeric' in df.columns:
            df['bb_regime_interaction'] = df['bb_position'] * df['regime_numeric']
        
        # Price momentum (multiple windows)
        df['price_mom_5'] = df['close'].pct_change(5)
        df['price_mom_10'] = df['close'].pct_change(10)
        df['price_mom_20'] = df['close'].pct_change(20)
        
        # Volume momentum
        if 'volume' in df.columns:
            df['volume_mom_5'] = df['volume'].pct_change(5)
        
        # ATR-normalized price changes
        if 'atr_14' in df.columns:
            df['price_change_atr'] = (df['close'] - df['close'].shift(1)) / df['atr_14']
        
        return df
    
    def _get_model_features(self) -> List[str]:
        """
        Get list of features for ML models
        Exclude timestamp, OHLC, and intermediate calculations
        """
        exclude_patterns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'bb_upper', 'bb_middle', 'bb_lower',  # Use bb_position instead
            'keltner_upper', 'keltner_lower',
            'sma_', 'ema_',  # Use price_vs_sma instead
            'close_20h', 'close_20l'  # Intermediate calculations
        ]
        
        # Start with technical feature names
        features = TechnicalIndicators.get_feature_names()
        
        # Add regime features
        features.extend(['regime_numeric'])
        
        # Add derived features
        features.extend([
            'rsi_atr_interaction',
            'adx_volume_interaction', 
            'bb_regime_interaction',
            'price_mom_5', 'price_mom_10', 'price_mom_20',
            'volume_mom_5',
            'price_change_atr'
        ])
        
        # Filter out excluded patterns
        filtered = []
        for feat in features:
            exclude = False
            for pattern in exclude_patterns:
                if pattern in feat:
                    exclude = True
                    break
            if not exclude:
                filtered.append(feat)
        
        return filtered
    
    def get_feature_vector(self, df: pd.DataFrame, index: int = -1) -> np.ndarray:
        """
        Get feature vector for a specific row (for live inference)
        
        Args:
            df: DataFrame with features
            index: Row index (-1 for latest)
            
        Returns:
            Feature vector as numpy array
        """
        if self.feature_names is None:
            raise ValueError("Features not initialized. Call engineer_features first.")
        
        # Get row
        row = df.iloc[index]
        
        # Extract features
        features = []
        for feat_name in self.feature_names:
            if feat_name in row:
                value = row[feat_name]
                # Handle NaN
                if pd.isna(value):
                    value = 0.0
                features.append(value)
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_importance_map(self, importances: np.ndarray) -> Dict[str, float]:
        """Map feature importances to names"""
        if self.feature_names is None or len(importances) != len(self.feature_names):
            return {}
        
        return dict(zip(self.feature_names, importances))


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='5min')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(300) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(300) * 0.5) + np.random.rand(300) * 2,
        'low': 100 + np.cumsum(np.random.randn(300) * 0.5) - np.random.rand(300) * 2,
        'close': 100 + np.cumsum(np.random.randn(300) * 0.5),
        'volume': np.random.randint(1000, 10000, 300)
    })
    
    # Engineer features
    store = FeatureStore()
    df_featured = store.engineer_features(df)
    
    print(f"✅ Feature engineering complete")
    print(f"   Total features: {len(store.feature_names)}")
    print(f"   Original shape: {df.shape}")
    print(f"   Featured shape: {df_featured.shape}")
    
    # Get feature vector for latest bar
    feature_vec = store.get_feature_vector(df_featured)
    print(f"\n📊 Latest feature vector shape: {feature_vec.shape}")
    print(f"   Sample features: {feature_vec[:5]}")
