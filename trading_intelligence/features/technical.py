"""
Technical Indicators - Comprehensive Feature Set
50+ indicators for ML model inputs
Using pandas_ta for cross-platform compatibility
"""
import pandas as pd
import numpy as np
from typing import Dict
import pandas_ta as ta


class TechnicalIndicators:
    """Calculate technical indicators for feature engineering"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Ensure we have enough data
        if len(df) < 200:
            return df
        
        # Momentum Indicators
        df = TechnicalIndicators._momentum_features(df)
        
        # Trend Indicators
        df = TechnicalIndicators._trend_features(df)
        
        # Volatility Indicators
        df = TechnicalIndicators._volatility_features(df)
        
        # Volume Indicators
        df = TechnicalIndicators._volume_features(df)
        
        # Price Action
        df = TechnicalIndicators._price_action_features(df)
        
        return df
    
    @staticmethod
    def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators"""
        
        # RSI (multiple periods)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_21'] = ta.rsi(df['close'], length=21)
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # ROC (Rate of Change)
        df['roc_10'] = ta.roc(df['close'], length=10)
        df['roc_20'] = ta.roc(df['close'], length=20)
        
        # Williams %R
        df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        return df
    
    @staticmethod
    def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Trend indicators"""
        close = df['close']
        
        # Moving Averages
        df['sma_20'] = ta.sma(close, length=20)
        df['sma_50'] = ta.sma(close, length=50)
        df['sma_100'] = ta.sma(close, length=100)
        df['sma_200'] = ta.sma(close, length=200)
        
        df['ema_9'] = ta.ema(close, length=9)
        df['ema_20'] = ta.ema(close, length=20)
        df['ema_50'] = ta.ema(close, length=50)
        
        # ADX (Trend Strength)
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_result is not None:
            df['adx'] = adx_result['ADX_14']
            df['adx_plus'] = adx_result['DMP_14']
            df['adx_minus'] = adx_result['DMN_14']
        
        # Parabolic SAR
        sar = ta.psar(df['high'], df['low'], df['close'])
        if sar is not None:
            df['sar'] = sar['PSARl_0.02_0.2'].fillna(sar['PSARs_0.02_0.2'])
        
        # Price vs MA distance
        df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (close - df['sma_50']) / df['sma_50']
        
        return df
    
    @staticmethod
    def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators"""
        close = df['close']
        
        # ATR (Average True Range)
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_20'] = ta.atr(df['high'], df['low'], df['close'], length=20)
        
        # Bollinger Bands
        bbands = ta.bbands(close, length=20, std=2)
        if bbands is not None and not bbands.empty:
            # Column names vary by pandas_ta version
            bb_cols = bbands.columns.tolist()
            if len(bb_cols) >= 3:
                df['bb_upper'] = bbands.iloc[:, 0]  # First column
                df['bb_middle'] = bbands.iloc[:, 1]  # Middle
                df['bb_lower'] = bbands.iloc[:, 2]  # Lower
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
        if kc is not None and not kc.empty:
            kc_cols = kc.columns.tolist()
            if len(kc_cols) >= 2:
                df['keltner_upper'] = kc.iloc[:, 0]
                df['keltner_lower'] = kc.iloc[:, 1]
                if 'ema_20' in df.columns:
                    df['keltner_width'] = (df['keltner_upper'] - df['keltner_lower']) / df['ema_20']
        
        # Historical Volatility
        returns = np.log(close / close.shift(1))
        df['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        
        # ATR as % of price (normalized volatility)
        df['atr_pct'] = df['atr_14'] / close
        
        return df
    
    @staticmethod
    def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volume indicators"""
        close = df['close']
        volume = df['volume']
        
        # OBV (On Balance Volume)
        df['obv'] = ta.obv(close, volume)
        if df['obv'] is not None:
            df['obv_ema'] = ta.ema(df['obv'], length=20)
        
        # Volume SMA
        df['volume_sma_20'] = ta.sma(volume, length=20)
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # MFI (Money Flow Index)
        df['mfi'] = ta.mfi(df['high'], df['low'], close, volume, length=14)
        
        # VWAP
        df['vwap'] = ta.vwap(df['high'], df['low'], close, volume)
        
        # AD (Accumulation/Distribution)
        df['ad'] = ta.ad(df['high'], df['low'], close, volume)
        
        return df
    
    @staticmethod
    def _price_action_features(df: pd.DataFrame) -> pd.DataFrame:
        """Price action patterns"""
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Candle body and wicks
        df['body_size'] = np.abs(close - open_price)
        df['upper_wick'] = high - np.maximum(open_price, close)
        df['lower_wick'] = np.minimum(open_price, close) - low
        df['body_to_range'] = df['body_size'] / (high - low + 1e-10)
        
        # Gap detection
        df['gap'] = open_price - np.roll(close, 1)
        df['gap_pct'] = df['gap'] / np.roll(close, 1)
        
        # Higher highs / Lower lows
        df['hh'] = (high > np.roll(high, 1)).astype(int)
        df['ll'] = (low < np.roll(low, 1)).astype(int)
        
        # Range expansion/contraction
        current_range = high - low
        prev_range = np.roll(high, 1) - np.roll(low, 1)
        df['range_expansion'] = current_range / (prev_range + 1e-10)
        
        return df
    
    @staticmethod
    def get_feature_names() -> list:
        """Get list of all feature names"""
        return [
            # Momentum
            'rsi_14', 'rsi_21', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'roc_10', 'roc_20', 'willr', 'cci',
            # Trend
            'sma_20', 'sma_50', 'sma_100', 'sma_200', 'ema_9', 'ema_20', 'ema_50',
            'adx', 'adx_plus', 'adx_minus', 'sar', 'price_vs_sma20', 'price_vs_sma50',
            # Volatility
            'atr_14', 'atr_20', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'keltner_upper', 'keltner_lower', 'keltner_width', 'hist_vol_20', 'atr_pct',
            # Volume
            'obv', 'obv_ema', 'volume_sma_20', 'volume_ratio', 'mfi', 'vwap', 'ad',
            # Price Action
            'body_size', 'upper_wick', 'lower_wick', 'body_to_range', 'gap', 'gap_pct',
            'hh', 'll', 'range_expansion'
        ]


# Demo
if __name__ == "__main__":
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
    
    # Calculate indicators
    df_with_features = TechnicalIndicators.calculate_all(df)
    
    print(f"✅ Calculated {len(TechnicalIndicators.get_feature_names())} features")
    print(f"\nSample features:")
    print(df_with_features[['close', 'rsi_14', 'macd', 'adx', 'bb_position']].tail())
