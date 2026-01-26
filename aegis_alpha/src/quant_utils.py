import pandas as pd
import numpy as np

def calc_position_size(balance: float, risk_pct: float, entry: float, stop_loss: float) -> float:
    """
    Calculate position size based on risk percentage and stop loss distance.
    """
    if entry <= 0 or stop_loss <= 0:
        return 0.0
        
    risk_amount = balance * risk_pct
    stop_distance = abs(entry - stop_loss)
    
    if stop_distance == 0:
        return 0.0
        
    position_size = risk_amount / stop_distance
    return position_size

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    THE MASTER ENGINE: Standard TERMINAL Feature Engineering.
    Converts raw OHLCV into a 13-dimension vector for Neural Inference.
    """
    try:
        df = df.copy()
        
        # 0. AGGRESSIVE NORMALIZATION (Handles YFinance V3 MultiIndex)
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)
        
        # Convert all to lower and strip to avoid any whitespace spirits
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Ensure we have the critical 5
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                # If 'adj close' exists but 'close' doesn't, alias it
                if col == 'close' and 'adj close' in df.columns:
                    df['close'] = df['adj close']
                else:
                    raise KeyError(f"Critical Column Missing: {col}")
        
        # 1. BASE FEATURES
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        for p in [5, 10, 20, 50]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'sma_ratio_{p}'] = df['close'] / df[f'sma_{p}']
            
        # RSI (Wilder's Smoothing simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100
        
        # MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - (sma20 - 2*std20)) / (4*std20)
        
        # ATR / Volume
        tr = pd.DataFrame({'hl': df['high'] - df['low'], 'hc': abs(df['high'] - df['close'].shift(1)), 'lc': abs(df['low'] - df['close'].shift(1))}).max(axis=1)
        df['atr_ratio'] = tr.rolling(14).mean() / df['close']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
        ]
        
        # Take the last row and handle NaNs from rolling starts
        features = df[cols].fillna(0).iloc[-1].values.reshape(1, -1)
        return features
    except Exception as e:
        print(f"❌ Feature Engineering Failed: {e}")
        return np.zeros((1, 13))
