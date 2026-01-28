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

def calculate_kelly_fraction(probability: float, win_loss_ratio: float, fraction_limit: float = 0.5) -> float:
    """
    Calculate the Kelly Criterion fraction for position sizing.
    f* = p/a - q/b = p - (1-p)/b
    p: probability of success (model confidence)
    b: win/loss ratio (avg win / avg loss)
    """
    if win_loss_ratio <= 0 or probability <= 0:
        return 0.0
    
    # Kelly Formula: f = p - (1-p)/b
    q = 1.0 - probability
    kelly_f = probability - (q / win_loss_ratio)
    
    # Apply a fraction limit (Half-Kelly or Quarter-Kelly) for safety
    return max(0.0, min(kelly_f * fraction_limit, 1.0))

def adaptive_threshold_logic(base_threshold: float, vol_z_score: float) -> float:
    """
    Adjust confidence threshold based on market volatility (Z-Score).
    Tightens during choppy markets, loosens during breakouts.
    """
    # If volatility is extreme (Z > 2), we want higher conviction
    if vol_z_score > 2.0:
        return min(0.95, base_threshold + 0.05)
    # If volatility is low (Z < -1), we want higher conviction (avoid chop)
    elif vol_z_score < -1.0:
        return min(0.95, base_threshold + 0.10)
    
    return base_threshold

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

        # --- HYPER-OPTIMIZATION: CONTEXT AWARENESS (Matching matrix_trainer.py) ---
        # 1. ADX (Trend Strength)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        deltas_up = df['high'].diff()
        deltas_down = -df['low'].diff()
        
        alpha = 1/14
        
        # Use numpy where for safe vectorization
        plus_dm = np.where((deltas_up > deltas_down) & (deltas_up > 0), deltas_up, 0)
        minus_dm = np.where((deltas_down > deltas_up) & (deltas_down > 0), deltas_down, 0)
        
        # Smooth with EMA
        tr_smooth = pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean()
        pdm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        mdm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        
        plus_di = 100 * (pdm_smooth / tr_smooth)
        minus_di = 100 * (mdm_smooth / tr_smooth)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # 2. Volatility Regime (Z-Score)
        rolling_vol = df['close'].pct_change().rolling(20).std()
        df['vol_z_score'] = (rolling_vol - rolling_vol.rolling(50).mean()) / rolling_vol.rolling(50).std()
        
        cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio',
            'adx', 'vol_z_score'
        ]
        
        # Take the last row and handle NaNs from rolling starts
        # If last row is NaN, we can't trade. But we must return valid shape.
        final_row = df[cols].iloc[-1].fillna(0)
        features = final_row.values.reshape(1, -1)
        return features
    except Exception as e:
        print(f"❌ Feature Engineering Failed: {e}")
        return np.zeros((1, 15))
