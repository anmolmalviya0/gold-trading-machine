"""
TERMINAL - Multi-Asset Trainer (The Fleet)
===========================================
Trains LightGBM models for multiple assets: SOL, BNB, ETH.
Reuses the proven Switchblade logic from lgbm_trainer.py.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
# from feature_engine import engineer_features, create_triple_barrier_labels # Removed invalid import

# We'll inline the logic since feature_engine.py doesn't exist yet
# This keeps it self-contained for now to avoid import errors

def engineer_logic(df):
    """Clone of lgbm_trainer.py feature engineering"""
    df = df.copy()
    
    # Drops NaNs upfront
    df = df.dropna().reset_index(drop=True)
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = df['rsi'] / 100
    
    # MACD
    exp12 = df['close'].ewm(span=12).mean()
    exp26 = df['close'].ewm(span=26).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

def label_logic(df, tp_atr=2.0, sl_atr=1.5, max_bars=20):
    """Clone of Triple-Barrier Labeling"""
    labels = []
    
    for i in range(len(df) - max_bars):
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(atr) or atr == 0:
            labels.append(0)
            continue
        
        take_profit = entry_price + tp_atr * atr
        stop_loss = entry_price - sl_atr * atr
        
        future_prices = df['close'].iloc[i+1:i+1+max_bars]
        
        hit_tp = (future_prices >= take_profit).any()
        hit_sl = (future_prices <= stop_loss).any()
        
        if hit_tp and hit_sl:
            tp_idx = (future_prices >= take_profit).idxmax()
            sl_idx = (future_prices <= stop_loss).idxmax()
            label = 1 if tp_idx < sl_idx else 0
        elif hit_tp:
            label = 1
        else:
            label = 0
            
        labels.append(label)
        
    labels.extend([0] * max_bars)
    df['target'] = labels
    return df

def train_asset(symbol, csv_path):
    print(f"\n📉 Processing {symbol}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return False
        
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} candles")
        
        # 1. Feature Engineering
        df = engineer_logic(df)
        
        # 2. Labeling
        df = label_logic(df)
        
        # 3. Clean
        df = df.dropna()
        feature_cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
        ]
        
        X = df[feature_cols]
        y = df['target']
        
        positive_pct = y.mean() * 100
        print(f"   Positives: {positive_pct:.1f}%")
        
        if len(X) < 1000:
            print("   ⚠️ Not enough data to train")
            return False
            
        # 4. Train LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            objective='binary',
            metric='auc',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        split = int(len(X)*0.8)
        model.fit(
            X.iloc[:split], y.iloc[:split],
            eval_set=[(X.iloc[split:], y.iloc[split:])],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 5. Save
        model_path = f'models/{symbol}_lgbm.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Model Saved: {model_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error training {symbol}: {e}")
        return False

def main():
    print("⚔️ FORGING NEW BLADES (Multi-Asset Training)")
    
    # We look for the files fetched by fetch_new_assets.py
    # Filenames: SOLUSDT_5m.csv, BNBUSDT_5m.csv, ETHUSDT_5m.csv
    # Note: fetch_new_assets.py saves as SOLUSDT_5m.csv (removed hyphens)
    
    targets = [
        ('SOL', 'market_data/SOLUSDT_5m.csv'),
        ('BNB', 'market_data/BNBUSDT_5m.csv'),
        ('ETH', 'market_data/ETHUSDT_5m.csv')
    ]
    
    for symbol, path in targets:
        train_asset(symbol, path)
        
    print("\n🏁 FLEET READY.")

if __name__ == "__main__":
    main()
