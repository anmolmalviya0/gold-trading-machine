"""
AEGIS V21 - LightGBM Trainer (Switchblade Protocol)
====================================================
Fast gradient boosting classifier as alternative to LSTM.
Reuses existing data infrastructure and Triple-Barrier labeling.
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_and_prepare_data():
    """Load market data and apply Triple-Barrier labeling"""
    print("📂 Loading market data...")
    
    data_dir = '/Users/anmol/Desktop/gold/market_data'
    dfs = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.lower()
                column_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
                df = df.rename(columns=column_map)
                dfs.append(df)
                print(f"   ✓ Loaded {filename}: {len(df)} rows")
            except Exception as e:
                print(f"   ⚠️ Skipped {filename}: {e}")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total samples: {len(df)}")
    
    return df


def engineer_features(df):
    """Create trading features"""
    print("⚙️ Engineering features...")
    
    df = df.copy()
    
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
    
    print(f"   ✅ Features created")
    return df


def create_triple_barrier_labels(df, tp_atr=2.0, sl_atr=1.5, max_bars=20):
    """Apply Triple-Barrier Method (De Prado)"""
    print("🏷️ Applying Triple-Barrier Labeling...")
    
    labels = []
    
    for i in range(len(df) - max_bars):
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(atr) or atr == 0:
            labels.append(0)
            continue
        
        # Define barriers
        take_profit = entry_price + tp_atr * atr
        stop_loss = entry_price - sl_atr * atr
        
        # Look ahead
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
    
    # Pad remaining rows
    labels.extend([0] * max_bars)
    
    df['target'] = labels
    
    positive_pct = (df['target'] == 1).sum() / len(df) * 100
    print(f"   ✅ Labels: {positive_pct:.1f}% positive")
    
    return df


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          AEGIS V21 - LIGHTGBM TRAINER                    ║
    ║              The Switchblade Protocol                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 1. Load data
    df = load_and_prepare_data()
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Create labels
    df = create_triple_barrier_labels(df, tp_atr=2.0, sl_atr=1.5, max_bars=20)
    
    # 4. Drop NaN rows
    df = df.dropna()
    print(f"📊 Clean samples: {len(df)}")
    
    # 5. Prepare features and target
    feature_cols = [
        'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
        'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
        'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"🔹 Features: {len(feature_cols)} | Target Balance: {y.mean():.1%}")
    
    # 6. Split (time-based, no shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📦 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 7. Train LightGBM
    print("\n⚡ Training LightGBM (The Speed Run)...")
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
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # 8. Evaluate
    print("\n📊 EVALUATION:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=['HOLD', 'BUY']))
    
    # 9. Check Win Rate at High Confidence
    probs = model.predict_proba(X_test)[:, 1]
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        high_conf_idx = probs > threshold
        count = high_conf_idx.sum()
        
        if count > 0:
            real_wins = y_test[high_conf_idx]
            win_rate = real_wins.mean() * 100
            status = "✅" if win_rate > 60 else "⚠️" if win_rate > 50 else "❌"
            print(f"{status} Conf>{threshold}: {count:4d} trades, {win_rate:5.1f}% win rate")
        else:
            print(f"⚪ Conf>{threshold}: No trades")
    
    # 10. Save model
    model_dir = '/Users/anmol/Desktop/gold/aegis_alpha/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'aegis_lgbm.pkl')
    
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # 11. Feature importance
    print("\n📈 TOP 10 FEATURES:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importances.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:6.0f}")
    
    print("\n✅ LightGBM Training Complete!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
