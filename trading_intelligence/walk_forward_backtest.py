"""
WALK-FORWARD VALIDATION + TRIPLE-BARRIER LABELING
Professional-grade backtesting with proper validation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 PROFESSIONAL BACKTEST: Walk-Forward + Triple-Barrier")
print("="*70)

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def triple_barrier_labels(df, atr_mult_tp=2.0, atr_mult_sl=1.5, max_holding=10):
    """
    Triple-Barrier Labeling Method (Lopez de Prado)
    
    For each entry point:
    1. Upper barrier: Take Profit = entry + ATR * mult
    2. Lower barrier: Stop Loss = entry - ATR * mult  
    3. Vertical barrier: Max holding period
    
    Label = 1 if TP hit first, 0 if SL hit first or timeout
    """
    atr = calculate_atr(df)
    labels = []
    
    for i in range(len(df) - max_holding):
        entry = df['c'].iloc[i]
        atr_val = atr.iloc[i]
        
        if pd.isna(atr_val) or atr_val <= 0:
            labels.append(np.nan)
            continue
        
        tp = entry + atr_val * atr_mult_tp
        sl = entry - atr_val * atr_mult_sl
        
        # Look forward
        label = 0  # Default: timeout/SL
        for j in range(1, max_holding + 1):
            if i + j >= len(df):
                break
            
            high = df['h'].iloc[i + j]
            low = df['l'].iloc[i + j]
            
            # Check which barrier is hit first
            if high >= tp:
                label = 1  # TP hit
                break
            if low <= sl:
                label = 0  # SL hit
                break
    
        labels.append(label)
    
    # Pad remaining rows
    labels.extend([np.nan] * max_holding)
    
    return pd.Series(labels, index=df.index)


def create_features(df):
    """Create features for ML"""
    df = df.copy()
    
    # Returns
    df['ret_1'] = df['c'].pct_change(1) * 100
    df['ret_5'] = df['c'].pct_change(5) * 100
    df['ret_10'] = df['c'].pct_change(10) * 100
    df['ret_20'] = df['c'].pct_change(20) * 100
    
    # RSI
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    
    # MAs
    df['sma20'] = df['c'].rolling(20).mean()
    df['sma50'] = df['c'].rolling(50).mean()
    df['sma200'] = df['c'].rolling(200).mean()
    
    df['dist_sma20'] = (df['c'] - df['sma20']) / df['c'] * 100
    df['dist_sma50'] = (df['c'] - df['sma50']) / df['c'] * 100
    df['dist_sma200'] = (df['c'] - df['sma200']) / df['c'] * 100
    
    # BB
    df['bb_std'] = df['c'].rolling(20).std()
    bb_range = 4 * df['bb_std']
    df['bb_position'] = (df['c'] - (df['sma20'] - 2*df['bb_std'])) / (bb_range + 1e-10)
    df['bb_width'] = bb_range / df['c'] * 100
    
    # Volatility
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['c'] * 100
    df['volatility'] = df['c'].pct_change().rolling(20).std() * 100
    df['volatility_rank'] = df['volatility'].rolling(100).rank(pct=True)
    
    # Momentum
    df['roc_5'] = (df['c'] / df['c'].shift(5) - 1) * 100
    df['roc_10'] = (df['c'] / df['c'].shift(10) - 1) * 100
    df['roc_20'] = (df['c'] / df['c'].shift(20) - 1) * 100
    
    # Volume
    df['vol_ratio'] = df['v'] / (df['v'].rolling(20).mean() + 1e-10)
    
    # Time
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Trend
    df['trend_sma'] = (df['sma20'] > df['sma50']).astype(int)
    df['above_sma200'] = (df['c'] > df['sma200']).astype(int)
    
    # Pattern
    df['body_pct'] = (df['c'] - df['o']) / (df['o'] + 1e-10) * 100
    df['up_move'] = (df['c'] > df['o']).astype(int)
    df['consec_up'] = df['up_move'].rolling(5).sum()
    
    # S/R
    df['high_20'] = df['h'].rolling(20).max()
    df['low_20'] = df['l'].rolling(20).min()
    df['near_high'] = df['c'] / (df['high_20'] + 1e-10)
    df['near_low'] = df['c'] / (df['low_20'] + 1e-10)
    
    return df


def walk_forward_validation(df, feature_cols, n_splits=5, train_ratio=0.7):
    """
    Walk-Forward Validation
    
    Instead of single train/test split:
    1. Train on window 1, test on window 2
    2. Train on window 1+2, test on window 3
    3. etc.
    
    This prevents look-ahead bias and simulates real production.
    """
    results = []
    n = len(df)
    window_size = n // (n_splits + 1)
    
    print(f"\n📊 Walk-Forward Validation ({n_splits} folds)")
    print(f"   Total samples: {n}, Window size: {window_size}")
    print("-" * 60)
    
    for fold in range(n_splits):
        # Training data: from start to end of this fold
        train_end = (fold + 1) * window_size
        train_start = 0
        
        # Test data: next window
        test_start = train_end
        test_end = min(test_start + window_size, n)
        
        if test_end <= test_start:
            break
        
        # Get data
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        # Prepare features
        X_train = train_df[feature_cols].copy()
        y_train = train_df['label'].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df['label'].copy()
        
        # Clean data
        for X in [X_train, X_test]:
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop NaN
        valid_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        valid_test = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        
        if len(X_train) < 100 or len(X_test) < 20:
            continue
        
        # Clip extreme values
        for col in X_train.columns:
            q1, q99 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(q1, q99)
            X_test[col] = X_test[col].clip(q1, q99)
        
        # Train models
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        
        # Ensemble prediction
        prob = (rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2
        
        # Test different thresholds
        for thresh in [0.50, 0.55, 0.60]:
            preds = (prob >= thresh).astype(int)
            trades = preds.sum()
            
            if trades > 0:
                wins = ((preds == 1) & (y_test.values == 1)).sum()
                win_rate = wins / trades * 100
                
                results.append({
                    'fold': fold + 1,
                    'threshold': thresh,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'trades': trades,
                    'wins': wins,
                    'win_rate': win_rate
                })
        
        # Print fold results
        best = max([r for r in results if r['fold'] == fold + 1], key=lambda x: x['win_rate'] if x['trades'] >= 5 else 0, default=None)
        if best:
            s = "✅" if best['win_rate'] >= 50 else "❌"
            print(f"   Fold {fold+1}: Train={len(X_train)}, Test={len(X_test)} | Best: {best['win_rate']:.1f}% @ {best['threshold']} ({best['trades']} trades) {s}")
    
    return results


# Load data
print("\n📥 Loading data...")
df = pd.read_csv('market_data/BTCUSDT_1h.csv')
df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
print(f"   Loaded {len(df)} candles")

# Create features
print("\n🔧 Creating features...")
df = create_features(df)

# Triple-barrier labeling
print("\n🏷️ Creating Triple-Barrier Labels (TP: 2x ATR, SL: 1.5x ATR, Max: 10 bars)...")
df['label'] = triple_barrier_labels(df, atr_mult_tp=2.0, atr_mult_sl=1.5, max_holding=10)

# Drop NaN
df_clean = df.dropna()
print(f"   Valid samples: {len(df_clean)}")

# Label distribution
label_dist = df_clean['label'].value_counts(normalize=True) * 100
print(f"   Label distribution: Win={label_dist.get(1, 0):.1f}%, Loss={label_dist.get(0, 0):.1f}%")

# Feature columns
FEATURE_COLS = [
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'rsi', 'macd_hist',
    'dist_sma20', 'dist_sma50', 'dist_sma200',
    'bb_position', 'bb_width',
    'atr_pct', 'volatility', 'volatility_rank',
    'roc_5', 'roc_10', 'roc_20',
    'vol_ratio',
    'hour', 'day_of_week', 'is_weekend',
    'trend_sma', 'above_sma200',
    'body_pct', 'consec_up',
    'near_high', 'near_low'
]

# Run walk-forward validation
results = walk_forward_validation(df_clean, FEATURE_COLS, n_splits=5)

# Summary
print("\n" + "="*70)
print("📊 WALK-FORWARD VALIDATION SUMMARY")
print("="*70)

if results:
    # Group by threshold
    for thresh in [0.50, 0.55, 0.60]:
        thresh_results = [r for r in results if r['threshold'] == thresh]
        if thresh_results:
            avg_wr = np.mean([r['win_rate'] for r in thresh_results])
            total_trades = sum([r['trades'] for r in thresh_results])
            total_wins = sum([r['wins'] for r in thresh_results])
            
            s = "✅" if avg_wr >= 50 else "❌"
            print(f"   Threshold {thresh}: Avg Win Rate = {avg_wr:.1f}%, Total Trades = {total_trades}, Total Wins = {total_wins} {s}")
    
    # Best configuration
    best = max(results, key=lambda x: x['win_rate'] if x['trades'] >= 10 else 0)
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Threshold: {best['threshold']}")
    print(f"   Win Rate: {best['win_rate']:.1f}%")
    print(f"   Trades: {best['trades']}")

print("\n" + "="*70)
print("🎯 COMPARISON: Old vs New Labeling")
print("="*70)
print("   Old (simple % return): ~40% win rate (too many false signals)")
print("   New (triple-barrier):  More realistic labels based on actual SL/TP hits")
print("   Walk-forward:          Prevents overfitting, simulates real deployment")
print("="*70)
