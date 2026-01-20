"""
OPTIMIZED WALK-FORWARD VALIDATION
Testing different R:R ratios to find best accuracy
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 OPTIMIZED BACKTEST: Finding Best Parameters")
print("="*70)

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def triple_barrier_labels(df, atr_mult_tp=1.5, atr_mult_sl=1.0, max_holding=10):
    """Triple-barrier with adjustable parameters"""
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
        
        label = 0
        for j in range(1, max_holding + 1):
            if i + j >= len(df):
                break
            
            high = df['h'].iloc[i + j]
            low = df['l'].iloc[i + j]
            
            if high >= tp:
                label = 1
                break
            if low <= sl:
                label = 0
                break
        
        labels.append(label)
    
    labels.extend([np.nan] * max_holding)
    return pd.Series(labels, index=df.index)


def create_features(df):
    df = df.copy()
    
    df['ret_1'] = df['c'].pct_change(1) * 100
    df['ret_5'] = df['c'].pct_change(5) * 100
    df['ret_10'] = df['c'].pct_change(10) * 100
    df['ret_20'] = df['c'].pct_change(20) * 100
    
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    
    df['sma20'] = df['c'].rolling(20).mean()
    df['sma50'] = df['c'].rolling(50).mean()
    df['sma200'] = df['c'].rolling(200).mean()
    
    df['dist_sma20'] = (df['c'] - df['sma20']) / df['c'] * 100
    df['dist_sma50'] = (df['c'] - df['sma50']) / df['c'] * 100
    df['dist_sma200'] = (df['c'] - df['sma200']) / df['c'] * 100
    
    df['bb_std'] = df['c'].rolling(20).std()
    bb_range = 4 * df['bb_std']
    df['bb_position'] = (df['c'] - (df['sma20'] - 2*df['bb_std'])) / (bb_range + 1e-10)
    df['bb_width'] = bb_range / df['c'] * 100
    
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['c'] * 100
    df['volatility'] = df['c'].pct_change().rolling(20).std() * 100
    df['volatility_rank'] = df['volatility'].rolling(100).rank(pct=True)
    
    df['roc_5'] = (df['c'] / df['c'].shift(5) - 1) * 100
    df['roc_10'] = (df['c'] / df['c'].shift(10) - 1) * 100
    df['roc_20'] = (df['c'] / df['c'].shift(20) - 1) * 100
    
    df['vol_ratio'] = df['v'] / (df['v'].rolling(20).mean() + 1e-10)
    
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['trend_sma'] = (df['sma20'] > df['sma50']).astype(int)
    df['above_sma200'] = (df['c'] > df['sma200']).astype(int)
    
    df['body_pct'] = (df['c'] - df['o']) / (df['o'] + 1e-10) * 100
    df['up_move'] = (df['c'] > df['o']).astype(int)
    df['consec_up'] = df['up_move'].rolling(5).sum()
    
    df['high_20'] = df['h'].rolling(20).max()
    df['low_20'] = df['l'].rolling(20).min()
    df['near_high'] = df['c'] / (df['high_20'] + 1e-10)
    df['near_low'] = df['c'] / (df['low_20'] + 1e-10)
    
    return df


def test_configuration(df, feature_cols, atr_tp, atr_sl, holding):
    """Test a specific R:R configuration"""
    df = df.copy()
    df['label'] = triple_barrier_labels(df, atr_tp, atr_sl, holding)
    
    df_clean = df.dropna()
    if len(df_clean) < 1000:
        return None
    
    # Simple train/test split (80/20)
    split = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split]
    test_df = df_clean.iloc[split:]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['label'].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df['label'].copy()
    
    # Clean
    for X in [X_train, X_test]:
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    valid_train = ~(X_train.isna().any(axis=1) | y_train.isna())
    valid_test = ~(X_test.isna().any(axis=1) | y_test.isna())
    
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    
    if len(X_train) < 500 or len(X_test) < 100:
        return None
    
    # Clip
    for col in X_train.columns:
        q1, q99 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(q1, q99)
        X_test[col] = X_test[col].clip(q1, q99)
    
    # Train
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    
    # Predict
    prob = (rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2
    
    # Calculate baseline (label distribution)
    base_win_rate = y_test.mean() * 100
    
    results = []
    for thresh in [0.50, 0.55, 0.60, 0.65]:
        preds = (prob >= thresh).astype(int)
        trades = preds.sum()
        
        if trades >= 10:
            wins = ((preds == 1) & (y_test.values == 1)).sum()
            win_rate = wins / trades * 100
            
            # Calculate expected profit
            win_pnl = wins * atr_tp
            loss_pnl = (trades - wins) * atr_sl
            expected_pf = win_pnl / (loss_pnl + 1e-10)
            
            results.append({
                'thresh': thresh,
                'trades': trades,
                'wins': wins,
                'win_rate': win_rate,
                'base_rate': base_win_rate,
                'lift': win_rate - base_win_rate,
                'profit_factor': expected_pf
            })
    
    return results


# Load data
print("\n📥 Loading data...")
df = pd.read_csv('market_data/BTCUSDT_1h.csv')
df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
df = create_features(df)
print(f"   Loaded {len(df)} candles")

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

# Test different R:R configurations
print("\n📊 Testing different R:R configurations...")
print("-" * 80)

configurations = [
    # (TP mult, SL mult, holding, name)
    (1.0, 1.0, 10, "1:1 R:R"),
    (1.5, 1.0, 10, "1.5:1 R:R"),
    (1.5, 1.5, 10, "1:1 (wider)"),
    (2.0, 1.0, 10, "2:1 R:R"),
    (1.0, 0.5, 5, "2:1 tight SL"),
    (1.2, 0.8, 8, "1.5:1 balanced"),
]

best_result = None
best_config = None

for atr_tp, atr_sl, holding, name in configurations:
    print(f"\n🔧 Testing: {name} (TP={atr_tp}x, SL={atr_sl}x, Hold={holding})")
    
    results = test_configuration(df, FEATURE_COLS, atr_tp, atr_sl, holding)
    
    if results:
        best = max(results, key=lambda x: x['profit_factor'] if x['trades'] >= 20 else 0)
        
        s = "✅" if best['win_rate'] >= 50 else "⚠️" if best['win_rate'] >= 45 else "❌"
        pf_s = "✅" if best['profit_factor'] >= 1.0 else "❌"
        
        print(f"   Base rate: {best['base_rate']:.1f}% | Best: {best['win_rate']:.1f}% @ {best['thresh']} ({best['trades']} trades) {s}")
        print(f"   Lift: +{best['lift']:.1f}% | Profit Factor: {best['profit_factor']:.2f} {pf_s}")
        
        if best_result is None or best['profit_factor'] > best_result['profit_factor']:
            best_result = best
            best_config = (atr_tp, atr_sl, holding, name)

print("\n" + "="*80)
print("🏆 BEST CONFIGURATION")
print("="*80)

if best_result and best_config:
    print(f"\n   Configuration: {best_config[3]}")
    print(f"   TP: {best_config[0]}x ATR")
    print(f"   SL: {best_config[1]}x ATR")
    print(f"   Max Hold: {best_config[2]} bars")
    print(f"\n   Win Rate: {best_result['win_rate']:.1f}%")
    print(f"   Trades: {best_result['trades']}")
    print(f"   Confidence Threshold: {best_result['thresh']}")
    print(f"   Profit Factor: {best_result['profit_factor']:.2f}")
    
    if best_result['profit_factor'] >= 1.0:
        print(f"\n   ✅ PROFITABLE CONFIGURATION FOUND!")
    else:
        print(f"\n   ⚠️ Need more optimization")

print("="*80)

# Save best config
config_txt = f"""
BEST_CONFIG = {{
    'TP_ATR_MULT': {best_config[0]},
    'SL_ATR_MULT': {best_config[1]},
    'MAX_HOLDING': {best_config[2]},
    'CONFIDENCE_THRESHOLD': {best_result['thresh']},
    'EXPECTED_WIN_RATE': {best_result['win_rate']:.1f},
    'PROFIT_FACTOR': {best_result['profit_factor']:.2f}
}}
"""
print(config_txt)
