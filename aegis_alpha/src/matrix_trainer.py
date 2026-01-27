import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')
DATA_DIR = "/Users/anmol/Desktop/gold/market_data"

def engineer_features(df, timeframe='1m', tp_mult=1.5, sl_mult=1.2):
    """
    Standard Feature Engineering (The Switchblade Protocol)
    Applies consistent logic across all timeframes.
    """
    df = df.copy()
    
    # NORMALIZE COLUMNS 
    df.columns = [str(c).lower().strip() for c in df.columns]
    rename_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
    df = df.rename(columns=rename_map)
    
    # 1. Base Features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # 2. Moving Averages
    for p in [5, 10, 20, 50]:
        df[f'sma_{p}'] = df['close'].rolling(p).mean()
        df[f'sma_ratio_{p}'] = df['close'] / df[f'sma_{p}']
    
    # 3. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100
    
    # 4. MACD
    exp12 = df['close'].ewm(span=12).mean()
    exp26 = df['close'].ewm(span=26).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 5. Bollinger
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - (sma20 - 2*std20)) / (4*std20)
    
    # 6. ATR/Vol
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'], 
        'hc': abs(df['high'] - df['close'].shift(1)), 
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    df['atr_ratio'] = tr.rolling(14).mean() / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    atr_vals = tr.rolling(14).mean().values

    # --- DYNAMIC TARGET WINDOWING ---
    window_map = {'1m': 60, '5m': 60, '15m': 48, '30m': 48, '1h': 24, '1d': 7}
    future_window = window_map.get(timeframe, 20)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    labels = []
    
    for i in range(len(df)):
        if i >= len(df) - future_window:
            labels.append(0)
            continue
        entry = closes[i]
        atr = atr_vals[i]
        if np.isnan(atr) or atr == 0:
            labels.append(0)
            continue
        tp = entry + tp_mult * atr
        sl = entry - sl_mult * atr
        window_highs = highs[i+1 : i+1+future_window]
        window_lows = lows[i+1 : i+1+future_window]
        hit_tp = np.any(window_highs >= tp)
        hit_sl = np.any(window_lows <= sl)
        if hit_tp and hit_sl:
            tp_idx = np.argmax(window_highs >= tp)
            sl_idx = np.argmax(window_lows <= sl)
            label = 1 if tp_idx < sl_idx else 0
        elif hit_tp:
            label = 1
        else:
            label = 0
        labels.append(label)
        
    df['target'] = labels
    
    # ADX and Z-Score
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    deltas_up = df['high'].diff()
    deltas_down = -df['low'].diff()
    alpha = 1/14
    plus_dm = np.where((deltas_up > deltas_down) & (deltas_up > 0), deltas_up, 0)
    minus_dm = np.where((deltas_down > deltas_up) & (deltas_down > 0), deltas_down, 0)
    tr_smooth = pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean()
    pdm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    mdm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (pdm_smooth / tr_smooth)
    minus_di = 100 * (mdm_smooth / tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
    rolling_vol = df['close'].pct_change().rolling(20).std()
    df['vol_z_score'] = (rolling_vol - rolling_vol.rolling(50).mean()) / rolling_vol.rolling(50).std()
    
    df = df.dropna()
    return df

def train_node(csv_path):
    try:
        filename = os.path.basename(csv_path)
        parts = filename.replace('.csv', '').split('_')
        symbol = parts[0]
        timeframe = parts[1] if len(parts) > 1 else '1m'
        if "deep" in filename: timeframe = "1m"
        
        print(f"⚙️  TRAINING NODE: {symbol} [{timeframe}]")
        raw_df = pd.read_csv(csv_path)
        
        # STRATEGY DESCENT LOOP
        # We try multipliers from Institutional (2.0) down to Scalp (0.8)
        mult_pairs = [(2.0, 1.5), (1.5, 1.2), (1.2, 1.0), (1.0, 0.8)]
        best_node_res = None
        
        for tp_m, sl_m in mult_pairs:
            # We don't want to re-run everything if we don't need to, 
            # but targets rely on engineer_features
            df = engineer_features(raw_df.copy(), timeframe=timeframe, tp_mult=tp_m, sl_mult=sl_m)
            
            feature_cols = [
                'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio',
                'adx', 'vol_z_score'
            ]
            
            if len(df) < 500: continue
            X, y = df[feature_cols], df['target']
            
            # Simple check: Does it have signals?
            if y.mean() < 0.05: # Less than 5% win rate in dataset is usually noise for these targets
                continue
                
            split = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]
            
            model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
            
            preds = model.predict_proba(X_val)[:, 1]
            tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '1d': 1440}
            m_per_tf = tf_minutes.get(timeframe, 1)
            val_weeks = max(0.1, (len(X_val) * m_per_tf) / 10080)
            
            # Calibration: Try to get at least 2 sigs/week on LFT, 25 sigs/week on HFT
            target_freq = 25 if timeframe in ['1m', '5m'] else 2.5
            total_target = max(1, int(target_freq * val_weeks))
            
            if len(preds) > total_target:
                threshold = np.percentile(preds, 100 * (1 - total_target / len(preds)))
            else: threshold = 0.52
            threshold = max(0.51, min(0.85, threshold))
            
            top_preds = (preds >= threshold).astype(int)
            sigs = int(np.sum(top_preds))
            if sigs == 0: continue
            
            accuracy = np.sum((top_preds == 1) & (y_val == 1)) / sigs
            
            net_profit_r = 0
            rrr = tp_m / sl_m
            for idx in range(len(top_preds)):
                if top_preds[idx] == 1:
                    net_profit_r += rrr if y_val.values[idx] == 1 else -1.0
            
            if net_profit_r > 0:
                print(f"   ✅ descent fixed. mult: {tp_m} | wr: {accuracy*100:.1f}% | net: {net_profit_r:.2f}r")
                joblib.dump(model, os.path.join(MODELS_DIR, f"{symbol}_{timeframe}_lgbm.pkl"))
                return {"node": f"{symbol}_{timeframe}", "win_rate": accuracy, "signals": sigs, "net_profit": round(net_profit_r, 2), "tp": tp_m}

        # If we reach here, we found no profitable strategy with descent
        print(f"   ⚠️ node silent after descent.")
        return {"node": f"{symbol}_{timeframe}", "win_rate": 0, "signals": 0, "net_profit": 0, "tp": 0}
        
    except Exception as e:
        print(f"   ❌ error: {e}")
        return None

def main():
    import sys
    asset_filter = sys.argv[1:] if len(sys.argv) > 1 else None
    print("🧠 PROTOCOL INFINITY: OMNISCIENT TRAINER (WAR MACHINE EDITION)")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_*.csv"))
    if asset_filter: csv_files = [f for f in csv_files if any(a in os.path.basename(f) for a in asset_filter)]
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = []
    for f in csv_files:
        res = train_node(f)
        if res: results.append(res)
    
    # Generate Final Report
    with open("logs/war_machine_status.txt", "w") as f:
        f.write("WAR MACHINE CERTIFICATION REPORT\n")
        f.write("===============================\n")
        for r in results: 
            f.write(f"{r['node']}: WR={r['win_rate']*100:.1f}%, Profit={r['net_profit']}R, TP_Mult={r.get('tp',0)}\n")

if __name__ == "__main__":
    main()
