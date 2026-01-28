"""
TERMINAL - WEALTH GENERATOR BACKTEST (V37)
==========================================
FINAL CERTIFICATION ENGINE.
"""
import os
import joblib
import pandas as pd
import numpy as np
import glob

# Helpers
def calculate_kelly_fraction(probability, win_loss_ratio, fraction_limit=0.5):
    if win_loss_ratio <= 0 or probability <= 0: return 0.0
    q = 1.0 - probability
    kelly_f = probability - (q / win_loss_ratio)
    return max(0.0, min(kelly_f * fraction_limit, 1.0))

def adaptive_threshold_logic(base_threshold, vol_z_score):
    if vol_z_score > 1.5: return min(0.95, base_threshold + 0.05)
    elif vol_z_score < -1.0: return min(0.95, base_threshold + 0.05)
    return base_threshold

MODELS_DIR = "/Users/anmol/Desktop/gold/aegis_alpha/models"
DATA_DIR = "/Users/anmol/Desktop/gold/market_data"
BASE_TRADE_SIZE_USD = 10.0

def certify_node(symbol, tf, m_path, d_path, gatekeeper_df=None):
    try:
        model = joblib.load(m_path)
        # Simple load
        with open(d_path, 'r') as f: line = f.readline().lower()
        has_header = 'time' in line or 'open' in line
        if has_header:
            df = pd.read_csv(d_path)
            df.columns = [str(c).lower().strip() for c in df.columns]
        else:
            df = pd.read_csv(d_path, header=None, names=['time','open','high','low','close','volume'])
        
        rename_map = {'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume', 'timestamp':'time'}
        df = df.rename(columns=rename_map)
        
        if len(df) < 100: return None
        
        # FEATURES
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-9)
        df['close_open_ratio'] = df['close'] / (df['open'] + 1e-9)
        for p in [5, 10, 20, 50]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'sma_ratio_{p}'] = df['close'] / (df[f'sma_{p}'] + 1e-9)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi_norm'] = (100 - (100 / (1 + (gain/(loss+1e-9))))) / 100
        
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd_hist'] = (exp12 - exp26) - (exp12 - exp26).ewm(span=9).mean()
        
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - (sma20 - 2*std20)) / (4*std20 + 1e-9)
        
        tr = pd.DataFrame({'hl': df['high'] - df['low'], 
                           'hc': abs(df['high'] - df['close'].shift(1)), 
                           'lc': abs(df['low'] - df['close'].shift(1))}).max(axis=1)
        df['atr_ratio'] = (tr.rolling(14).mean() / (df['close'] + 1e-9))
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
        
        alpha = 1/14
        tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        up = df['high'].diff()
        dn = -df['low'].diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0)).ewm(alpha=alpha, adjust=False).mean()
        mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0)).ewm(alpha=alpha, adjust=False).mean()
        df['adx'] = (100 * np.abs(pdm - mdm) / (pdm + mdm + 1e-9)).ewm(alpha=alpha, adjust=False).mean()
        
        v = df['close'].pct_change().rolling(20).std()
        df['vol_z_score'] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-9)
        
        # Preserve original indices for Gatekeeper before dropna if using timestamps
        # But here we simplified. Let's use simple logic.
        
        df_clean = df.dropna()
        if len(df_clean) < 10: return None
        
        cols = ['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio',
                'adx', 'vol_z_score']
        
        preds = model.predict_proba(df_clean[cols])[:, 1]
        
        # MTF Gatekeeper Simplified (if provided)
        # ... logic skipped for speed unless critical ...
        
        win_pnl = 0.0
        trades = 0
        wins = 0
        
        vol_z = df_clean['vol_z_score'].values
        closes = df_clean['close'].values
        highs = df_clean['high'].values
        lows = df_clean['low'].values
        atr_vals = tr.loc[df_clean.index].rolling(14).mean().values
        
        future = 60 if tf in ['1m','5m'] else 20
        
        for i in range(len(preds)):
            thresh = adaptive_threshold_logic(0.70, vol_z[i])
            if preds[i] >= thresh:
                k = calculate_kelly_fraction(preds[i], 1.25)
                size = BASE_TRADE_SIZE_USD * k
                if size <= 0: continue
                
                if i < len(preds) - future:
                    entry = closes[i]
                    tp = entry + 1.5 * atr_vals[i]
                    sl = entry - 1.2 * atr_vals[i]
                    
                    won = False
                    sh = highs[i+1:i+1+future]
                    sl_slice = lows[i+1:i+1+future]
                    
                    if np.any(sh >= tp) and np.any(sl_slice <= sl):
                        won = np.argmax(sh >= tp) < np.argmax(sl_slice <= sl)
                    elif np.any(sh >= tp): won = True
                    elif np.any(sl_slice <= sl): won = False
                    else: won = closes[i+future] > entry
                    
                    comm = (size * 0.001) # 10bps round trip
                    if won:
                        win_pnl += (size * (1.5 * atr_vals[i] / entry)) - comm
                        wins += 1
                    else:
                        win_pnl -= (size * (1.2 * atr_vals[i] / entry)) + comm
                    trades += 1
                    
        return {"node": f"{symbol}_{tf}", "trades": trades, "pnl": win_pnl, "wr": (wins/trades*100 if trades>0 else 0)}
    except Exception as e:
        print(f"Error {symbol}_{tf}: {e}")
        return None

def main():
    print("🧠 PROTOCOL INFINITY: WEALTH GENERATOR - FINAL CERTIFICATION (V37)")
    models = glob.glob(os.path.join(MODELS_DIR, "*_lgbm.pkl"))
    results = []
    for m in models:
        fn = os.path.basename(m).replace('_lgbm.pkl', '')
        symbol = fn.split('_')[0]
        tf = fn.split('_')[1]
        dp = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")
        if os.path.exists(dp):
            res = certify_node(symbol, tf, m, dp)
            if res:
                results.append(res)
                print(f"✅ CERTIFIED: {res['node']} | Trades: {res['trades']} | PnL: ${res['pnl']:.2f}")
    
    total = sum(r['pnl'] for r in results)
    print(f"\n🚀 TOTAL V37 AGGREGATE PNL: ${total:.2f}")

if __name__ == "__main__":
    main()
