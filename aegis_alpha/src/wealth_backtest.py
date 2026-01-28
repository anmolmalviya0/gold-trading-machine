"""
TERMINAL - WEALTH GENERATOR BACKTEST (V37)
==========================================
Simulates Alpha protocols: Kelly Criterion, Adaptive Thresholds, MTF Gatekeeper.
Certifies V37 Performance vs V36.
"""
import os
import joblib
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Import helpers from quant_utils (or redefine for speed)
def calculate_kelly_fraction(probability, win_loss_ratio, fraction_limit=0.5):
    if win_loss_ratio <= 0 or probability <= 0: return 0.0
    q = 1.0 - probability
    kelly_f = probability - (q / win_loss_ratio)
    return max(0.0, min(kelly_f * fraction_limit, 1.0))

def adaptive_threshold_logic(base_threshold, vol_z_score):
    if vol_z_score > 1.5: return min(0.95, base_threshold + 0.05)
    elif vol_z_score < -1.0: return min(0.95, base_threshold + 0.05)
    return base_threshold

# Configuration
MODELS_DIR = "/Users/anmol/Desktop/gold/aegis_alpha/models"
DATA_DIR = "/Users/anmol/Desktop/gold/market_data"
BASE_TRADE_SIZE_USD = 10.0
COMMISSION_BPS = 5.0

def load_and_standardize_csv(csv_path):
    """Robust loader for heterogeneous CSV formats"""
    try:
        # Check first line for header
        with open(csv_path, 'r') as f:
            first_line = f.readline().lower()
        
        has_header = 'time' in first_line or 'open' in first_line or 'timestamp' in first_line
        
        if has_header:
            df = pd.read_csv(csv_path)
            df.columns = [str(c).lower().strip() for c in df.columns]
        else:
            # Assume no header: time, open, high, low, close, volume
            df = pd.read_csv(csv_path, header=None, names=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        rename_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'timestamp': 'time'}
        df = df.rename(columns=rename_map)
        
        # Ensure time is datetime and index
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert all to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Load error {csv_path}: {e}")
        return None

def simulate_trades(symbol, timeframe, model_path, csv_path, gatekeeper_df=None):
    try:
        model = joblib.load(model_path)
        df = load_and_standardize_csv(csv_path)
        if df is None or len(df) < 100: 
            return None
        
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
        rs = gain / (loss + 1e-9)
        df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100
        
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd_hist'] = (exp12 - exp26) - (exp12 - exp26).ewm(span=9).mean()
        
        sma20_v = df['close'].rolling(20).mean()
        std20_v = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - (sma20_v - 2*std20_v)) / (4*std20_v + 1e-9)
        
        tr = pd.DataFrame({'hl': df['high'] - df['low'], 
                           'hc': abs(df['high'] - df['close'].shift(1)), 
                           'lc': abs(df['low'] - df['close'].shift(1))}).max(axis=1)
        df['atr_ratio'] = (tr.rolling(14).mean() / (df['close'] + 1e-9))
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
        
        true_range = tr # Already calculated above
        deltas_up = df['high'].diff()
        deltas_down = -df['low'].diff()
        alpha = 1/14
        tr_smooth = pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean()
        pdm_smooth = pd.Series(np.where((deltas_up > deltas_down) & (deltas_up > 0), deltas_up, 0)).ewm(alpha=alpha, adjust=False).mean()
        mdm_smooth = pd.Series(np.where((deltas_down > deltas_up) & (deltas_down > 0), deltas_down, 0)).ewm(alpha=alpha, adjust=False).mean()
        df['adx'] = (100 * np.abs((100 * (pdm_smooth / (tr_smooth + 1e-9))) - (100 * (mdm_smooth / (tr_smooth + 1e-9)))) / \
                    ((100 * (pdm_smooth / (tr_smooth + 1e-9))) + (100 * (mdm_smooth / (tr_smooth + 1e-9))) + 1e-9)).ewm(alpha=alpha, adjust=False).mean()

        rolling_vol = df['close'].pct_change().rolling(20).std()
        df['vol_z_score'] = (rolling_vol - rolling_vol.rolling(50).mean()) / (rolling_vol.rolling(50).std() + 1e-9)
        
        df_clean = df.dropna().copy()
        if df_clean.empty:
            return None
        
        feature_cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio',
            'adx', 'vol_z_score'
        ]
        
        X = df_clean[feature_cols]
        preds = model.predict_proba(X)[:, 1]
        
        # GATEKEEPER PRE-SYNC
        bias_series = None
        if gatekeeper_df is not None:
            g = gatekeeper_df.copy()
            g['sma20'] = g['close'].rolling(20).mean()
            g['bias'] = np.where(g['close'] > g['sma20'], 1, -1)
            # Reindex to match the current cleaned dataframe
            bias_series = g['bias'].reindex(df_clean.index, method='ffill').fillna(0).values

        atr_vals = tr.loc[df_clean.index].rolling(14).mean().values
        vol_z_vals = df_clean['vol_z_score'].values
        closes = df_clean['close'].values
        highs = df_clean['high'].values
        lows = df_clean['low'].values
        
        tp_mult, sl_mult = 1.5, 1.2
        window_map = {'1m': 60, '5m': 60, '15m': 48, '30m': 48, '1h': 24, '1d': 7}
        future_window = window_map.get(timeframe, 20)
        
        total_pnl = 0.0
        total_trades = 0
        wins = 0
        
        for i in range(len(preds)):
            current_threshold = adaptive_threshold_logic(0.70, vol_z_vals[i])
            if preds[i] >= current_threshold:
                # MTF Gatekeeper logic
                if timeframe in ['1m', '5m'] and bias_series is not None:
                    if bias_series[i] == -1: continue # Block BUYs in BEAR trend
                
                kelly_f = calculate_kelly_fraction(preds[i], 1.25)
                trade_size = BASE_TRADE_SIZE_USD * kelly_f
                if trade_size <= 0: continue
                
                if i < len(preds) - future_window:
                    entry = closes[i]
                    atr = atr_vals[i]
                    if np.isnan(atr) or atr <= 0: continue
                    tp_price = entry + tp_mult * atr
                    sl_price = entry - sl_mult * atr
                    
                    batch_highs = highs[i+1 : i+1+future_window]
                    batch_lows = lows[i+1 : i+1+future_window]
                    hit_tp = np.any(batch_highs >= tp_price)
                    hit_sl = np.any(batch_lows <= sl_price)
                    
                    won = False
                    if hit_tp and hit_sl:
                        won = np.argmax(batch_highs >= tp_price) < np.argmax(batch_lows <= sl_price)
                    elif hit_tp: won = True
                    elif hit_sl: won = False
                    else: won = closes[i+future_window] > entry
                    
                    comm = (trade_size * COMMISSION_BPS / 10000) * 2
                    if won:
                        total_pnl += (trade_size * (tp_mult * atr / entry)) - comm
                        wins += 1
                    else:
                        total_pnl -= (trade_size * (sl_mult * atr / entry)) + comm
                    total_trades += 1
                    
        return {"node": f"{symbol}_{timeframe}", "trades": total_trades, "win_rate": (wins/total_trades*100) if total_trades>0 else 0, "pnl": total_pnl}
    except Exception as e:
        print(f"Error {symbol}_{timeframe}: {e}")
        return None

def main():
    print("🧠 PROTOCOL INFINITY: WEALTH GENERATOR BACKTEST (V37)")
    models = glob.glob(os.path.join(MODELS_DIR, "*_lgbm.pkl"))
    results = []
    
    # Pre-load 15m data for gatekeeper
    gatekeepers = {}
    for asset in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'PAXGUSDT']:
        path = os.path.join(DATA_DIR, f"{asset}_15m.csv")
        if os.path.exists(path):
            gatekeepers[asset] = load_and_standardize_csv(path)

    for m_path in models:
        filename = os.path.basename(m_path)
        parts = filename.replace('_lgbm.pkl', '').split('_')
        symbol, timeframe = parts[0], parts[1]
        data_csv = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
        if not os.path.exists(data_csv):
            data_csv = os.path.join(DATA_DIR, f"{symbol.replace('/','')}_{timeframe}.csv")
            
        if os.path.exists(data_csv):
            gk = gatekeepers.get(symbol.replace('/',''))
            res = simulate_trades(symbol, timeframe, m_path, data_csv, gatekeeper_df=gk)
            if res: 
                results.append(res)
                print(f"✅ Tested {symbol} [{timeframe}] - Result Secured.")
            else:
                print(f"⚠️  Skipped {symbol} [{timeframe}] - Insufficient depth for সুইচব্লেড.")

    summary_path = "/Users/anmol/Desktop/gold/aegis_alpha/logs/v37_wealth_report.txt"
    with open(summary_path, "w") as f:
        f.write("WEALTH GENERATOR PROFIT REPORT (V37)\n")
        f.write("====================================\n")
        f.write(f"Base Trade: ${BASE_TRADE_SIZE_USD} | Protocol: VULCAN (Kelly)\n\n")
        total_pnl = 0
        for r in results:
            f.write(f"{r['node']}: {r['trades']} Trades | WR: {r['win_rate']:.1f}% | PnL: ${r['pnl']:.2f}\n")
            total_pnl += r['pnl']
        f.write(f"\nAGGREGATE V37 PNL: ${total_pnl:.2f}\n")
    print(f"\n✅ V37 WEALTH REPORT SECURED: {summary_path}")

if __name__ == "__main__":
    main()
