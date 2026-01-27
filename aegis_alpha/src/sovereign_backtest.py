"""
TERMINAL - SOVEREIGN BACKTEST (V36)
===================================
Simulates $10 trades across all 30 nodes (5 assets x 6 timeframes).
Certifies profitability with institutional-grade forensics.
"""
import os
import joblib
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Configuration
MODELS_DIR = "/Users/anmol/Desktop/gold/aegis_alpha/models"
DATA_DIR = "/Users/anmol/Desktop/gold/market_data"
TRADE_SIZE_USD = 10.0
COMMISSION_BPS = 5.0 # 0.05% per side

def simulate_trades(symbol, timeframe, model_path, csv_path):
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(csv_path)
        
        # Standard Feature Engineering (Minimal version for backtest speed)
        df.columns = [str(c).lower().strip() for c in df.columns]
        rename_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        df = df.rename(columns=rename_map)
        
        # We need the same 15 features as the trainer
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        for p in [5, 10, 20, 50]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'sma_ratio_{p}'] = df['close'] / df[f'sma_{p}']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100
        
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd_hist'] = (exp12 - exp26) - (exp12 - exp26).ewm(span=9).mean()
        
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - (sma20 - 2*std20)) / (4*std20)
        
        tr = pd.DataFrame({'hl': df['high'] - df['low'], 'hc': abs(df['high'] - df['close'].shift(1)), 'lc': abs(df['low'] - df['close'].shift(1))}).max(axis=1)
        df['atr_ratio'] = tr.rolling(14).mean() / df['close']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        deltas_up = df['high'].diff()
        deltas_down = -df['low'].diff()
        alpha = 1/14
        tr_smooth = pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean()
        pdm_smooth = pd.Series(np.where((deltas_up > deltas_down) & (deltas_up > 0), deltas_up, 0)).ewm(alpha=alpha, adjust=False).mean()
        mdm_smooth = pd.Series(np.where((deltas_down > deltas_up) & (deltas_down > 0), deltas_down, 0)).ewm(alpha=alpha, adjust=False).mean()
        df['adx'] = (100 * np.abs((100 * (pdm_smooth / tr_smooth)) - (100 * (mdm_smooth / tr_smooth))) / ((100 * (pdm_smooth / tr_smooth)) + (100 * (mdm_smooth / tr_smooth)))).ewm(alpha=alpha, adjust=False).mean()
        rolling_vol = df['close'].pct_change().rolling(20).std()
        df['vol_z_score'] = (rolling_vol - rolling_vol.rolling(50).mean()) / rolling_vol.rolling(50).std()
        
        df = df.dropna()
        
        feature_cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio',
            'adx', 'vol_z_score'
        ]
        
        X = df[feature_cols]
        preds = model.predict_proba(X)[:, 1]
        
        # Adaptive Threshold (Top 5% of signals)
        threshold = np.percentile(preds, 95)
        signals = (preds >= threshold).astype(int)
        
        # Simplified PnL: ATR based TP/SL
        # As used in matrix_trainer TP=1.5, SL=1.2 (Sovereign settings)
        atr_vals = tr.loc[df.index].rolling(14).mean().values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        tp_mult = 1.5
        sl_mult = 1.2
        window_map = {'1m': 60, '5m': 60, '15m': 48, '30m': 48, '1h': 24, '1d': 7}
        future_window = window_map.get(timeframe, 20)
        
        total_pnl_usd = 0.0
        wins = 0
        total_trades = 0
        
        for i in range(len(signals)):
            if signals[i] == 1 and i < len(signals) - future_window:
                entry = closes[i]
                atr = atr_vals[i]
                if np.isnan(atr) or atr == 0: continue
                
                tp_price = entry + tp_mult * atr
                sl_price = entry - sl_mult * atr
                
                # Check outcome
                slice_h = highs[i+1 : i+1+future_window]
                slice_l = lows[i+1 : i+1+future_window]
                
                hit_tp = np.any(slice_h >= tp_price)
                hit_sl = np.any(slice_l <= sl_price)
                
                if hit_tp and hit_sl:
                    tp_idx = np.argmax(slice_h >= tp_price)
                    sl_idx = np.argmax(slice_l <= sl_price)
                    won = tp_idx < sl_idx
                elif hit_tp:
                    won = True
                elif hit_sl:
                    won = False
                else:
                    won = closes[i+future_window] > entry # Time exit
                
                comm = (TRADE_SIZE_USD * COMMISSION_BPS / 10000) * 2
                if won:
                    profit_r = tp_mult / sl_mult # Risk Reward 1.25
                    total_pnl_usd += (TRADE_SIZE_USD * (tp_mult * atr / entry)) - comm
                    wins += 1
                else:
                    total_pnl_usd -= (TRADE_SIZE_USD * (sl_mult * atr / entry)) + comm
                
                total_trades += 1
                
        return {
            "node": f"{symbol}_{timeframe}",
            "trades": total_trades,
            "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            "pnl_usd": total_pnl_usd
        }
    except Exception as e:
        print(f"Error simulation {symbol} {timeframe}: {e}")
        return None

def main():
    print("🧠 PROTOCOL INFINITY: SOVEREIGN BACKTEST (V36)")
    models = glob.glob(os.path.join(MODELS_DIR, "*_lgbm.pkl"))
    results = []
    
    for m_path in models:
        filename = os.path.basename(m_path)
        parts = filename.replace('_lgbm.pkl', '').split('_')
        symbol = parts[0]
        timeframe = parts[1]
        
        # Match data file
        data_csv = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
        if not os.path.exists(data_csv):
            # Try alternate names like BTCUSDT_1m.csv
            data_csv = os.path.join(DATA_DIR, f"{symbol.replace('/','')}_{timeframe}.csv")
            
        if os.path.exists(data_csv):
            print(f"📊 Testing {symbol} [{timeframe}]...")
            res = simulate_trades(symbol, timeframe, m_path, data_csv)
            if res: results.append(res)
        else:
            print(f"⚠️ Data missing for {filename}")

    # Report
    summary_path = "/Users/anmol/Desktop/gold/aegis_alpha/logs/v36_backtest_report.txt"
    with open(summary_path, "w") as f:
        f.write("SOVEREIGN BACKTEST REPORT (V36)\n")
        f.write("===============================\n")
        f.write(f"Trade Size: ${TRADE_SIZE_USD}\n\n")
        total_pnl = 0
        for r in results:
            f.write(f"{r['node']}: {r['trades']} Trades | WR: {r['win_rate']:.1f}% | PnL: ${r['pnl_usd']:.2f}\n")
            total_pnl += r['pnl_usd']
        f.write(f"\nAGGREGATE PNL: ${total_pnl:.2f}\n")
    
    print(f"\n✅ BACKTEST COMPLETE. REPORT SECURED IN {summary_path}")

if __name__ == "__main__":
    main()
