"""
PROFESSIONAL TRADING TERMINAL v1.0
Features:
- Live BTC + Gold predictions
- Auto-retraining ML model every 4 hours
- Paper trading mode with trade journal
- Fees calculation (0.1%)
- Win rate tracking
- Professional UI
"""
import asyncio
import json
import time
import os
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import joblib
import aiohttp
import ssl
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
# === CONFIGURATION ===
CONFIG = {
    'FEE_RATE': 0.001,  # 0.1% per trade
    'RETRAIN_HOURS': 4,  # Retrain every 4 hours
    'MIN_CONFIDENCE': 0.60,  # 60% threshold (improved from 55%)
    'RISK_PERCENT': 0.02,  # 2% risk per trade
    'SL_ATR_MULT': 1.5,  # Stop loss = 1.5 * ATR
    'TP_ATR_MULT': 2.5,  # Take profit = 2.5 * ATR
    'LOOKBACK_DAYS': 14,  # Days of data for training
}

# Telegram config (set your credentials here)
TELEGRAM = {
    'BOT_TOKEN': None,  # Get from @BotFather
    'CHAT_ID': None,    # Get from /getUpdates
    'ENABLED': False,   # Set True when configured
}

# Auto-trading config
AUTO_TRADE = {
    'ENABLED': True,    # Auto-open paper trades on signals
    'MIN_CONFIDENCE': 65,  # Only auto-trade above this confidence
    'COOLDOWN': 60,     # Seconds between auto-trades per symbol
}

last_auto_trade = {'BTCUSDT': 0, 'PAXGUSDT': 0}

# SSL
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# Note: app and lifespan defined after all dependencies

# === DIRECTORIES ===
BASE_DIR = '/Users/anmol/Desktop/gold/trading_intelligence'
MODEL_DIR = f'{BASE_DIR}/models/saved'
LOG_DIR = f'{BASE_DIR}/logs'
os.makedirs(LOG_DIR, exist_ok=True)

# === GLOBAL STATE ===
state = {
    'BTCUSDT': {
        'price': 0, 'open': 0, 'high': 0, 'low': 0, 
        'change': 0, 'volume': 0, 'history': deque(maxlen=500), 'ts': 0
    },
    'PAXGUSDT': {
        'price': 0, 'open': 0, 'high': 0, 'low': 0,
        'change': 0, 'volume': 0, 'history': deque(maxlen=500), 'ts': 0
    }
}

# ML Models
ml_state = {
    'rf_model': None,
    'gb_model': None,
    'features': None,
    'last_train': None,
    'train_accuracy': 0,
    'status': 'Loading...'
}

# Paper Trading
paper_trades = {
    'BTCUSDT': [],
    'PAXGUSDT': [],
    'active': {'BTCUSDT': None, 'PAXGUSDT': None}
}

# Statistics
stats = {
    'total_trades': 0,
    'wins': 0,
    'losses': 0,
    'total_pnl': 0,
    'predictions_made': 0
}

news = []

# === FEATURE ENGINEERING ===
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


def create_features(df):
    """Create all ML features"""
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
    df['sma200'] = df['c'].rolling(min(200, len(df))).mean()
    
    df['dist_sma20'] = (df['c'] - df['sma20']) / df['c'] * 100
    df['dist_sma50'] = (df['c'] - df['sma50']) / df['c'] * 100
    df['dist_sma200'] = (df['c'] - df['sma200']) / df['c'] * 100
    
    # Bollinger Bands
    df['bb_std'] = df['c'].rolling(20).std()
    bb_range = 4 * df['bb_std']
    df['bb_position'] = (df['c'] - (df['sma20'] - 2*df['bb_std'])) / (bb_range + 1e-10)
    df['bb_width'] = bb_range / df['c'] * 100
    
    # Volatility
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['c'] * 100
    df['volatility'] = df['c'].pct_change().rolling(20).std() * 100
    df['volatility_rank'] = df['volatility'].rolling(min(100, len(df))).rank(pct=True)
    
    # Momentum
    df['roc_5'] = (df['c'] / df['c'].shift(5) - 1) * 100
    df['roc_10'] = (df['c'] / df['c'].shift(10) - 1) * 100
    df['roc_20'] = (df['c'] / df['c'].shift(20) - 1) * 100
    
    # Volume
    df['vol_ratio'] = df['v'] / (df['v'].rolling(20).mean() + 1e-10)
    
    # Time
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    else:
        now = datetime.now()
        df['hour'] = now.hour
        df['day_of_week'] = now.weekday()
        df['is_weekend'] = 1 if now.weekday() >= 5 else 0
    
    # Trend
    df['trend_sma'] = (df['sma20'] > df['sma50']).astype(int)
    df['above_sma200'] = (df['c'] > df['sma200']).astype(int)
    
    # Pattern
    df['body_pct'] = (df['c'] - df['o']) / (df['o'] + 1e-10) * 100
    df['up_move'] = (df['c'] > df['o']).astype(int)
    df['consec_up'] = df['up_move'].rolling(5).sum()
    
    # Support/Resistance
    df['high_20'] = df['h'].rolling(20).max()
    df['low_20'] = df['l'].rolling(20).min()
    df['near_high'] = df['c'] / (df['high_20'] + 1e-10)
    df['near_low'] = df['c'] / (df['low_20'] + 1e-10)
    
    return df


async def fetch_historical_klines(symbol, interval='1h', limit=500):
    """Fetch historical data from Binance"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, ssl=False, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    df = pd.DataFrame(data, columns=[
                        'time', 'o', 'h', 'l', 'c', 'v',
                        'close_time', 'qav', 'trades', 'taker_base', 'taker_quote', 'ignore'
                    ])
                    df = df[['time', 'o', 'h', 'l', 'c', 'v']].astype(float)
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    return df
    except Exception as e:
        print(f"Error fetching klines: {e}")
    return None


async def train_model():
    """Train/retrain ML model on recent data"""
    global ml_state
    
    try:
        ml_state['status'] = 'Training...'
        print(f"🔄 Retraining model at {datetime.now().strftime('%H:%M:%S')}")
        
        # Fetch recent data
        df_btc = await fetch_historical_klines('BTCUSDT', '1h', 500)
        if df_btc is None or len(df_btc) < 100:
            ml_state['status'] = 'Training failed - no data'
            return
        
        # Create features
        df = create_features(df_btc)
        
        # Create labels (price up 0.5% in next 10 candles)
        df['future_ret'] = df['c'].shift(-10) / df['c'] - 1
        df['label'] = (df['future_ret'] > 0.005).astype(int)
        
        # Clean data
        df_clean = df.dropna()
        X = df_clean[FEATURE_COLS].copy()
        y = df_clean['label'].copy()
        
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_idx = ~X.isna().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            ml_state['status'] = 'Training failed - insufficient data'
            return
        
        # Clip extreme values
        for col in X.columns:
            q1, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
            X[col] = X[col].clip(q1, q99)
        
        # Train models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        # Time-series split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        y_pred = (rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2 >= 0.5
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        # Save models
        ml_state['rf_model'] = rf
        ml_state['gb_model'] = gb
        ml_state['features'] = FEATURE_COLS
        ml_state['last_train'] = datetime.now()
        ml_state['train_accuracy'] = accuracy
        ml_state['status'] = f'Ready ({accuracy:.1f}%)'
        
        # Save to disk
        joblib.dump(rf, f'{MODEL_DIR}/ml_rf_model.pkl')
        joblib.dump(gb, f'{MODEL_DIR}/ml_gb_model.pkl')
        
        print(f"✅ Model trained: {accuracy:.1f}% accuracy on {len(X)} samples")
        
    except Exception as e:
        ml_state['status'] = f'Error: {str(e)[:30]}'
        print(f"❌ Training error: {e}")


async def retrain_loop():
    """Background task to retrain model periodically"""
    while True:
        await train_model()
        await asyncio.sleep(CONFIG['RETRAIN_HOURS'] * 3600)


async def fetch_prices():
    """Fetch live prices"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in ['BTCUSDT', 'PAXGUSDT']:
                    async with session.get(f"{url}?symbol={symbol}", ssl=False, timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            state[symbol]['price'] = float(data['lastPrice'])
                            state[symbol]['open'] = float(data['openPrice'])
                            state[symbol]['high'] = float(data['highPrice'])
                            state[symbol]['low'] = float(data['lowPrice'])
                            state[symbol]['change'] = float(data['priceChangePercent'])
                            state[symbol]['volume'] = float(data['volume'])
                            state[symbol]['ts'] = int(time.time() * 1000)
                            
                            # Add to history for live prediction
                            state[symbol]['history'].append({
                                'time': datetime.now(),
                                'o': float(data['openPrice']),
                                'h': float(data['highPrice']),
                                'l': float(data['lowPrice']),
                                'c': float(data['lastPrice']),
                                'v': float(data['volume'])
                            })
        except Exception as e:
            print(f"Price error: {e}")
        
        # Check paper trades
        check_paper_trades()
        
        await asyncio.sleep(0.5)  # 500ms updates


async def fetch_news():
    """Fetch market news"""
    global news
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,Gold"
                async with session.get(url, ssl=False, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        news = [{'title': n['title'], 'source': n['source'], 'time': n.get('published_on', 0)} 
                                for n in data.get('Data', [])[:5]]
        except:
            pass
        await asyncio.sleep(120)


def check_paper_trades():
    """Check if paper trades hit TP or SL"""
    for symbol in ['BTCUSDT', 'PAXGUSDT']:
        trade = paper_trades['active'][symbol]
        if trade is None:
            continue
        
        price = state[symbol]['price']
        high = state[symbol]['high']
        low = state[symbol]['low']
        
        if trade['side'] == 'BUY':
            if high >= trade['tp']:
                # TP hit
                pnl = (trade['tp'] - trade['entry']) / trade['entry'] * trade['size']
                pnl_after_fees = pnl - (trade['size'] * CONFIG['FEE_RATE'] * 2)
                close_trade(symbol, 'TP', pnl_after_fees)
            elif low <= trade['sl']:
                # SL hit
                pnl = (trade['sl'] - trade['entry']) / trade['entry'] * trade['size']
                pnl_after_fees = pnl - (trade['size'] * CONFIG['FEE_RATE'] * 2)
                close_trade(symbol, 'SL', pnl_after_fees)
        else:  # SELL
            if low <= trade['tp']:
                pnl = (trade['entry'] - trade['tp']) / trade['entry'] * trade['size']
                pnl_after_fees = pnl - (trade['size'] * CONFIG['FEE_RATE'] * 2)
                close_trade(symbol, 'TP', pnl_after_fees)
            elif high >= trade['sl']:
                pnl = (trade['entry'] - trade['sl']) / trade['entry'] * trade['size']
                pnl_after_fees = pnl - (trade['size'] * CONFIG['FEE_RATE'] * 2)
                close_trade(symbol, 'SL', pnl_after_fees)


def close_trade(symbol, reason, pnl):
    """Close a paper trade"""
    trade = paper_trades['active'][symbol]
    if trade is None:
        return
    
    trade['close_time'] = datetime.now().isoformat()
    trade['close_reason'] = reason
    trade['pnl'] = pnl
    trade['status'] = 'WIN' if pnl > 0 else 'LOSS'
    
    paper_trades[symbol].append(trade)
    paper_trades['active'][symbol] = None
    
    stats['total_trades'] += 1
    stats['total_pnl'] += pnl
    if pnl > 0:
        stats['wins'] += 1
    else:
        stats['losses'] += 1
    
    # Log trade
    log_trade(symbol, trade)
    
    print(f"📊 Trade closed: {symbol} {trade['side']} - {reason} - PnL: ${pnl:.2f}")


def log_trade(symbol, trade):
    """Log trade to file"""
    log_file = f"{LOG_DIR}/trades_{datetime.now().strftime('%Y%m%d')}.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append({
            'symbol': symbol,
            **trade
        })
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    except Exception as e:
        print(f"Log error: {e}")


def create_live_features(history):
    """Create features from live price history"""
    if len(history) < 50:
        return None
    
    df = pd.DataFrame(list(history))
    df = create_features(df)
    
    return df.iloc[-1] if len(df) > 0 else None


def ml_predict(symbol):
    """Generate ML prediction with TP/SL"""
    history = state[symbol]['history']
    price = state[symbol]['price']
    
    stats['predictions_made'] += 1
    
    # Check if models loaded
    if ml_state['rf_model'] is None:
        return {
            'signal': 'LOADING', 'prob': 0, 'conf': 0,
            'reason': ml_state['status'], 'color': '#666666'
        }
    
    # Check data sufficiency
    if len(history) < 50:
        return {
            'signal': 'SYNCING', 'prob': 0, 'conf': 0,
            'reason': f'Need {50 - len(history)} more data points', 'color': '#666666'
        }
    
    try:
        features = create_live_features(history)
        if features is None:
            return {'signal': 'PROCESSING', 'prob': 0, 'conf': 0, 'reason': 'Building features', 'color': '#666666'}
        
        # Get feature values
        X = []
        for col in FEATURE_COLS:
            val = features.get(col, 0)
            if pd.isna(val) or np.isinf(val):
                val = 0
            X.append(val)
        X = np.array([X])
        
        # Ensemble prediction
        prob_rf = ml_state['rf_model'].predict_proba(X)[0][1]
        prob_gb = ml_state['gb_model'].predict_proba(X)[0][1]
        prob = (prob_rf + prob_gb) / 2
        
        # Generate signal
        if prob >= 0.65:
            signal, color = 'STRONG BUY', '#00ff88'
            conf = int(prob * 100)
        elif prob >= CONFIG['MIN_CONFIDENCE']:
            signal, color = 'BUY', '#00cc66'
            conf = int(prob * 100)
        elif prob <= 0.35:
            signal, color = 'STRONG SELL', '#ff4444'
            conf = int((1 - prob) * 100)
        elif prob <= 1 - CONFIG['MIN_CONFIDENCE']:
            signal, color = 'SELL', '#ff6666'
            conf = int((1 - prob) * 100)
        else:
            signal, color = 'HOLD', '#888888'
            conf = 50
        
        # Calculate TP/SL
        atr = features.get('atr', price * 0.01)
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.01
        
        fee_pct = CONFIG['FEE_RATE'] * 100  # 0.1%
        
        if 'BUY' in signal:
            entry = price
            sl = price - (atr * CONFIG['SL_ATR_MULT'])
            tp = price + (atr * CONFIG['TP_ATR_MULT'])
            risk = (price - sl) / price * 100
            reward = (tp - price) / price * 100
        elif 'SELL' in signal:
            entry = price
            sl = price + (atr * CONFIG['SL_ATR_MULT'])
            tp = price - (atr * CONFIG['TP_ATR_MULT'])
            risk = (sl - price) / price * 100
            reward = (price - tp) / price * 100
        else:
            entry = sl = tp = price
            risk = reward = 0
        
        # Indicators for display
        rsi = features.get('rsi', 50)
        macd = features.get('macd_hist', 0)
        atr_pct = features.get('atr_pct', 0)
        
        return {
            'signal': signal,
            'prob': round(prob, 3),
            'conf': conf,
            'color': color,
            'reason': f'ML: {prob*100:.0f}% | Model: {ml_state["train_accuracy"]:.0f}%',
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'risk': risk,
            'reward': reward,
            'fee': fee_pct,
            'ind': {
                'rsi': float(rsi) if not pd.isna(rsi) else 50,
                'macd': float(macd) if not pd.isna(macd) else 0,
                'atr': float(atr_pct) if not pd.isna(atr_pct) else 0
            }
        }
    except Exception as e:
        return {'signal': 'ERROR', 'prob': 0, 'conf': 0, 'reason': str(e)[:40], 'color': '#ff4444'}


def open_paper_trade(symbol, signal, entry, sl, tp):
    """Open a paper trade"""
    if paper_trades['active'][symbol] is not None:
        return False  # Already have active trade
    
    trade = {
        'symbol': symbol,
        'side': 'BUY' if 'BUY' in signal else 'SELL',
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'size': 1000,  # $1000 per trade
        'open_time': datetime.now().isoformat(),
        'signal': signal
    }
    
    paper_trades['active'][symbol] = trade
    print(f"📈 Paper trade opened: {symbol} {trade['side']} @ ${entry:.2f}")
    return True


# === HTML TEMPLATE ===
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>TRADING TERMINAL PRO</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #000;
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .header {
            background: #050505;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1a1a1a;
        }
        .logo { color: #fff; font-size: 1em; font-weight: 600; }
        .badge { 
            background: linear-gradient(90deg, #00ff88, #00aa55);
            color: #000; padding: 2px 8px; border-radius: 3px;
            font-size: 0.65em; font-weight: bold; margin-left: 8px;
        }
        .badge.training { background: #ff8800; }
        #clock { font-size: 1.5em; color: #00ff88; }
        .stats-bar { display: flex; gap: 20px; font-size: 0.75em; }
        .stat { text-align: center; }
        .stat-label { color: #555; font-size: 0.8em; }
        .stat-value { color: #00ff88; }
        .stat-value.loss { color: #ff4444; }
        
        .main {
            display: grid;
            grid-template-columns: 1fr 1fr 300px;
            gap: 1px;
            background: #111;
            height: calc(100vh - 45px);
        }
        
        .card {
            background: #050505;
            padding: 15px;
            display: flex;
            flex-direction: column;
        }
        
        .asset-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 10px;
            border-bottom: 1px solid #1a1a1a;
        }
        .asset-name { font-size: 0.85em; color: #fff; font-weight: 600; }
        .live-dot { width: 6px; height: 6px; border-radius: 50%; background: #00ff88; animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        
        .price-section { text-align: center; padding: 15px 0; }
        .price { font-size: 2.2em; font-weight: 300; }
        .price.up { color: #00ff88; }
        .price.down { color: #ff4444; }
        .change { font-size: 0.85em; margin-top: 3px; }
        
        .indicators {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
            padding: 10px 0;
            border-top: 1px solid #111;
            border-bottom: 1px solid #111;
        }
        .ind { text-align: center; }
        .ind-label { font-size: 0.6em; color: #444; }
        .ind-value { font-size: 0.75em; }
        
        .prediction {
            background: #0a0a0a;
            border: 1px solid #1a1a1a;
            border-radius: 4px;
            padding: 12px;
            text-align: center;
            margin: 10px 0;
        }
        .pred-label { font-size: 0.55em; color: #00ff88; text-transform: uppercase; letter-spacing: 2px; }
        .pred-signal { font-size: 1.4em; font-weight: 600; margin: 6px 0; }
        .pred-conf { font-size: 0.7em; color: #666; }
        .pred-reason { font-size: 0.6em; color: #444; margin-top: 3px; }
        
        .levels {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 5px;
            margin-top: 10px;
        }
        .level { text-align: center; padding: 8px; background: #080808; border-radius: 3px; }
        .level-label { font-size: 0.5em; color: #444; text-transform: uppercase; }
        .level-value { font-size: 0.7em; }
        
        .trade-btn {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            border: none;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.8em;
        }
        .trade-btn.buy { background: #00ff88; color: #000; }
        .trade-btn.sell { background: #ff4444; color: #fff; }
        .trade-btn.hold { background: #333; color: #888; cursor: not-allowed; }
        
        .risk-info { font-size: 0.6em; color: #555; margin-top: 5px; text-align: center; }
        
        .sidebar { overflow-y: auto; }
        .sidebar-section { padding: 15px; border-bottom: 1px solid #111; }
        .section-title { font-size: 0.7em; color: #555; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
        
        .trade-item { padding: 8px 0; border-bottom: 1px solid #111; font-size: 0.75em; }
        .trade-item.win { border-left: 2px solid #00ff88; padding-left: 8px; }
        .trade-item.loss { border-left: 2px solid #ff4444; padding-left: 8px; }
        
        .news-item { padding: 10px 0; border-bottom: 1px solid #111; }
        .news-title { font-size: 0.75em; color: #888; line-height: 1.3; }
        .news-meta { font-size: 0.6em; color: #444; margin-top: 3px; }
        
        .model-status { 
            background: #0a0a0a; 
            padding: 10px; 
            border-radius: 4px; 
            font-size: 0.7em;
            margin-top: 10px;
        }
        .model-status .label { color: #555; }
        .model-status .value { color: #00ff88; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            TRADING TERMINAL
            <span class="badge" id="model-badge">ML READY</span>
        </div>
        <div id="clock">--:--:--</div>
        <div class="stats-bar">
            <div class="stat" title="Total paper trades taken"><div class="stat-label">Trades</div><div class="stat-value" id="total-trades">0</div></div>
            <div class="stat" title="Percentage of winning trades"><div class="stat-label">Win Rate</div><div class="stat-value" id="win-rate">--%</div></div>
            <div class="stat" title="Total profit/loss from paper trades"><div class="stat-label">PnL</div><div class="stat-value" id="total-pnl">$0.00</div></div>
            <div class="stat" title="Network latency to server"><div class="stat-label">Latency</div><div class="stat-value" id="latency">--</div></div>
            <div class="stat" title="Auto-trading is ON (≥65% confidence)"><div class="stat-label">Auto</div><div class="stat-value" style="color:#00ff88">ON</div></div>
        </div>
    </div>
    
    <div class="main">
        <div class="card" id="btc"></div>
        <div class="card" id="paxg"></div>
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="section-title">📊 Model Status</div>
                <div class="model-status">
                    <div title="Training accuracy - how well model learned patterns"><span class="label">Status:</span> <span class="value" id="ml-status">Loading...</span></div>
                    <div title="When model was last retrained"><span class="label">Last Train:</span> <span class="value" id="last-train">--</span></div>
                    <div title="Next automatic retraining"><span class="label">Next Train:</span> <span class="value" id="next-train">--</span></div>
                </div>
                <div class="model-status" style="margin-top:8px;font-size:0.65em;color:#666">
                    ℹ️ Hover over any stat for explanation
                </div>
            </div>
            <div class="sidebar-section">
                <div class="section-title">📈 Recent Trades</div>
                <div id="trades">No trades yet</div>
            </div>
            <div class="sidebar-section">
                <div class="section-title">📰 News</div>
                <div id="news">Loading...</div>
            </div>
        </div>
    </div>
    
    <script>
        setInterval(() => {
            document.getElementById('clock').textContent = new Date().toTimeString().split(' ')[0];
        }, 100);
        
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onclose = () => document.getElementById('model-badge').textContent = 'OFFLINE';
        
        ws.onmessage = (e) => {
            const d = JSON.parse(e.data);
            document.getElementById('latency').textContent = (Date.now() - d.ts) + 'ms';
            
            // Update stats
            document.getElementById('total-trades').textContent = d.stats.total_trades;
            const wr = d.stats.total_trades > 0 ? (d.stats.wins / d.stats.total_trades * 100).toFixed(1) : 0;
            document.getElementById('win-rate').textContent = wr + '%';
            document.getElementById('win-rate').className = 'stat-value' + (wr >= 50 ? '' : ' loss');
            document.getElementById('total-pnl').textContent = '$' + d.stats.total_pnl.toFixed(2);
            document.getElementById('total-pnl').className = 'stat-value' + (d.stats.total_pnl >= 0 ? '' : ' loss');
            
            // Model status
            document.getElementById('ml-status').textContent = d.ml_status;
            document.getElementById('last-train').textContent = d.last_train || '--';
            document.getElementById('next-train').textContent = d.next_train || '--';
            document.getElementById('model-badge').textContent = d.ml_status.includes('Ready') ? 'ML ' + d.ml_status.match(/\\d+/)?.[0] + '%' : d.ml_status;
            document.getElementById('model-badge').className = 'badge' + (d.ml_status.includes('Training') ? ' training' : '');
            
            render('btc', 'BITCOIN (BTC/USD)', d.btc, 'BTCUSDT');
            render('paxg', 'GOLD (PAXG/USD)', d.paxg, 'PAXGUSDT');
            renderTrades(d.recent_trades);
            renderNews(d.news);
        };
        
        function render(id, name, d, symbol) {
            const up = d.change >= 0;
            
            let ind = '';
            if (d.ind) {
                ind = `
                    <div class="indicators">
                        <div class="ind"><div class="ind-label">RSI</div><div class="ind-value" style="color:${d.ind.rsi<30?'#00ff88':d.ind.rsi>70?'#ff4444':'#888'}">${d.ind.rsi?.toFixed(1)||'--'}</div></div>
                        <div class="ind"><div class="ind-label">MACD</div><div class="ind-value" style="color:${d.ind.macd>0?'#00ff88':'#ff4444'}">${d.ind.macd?.toFixed(2)||'--'}</div></div>
                        <div class="ind"><div class="ind-label">ATR%</div><div class="ind-value">${d.ind.atr?.toFixed(2)||'--'}%</div></div>
                    </div>
                `;
            }
            
            let levels = '';
            let btn = '<button class="trade-btn hold" disabled>WAITING</button>';
            
            if (d.signal?.includes('BUY') || d.signal?.includes('SELL')) {
                levels = `
                    <div class="levels">
                        <div class="level"><div class="level-label">Entry</div><div class="level-value" style="color:#fff">$${d.entry?.toFixed(2)||'--'}</div></div>
                        <div class="level"><div class="level-label">Stop Loss</div><div class="level-value" style="color:#ff4444">$${d.sl?.toFixed(2)||'--'}</div></div>
                        <div class="level"><div class="level-label">Take Profit</div><div class="level-value" style="color:#00ff88">$${d.tp?.toFixed(2)||'--'}</div></div>
                    </div>
                    <div class="risk-info">Risk: ${d.risk?.toFixed(2)||0}% | Reward: ${d.reward?.toFixed(2)||0}% | Fee: ${d.fee||0.1}%</div>
                `;
                const btnClass = d.signal.includes('BUY') ? 'buy' : 'sell';
                btn = `<button class="trade-btn ${btnClass}" onclick="openTrade('${symbol}', '${d.signal}', ${d.entry}, ${d.sl}, ${d.tp})">PAPER ${d.signal}</button>`;
            }
            
            document.getElementById(id).innerHTML = `
                <div class="asset-header">
                    <span class="asset-name">${name}</span>
                    <div class="live-dot"></div>
                </div>
                <div class="price-section">
                    <div class="price ${up?'up':'down'}">$${d.price?.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})||'--'}</div>
                    <div class="change" style="color:${up?'#00ff88':'#ff4444'}">${up?'▲':'▼'} ${Math.abs(d.change||0).toFixed(2)}%</div>
                </div>
                ${ind}
                <div class="prediction">
                    <div class="pred-label">ML Prediction</div>
                    <div class="pred-signal" style="color:${d.color||'#888'}">${d.signal||'LOADING'}</div>
                    <div class="pred-conf">${d.conf||0}% confidence</div>
                    <div class="pred-reason">${d.reason||''}</div>
                </div>
                ${levels}
                ${btn}
            `;
        }
        
        function renderTrades(trades) {
            if (!trades?.length) {
                document.getElementById('trades').innerHTML = '<div style="color:#444;font-size:0.7em">No trades yet</div>';
                return;
            }
            document.getElementById('trades').innerHTML = trades.slice(0, 10).map(t => 
                `<div class="trade-item ${t.status?.toLowerCase()}">
                    <div>${t.symbol} ${t.side} - ${t.close_reason}</div>
                    <div style="color:${t.pnl>=0?'#00ff88':'#ff4444'}">$${t.pnl?.toFixed(2)}</div>
                </div>`
            ).join('');
        }
        
        function renderNews(items) {
            if (!items?.length) return;
            document.getElementById('news').innerHTML = items.map(n => {
                const ago = Math.floor((Date.now()/1000 - n.time)/60);
                return `<div class="news-item"><div class="news-title">${n.title}</div><div class="news-meta">${n.source} • ${ago}m</div></div>`;
            }).join('');
        }
        
        function openTrade(symbol, signal, entry, sl, tp) {
            fetch(`/trade?symbol=${symbol}&signal=${signal}&entry=${entry}&sl=${sl}&tp=${tp}`, {method: 'POST'});
        }
    </script>
</body>
</html>
'''


# === LIFESPAN AND APP ===
@asynccontextmanager
async def lifespan(app):
    # Startup
    try:
        ml_state['rf_model'] = joblib.load(f'{MODEL_DIR}/ml_rf_model.pkl')
        ml_state['gb_model'] = joblib.load(f'{MODEL_DIR}/ml_gb_model.pkl')
        ml_state['features'] = FEATURE_COLS
        ml_state['last_train'] = datetime.now()
        ml_state['status'] = 'Ready (loaded)'
        print("✅ Loaded existing models")
    except:
        print("⚠️ No existing models, will train fresh")
    
    asyncio.create_task(fetch_prices())
    asyncio.create_task(fetch_news())
    asyncio.create_task(retrain_loop())
    
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get():
    return HTMLResponse(HTML)


@app.post("/trade")
async def trade(symbol: str, signal: str, entry: float, sl: float, tp: float):
    success = open_paper_trade(symbol, signal, entry, sl, tp)
    return {"success": success}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            btc = ml_predict('BTCUSDT')
            btc['price'] = state['BTCUSDT']['price']
            btc['change'] = state['BTCUSDT']['change']
            
            paxg = ml_predict('PAXGUSDT')
            paxg['price'] = state['PAXGUSDT']['price']
            paxg['change'] = state['PAXGUSDT']['change']
            
            # Auto-trading logic
            if AUTO_TRADE['ENABLED']:
                now = time.time()
                
                # Check BTC
                if btc.get('conf', 0) >= AUTO_TRADE['MIN_CONFIDENCE']:
                    if 'BUY' in btc.get('signal', '') or 'SELL' in btc.get('signal', ''):
                        if now - last_auto_trade['BTCUSDT'] > AUTO_TRADE['COOLDOWN']:
                            if paper_trades['active']['BTCUSDT'] is None:
                                open_paper_trade('BTCUSDT', btc['signal'], btc['entry'], btc['sl'], btc['tp'])
                                last_auto_trade['BTCUSDT'] = now
                
                # Check PAXG
                if paxg.get('conf', 0) >= AUTO_TRADE['MIN_CONFIDENCE']:
                    if 'BUY' in paxg.get('signal', '') or 'SELL' in paxg.get('signal', ''):
                        if now - last_auto_trade['PAXGUSDT'] > AUTO_TRADE['COOLDOWN']:
                            if paper_trades['active']['PAXGUSDT'] is None:
                                open_paper_trade('PAXGUSDT', paxg['signal'], paxg['entry'], paxg['sl'], paxg['tp'])
                                last_auto_trade['PAXGUSDT'] = now
            
            # Get recent trades
            recent_trades = paper_trades['BTCUSDT'][-5:] + paper_trades['PAXGUSDT'][-5:]
            recent_trades = sorted(recent_trades, key=lambda x: x.get('close_time', ''), reverse=True)[:10]
            
            # Calculate next train time
            next_train = '--'
            if ml_state['last_train']:
                next_dt = ml_state['last_train'] + timedelta(hours=CONFIG['RETRAIN_HOURS'])
                next_train = next_dt.strftime('%H:%M')
            
            await websocket.send_json({
                'ts': int(time.time() * 1000),
                'btc': btc,
                'paxg': paxg,
                'news': news,
                'stats': stats,
                'ml_status': ml_state['status'],
                'last_train': ml_state['last_train'].strftime('%H:%M') if ml_state['last_train'] else None,
                'next_train': next_train,
                'recent_trades': recent_trades
            })
            await asyncio.sleep(0.5)  # 500ms updates
    except WebSocketDisconnect:
        pass




if __name__ == "__main__":
    print("="*60)
    print("🚀 PROFESSIONAL TRADING TERMINAL v1.0")
    print("="*60)
    print(f"   Features:")
    print(f"   ✅ Live BTC + Gold prices")
    print(f"   ✅ ML predictions (RF + GB ensemble)")
    print(f"   ✅ Auto-retrain every {CONFIG['RETRAIN_HOURS']} hours")
    print(f"   ✅ Paper trading with PnL tracking")
    print(f"   ✅ Trade logging to {LOG_DIR}")
    print(f"   ✅ Fees: {CONFIG['FEE_RATE']*100}% per trade")
    print(f"   ✅ Risk: {CONFIG['RISK_PERCENT']*100}% per trade")
    print(f"   📡 URL: http://localhost:8000")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
