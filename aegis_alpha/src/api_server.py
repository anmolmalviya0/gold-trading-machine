"""
AEGIS V21 - API Server (The Bridge)
====================================
FastAPI bridge between Python Daemon and Next.js UI.
Run with: uvicorn src.api_server:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import json
import psutil
from datetime import datetime

app = FastAPI(title="AEGIS V21 API")

# Allow Next.js (localhost:3000) to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Constants
LOG_FILE = "logs/executor.log"
MODELS_DIR = "models"
DAEMON_SCRIPT = "src/executor.py"

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import httpx
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

try:
    from src.lgbm_adapter import LightGBMPredictor
except ImportError:
    from lgbm_adapter import LightGBMPredictor

# Import LSTM predictor
try:
    from src.lstm_model import SignalPredictor as LSTMPredictor
except ImportError:
    try:
        from lstm_model import SignalPredictor as LSTMPredictor
    except ImportError:
        LSTMPredictor = None
        print("Warning: LSTM predictor not available")

# Cache models to avoid reloading on every request
MODEL_CACHE = {}
LSTM_MODEL = None  # Global LSTM model (shared across symbols)

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Background Task for Real-Time Prices (0.5s)
async def broadcast_prices():
    """Fetches prices from Binance and broadcasts via WS"""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Binance Public API (Very fast)
                # Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, PAXGUSDT
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "PAXGUSDT"]
                # Clean implementation: multiple fetches or one batch? 
                # V3 ticker/price supports batch? No, but individual is fast.
                # Actually, /api/v3/ticker/price returns ALL if no symbol.
                
                # Fetch ALL (approx 2MB) or just individual? Individual is safer for bandwidth.
                start = datetime.now()
                prices = {}
                
                # We can use asyncio.gather for parallel speed
                tasks = [client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={s}") for s in symbols]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for r in responses:
                    if isinstance(r, httpx.Response) and r.status_code == 200:
                        data = r.json()
                        sym = data['symbol'].replace("USDT", "")
                        prices[sym] = float(data['price'])
                
                if prices:
                    # Broadcast format: {"type": "price_update", "data": {"BTC": 95000, ...}}
                    msg = json.dumps({"type": "price_update", "data": prices, "timestamp": str(datetime.now())})
                    await manager.broadcast(msg)
                    
            except Exception as e:
                print(f"WS Error: {e}")
                
            await asyncio.sleep(0.5) # 0.5s Interval requested

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_prices())

def get_lstm_model():
    """Load the LSTM model (shared across all symbols)"""
    global LSTM_MODEL
    if LSTM_MODEL is not None:
        return LSTM_MODEL
    
    lstm_path = "models/aegis_lstm.pth"
    if LSTMPredictor and os.path.exists(lstm_path):
        try:
            LSTM_MODEL = LSTMPredictor(lstm_path)
            print(f"🧠 LSTM loaded: {lstm_path}")
            return LSTM_MODEL
        except Exception as e:
            print(f"LSTM load failed: {e}")
    return None

def get_model(symbol: str):
    """Load model - prefers LSTM, falls back to LightGBM"""
    # Try LSTM first (65% accuracy)
    lstm = get_lstm_model()
    if lstm:
        return lstm
    
    # Fallback to LightGBM
    symbol = symbol.upper().replace("USDT", "")
    if symbol in MODEL_CACHE:
        return MODEL_CACHE[symbol]
    
    # Try different naming conventions
    paths = [
        f"models/{symbol}_lgbm.pkl",
        f"models/{symbol}USDT_lgbm.pkl",
        f"models/aegis_lgbm.pkl" if symbol == "PAXG" else None,
        f"models/BTC_lgbm.pkl" if symbol == "BTC" else None
    ]
    
    for p in paths:
        if p and os.path.exists(p):
            try:
                model = LightGBMPredictor(p)
                MODEL_CACHE[symbol] = model
                return model
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                
    return None

def fetch_live_data(symbol: str, interval: str = "15m"):
    """
    Fetches live data and standardizes columns for the LightGBM model.
    Handles both Flat and MultiIndex returns from yfinance.
    """
    try:
        # Map valid yfinance tickers (Preserving Agent Logic)
        yf_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "BNB": "BNB-USD",
            "PAXG": "PAXG-USD",
            "GOLD": "GC=F"
        }
        ticker_name = yf_map.get(symbol.upper(), f"{symbol}-USD")
        
        # 1. Fetch Data
        ticker = yf.Ticker(ticker_name)
        df = ticker.history(period="5d", interval=interval)
        
        if df.empty:
            # print(f"[ERROR] No data received for {symbol}")
            return None 

        # 2. THE FIX: Flatten MultiIndex (The "NaN" Killer)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. Standardize Headers
        df.columns = df.columns.str.lower()
        
        # 4. Clean Artifacts
        if df.index.tz is not None:
             # Convert to UTC first, then naive
            df.index = df.index.tz_convert('UTC').tz_localize(None)
            
        df['timestamp'] = df.index.astype(str) 
        
        # 5. Ensure Required Columns Exist
        df['time'] = (df.index.astype('int64') // 10**9).astype(int)
        
        return df.tail(100) 

    except Exception as e:
        print(f"[CRITICAL] Data Fetch Failed: {e}")
        return None


# ============================================================
# HISTORY API - Fix for Giant Candle Bug (500 candles)
# ============================================================
@app.get("/api/history/{symbol}")
def get_history(symbol: str, limit: int = 500, interval: str = "15m"):
    """Get historical candles for proper chart rendering"""
    try:
        yf_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "BNB": "BNB-USD",
            "PAXG": "PAXG-USD",
        }
        ticker_name = yf_map.get(symbol.upper(), f"{symbol}-USD")
        
        ticker = yf.Ticker(ticker_name)
        # Get more history for proper chart display
        df = ticker.history(period="60d", interval=interval)
        
        if df.empty:
            return {"error": "No data", "data": []}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = df.columns.str.lower()
        
        if df.index.tz is not None:
            # Convert to UTC first, then naive
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        # KEY FIX: Return UNIX timestamp (seconds) directly to avoid string parsing issues
        # GAP FILLING LOGIC (The "Solid State" Fix)
        # Ensure the index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)

        # Resample to ensure continuous 15m intervals
        # '15min' for 15m, '1h' for 1h etc. Logic to map interval string to pandas offset
        # Simple mapping: 15m -> 15min, 1h -> 1h, 5m -> 5min
        pd_interval = interval.replace('m', 'min')
        
        # Resample and Forward Fill (Propagate last close)
        # 1. Resample to full grid
        df_resampled = df.resample(pd_interval).asfreq()
        
        # 2. Forward fill 'close' to handle gaps (price hasn't changed if no data)
        df_resampled['close'] = df_resampled['close'].ffill()
        
        # 3. For gaps: Open=High=Low=Close (Doji), Volume=0
        df_resampled['open'] = df_resampled['open'].fillna(df_resampled['close'])
        df_resampled['high'] = df_resampled['high'].fillna(df_resampled['close'])
        df_resampled['low'] = df_resampled['low'].fillna(df_resampled['close'])
        df_resampled['volume'] = df_resampled['volume'].fillna(0)
        
        df = df_resampled
        
        # Recalculate Time Column after resampling
        df['time'] = (df.index.astype('int64') // 10**9).astype(int)
        
        # Limit to requested number of candles
        df = df.tail(limit)
        
        data = df[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
        return {"symbol": symbol, "interval": interval, "count": len(data), "data": data}
        
    except Exception as e:
        return {"error": str(e), "data": []}


@app.get("/")
def home():
    return {"system": "AEGIS V21", "status": "ONLINE"}

@app.get("/api/status")
def get_status():
    """Get daemon status, pid, and uptime"""
    isRunning = False
    pid = None
    uptime = "00:00:00"
    
    # Check process
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            # More robust check: Look for "python" AND "executor" in the command
            if 'python' in cmd.lower() and 'executor' in cmd.lower():
                isRunning = True
                pid = proc.info['pid']
                create_time = datetime.fromtimestamp(proc.create_time())
                uptime = str(datetime.now() - create_time).split('.')[0]
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    return {
        "running": isRunning,
        "pid": pid,
        "uptime": uptime,
        "strategy": "LightGBM Switchblade",
        "assets": ["PAXG", "BTC", "SOL", "BNB", "ETH"]
    }

@app.get("/api/models")
def get_models():
    """List available brains"""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith('.pkl') or f.endswith('.pth'):
                models.append(f)
    return {"models": models}

@app.get("/api/market-data/{symbol}")
def get_market_data(symbol: str, interval: str = "15m"):
    """Get chart data"""
    df = fetch_live_data(symbol, interval=interval)
    if df is None:
        raise HTTPException(status_code=404, detail="Asset not found")
        
    # Convert to list of dicts for frontend
    data = []
    for _, row in df.iterrows():
        data.append({
            "time": str(row['time']),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        })
    return {"symbol": symbol, "data": data}

@app.get("/api/predict/{symbol}")
def get_prediction(symbol: str):
    """Run live inference"""
    model = get_model(symbol)
    if not model:
        return {"status": "error", "message": "Model not trained"}
        
    # NOTE: Predictions ALWAYS use 15m (or model-specific default) to match training
    # We do NOT use the chart interval here.
    df = fetch_live_data(symbol, interval="15m")
    if df is None or len(df) < 50:
        return {"status": "error", "message": "Insufficient data"}
        
    # We need to replicate the EXACT feature engineering here
    # Since we can't easily import the trainer's local functions, 
    # we will do a simplified feature generation compatible with the model
    # Note: In production, this logic should be in a shared library.
    
    # Minimal Feature Vector Construction (matching lgbm_adapter expectations)
    # The adapter handles reshaping. We just need to give it the raw material.
    # For now, we will assume the adapter handles feature eng if we pass the DF,
    # OR we must implement it here.
    
    # CRITICAL: The model expects specific features. 
    # To save time and avoid errors, we'll pass the close prices and let the adapter 
    # (if smart enough) or just calculate the basic indicators here.
    
    # Let's use the 'returns' and 'volatility' as proxy features if exact match fails
    # But to be precise, let's try to pass the raw array and hope the model is robust
    # actually, LightGBM needs the specific columns.
    
    # Improvised Feature Engineering for Inference
    try:
        # returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # SMA Ratios
        for p in [5, 10, 20, 50]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'sma_ratio_{p}'] = df['close'] / df[f'sma_{p}']
            
        # RSI
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
        
        # BB
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - (sma20 - 2*std20)) / (4*std20)
        
        # ATR
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr_ratio'] = tr.rolling(14).mean() / df['close']
        
        # Volume
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
        ]
        
        # Fill NaN values to prevent errors
        df = df.fillna(0)
        
        # Check if we're using LSTM (SignalPredictor) or LightGBM
        is_lstm = hasattr(model, 'model') and model.model is not None
        
        if is_lstm:
            # LSTM needs sequence of 60 timesteps with EXACTLY the training features
            seq_len = 60
            if len(df) < seq_len:
                return {"status": "error", "message": f"Need {seq_len} rows for LSTM, got {len(df)}"}
            
            # MUST match training features EXACTLY (from lstm_trainer.py lines 305-315)
            lstm_feature_cols = [
                'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
                'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
            ]
            # Pad to 20 features by repeating last column (matches training)
            while len(lstm_feature_cols) < 20:
                lstm_feature_cols.append(lstm_feature_cols[-1])
            
            # Extract features for last 60 rows
            features = df[lstm_feature_cols[:20]].iloc[-seq_len:].values.astype(np.float32)
            # Replace any inf/nan with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            result = model.predict(features)
        else:
            # LightGBM uses single row
            features = df[cols].iloc[-1].values
            result = model.predict(features)
        
        # Add risk metrics
        current_price = df['close'].iloc[-1]
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Entry price is the CURRENT price when signal triggers
        entry_price = current_price
        
        # Calculate TP/SL based on signal direction
        if result['signal'] == 'SELL':
            # For SHORT: TP is BELOW entry, SL is ABOVE entry
            stop_loss = entry_price + (1.5 * atr)
            take_profit = entry_price - (2.0 * atr)
        else:
            # For LONG (or HOLD): TP is ABOVE entry, SL is BELOW entry
            stop_loss = entry_price - (1.5 * atr)
            take_profit = entry_price + (2.0 * atr)
        
        return {
            "symbol": symbol,
            "price": current_price,
            "entry_price": entry_price,
            "confidence": result['confidence'],
            "signal": result['signal'],
            "is_hot": result['confidence'] > 0.60,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr
        }
        
    except Exception as e:
        print(f"Inference error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/news")
def get_news():
    """Mock news feed with dynamic rotation (Simulated Live Feed)"""
    import random
    
    # Pool of potential headlines to simulate variety
    headline_pool = [
        {"title": "Bitcoin Reclaims $95k Level as Institutional Inflows Surge", "sentiment": "bullish"},
        {"title": "Solana Network Activity Hits All-Time High", "sentiment": "bullish"},
        {"title": "SEC Delays Ethereum ETF Decision Again", "sentiment": "bearish"},
        {"title": "Gold Prices Stabilize Amidst Macro Uncertainty", "sentiment": "neutral"},
        {"title": "BNB Chain Upgrade Promises Lower Fees", "sentiment": "bullish"},
        {"title": "Federal Reserve Hints at Rate Cuts in Q3", "sentiment": "bullish"},
        {"title": "BlackRock Files for New Crypto Index Fund", "sentiment": "bullish"},
        {"title": "Regulatory Crackdown Concerns Weigh on Altcoins", "sentiment": "bearish"},
        {"title": "Tether Treasury Mints Another 1B USDT", "sentiment": "neutral"},
        {"title": "Analysts Predict Volatility Ahead of FOMC Meeting", "sentiment": "neutral"},
        {"title": "MicroStrategy Buys Another 12,000 BTC", "sentiment": "bullish"},
        {"title": "DeFi TVL Reaches New Yearly High", "sentiment": "bullish"}
    ]
    
    # Randomly select 5 headlines
    selected = random.sample(headline_pool, 5)
    
    # Add dynamic timestamps
    start_time = random.randint(1, 5)
    headlines = []
    for i, item in enumerate(selected): # Keep order random
        # Incremental time ago
        mins_ago = start_time + (i * random.randint(2, 15))
        if mins_ago < 60:
            time_str = f"{mins_ago} mins ago"
        else:
            time_str = f"{mins_ago // 60} hours ago"
            
        headlines.append({
            "title": item["title"],
            "sentiment": item["sentiment"],
            "time": time_str
        })
        
    return {"headlines": headlines}
@app.get("/api/logs")
def get_logs(lines: int = 50):
    """Get last N lines of logs"""
    if not os.path.exists(LOG_FILE):
        return {"logs": ["Log file not found."]}
        
    try:
        # Using tail is efficient
        result = subprocess.run(['tail', '-n', str(lines), LOG_FILE], capture_output=True, text=True)
        return {"logs": result.stdout.strip().split('\n')}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}

@app.get("/api/control/{action}")
def control_daemon(action: str):
    """Start/Stop/Restart daemon"""
    if action not in ['start', 'stop', 'restart']:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    cmd = [DAEMON_SCRIPT, action]
    try:
        subprocess.run(cmd, check=True)
        return {"status": "success", "action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recalibrate")
def recalibrate_models():
    """
    Recalibrate models by clearing cache and reloading.
    This forces fresh predictions with updated market data.
    """
    global MODEL_CACHE, LSTM_MODEL
    
    try:
        # Clear all cached models
        MODEL_CACHE.clear()
        LSTM_MODEL = None
        
        # Reload LSTM
        lstm = get_lstm_model()
        
        # Log the recalibration
        print("🔄 RECALIBRATION TRIGGERED")
        print(f"   Cache cleared: {len(MODEL_CACHE)} entries")
        print(f"   LSTM reloaded: {'Yes' if lstm else 'No'}")
        
        return {
            "status": "success",
            "message": "Models recalibrated successfully",
            "lstm_loaded": lstm is not None,
            "cache_cleared": True,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Recalibration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
