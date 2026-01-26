"""
AEGIS V21 - THE MASTER BRIDGE
==============================
Final Institutional Grade API Server.
Equipped with:
1. High-Frequency Price Broadcaster (0.5s)
2. Forensic WebSocket Manager (Mobile Ready)
3. Glass Box AI Inference
4. News & History Omni-Feed
"""
import os
import sys
import json
import psutil
import asyncio
import httpx
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import List, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add parent dir to path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.lgbm_adapter import LightGBMPredictor
except ImportError:
    from lgbm_adapter import LightGBMPredictor

app = FastAPI(title="AEGIS V21 MASTER BRIDGE")

# Universal Access Rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
LOG_FILE = "logs/executor.log"
DAEMON_SCRIPT = "src/executor.py"
MODELS_DIR = "models"
MODEL_CACHE = {}
startup_time = datetime.now()

# ------------------------------------------------------------
# FORENSIC WEBSOCKET MANAGER
# ------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 WS CONNECTED: {websocket.client} | Active Nodes: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 WS DISCONNECTED | Active Nodes: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# ------------------------------------------------------------
# HIGH-FREQUENCY BROADCASTER
# ------------------------------------------------------------
async def broadcast_prices():
    """Fetches fast prices from Binance Public API and broadcasts via WS"""
    async with httpx.AsyncClient() as client:
        print("🚀 Price Broadcaster System: ONLINE (0.5s Heartbeat)")
        while True:
            try:
                symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "PAXGUSDT"]
                tasks = [client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={s}", timeout=2.0) for s in symbols]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                prices = {}
                for r in responses:
                    if isinstance(r, httpx.Response) and r.status_code == 200:
                        data = r.json()
                        sym = data['symbol'].replace("USDT", "")
                        prices[sym] = float(data['price'])
                
                if prices:
                    # Only log if listeners exist to prevent log spam
                    if len(manager.active_connections) > 0:
                        print(f"📡 Pulse: {len(manager.active_connections)} nodes | BTC: {prices.get('BTC')}")
                    
                    msg = json.dumps({"type": "price_update", "data": prices, "timestamp": str(datetime.now())})
                    await manager.broadcast(msg)
                    
            except Exception as e:
                print(f"⚠️ Broadcast Error: {e}")
                
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_prices())

# ------------------------------------------------------------
# CORE ENDPOINTS
# ------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    executor_found = any(DAEMON_SCRIPT in " ".join(proc.info['cmdline'] or []) for proc in psutil.process_iter(['cmdline']))
    return {
        "api": "online",
        "executor": "online" if executor_found else "offline",
        "uptime": str(datetime.now() - startup_time),
        "listeners": len(manager.active_connections)
    }

@app.get("/api/history/{symbol}")
async def get_history(symbol: str, interval: str = "15m", limit: int = 500):
    """Serve historical data for the Pro Visor"""
    print(f"📥 History Request: {symbol} | Int: {interval} | Lim: {limit}")
    yf_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD", "PAXG": "PAXG-USD"}
    ticker_sym = yf_map.get(symbol, f"{symbol}-USD")
    
    # Map timeframe for yfinance
    period = "1d"
    if interval == "1h": period = "7d"
    elif interval == "1d": period = "1mo"
    elif interval == "15m": period = "2d"
    
    try:
        # Avoid async conflicts with yfinance
        df = await asyncio.to_thread(yf.download, ticker_sym, period=period, interval=interval, progress=False)
        # Formatting
        data = []
        for idx, row in df.iterrows():
            data.append({
                "time": int(idx.timestamp()),
                "open": float(row['open'].iloc[0] if isinstance(row['open'], pd.Series) else row['open']),
                "high": float(row['high'].iloc[0] if isinstance(row['high'], pd.Series) else row['high']),
                "low": float(row['low'].iloc[0] if isinstance(row['low'], pd.Series) else row['low']),
                "close": float(row['close'].iloc[0] if isinstance(row['close'], pd.Series) else row['close']),
                "volume": float(row['volume'].iloc[0] if isinstance(row['volume'], pd.Series) else row['volume'])
            })
        return {"symbol": symbol, "data": data}
    except Exception as e:
        print(f"❌ History Error: {e}")
        return {"status": "error", "message": str(e)}

from quant_utils import prepare_features

@app.get("/api/predict/{symbol}")
async def get_prediction(symbol: str, interval: str = "15m"):
    """Run live inference using the Switchblade Model"""
    try:
        # 1. Resolve Model
        model_path = os.path.join(MODELS_DIR, f"{symbol}_lgbm.pkl")
        if symbol not in MODEL_CACHE:
            if os.path.exists(model_path):
                MODEL_CACHE[symbol] = joblib.load(model_path)
            else:
                return {"status": "error", "message": "Model not found"}
        
        # 2. Fetch Fresh Data (Need 100 candles for SMA_50)
        yf_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD", "PAXG": "PAXG-USD"}
        ticker_sym = yf_map.get(symbol, f"{symbol}-USD")
        
        # Fetch slightly more than needed to ensure indicators calculate
        df = await asyncio.to_thread(yf.download, ticker_sym, period="2d", interval=interval, progress=False)
        if df is None or df.empty:
            return {"status": "error", "message": "Failed to fetch data for prediction"}
        
        # 3. Neural Math (Sovereign Engine)
        features = prepare_features(df)
        
        # 4. Inference
        model_wrapper = LightGBMPredictor(model_path) # Using wrapper for standard output
        result = model_wrapper.predict(features)
        
        return {
            "symbol": symbol,
            "price": float(df['close'].iloc[-1]),
            "confidence": result['confidence'],
            "signal": result['signal'],
            "reason": result['reason'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Prediction API Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/news")
async def get_news():
    """Institutional Intel Feed"""
    try:
        ticker = yf.Ticker("BTC-USD")
        raw_news = ticker.news[:5]
        headlines = [{"title": n['title'], "link": n['link'], "source": n['publisher'], "type": "neutral"} for n in raw_news]
        return {"headlines": headlines}
    except:
        return {"headlines": []}

@app.get("/api/logs")
async def get_logs(lines: int = 20):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return {"logs": f.readlines()[-lines:]}
    return {"logs": ["No logs found"]}

# ------------------------------------------------------------
# WEBSOCKET GATEWAY
# ------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
