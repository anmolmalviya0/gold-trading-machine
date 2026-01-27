"""
TERMINAL - THE MASTER BRIDGE
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
from pathlib import Path
from typing import List, Dict
from notifier import sentinel
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add parent dir to path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "market_data"
LOG_DIR = BASE_DIR / "logs"
CALENDAR_FILE = DATA_DIR / "economic_calendar.json"

try:
    from src.lgbm_adapter import LightGBMPredictor
except ImportError:
    from lgbm_adapter import LightGBMPredictor

app = FastAPI(title="TERMINAL MASTER BRIDGE")

# Universal Access Rules (Hardened for V21)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Must be False if using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
LOG_FILE = "logs/executor.log"
IGNITION_FILE = "logs/.ignition_time"
DAEMON_SCRIPT = "src/executor.py"
MODEL_CACHE = {}
INTEL_BUFFER = {
    "sentiment_score": 0.0,
    "macro_bias": "NEUTRAL",
    "high_impact_events": [],
    "last_update": None
}

# THE ETERNAL IGNITION PROTOCOL
def get_ignition_time():
    """Retrieve or initialize the sovereign birth time of the system"""
    try:
        if os.path.exists(IGNITION_FILE):
            with open(IGNITION_FILE, "r") as f:
                return datetime.fromisoformat(f.read().strip())
    except:
        pass
    
    # Initialize if missing
    now = datetime.now()
    try:
        os.makedirs(os.path.dirname(IGNITION_FILE), exist_ok=True)
        with open(IGNITION_FILE, "w") as f:
            f.write(now.isoformat())
    except:
        pass
    return now

startup_time = get_ignition_time()

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
        # Iterate over a slice to allow safe removal of dead nodes
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

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
    sentinel.notify("IGNITION", "TERMINAL Master Bridge is ONLINE 🛰️", tags="satellite,zap", priority=4)
    asyncio.create_task(broadcast_prices())

# ------------------------------------------------------------
# CORE ENDPOINTS
# ------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """System Pulse Telemetry (Hardened)"""
    executor_found = any(DAEMON_SCRIPT in " ".join(proc.info['cmdline'] or []) for proc in psutil.process_iter(['cmdline']))
    return {
        "status": "online",
        "api": "online",
        "running": True, 
        "executor": "online" if executor_found else "offline",
        "uptime": str(datetime.now() - startup_time),
        "listeners": len(manager.active_connections),
        "pid": os.getpid()
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
        if df is None or df.empty: 
            return {"status": "error", "data": []}
            
        # NORMALIZE (Immediate & Robust)
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Formatting (Fast extraction)
        times = [int(t.timestamp()) for t in df.index]
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        vols = df['volume'].values
        
        data = []
        for i in range(len(times)):
            data.append({
                "time": times[i],
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(vols[i])
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
        # Map symbol "BTC" -> "BTCUSDT" for filename
        filesym = f"{symbol}USDT"
        model_path = os.path.join(MODELS_DIR, f"{filesym}_{interval}_lgbm.pkl")
        
        if not os.path.exists(model_path):
            # Fallback for old naming or slightly different format
            model_path = os.path.join(MODELS_DIR, f"{symbol}_{interval}_lgbm.pkl")
            
        if symbol not in MODEL_CACHE:
            if os.path.exists(model_path):
                MODEL_CACHE[symbol] = joblib.load(model_path)
            else:
                return {"status": "error", "message": f"Model not found: {model_path}"}
        
        # 2. Fetch Fresh Data (Need 100 candles for SMA_50)
        yf_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD", "PAXG": "PAXG-USD"}
        ticker_sym = yf_map.get(symbol, f"{symbol}-USD")
        
        cache_key = f"predict_df_{symbol}_{interval}"
        if cache_key in MODEL_CACHE:
            cached_df, timestamp = MODEL_CACHE[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=60):
                df = cached_df
            else:
                df = await asyncio.to_thread(yf.download, ticker_sym, period="2d", interval=interval, progress=False)
                MODEL_CACHE[cache_key] = (df, datetime.now())
        else:
            df = await asyncio.to_thread(yf.download, ticker_sym, period="2d", interval=interval, progress=False)
            MODEL_CACHE[cache_key] = (df, datetime.now())

        if df is None or df.empty:
            return {"status": "error", "message": "Failed to fetch data for prediction"}
        
        # NORMALIZE (Immediate & Robust)
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # 3. Neural Math (Sovereign Engine)
        features = prepare_features(df)
        
        # 4. Inference
        model_wrapper = LightGBMPredictor(model_path)
        result = model_wrapper.predict(features)
        
        return {
            "symbol": symbol,
            "price": float(df['close'].values[-1]),
            "confidence": result['confidence'],
            "signal": result['signal'],
            "reason": result['reason'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Prediction API Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/news")
async def get_news(symbol: str = None):
    """Institutional Intel Feed (Contextual Pivot)"""
    try:
        # 🐉 NASA-GRADE: Contextual Intel Pivot
        # If symbol is provided (e.g., BTC), prioritize it
        yf_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "PAXG": "PAXG-USD", "BNB": "BNB-USD"}
        
        main_ticker = yf_map.get(symbol)
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "PAXG-USD"]
        
        # If we have a specific symbol not in the primary list, add it to the scan
        if main_ticker and main_ticker not in tickers:
            tickers.insert(0, main_ticker)
        elif main_ticker:
            # Move priority ticker to front
            tickers.remove(main_ticker)
            tickers.insert(0, main_ticker)
            
        all_raw_news = []
        
        async def fetch_ticker_news(sym):
            try:
                t = yf.Ticker(sym)
                news_items = await asyncio.to_thread(lambda: t.news[:5])
                # Tag items with their origin
                for item in news_items:
                    item['_origin_asset'] = sym.split('-')[0]
                return news_items
            except:
                return []
        
        results = await asyncio.gather(*(fetch_ticker_news(s) for s in tickers))
        for res in results:
            all_raw_news.extend(res)
        
        if all_raw_news:
            headlines = []
            for n in all_raw_news:
                content = n.get('content', n)
                title = content.get('title', n.get('title', "Market Event"))
                link_obj = n.get('clickThroughUrl', {})
                link = link_obj.get('url', n.get('link', "#"))
                provider = n.get('provider', {})
                source = provider.get('displayName', n.get('publisher', "SYSTEM"))
                asset_tag = n.get('_origin_asset', "MARKET")
                
                display_time = datetime.now().strftime("%H:%M")
                original_timestamp = datetime.now().timestamp()
                minutes_ago = 0
                
                try:
                    p_date = content.get('pubDate', n.get('pubDate'))
                    if p_date:
                        dt_obj = None
                        if isinstance(p_date, str):
                            if 'T' in p_date and 'Z' in p_date:
                                dt_obj = datetime.fromisoformat(p_date.replace('Z', '+00:00'))
                            elif 'T' in p_date:
                                dt_obj = datetime.fromisoformat(p_date)
                        elif isinstance(p_date, (int, float)):
                            dt_obj = datetime.fromtimestamp(p_date)
                        
                        if dt_obj:
                            display_time = dt_obj.strftime("%H:%M")
                            original_timestamp = dt_obj.timestamp()
                            delta = datetime.now().timestamp() - original_timestamp
                            minutes_ago = max(0, int(delta // 60))
                except:
                    pass 
                
                # Multi-Parameter Sentiment Analysis (Keyword Based - Zero Latency)
                sentiment = "neutral"
                s_weight = 0
                bullish_k = ["increase", "surge", "gain", "high", "support", "rally", "launch", "all-time high", "growth", "bullish", "adoption"]
                bearish_k = ["drop", "falls", "crash", "hack", "ban", "regulatory", "dump", "selloff", "bearish", "fud", "scam"]
                
                title_lower = title.lower()
                if any(k in title_lower for k in bullish_k):
                    sentiment = "bullish"
                    s_weight = 1
                elif any(k in title_lower for k in bearish_k):
                    sentiment = "bearish"
                    s_weight = -1
                
                headlines.append({
                    "title": f"[{asset_tag}] {title}", 
                    "link": link, 
                    "source": source, 
                    "time": display_time,
                    "minutes_ago": minutes_ago,
                    "asset": asset_tag,
                    "sentiment": sentiment,
                    "s_weight": s_weight,
                    "original_timestamp": original_timestamp
                })
            
            # 🐉 NASA-GRADE Pivot Logic:
            # Sort Parameters: 
            # 1. Matches targeted asset vector (symbol)
            # 2. Sentiment Magnitude (Absolute s_weight) - Priority for market moving news
            # 3. Recency (timestamp)
            unique_news = []
            seen = set()
            for h in headlines:
                if h['title'] not in seen:
                    unique_news.append(h)
                    seen.add(h['title'])
            
            def sort_key(h):
                is_priority = 2 if symbol and h['asset'] == symbol else 0
                # Give high impact news (bullish/bearish) a small boost
                impact = 1 if h['s_weight'] != 0 else 0
                return (is_priority, impact, h['original_timestamp'])
                
            unique_news.sort(key=sort_key, reverse=True)
            return {"headlines": unique_news[:10]}
        
        # Fallback (Purged of legacy Terminal strings)
        current_time = datetime.now().strftime("%H:%M")
        return {"headlines": [
            {"title": "[SYSTEM] TERMINAL: Market Sentiment is Stable", "link": "#", "source": "SENTINEL_CORE", "time": current_time, "minutes_ago": 0, "asset": "SYSTEM", "sentiment": "neutral"},
            {"title": "[MARKET] Volatility Shield is active for Global Indices", "link": "#", "source": "SENTINEL", "time": current_time, "minutes_ago": 0, "asset": "SYSTEM", "sentiment": "neutral"}
        ]}
    except Exception as e:
        print(f"⚠️ News Feed Error: {e}")
        return {"headlines": [{"title": "Intel Feed Synchronizing...", "link": "#", "source": "SYSTEM", "time": "---", "sentiment": "neutral", "minutes_ago": 0, "asset": "SYSTEM"}]}

@app.get("/api/calendar")
async def get_calendar():
    """Economic Pulse Feed (Placeholder logic for high-impact events)"""
    try:
        now = datetime.now()
        # Simulated events for the Forensic Visor
        events = [
            {"time": (now + timedelta(minutes=45)).strftime("%H:%M"), "event": "USD Core Durable Goods Orders", "currency": "USD", "color": "#ef4444", "minutes": 45},
            {"time": (now + timedelta(minutes=120)).strftime("%H:%M"), "event": "FOMC Member Speech", "currency": "USD", "color": "#eab308", "minutes": 120},
            {"time": (now - timedelta(minutes=15)).strftime("%H:%M"), "event": "EUR Consumer Confidence", "currency": "EUR", "color": "#ef4444", "minutes": -15}
        ]
        return {"events": events}
    except Exception as e:
        print(f"⚠️ Calendar Error: {e}")
        return {"events": []}

@app.post("/api/intel/update")
async def update_intel(data: Dict):
    """Internal endpoint for Intel Scout to push sovereign intelligence"""
    global INTEL_BUFFER
    INTEL_BUFFER.update(data)
    # Broadcast to WS visors
    await manager.broadcast(json.dumps({
        "type": "intel_update",
        "data": INTEL_BUFFER
    }))
    return {"status": "success"}

@app.get("/api/intel")
async def get_intel():
    """Expose the Sovereign Intel Buffer to the Visor"""
    return INTEL_BUFFER

@app.post("/api/recalibrate")
async def recalibrate():
    """Recalibrate all LightGBM models via the Strategy Optimizer"""
    try:
        sentinel.notify("RECALIBRATION", "Initiating global model recalibration...", priority=3)
        # Execute the strategic recalibration in a non-blocking process
        os.system("python3 src/strategy_optimizer.py --mode force_recalibrate &")
        return {"status": "success", "message": "Recalibration sequence initiated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
    # Enforce 0.0.0.0 for external (iPhone) access
    print("🦅 MASTER BRIDGE: Binding to 0.0.0.0:8000 (Open for iPhone Discovery)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
