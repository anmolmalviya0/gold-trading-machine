"""
TRADINGVIEW WEBHOOK INTEGRATION
===============================
Receives TradingView alerts and executes trades.

Features:
1. Webhook endpoint for TradingView alerts
2. Parse alert format (symbol, action, price)
3. Execute trades via broker adapter
4. Generate Pine Script for export

Usage:
    1. Add webhook URL to TradingView alert: http://your-server:8001/webhook
    2. Alert message format: {"symbol":"BTCUSDT","action":"buy","price":95000}
"""
import asyncio
from pathlib import Path
from datetime import datetime
import json
import hmac
import hashlib
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import sqlite3

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'tradingview_signals.db'

CONFIG = {
    'SECRET_KEY': 'your-webhook-secret-key-change-me',  # For webhook validation
    'MAX_SIGNALS_PER_MINUTE': 10,
    'SUPPORTED_SYMBOLS': ['BTCUSDT', 'PAXGUSDT', 'ETHUSDT', 'AAPL', 'TSLA', 'SPY'],
}

# === DATABASE ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT DEFAULT 'tradingview',
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL,
            quantity REAL,
            strategy TEXT,
            executed INTEGER DEFAULT 0,
            result TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS executed_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            broker TEXT,
            order_id TEXT,
            fill_price REAL,
            status TEXT,
            pnl REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# === SIGNAL PROCESSING ===
class SignalProcessor:
    """Process and execute signals from TradingView"""
    
    def __init__(self):
        self.recent_signals = []
        self.brokers = {}
    
    def validate_webhook(self, payload: dict, signature: str = None) -> bool:
        """Validate webhook signature"""
        if not signature:
            return True  # Allow unsigned for testing
        
        expected = hmac.new(
            CONFIG['SECRET_KEY'].encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    def rate_limit_check(self) -> bool:
        """Check if rate limit exceeded"""
        now = datetime.now()
        self.recent_signals = [s for s in self.recent_signals 
                              if (now - s).seconds < 60]
        
        if len(self.recent_signals) >= CONFIG['MAX_SIGNALS_PER_MINUTE']:
            return False
        
        self.recent_signals.append(now)
        return True
    
    def parse_alert(self, data: dict) -> dict:
        """Parse TradingView alert format"""
        # Support multiple formats
        
        # Format 1: Simple {"symbol":"BTC","action":"buy"}
        if 'symbol' in data and 'action' in data:
            return {
                'symbol': data['symbol'].upper().replace('/', ''),
                'action': data['action'].lower(),
                'price': data.get('price'),
                'quantity': data.get('quantity', 1),
                'strategy': data.get('strategy', 'tradingview'),
                'stop_loss': data.get('stop_loss'),
                'take_profit': data.get('take_profit'),
            }
        
        # Format 2: Pine Script {{strategy.order.action}}
        if 'strategy.order.action' in data:
            return {
                'symbol': data.get('ticker', 'UNKNOWN'),
                'action': data['strategy.order.action'].lower(),
                'price': data.get('strategy.order.price'),
                'quantity': data.get('strategy.order.contracts', 1),
                'strategy': data.get('strategy.name', 'pine'),
            }
        
        # Format 3: Text message parsing
        if 'message' in data:
            msg = data['message'].upper()
            action = 'buy' if 'BUY' in msg else ('sell' if 'SELL' in msg else None)
            
            # Extract symbol
            symbol = None
            for sym in CONFIG['SUPPORTED_SYMBOLS']:
                if sym in msg:
                    symbol = sym
                    break
            
            if action and symbol:
                return {
                    'symbol': symbol,
                    'action': action,
                    'price': None,
                    'quantity': 1,
                    'strategy': 'text_alert',
                }
        
        return None
    
    def log_signal(self, signal: dict):
        """Log signal to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, action, price, quantity, strategy)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            signal['symbol'],
            signal['action'],
            signal.get('price'),
            signal.get('quantity', 1),
            signal.get('strategy', 'tradingview')
        ))
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    
    async def execute_signal(self, signal: dict, signal_id: int):
        """Execute signal via broker"""
        print(f"📈 Executing: {signal['action'].upper()} {signal['symbol']}")
        
        # Here you would integrate with your broker
        # For now, simulate execution
        await asyncio.sleep(0.1)
        
        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE signals SET executed = 1, result = ? WHERE id = ?",
            ('simulated', signal_id)
        )
        cursor.execute('''
            INSERT INTO executed_trades (signal_id, broker, status)
            VALUES (?, ?, ?)
        ''', (signal_id, 'paper', 'filled'))
        conn.commit()
        conn.close()
        
        return {'status': 'executed', 'signal_id': signal_id}

processor = SignalProcessor()

# === PINE SCRIPT GENERATOR ===
def generate_pine_script(strategy_params: dict) -> str:
    """Generate Pine Script from strategy parameters"""
    
    rsi_oversold = strategy_params.get('rsi_oversold', 30)
    rsi_overbought = strategy_params.get('rsi_overbought', 70)
    sma_fast = strategy_params.get('sma_fast', 20)
    sma_slow = strategy_params.get('sma_slow', 50)
    take_profit_pct = strategy_params.get('take_profit_pct', 2.0)
    stop_loss_pct = strategy_params.get('stop_loss_pct', 1.0)
    
    script = f'''
//@version=5
strategy("AI Trading Bot Signal", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// === INPUTS ===
rsiLength = input.int(14, "RSI Length")
rsiOversold = input.int({rsi_oversold}, "RSI Oversold")
rsiOverbought = input.int({rsi_overbought}, "RSI Overbought")
smaFast = input.int({sma_fast}, "SMA Fast")
smaSlow = input.int({sma_slow}, "SMA Slow")
takeProfitPct = input.float({take_profit_pct}, "Take Profit %")
stopLossPct = input.float({stop_loss_pct}, "Stop Loss %")

// === INDICATORS ===
rsi = ta.rsi(close, rsiLength)
smaFastLine = ta.sma(close, smaFast)
smaSlowLine = ta.sma(close, smaSlow)

trend = smaFastLine > smaSlowLine ? 1 : smaFastLine < smaSlowLine ? -1 : 0

// === SIGNALS ===
bullish = rsi < rsiOversold and trend == -1  // Oversold in downtrend = mean reversion
bearish = rsi > rsiOverbought and trend == 1  // Overbought in uptrend = take profit

// === ENTRIES ===
if (bullish)
    strategy.entry("Long", strategy.long)
    alert("{{\\\"symbol\\\":\\\\"" + syminfo.ticker + "\\\\",\\\"action\\\":\\\"buy\\\",\\\"price\\\":" + str.tostring(close) + "}}", alert.freq_once_per_bar)

if (bearish)
    strategy.entry("Short", strategy.short)
    alert("{{\\\"symbol\\\":\\\\"" + syminfo.ticker + "\\\\",\\\"action\\\":\\\"sell\\\",\\\"price\\\":" + str.tostring(close) + "}}", alert.freq_once_per_bar)

// === EXITS ===
strategy.exit("TP/SL Long", "Long", profit=close * takeProfitPct / 100 / syminfo.mintick, loss=close * stopLossPct / 100 / syminfo.mintick)
strategy.exit("TP/SL Short", "Short", profit=close * takeProfitPct / 100 / syminfo.mintick, loss=close * stopLossPct / 100 / syminfo.mintick)

// === PLOTTING ===
plot(smaFastLine, "SMA Fast", color=color.blue)
plot(smaSlowLine, "SMA Slow", color=color.orange)
bgcolor(bullish ? color.new(color.green, 90) : bearish ? color.new(color.red, 90) : na)
'''
    return script

# === API ===
@asynccontextmanager
async def lifespan(app):
    init_db()
    print("🔗 TradingView Webhook Server ready")
    print("   Webhook URL: http://your-ip:8001/webhook")
    yield

app = FastAPI(lifespan=lifespan, title="TradingView Webhook Server")

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive TradingView webhook alerts"""
    
    # Rate limit check
    if not processor.rate_limit_check():
        raise HTTPException(429, "Rate limit exceeded")
    
    # Parse body
    try:
        data = await request.json()
    except:
        body = await request.body()
        try:
            data = json.loads(body.decode())
        except:
            data = {'message': body.decode()}
    
    # Validate webhook (optional signature check)
    signature = request.headers.get('X-Webhook-Signature')
    if not processor.validate_webhook(data, signature):
        raise HTTPException(401, "Invalid signature")
    
    # Parse alert
    signal = processor.parse_alert(data)
    if not signal:
        return JSONResponse({'status': 'ignored', 'reason': 'could not parse alert'})
    
    # Log signal
    signal_id = processor.log_signal(signal)
    
    # Execute in background
    background_tasks.add_task(processor.execute_signal, signal, signal_id)
    
    print(f"📥 Signal received: {signal['action'].upper()} {signal['symbol']}")
    
    return JSONResponse({
        'status': 'received',
        'signal_id': signal_id,
        'signal': signal
    })

@app.get("/signals")
async def get_signals(limit: int = 50):
    """Get recent signals"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, symbol, action, price, executed, result
        FROM signals ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        'id': r[0], 'timestamp': r[1], 'symbol': r[2],
        'action': r[3], 'price': r[4], 'executed': r[5], 'result': r[6]
    } for r in rows]

@app.get("/pine-script")
async def get_pine_script(
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    sma_fast: int = 20,
    sma_slow: int = 50,
    take_profit: float = 2.0,
    stop_loss: float = 1.0
):
    """Generate Pine Script with custom parameters"""
    script = generate_pine_script({
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'sma_fast': sma_fast,
        'sma_slow': sma_slow,
        'take_profit_pct': take_profit,
        'stop_loss_pct': stop_loss,
    })
    return HTMLResponse(f"<pre>{script}</pre>")

@app.get("/")
async def home():
    """Home page with instructions"""
    return HTMLResponse('''
    <html>
    <head><title>TradingView Webhook Server</title>
    <style>body{background:#0a0e14;color:#e6e6e6;font-family:system-ui;padding:20px;}
    h1{color:#00ff88;}code{background:#1e252e;padding:2px 6px;border-radius:4px;}
    pre{background:#1e252e;padding:15px;border-radius:8px;overflow-x:auto;}</style>
    </head>
    <body>
    <h1>🔗 TradingView Webhook Server</h1>
    <h2>Endpoints</h2>
    <ul>
        <li><code>POST /webhook</code> - Receive TradingView alerts</li>
        <li><code>GET /signals</code> - View recent signals</li>
        <li><code>GET /pine-script</code> - Generate Pine Script</li>
    </ul>
    <h2>Alert Format</h2>
    <pre>{"symbol":"BTCUSDT","action":"buy","price":95000}</pre>
    <h2>TradingView Setup</h2>
    <ol>
        <li>Create alert on TradingView</li>
        <li>Set Webhook URL: <code>http://your-ip:8001/webhook</code></li>
        <li>Alert message: <code>{"symbol":"{{ticker}}","action":"buy","price":{{close}}}</code></li>
    </ol>
    </body></html>
    ''')

if __name__ == "__main__":
    print("="*60)
    print("🔗 TRADINGVIEW WEBHOOK SERVER")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
