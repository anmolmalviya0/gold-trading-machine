"""
STRATEGY MARKETPLACE (3Commas-Style)
=====================================
A professional marketplace to:
1. List available trading strategies
2. Rank by ROI, Sharpe, Win Rate
3. Auto-deploy the best performers
4. Simulate "Social Trading" aspect

Usage:
    Run: uvicorn strategy_marketplace:app --port 8002
"""
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import random
import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import json

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'marketplace.db'

# === DATABASE ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            author TEXT NOT NULL,
            description TEXT,
            timeframe TEXT,
            asset TEXT,
            type TEXT, -- Scalping, Swing, Grid, HFT
            roi_30d REAL,
            win_rate REAL,
            sharpe REAL,
            drawdown REAL,
            copiers INTEGER,
            price REAL, -- Monthly fee or 0 for free
            deployed INTEGER DEFAULT 0,
            config JSON,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Seed data if empty
    cursor.execute('SELECT count(*) FROM strategies')
    if cursor.fetchone()[0] == 0:
        seed_strategies(cursor)
    
    conn.commit()
    conn.close()

def seed_strategies(cursor):
    strategies = [
        ('Institutional Sniper', 'QuantElite', 'High-frequency mean reversion on 1m/5m', '5m', 'BTC/USDT', 'HFT', 45.2, 72.5, 2.4, 0.5, 1240, 99.00, '{"rsi_low": 25, "rsi_high": 75}'),
        ('Golden Grid', 'GoldMaster', 'Conservative grid trading for PAXG', '1h', 'PAXG/USDT', 'Grid', 12.5, 98.0, 3.1, 8.2, 850, 49.00, '{"grid_lines": 20, "spacing": 0.5}'),
        ('BTC Trend Surfer', 'SatoshiN', 'Classic trend following with trailing stop', '4h', 'BTC/USDT', 'Swing', 128.4, 45.0, 1.8, 15.4, 3200, 0.00, '{"ma_fast": 50, "ma_slow": 200}'),
        ('Eth RSI Scalper', 'VitalikFan', 'Quick scalps on RSI divergence', '15m', 'ETH/USDT', 'Scalping', 28.7, 65.0, 1.9, 1.2, 560, 29.00, '{"length": 14, "source": "close"}'),
        ('News Sentiment AI', 'AlphaSeeker', 'Trades on high-impact news events', '1m', 'ALL', 'Sentiment', 56.0, 60.0, 2.1, 5.0, 150, 199.00, '{"threshold": 0.8}'),
    ]
    cursor.executemany('''
        INSERT INTO strategies (name, author, description, timeframe, asset, type, roi_30d, win_rate, sharpe, drawdown, copiers, price, config)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', strategies)

# === MODELS ===
class Strategy(BaseModel):
    id: int
    name: str
    author: str
    description: str
    roi_30d: float
    win_rate: float
    copiers: int
    price: float
    deployed: bool

# === API ===
app = FastAPI(title="Strategy Marketplace")

@app.on_event("startup")
async def startup():
    init_db()
    print("🛒 Strategy Marketplace ready at http://localhost:8002")

@app.get("/api/strategies")
async def get_strategies(sort_by: str = 'roi_30d'):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Sort mapping
    order = {
        'roi': 'roi_30d DESC',
        'safe': 'drawdown ASC',
        'popular': 'copiers DESC'
    }.get(sort_by, 'roi_30d DESC')
    
    cursor.execute(f'SELECT * FROM strategies ORDER BY {order}')
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/deploy/{id}")
async def deploy_strategy(id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if exists
    cursor.execute('SELECT * FROM strategies WHERE id = ?', (id,))
    strat = cursor.fetchone()
    if not strat:
        conn.close()
        raise HTTPException(404, "Strategy not found")
    
    # "Deploy" (Update DB)
    cursor.execute('UPDATE strategies SET deployed = 0') # Undeploy others (simulated logic)
    cursor.execute('UPDATE strategies SET deployed = 1, copiers = copiers + 1 WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    
    # In a real system, this would notify the execution engine
    print(f"🚀 Deployed strategy: {strat[1]}")
    return {"status": "deployed", "name": strat[1]}

# === HTML UI ===
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Marketplace</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; font-family:-apple-system, system-ui, sans-serif; }
        body { background:#0a0e14; color:#e2e8f0; padding:20px; }
        
        .header { display:flex; justify-content:space-between; align-items:center; margin-bottom:30px; padding-bottom:20px; border-bottom:1px solid #1e293b; }
        .logo { font-size:24px; font-weight:800; background:linear-gradient(45deg, #3b82f6, #06b6d4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        
        .filters { display:flex; gap:10px; margin-bottom:20px; }
        .btn { padding:8px 16px; border-radius:6px; border:1px solid #334155; background:#0f172a; color:#94a3b8; cursor:pointer; font-size:14px; transition:all 0.2s; }
        .btn:hover, .btn.active { background:#3b82f6; color:white; border-color:#3b82f6; }
        
        .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:20px; }
        .card { background:#1e293b; border:1px solid #334155; border-radius:12px; overflow:hidden; transition:transform 0.2s; position:relative; }
        .card:hover { transform:translateY(-2px); border-color:#475569; }
        
        .card-header { padding:15px; background:rgba(255,255,255,0.03); border-bottom:1px solid #334155; display:flex; justify-content:space-between; }
        .badge { font-size:10px; padding:2px 6px; border-radius:4px; font-weight:700; text-transform:uppercase; }
        .badge.hft { background:rgba(239,68,68,0.2); color:#ef4444; }
        .badge.swing { background:rgba(59,130,246,0.2); color:#3b82f6; }
        .badge.grid { background:rgba(16,185,129,0.2); color:#10b981; }
        
        .card-body { padding:15px; }
        .strat-name { font-size:16px; font-weight:700; margin-bottom:4px; }
        .author { font-size:12px; color:#64748b; margin-bottom:12px; }
        
        .stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:15px; }
        .stat-box { background:#0f172a; padding:8px; border-radius:6px; text-align:center; }
        .stat-label { font-size:10px; color:#64748b; text-transform:uppercase; }
        .stat-val { font-size:14px; font-weight:700; }
        .green { color:#10b981; }
        
        .price-row { display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:10px; border-top:1px solid #334155; }
        .price { font-size:18px; font-weight:700; }
        .deploy-btn { background:#3b82f6; color:white; border:none; padding:6px 16px; border-radius:6px; font-weight:600; cursor:pointer; }
        .deploy-btn:hover { background:#2563eb; }
        .deploy-btn.deployed { background:#10b981; cursor:default; }
        
        .search-bar { width:100%; max-width:400px; padding:10px; background:#0f172a; border:1px solid #334155; border-radius:8px; color:white; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">STRATEGY MARKETPLACE</div>
        <input type="text" class="search-bar" placeholder="Search strategies (e.g. Scalping, BTC)...">
        <div>
            <span style="font-size:12px; color:#64748b; margin-right:10px;">Total Volume: $42.5M</span>
            <span style="font-size:12px; color:#64748b;">Active Bots: 12,450</span>
        </div>
    </div>
    
    <div class="filters">
        <button class="btn active" onclick="loadStrategies('roi')">Top ROI</button>
        <button class="btn" onclick="loadStrategies('safe')">Safest</button>
        <button class="btn" onclick="loadStrategies('popular')">Most Copied</button>
        <button class="btn" onclick="loadStrategies('new')">Newest</button>
    </div>
    
    <div class="grid" id="grid">
        <!-- Cards injected here -->
    </div>

    <script>
        async function loadStrategies(sort) {
            // Update buttons
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
            const res = await fetch(`/api/strategies?sort_by=${sort}`);
            const data = await res.json();
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            
            data.forEach(s => {
                const badgeClass = s.type.toLowerCase();
                const btnText = s.deployed ? 'ACTIVE' : 'DEPLOY';
                const btnClass = s.deployed ? 'deployed' : '';
                
                grid.innerHTML += `
                    <div class="card">
                        <div class="card-header">
                            <span class="badge ${badgeClass}">${s.type}</span>
                            <span style="font-size:11px; color:#94a3b8">${s.asset}</span>
                        </div>
                        <div class="card-body">
                            <div class="strat-name">${s.name}</div>
                            <div class="author">by ${s.author} • ${s.copiers} copiers</div>
                            
                            <div class="stats">
                                <div class="stat-box">
                                    <div class="stat-label">30d ROI</div>
                                    <div class="stat-val green">+${s.roi_30d}%</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-label">Win Rate</div>
                                    <div class="stat-val">${s.win_rate}%</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-label">Sharpe</div>
                                    <div class="stat-val">${s.sharpe}</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-label">Drawdown</div>
                                    <div class="stat-val" style="color:#ef4444">${s.drawdown}%</div>
                                </div>
                            </div>
                            
                            <div style="font-size:12px; color:#94a3b8; line-height:1.4; margin-bottom:10px;">
                                ${s.description}
                            </div>
                            
                            <div class="price-row">
                                <div class="price">${s.price > 0 ? '$'+s.price+'/mo' : 'FREE'}</div>
                                <button class="deploy-btn ${btnClass}" onclick="deploy(${s.id})">${btnText}</button>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        async function deploy(id) {
            const res = await fetch(`/api/deploy/${id}`, {method: 'POST'});
            const data = await res.json();
            if(res.ok) {
                alert(`Successfully deployed: ${data.name}`);
                loadStrategies('roi'); // Reload to update UI
            }
        }
        
        // Initial load
        loadStrategies('roi');
    </script>
</body>
</html>
'''

@app.get("/")
async def home():
    return HTMLResponse(HTML)

if __name__ == "__main__":
    print("="*60)
    print("🛒 STRATEGY MARKETPLACE SERVER")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
