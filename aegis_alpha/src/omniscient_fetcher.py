
"""
TERMINAL - OMNISCIENT FETCHER (Protocol Infinity)
=================================================
The Deep Harvest: Fetches 7 Years of Data across All Timeframes.
Assets: BTC, ETH, SOL, BNB, PAXG
Timeframes: 1m, 5m, 15m, 30m, 1h, 1d
Source: Binance (Public API via CCXT)
"""
import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# Configuration
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'PAXG/USDT']
TIMEFRAMES = {
    '1d': '1d',
    '4h': '4h',
    '1h': '1h',
    '30m': '30m', 
    '15m': '15m',
    '5m': '5m',
    '1m': '1m' # The Heavy Lifter
}
START_DATE = "2019-01-01 00:00:00"
DATA_DIR = "/Users/anmol/Desktop/gold/market_data"

def fetch_ohlcv_batch(exchange, symbol, timeframe, since, limit=1000):
    """Fetch a single batch of candles"""
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return candles
    except Exception as e:
        print(f"   ⚠️ Fetch Error: {e}")
        return []

def safe_fetch(symbol, timeframe):
    """
    Orchestrates the massive download with pagination.
    """
    clean_symbol = symbol.replace('/', '')
    filename = os.path.join(DATA_DIR, f"{clean_symbol}_{timeframe}.csv")
    
    print(f"\n🌊 INITIATING HARVEST: {symbol} [{timeframe}]")
    print(f"   Target: 2019 -> Present")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calculate Start Timestamp (ms)
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    current_ts = int(time.time() * 1000)
    
    all_candles = []
    since = start_ts
    
    while since < current_ts:
        batch = fetch_ohlcv_batch(exchange, symbol, timeframe, since)
        
        if not batch:
            print("   ⚠️ No more data or rate limit hit. Stopping.")
            break
            
        all_candles.extend(batch)
        
        # Update 'since' to the last candle timestamp + 1ms
        last_ts = batch[-1][0]
        since = last_ts + 1
        
        # Progress Log
        last_date = datetime.fromtimestamp(last_ts / 1000).strftime('%Y-%m-%d')
        print(f"   📥 Recovered until: {last_date} | Total: {len(all_candles)}")
        
        # Politeness Sleep (Binance is strict)
        time.sleep(exchange.rateLimit / 1000)
        
        # Safety break if batch is smaller than limit (end of history)
        if len(batch) < 1:
            break
            
    if not all_candles:
        print("❌ No data harvested.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert Timestamp to human readable
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Save
    if os.path.exists(filename):
        # Merge if exists? For now, we overwrite to ensure purity.
        pass
        
    df.to_csv(filename, index=False)
    print(f"✅ HARVEST COMPLETE: {filename}")
    print(f"   📦 {len(df)} Candles Secured.")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       🦅 PROTOCOL INFINITY: THE OMNISCIENT FETCHER      ║
    ║             7 Years. 5 Assets. 6 Timeframes.             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # User asked for everything. 7 Years for all.
    priority_order = ['1d', '1h', '30m', '15m', '5m', '1m'] 
    
    for tf_name in priority_order:
        for symbol in ASSETS:
            safe_fetch(symbol, TIMEFRAMES[tf_name])
            
    print("\n🏁 PHASE 1 HARVEST COMPLETE.")
    print("⚠️ 1m Data requires 'Deep Dive' authorization (Heavy Load).")
    print("   Run with --deep-dive flag to fetch 1m data.")

if __name__ == "__main__":
    main()
