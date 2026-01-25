"""
AEGIS V21 - Data Fetcher (Phase 2 Expansion)
=============================================
Fetches historical data for new assets: SOL, BNB, ETH.
Uses public APIs (Binance/Yahoo) to get 5m/1h data.
"""
import yfinance as yf
import pandas as pd
import os
import time

def fetch_data(symbol, period="60d", interval="5m"):
    print(f"📥 Fetching {symbol} ({interval})...")
    try:
        # Check if file already exists to avoid redownloading
        filename = f"market_data/{symbol.replace('-','')}USDT_{interval}.csv"
        
        # Yahoo Finance Tickers: SOL-USD, BNB-USD, ETH-USD
        yf_symbol = f"{symbol}-USD"
        
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        
        if df.empty:
            print(f"❌ No data for {symbol}")
            return None
            
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean and format
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        
        # Rename for compatibility
        rename_map = {
            'date': 'time',
            'datetime': 'time', 
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        df = df.rename(columns=rename_map)
        
        # Ensure 'time' column is present
        if 'time' not in df.columns and 'date' in df.columns:
             df = df.rename(columns={'date': 'time'})
             
        # Select valid columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # Save
        os.makedirs('market_data', exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"✅ Saved to {filename} ({len(df)} rows)")
        return df
        
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None

def main():
    print("🚀 AEGIS EXPANSION: DATA ACQUISITION")
    print("Targeting: SOL, BNB, ETH")
    
    assets = ['SOL', 'BNB', 'ETH']
    intervals = ['5m', '1h'] # High freq and Trend
    
    for asset in assets:
        for interval in intervals:
            fetch_data(asset, period="59d" if interval=="5m" else "2y", interval=interval)
            time.sleep(1) # Rate limit politeness

if __name__ == "__main__":
    main()
