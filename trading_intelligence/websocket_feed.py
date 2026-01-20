"""
BINANCE WEBSOCKET FEED - Production Grade
==========================================
Replaces REST polling with real-time WebSocket streams.

Features:
- Binance WebSocket for BTCUSDT + PAXGUSDT
- Auto-reconnect on disconnect
- Candle builder (aggregates ticks)
- Heartbeat monitoring
- No drift (precise timing)
"""
import asyncio
import json
import websockets
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Callable, Optional
import ssl

class BinanceWebSocketFeed:
    """
    Production-grade WebSocket feed for Binance.
    
    Handles:
    - Multiple symbols
    - Auto-reconnect
    - Candle building
    - Heartbeat/ping-pong
    """
    
    def __init__(self, symbols: list, on_candle: Callable = None):
        """
        Parameters:
        -----------
        symbols : List of symbols to track (e.g., ['BTCUSDT', 'PAXGUSDT'])
        on_candle : Callback function(symbol, candle_dict) when candle closes
        """
        self.symbols = [s.lower() for s in symbols]
        self.on_candle = on_candle or (lambda s, c: print(f"{s}: {c}"))
        
        self.ws = None
        self.running = False
        self.last_ping = datetime.now()
        
        # Current candle data
        self.candles = defaultdict(lambda: {
            'open': None,
            'high': -float('inf'),
            'low': float('inf'),
            'close': None,
            'volume': 0.0,
            'trades': 0,
            'start_time': None
        })
        
        # Price history
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
    async def connect(self):
        """Connect to Binance WebSocket"""
        # Build stream URL for multiple symbols
        streams = [f"{symbol}@trade" for symbol in self.symbols]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        # SECURITY FIX: Use proper SSL certificate verification
        # Import certifi for trusted CA bundle
        try:
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            # Fallback to system certificates
            ssl_context = ssl.create_default_context()
        
        # Ensure certificate verification is ENABLED (institutional security requirement)
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        print(f"🔌 Connecting to Binance WebSocket (SSL verified)...")
        self.ws = await websockets.connect(url, ssl=ssl_context, ping_interval=20)
        self.running = True
        print(f"✅ Connected: {', '.join([s.upper() for s in self.symbols])}")
        
    async def disconnect(self):
        """Graceful disconnect"""
        self.running = False
        if self.ws:
            await self.ws.close()
        print("🔌 Disconnected")
    
    async def heartbeat(self):
        """Send periodic pings to keep connection alive"""
        while self.running:
            try:
                if self.ws and not self.ws.closed:
                    await self.ws.ping()
                    self.last_ping = datetime.now()
                await asyncio.sleep(30)
            except Exception as e:
                print(f"⚠️ Heartbeat error: {e}")
                break
    
    async def handle_trade(self, data: dict):
        """Process incoming trade data"""
        symbol = data['s'].lower()
        price = float(data['p'])
        qty = float(data['q'])
        timestamp = datetime.fromtimestamp(data['T'] / 1000)
        
        # Update price history
        self.price_history[symbol].append({
            'time': timestamp,
            'price': price,
            'qty': qty
        })
        
        # Update current 1-minute candle
        candle = self.candles[symbol]
        
        # New candle check (every minute)
        current_minute = timestamp.replace(second=0, microsecond=0)
        
        if candle['start_time'] is None:
            # First tick of new candle
            candle['start_time'] = current_minute
            candle['open'] = price
            candle['high'] = price
            candle['low'] = price
            candle['close'] = price
            candle['volume'] = qty
            candle['trades'] = 1
        elif current_minute != candle['start_time']:
            # Candle closed - emit it
            closed_candle = {
                'time': candle['start_time'],
                'o': candle['open'],
                'h': candle['high'],
                'l': candle['low'],
                'c': candle['close'],
                'v': candle['volume'],
                'n': candle['trades']
            }
            
            # Callback
            if self.on_candle:
                await self.on_candle(symbol.upper(), closed_candle)
            
            # Start new candle
            candle['start_time'] = current_minute
            candle['open'] = price
            candle['high'] = price
            candle['low'] = price
            candle['close'] = price
            candle['volume'] = qty
            candle['trades'] = 1
        else:
            # Update current candle
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] += qty
            candle['trades'] += 1
    
    async def listen(self):
        """Main listening loop with auto-reconnect"""
        reconnect_delay = 1
        max_delay = 60
        
        while self.running:
            try:
                if not self.ws or self.ws.closed:
                    await self.connect()
                    reconnect_delay = 1
                
                # Listen for messages
                message = await self.ws.recv()
                data = json.loads(message)
                
                # Handle trade data
                if 'stream' in data and 'data' in data:
                    trade_data = data['data']
                    if trade_data.get('e') == 'trade':
                        await self.handle_trade(trade_data)
                
            except websockets.exceptions.ConnectionClosed:
                print(f"⚠️ Connection closed. Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)
            
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)
    
    async def run(self):
        """Start the feed"""
        print(f"🚀 Starting Binance WebSocket Feed...")
        
        # Run both listener and heartbeat
        await asyncio.gather(
            self.listen(),
            self.heartbeat()
        )
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        symbol = symbol.lower()
        candle = self.candles.get(symbol)
        return candle['close'] if candle and candle['close'] else None
    
    def get_current_candle(self, symbol: str) -> Optional[dict]:
        """Get current (incomplete) candle"""
        symbol = symbol.lower()
        candle = self.candles.get(symbol)
        
        if candle and candle['open'] is not None:
            return {
                'time': candle['start_time'],
                'o': candle['open'],
                'h': candle['high'],
                'l': candle['low'],
                'c': candle['close'],
                'v': candle['volume'],
                'n': candle['trades']
            }
        return None


# =============================================================================
#  EXAMPLE USAGE
# =============================================================================

async def on_new_candle(symbol: str, candle: dict):
    """Callback when 1-minute candle closes"""
    print(f"✅ {symbol} Candle Closed: O={candle['o']:.2f} H={candle['h']:.2f} L={candle['l']:.2f} C={candle['c']:.2f} V={candle['v']:.4f}")


async def test_feed():
    """Test the WebSocket feed"""
    feed = BinanceWebSocketFeed(
        symbols=['BTCUSDT', 'PAXGUSDT'],
        on_candle=on_new_candle
    )
    
    try:
        await feed.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping feed...")
        await feed.disconnect()


if __name__ == "__main__":
    print("="*70)
    print("🌐 BINANCE WEBSOCKET FEED - Test")
    print("="*70)
    print("\nPress Ctrl+C to stop\n")
    
    asyncio.run(test_feed())
