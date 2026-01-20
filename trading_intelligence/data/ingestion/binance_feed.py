"""
Binance WebSocket Feed - Production Grade
Real-time OHLCV data for BTC and PAXG with reliability features
"""
import asyncio
import websockets
import json
from datetime import datetime
from typing import Dict, Callable, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)


class BinanceFeed:
    """Production-grade Binance WebSocket feed"""
    
    def __init__(self, symbols: list, on_tick: Callable):
        """
        Args:
            symbols: List of Binance symbols (e.g., ['BTCUSDT', 'PAXGUSDT'])
            on_tick: Callback function(symbol, tick_data)
        """
        self.symbols = [s.lower() for s in symbols]
        self.on_tick = on_tick
        
        # WebSocket connection
        self.ws = None
        self.connected = False
        self.reconnect_count = 0
        self.max_reconnect = 10
        
        # Quality control
        self.last_update = {}
        self.message_buffer = deque(maxlen=1000)
        self.duplicate_count = 0
        
        # Stream URL
        streams = '/'.join([f"{s}@trade" for s in self.symbols])
        self.ws_url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    
    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                ssl=ssl_context
            )
            self.connected = True
            self.reconnect_count = 0
            logger.info(f"✅ Connected to Binance: {self.symbols}")
            
        except Exception as e:
            logger.error(f"❌ Binance connection failed: {e}")
            self.connected = False
            raise
    
    async def listen(self):
        """Main listening loop with auto-reconnect"""
        while True:
            try:
                if not self.connected:
                    await self.connect()
                
                # Listen for messages
                async for message in self.ws:
                    await self._handle_message(message)
            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️  Binance connection closed")
                await self._reconnect()
            
            except Exception as e:
                logger.error(f"❌ Binance error: {e}")
                await self._reconnect()
    
    async def _handle_message(self, message: str):
        """Process incoming tick data"""
        try:
            data = json.loads(message)
            
            if 'stream' not in data:
                return
            
            stream = data['stream']
            tick = data['data']
            
            # Extract symbol
            symbol = stream.split('@')[0].upper()
            
            # Deduplication check
            event_time = tick['E']
            if self._is_duplicate(symbol, event_time):
                self.duplicate_count += 1
                return
            
            # Parse tick
            tick_data = {
                'symbol': symbol,
                'price': float(tick['p']),
                'quantity': float(tick['q']),
                'timestamp': datetime.fromtimestamp(event_time / 1000),
                'trade_id': tick['t'],
                'is_buyer_maker': tick['m']
            }
            
            # Update tracking
            self.last_update[symbol] = datetime.now()
            self.message_buffer.append((symbol, event_time))
            
            # Callback
            await self.on_tick(symbol, tick_data)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _is_duplicate(self, symbol: str, event_time: int) -> bool:
        """Check for duplicate messages"""
        key = (symbol, event_time)
        
        # Check if in recent buffer
        if key in self.message_buffer:
            return True
        
        return False
    
    async def _reconnect(self):
        """Reconnection logic with exponential backoff"""
        self.connected = False
        self.reconnect_count += 1
        
        if self.reconnect_count > self.max_reconnect:
            logger.critical("🚨 Max reconnection attempts reached")
            raise Exception("Failed to reconnect to Binance")
        
        # Exponential backoff
        delay = min(2 ** self.reconnect_count, 60)
        logger.info(f"🔄 Reconnecting in {delay}s (attempt {self.reconnect_count})")
        
        await asyncio.sleep(delay)
        await self.connect()
    
    def is_healthy(self) -> Dict[str, bool]:
        """Health check for each symbol"""
        health = {}
        now = datetime.now()
        
        for symbol in [s.upper() for s in self.symbols]:
            if symbol not in self.last_update:
                health[symbol] = False
            else:
                age = (now - self.last_update[symbol]).total_seconds()
                health[symbol] = age < 60  # Stale if > 60s
        
        return health
    
    async def close(self):
        """Graceful shutdown"""
        if self.ws:
            await self.ws.close()
        
        logger.info("👋 Binance feed closed")


# Demo
async def demo():
    """Test Binance feed"""
    async def on_tick(symbol, tick):
        print(f"{symbol}: ${tick['price']:.2f} @ {tick['timestamp']}")
    
    feed = BinanceFeed(['BTCUSDT', 'PAXGUSDT'], on_tick)
    
    try:
        await feed.listen()
    except KeyboardInterrupt:
        await feed.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
