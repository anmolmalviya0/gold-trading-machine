"""
Live WebSocket Feed - Candle-Close Execution Only
"""

import websocket
import json
import logging
from typing import Callable, Optional, Dict
from datetime import datetime
from threading import Thread, Event
import time

logger = logging.getLogger(__name__)


class BinanceWebSocketFeed:
    """
    Binance WebSocket connection for live candles.
    
    Rules:
    - Only execute on candle close (x=true)
    - No repainting
    - Reconnect logic built-in
    """
    
    def __init__(self, config: Dict, on_candle_close: Callable):
        """
        Args:
            config: Full config dict
            on_candle_close: Callback fn(symbol, tf, ohlcv_dict)
        """
        self.config = config['ws']
        self.base_url = self.config['binance_url']
        self.on_candle_close = on_candle_close
        self.symbols = self.config.get('symbols', [])
        self.reconnect_timeout = self.config.get('reconnect_timeout', 5)
        
        self.ws = None
        self.running = False
        self.stop_event = Event()
    
    def start(self):
        """Start WebSocket connection"""
        self.running = True
        self.stop_event.clear()
        
        # Build stream string
        streams = []
        for symbol in self.symbols:
            # 5m, 15m, 30m, 1h streams
            for tf in ['5m', '15m', '30m', '1h']:
                streams.append(f"{symbol.lower()}@kline_{tf}")
        
        stream_param = "/".join(streams)
        url = f"{self.base_url}?streams={stream_param}"
        
        logger.info(f"Connecting to {url}")
        
        # Start WS in separate thread
        ws_thread = Thread(target=self._run_ws, args=(url,), daemon=True)
        ws_thread.start()
    
    def _run_ws(self, url: str):
        """WebSocket event loop with reconnect"""
        while self.running and not self.stop_event.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws.run_forever()
                
            except Exception as e:
                logger.error(f"WS error: {e}")
            
            if self.running and not self.stop_event.is_set():
                logger.info(f"Reconnecting in {self.reconnect_timeout}s...")
                time.sleep(self.reconnect_timeout)
    
    def _on_message(self, ws, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                kline_data = data['data'].get('k', {})
                
                # Extract symbol and timeframe from stream
                # Format: "symbol@kline_tf"
                parts = stream.split('@')
                if len(parts) != 2:
                    return
                
                symbol_raw = parts[0]
                tf_raw = parts[1].replace('kline_', '')
                
                # Normalize
                symbol = symbol_raw.upper()
                tf = tf_raw
                
                # Only process on candle close
                if not kline_data.get('x', False):
                    return  # Candle not closed yet
                
                # Extract OHLCV
                ohlcv = {
                    'open': float(kline_data.get('o', 0)),
                    'high': float(kline_data.get('h', 0)),
                    'low': float(kline_data.get('l', 0)),
                    'close': float(kline_data.get('c', 0)),
                    'volume': float(kline_data.get('v', 0)),
                    'open_time': int(kline_data.get('t', 0)),
                    'close_time': int(kline_data.get('T', 0)),
                    'is_complete': kline_data.get('x', False)
                }
                
                logger.debug(f"Candle close: {symbol} {tf} @ {ohlcv['close']}")
                
                # Trigger callback
                self.on_candle_close(symbol, tf, ohlcv)
        
        except Exception as e:
            logger.error(f"Message parse error: {e}")
    
    def _on_error(self, ws, error):
        logger.error(f"WS Error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WS Closed: {close_status_code} {close_msg}")
    
    def stop(self):
        """Stop WebSocket"""
        self.running = False
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        logger.info("WebSocket stopped")


class OrderBookSnapshot:
    """Fetch order book snapshot for conviction check"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def fetch(self, symbol: str, depth: int = 20) -> Optional[Dict]:
        """
        Fetch order book from Binance REST API.
        Returns: {
            'bid': float,
            'ask': float,
            'bids': [(price, amount), ...],
            'asks': [(price, amount), ...]
        }
        """
        try:
            from binance.client import Client
            # Note: This requires Binance API key (can be empty for public data)
            client = Client()
            
            book = client.get_order_book(symbol=symbol, limit=depth)
            
            bids = [(float(b[0]), float(b[1])) for b in book['bids']]
            asks = [(float(a[0]), float(a[1])) for a in book['asks']]
            
            return {
                'bid': bids[0][0] if bids else None,
                'ask': asks[0][0] if asks else None,
                'bids': bids,
                'asks': asks,
                'timestamp': int(datetime.utcnow().timestamp() * 1000)
            }
        
        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            return None
