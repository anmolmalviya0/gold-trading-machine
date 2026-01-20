"""
MetaTrader5 Integration - XAU/USD Feed
Requires MT5 terminal installed on macOS (via Wine or Parallels)
"""
import asyncio
from typing import Dict, Optional
from datetime import datetime
import logging
from .broker_adapter import BrokerAdapter

logger = logging.getLogger(__name__)


class MT5Feed(BrokerAdapter):
    """MetaTrader5 broker adapter"""
    
    def __init__(self, symbol: str = 'XAUUSD'):
        super().__init__(symbol)
        self.mt5 = None
    
    async def connect(self) -> bool:
        """Connect to MT5"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            
            # Initialize MT5
            if not self.mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Check symbol availability
            symbol_info = self.mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            # Enable symbol
            if not symbol_info.visible:
                if not self.mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to enable {self.symbol}")
                    return False
            
            self.connected = True
            logger.info(f"✅ MT5 connected: {self.symbol}")
            return True
        
        except ImportError:
            logger.warning("⚠️  MT5 Python package not installed")
            logger.info("Install: pip install MetaTrader5")
            return False
        
        except Exception as e:
            logger.error(f"MT5 connection failed: {e}")
            return False
    
    async def get_latest_tick(self) -> Optional[Dict]:
        """Get latest tick from MT5"""
        if not self.connected or self.mt5 is None:
            return None
        
        try:
            tick = self.mt5.symbol_info_tick(self.symbol)
            
            if tick is None:
                return None
            
            tick_data = {
                'symbol': self.symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'timestamp': datetime.fromtimestamp(tick.time)
            }
            
            self.last_tick = tick_data
            return tick_data
        
        except Exception as e:
            logger.error(f"Error getting MT5 tick: {e}")
            return None
    
    async def get_historical_ohlc(self, timeframe: str, count: int = 100) -> Optional[list]:
        """Get historical OHLC from MT5"""
        if not self.connected or self.mt5 is None:
            return None
        
        try:
            # Map timeframe
            tf_map = {
                '5m': self.mt5.TIMEFRAME_M5,
                '15m': self.mt5.TIMEFRAME_M15,
                '30m': self.mt5.TIMEFRAME_M30,
                '1h': self.mt5.TIMEFRAME_H1
            }
            
            mt5_tf = tf_map.get(timeframe)
            if mt5_tf is None:
                logger.error(f"Unsupported timeframe: {timeframe}")
                return None
            
            # Get rates
            rates = self.mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, count)
            
            if rates is None or len(rates) == 0:
                return None
            
            # Convert to standard format
            ohlc_data = []
            for rate in rates:
                ohlc_data.append({
                    'timestamp': datetime.fromtimestamp(rate['time']),
                    'open': rate['open'],
                    'high': rate['high'],
                    'low': rate['low'],
                    'close': rate['close'],
                    'volume': rate['tick_volume']
                })
            
            return ohlc_data
        
        except Exception as e:
            logger.error(f"Error getting MT5 history: {e}")
            return None
    
    async def disconnect(self):
        """Shutdown MT5 connection"""
        if self.mt5:
            self.mt5.shutdown()
        
        self.connected = False
        logger.info("👋 MT5 disconnected")


# Demo
async def demo():
    """Test MT5 feed"""
    feed = MT5Feed('XAUUSD')
    
    if await feed.connect():
        # Get tick
        tick = await feed.get_latest_tick()
        if tick:
            print(f"XAU/USD: Bid=${tick['bid']:.2f} Ask=${tick['ask']:.2f}")
        
        # Get history
        history = await feed.get_historical_ohlc('15m', count=10)
        if history:
            print(f"\nLast 10 bars (15m):")
            for bar in history[-5:]:
                print(f"  {bar['timestamp']}: O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}")
        
        await feed.disconnect()
    else:
        print("⚠️  MT5 not available, will use PAXG as proxy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
