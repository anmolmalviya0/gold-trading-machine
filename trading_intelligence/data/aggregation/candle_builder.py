"""
Candle Builder - Aggregate ticks into OHLCV candles
Supports 5m, 15m, 30m, 1h timeframes with proper alignment
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CandleBuilder:
    """Build OHLCV candles from tick data"""
    
    def __init__(self, symbols: List[str], timeframes: List[str], on_candle: Callable):
        """
        Args:
            symbols: List of symbols to track
            timeframes: List of timeframes ('5m', '15m', '30m', '1h')
            on_candle: Callback(symbol, timeframe, candle_dict)
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.on_candle = on_candle
        
        # Active candles being built
        self.active_candles = defaultdict(lambda: defaultdict(dict))
        
        # Completed candles (circular buffer)
        self.history = defaultdict(lambda: defaultdict(lambda: pd.DataFrame()))
        self.max_history = 500
        
        # Timeframe to minutes mapping
        self.tf_minutes = {
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60
        }
    
    async def process_tick(self, symbol: str, tick: Dict):
        """Process incoming tick data"""
        timestamp = tick['timestamp']
        price = tick['price']
        volume = tick.get('quantity', 0)
        
        for tf in self.timeframes:
            # Get candle period for this tick
            candle_start = self._get_candle_start(timestamp, tf)
            
            # Create or update candle
            candle = self.active_candles[symbol][tf].get(candle_start)
            
            if candle is None:
                # New candle
                candle = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'start_time': candle_start,
                    'tick_count': 1
                }
                self.active_candles[symbol][tf][candle_start] = candle
            else:
                # Update existing candle
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += volume
                candle['tick_count'] += 1
            
            # Check if candle is complete
            if self._is_candle_complete(candle_start, timestamp, tf):
                await self._finalize_candle(symbol, tf, candle)
    
    def _get_candle_start(self, timestamp: datetime, timeframe: str) -> datetime:
        """Align timestamp to candle start time"""
        minutes = self.tf_minutes[timeframe]
        
        # Floor to minute boundary
        aligned = timestamp.replace(second=0, microsecond=0)
        
        # Floor to timeframe boundary
        minute_offset = aligned.minute % minutes
        candle_start = aligned - timedelta(minutes=minute_offset)
        
        return candle_start
    
    def _is_candle_complete(self, candle_start: datetime, current_time: datetime, timeframe: str) -> bool:
        """Check if candle period has ended"""
        minutes = self.tf_minutes[timeframe]
        candle_end = candle_start + timedelta(minutes=minutes)
        
        return current_time >= candle_end
    
    async def _finalize_candle(self, symbol: str, timeframe: str, candle: Dict):
        """Complete and store candle"""
        # Remove from active
        candle_start = candle['start_time']
        del self.active_candles[symbol][timeframe][candle_start]
        
        # Add to history
        candle_df = pd.DataFrame([{
            'timestamp': candle_start,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }])
        
        # Append to history
        if self.history[symbol][timeframe].empty:
            self.history[symbol][timeframe] = candle_df
        else:
            self.history[symbol][timeframe] = pd.concat([
                self.history[symbol][timeframe],
                candle_df
            ], ignore_index=True)
        
        # Trim history
        if len(self.history[symbol][timeframe]) > self.max_history:
            self.history[symbol][timeframe] = self.history[symbol][timeframe].iloc[-self.max_history:]
        
        # Callback
        await self.on_candle(symbol, timeframe, candle)
        
        logger.debug(f"Candle complete: {symbol} {timeframe} @ {candle_start}")
    
    def get_history(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get historical candles"""
        df = self.history[symbol][timeframe]
        
        if df.empty:
            return pd.DataFrame()
        
        return df.iloc[-count:].copy()
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get most recent completed candle"""
        df = self.history[symbol][timeframe]
        
        if df.empty:
            return None
        
        row = df.iloc[-1]
        return row.to_dict()


# Demo
async def demo():
    """Test candle builder"""
    async def on_candle(symbol, tf, candle):
        print(f"{symbol} {tf}: O={candle['open']:.2f} H={candle['high']:.2f} L={candle['low']:.2f} C={candle['close']:.2f} V={candle['volume']}")
    
    builder = CandleBuilder(['BTCUSDT'], ['5m', '15m'], on_candle)
    
    # Simulate ticks
    base_time = datetime.now().replace(second=0, microsecond=0)
    base_price = 95000
    
    for i in range(100):
        tick = {
            'timestamp': base_time + timedelta(seconds=i*30),
            'price': base_price + (i % 10 - 5) * 10,
            'quantity': 0.1
        }
        await builder.process_tick('BTCUSDT', tick)
    
    # Get history
    history = builder.get_history('BTCUSDT', '5m')
    print(f"\n5m candles built: {len(history)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import asyncio
    asyncio.run(demo())
