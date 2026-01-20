"""
Broker Adapter Interface - Abstract base for XAU/USD feeds
Plug in MT5, IBKR, or OANDA
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BrokerAdapter(ABC):
    """Abstract interface for broker data feeds"""
    
    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.connected = False
        self.last_tick = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to broker
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_latest_tick(self) -> Optional[Dict]:
        """
        Get latest tick/quote
        
        Returns:
            {
                'symbol': str,
                'bid': float,
                'ask': float,
                'timestamp': datetime
            }
        """
        pass
    
    @abstractmethod
    async def get_historical_ohlc(self, timeframe: str, count: int = 100) -> Optional[list]:
        """
        Get historical OHLC data
        
        Args:
            timeframe: '5m', '15m', '30m', '1h'
            count: Number of bars
            
        Returns:
            List of OHLC dicts
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Clean disconnect"""
        pass
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        if not self.connected:
            return False
        
        if self.last_tick is None:
            return False
        
        # Check if data is stale (>2 min)
        age = (datetime.now() - self.last_tick['timestamp']).total_seconds()
        return age < 120
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price from last tick"""
        if self.last_tick is None:
            return None
        
        return (self.last_tick['bid'] + self.last_tick['ask']) / 2
