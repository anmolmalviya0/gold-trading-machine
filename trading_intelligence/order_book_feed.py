"""
ORDER BOOK & FUNDING RATE FEED
===============================
Advanced data feeds for HFT:
1. Order Book Imbalance (L2 depth)
2. Funding Rate (crypto perpetuals)
3. Tick Data Aggregation

Usage:
    from order_book_feed import OrderBookFeed, FundingRateFeed
"""
import asyncio
import aiohttp
import ssl
import certifi
from datetime import datetime
from typing import Dict, Callable, Optional
from collections import deque
import json


# =============================================================================
# ORDER BOOK IMBALANCE
# =============================================================================

class OrderBookFeed:
    """
    Real-time Order Book (L2 Depth) for imbalance calculation.
    
    Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
    Range: -1 (all asks) to +1 (all bids)
    
    Trading Signal:
    - Imbalance > 0.3: Buying pressure → Bullish
    - Imbalance < -0.3: Selling pressure → Bearish
    """
    
    def __init__(self, symbols: list, depth_levels: int = 10):
        self.symbols = symbols
        self.depth_levels = depth_levels
        self.order_books: Dict[str, dict] = {}
        self.imbalances: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # API endpoint
        self.base_url = "https://api.binance.com/api/v3"
    
    async def fetch_depth(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Fetch order book depth from Binance"""
        url = f"{self.base_url}/depth?symbol={symbol}&limit={self.depth_levels}"
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    return await r.json()
        except Exception as e:
            print(f"❌ Order book fetch failed for {symbol}: {e}")
        return None
    
    def calculate_imbalance(self, depth: dict) -> float:
        """
        Calculate order book imbalance.
        
        Returns:
        --------
        float: Imbalance ratio (-1 to +1)
        """
        if not depth or 'bids' not in depth or 'asks' not in depth:
            return 0.0
        
        # Sum bid volume (price * quantity)
        bid_volume = sum(float(b[0]) * float(b[1]) for b in depth['bids'][:self.depth_levels])
        
        # Sum ask volume (price * quantity)
        ask_volume = sum(float(a[0]) * float(a[1]) for a in depth['asks'][:self.depth_levels])
        
        # Calculate imbalance
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        imbalance = (bid_volume - ask_volume) / total
        return round(imbalance, 4)
    
    def get_spread(self, depth: dict) -> float:
        """Calculate bid-ask spread in basis points"""
        if not depth or not depth.get('bids') or not depth.get('asks'):
            return 0.0
        
        best_bid = float(depth['bids'][0][0])
        best_ask = float(depth['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2
        
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        return round(spread_bps, 2)
    
    async def update(self, session: aiohttp.ClientSession):
        """Update all order books"""
        for symbol in self.symbols:
            depth = await self.fetch_depth(session, symbol)
            if depth:
                self.order_books[symbol] = depth
                self.imbalances[symbol] = self.calculate_imbalance(depth)
                self.last_update[symbol] = datetime.now()
    
    def get_imbalance(self, symbol: str) -> float:
        """Get current imbalance for symbol"""
        return self.imbalances.get(symbol, 0.0)
    
    def get_signal_bias(self, symbol: str) -> str:
        """Get trading bias from imbalance"""
        imb = self.get_imbalance(symbol)
        if imb > 0.3:
            return 'BULLISH'
        elif imb < -0.3:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


# =============================================================================
# FUNDING RATE FEED
# =============================================================================

class FundingRateFeed:
    """
    Binance Perpetual Futures Funding Rate.
    
    Funding Rate indicates market sentiment:
    - Positive: Longs pay shorts → Market is overleveraged long → Bearish signal
    - Negative: Shorts pay longs → Market is overleveraged short → Bullish signal
    
    Extreme funding (|rate| > 0.01%) often precedes reversals.
    """
    
    def __init__(self, symbols: list):
        # Convert spot symbols to futures format
        self.symbols = symbols
        self.funding_rates: Dict[str, float] = {}
        self.next_funding_time: Dict[str, datetime] = {}
        
        # Futures API endpoint
        self.base_url = "https://fapi.binance.com/fapi/v1"
    
    async def fetch_funding(self, session: aiohttp.ClientSession, symbol: str) -> dict:
        """Fetch current funding rate"""
        url = f"{self.base_url}/premiumIndex?symbol={symbol}"
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    return await r.json()
        except Exception as e:
            print(f"❌ Funding rate fetch failed for {symbol}: {e}")
        return None
    
    async def update(self, session: aiohttp.ClientSession):
        """Update all funding rates"""
        for symbol in self.symbols:
            data = await self.fetch_funding(session, symbol)
            if data:
                # Funding rate as percentage
                rate = float(data.get('lastFundingRate', 0)) * 100
                self.funding_rates[symbol] = round(rate, 4)
                
                # Next funding time
                next_ts = data.get('nextFundingTime', 0)
                if next_ts:
                    self.next_funding_time[symbol] = datetime.fromtimestamp(next_ts / 1000)
    
    def get_rate(self, symbol: str) -> float:
        """Get current funding rate (%)"""
        return self.funding_rates.get(symbol, 0.0)
    
    def get_signal_bias(self, symbol: str) -> str:
        """Get trading bias from funding rate"""
        rate = self.get_rate(symbol)
        if rate > 0.01:
            return 'BEARISH'  # Overleveraged long
        elif rate < -0.01:
            return 'BULLISH'  # Overleveraged short
        else:
            return 'NEUTRAL'


# =============================================================================
# TICK DATA AGGREGATOR
# =============================================================================

class TickAggregator:
    """
    Aggregates raw tick data into VWAP and volume profiles.
    
    For HFT strategies, tick-level data provides:
    - True VWAP (volume-weighted average price)
    - Trade flow direction
    - Large order detection
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.ticks: Dict[str, deque] = {}
        self.vwap: Dict[str, float] = {}
        self.buy_volume: Dict[str, float] = {}
        self.sell_volume: Dict[str, float] = {}
    
    def add_tick(self, symbol: str, price: float, quantity: float, is_buyer_maker: bool):
        """Add a new tick"""
        if symbol not in self.ticks:
            self.ticks[symbol] = deque(maxlen=self.window_size)
            self.buy_volume[symbol] = 0.0
            self.sell_volume[symbol] = 0.0
        
        tick = {
            'price': price,
            'quantity': quantity,
            'is_buyer_maker': is_buyer_maker,
            'timestamp': datetime.now()
        }
        
        self.ticks[symbol].append(tick)
        
        # Update volume counters
        if is_buyer_maker:
            self.sell_volume[symbol] += quantity  # Buyer maker = seller aggressor
        else:
            self.buy_volume[symbol] += quantity   # Seller maker = buyer aggressor
        
        # Recalculate VWAP
        self._calculate_vwap(symbol)
    
    def _calculate_vwap(self, symbol: str):
        """Calculate VWAP from recent ticks"""
        if symbol not in self.ticks or len(self.ticks[symbol]) == 0:
            return
        
        total_value = sum(t['price'] * t['quantity'] for t in self.ticks[symbol])
        total_volume = sum(t['quantity'] for t in self.ticks[symbol])
        
        if total_volume > 0:
            self.vwap[symbol] = round(total_value / total_volume, 2)
    
    def get_vwap(self, symbol: str) -> float:
        """Get current VWAP"""
        return self.vwap.get(symbol, 0.0)
    
    def get_volume_ratio(self, symbol: str) -> float:
        """Get buy/sell volume ratio (>1 = bullish, <1 = bearish)"""
        buy = self.buy_volume.get(symbol, 0.0)
        sell = self.sell_volume.get(symbol, 0.0)
        
        if sell == 0:
            return 2.0 if buy > 0 else 1.0
        
        return round(buy / sell, 2)


# =============================================================================
# COMBINED ADVANCED FEED
# =============================================================================

class AdvancedMarketFeed:
    """
    Combined feed for all advanced market data.
    
    Provides:
    - Order Book Imbalance
    - Funding Rate
    - Tick Aggregation
    - Combined Signal
    """
    
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.order_book = OrderBookFeed(symbols)
        self.funding = FundingRateFeed(symbols)
        self.ticks = TickAggregator()
        
        self.session = None
    
    async def start(self):
        """Start the feed"""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(connector=conn)
        
        print("📊 Advanced Market Feed started")
    
    async def stop(self):
        """Stop the feed"""
        if self.session:
            await self.session.close()
    
    async def update(self):
        """Update all feeds"""
        if not self.session:
            await self.start()
        
        await self.order_book.update(self.session)
        await self.funding.update(self.session)
    
    def get_combined_signal(self, symbol: str) -> dict:
        """
        Get combined signal from all advanced data sources.
        
        Returns:
        --------
        dict with 'bias', 'confidence', 'imbalance', 'funding', 'vwap'
        """
        ob_bias = self.order_book.get_signal_bias(symbol)
        fr_bias = self.funding.get_signal_bias(symbol)
        
        # Score calculation
        score = 0
        if ob_bias == 'BULLISH':
            score += 1
        elif ob_bias == 'BEARISH':
            score -= 1
        
        if fr_bias == 'BULLISH':
            score += 1
        elif fr_bias == 'BEARISH':
            score -= 1
        
        # Volume ratio
        vol_ratio = self.ticks.get_volume_ratio(symbol)
        if vol_ratio > 1.2:
            score += 1
        elif vol_ratio < 0.8:
            score -= 1
        
        # Final bias
        if score >= 2:
            bias = 'STRONG_BUY'
            confidence = 80
        elif score == 1:
            bias = 'BUY'
            confidence = 65
        elif score <= -2:
            bias = 'STRONG_SELL'
            confidence = 80
        elif score == -1:
            bias = 'SELL'
            confidence = 65
        else:
            bias = 'NEUTRAL'
            confidence = 50
        
        return {
            'bias': bias,
            'confidence': confidence,
            'imbalance': self.order_book.get_imbalance(symbol),
            'funding': self.funding.get_rate(symbol),
            'vwap': self.ticks.get_vwap(symbol),
            'volume_ratio': vol_ratio,
            'ob_bias': ob_bias,
            'fr_bias': fr_bias
        }


# =============================================================================
# TEST
# =============================================================================

async def test_feeds():
    """Test the advanced feeds"""
    feed = AdvancedMarketFeed(['BTCUSDT', 'PAXGUSDT'])
    await feed.start()
    
    for i in range(3):
        await feed.update()
        
        for symbol in feed.symbols:
            signal = feed.get_combined_signal(symbol)
            print(f"\n{symbol}:")
            print(f"  Imbalance: {signal['imbalance']}")
            print(f"  Funding: {signal['funding']}%")
            print(f"  OB Bias: {signal['ob_bias']}")
            print(f"  FR Bias: {signal['fr_bias']}")
            print(f"  Combined: {signal['bias']} ({signal['confidence']}%)")
        
        await asyncio.sleep(2)
    
    await feed.stop()


if __name__ == "__main__":
    print("="*70)
    print("📊 ADVANCED MARKET FEED - Test")
    print("="*70)
    asyncio.run(test_feeds())
