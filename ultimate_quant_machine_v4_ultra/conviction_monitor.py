"""
Conviction Monitor - USD-Based Gatekeeper
(Spread, OBI, Depth, Slippage Impact)
"""

import logging
from typing import Optional, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ConvictionMonitor:
    """
    Real-time conviction gates:
    - Spread filter (in %)
    - USD depth availability
    - Slippage impact estimation
    - OBI (Order Book Imbalance) in USD
    """
    
    def __init__(self, config: Dict):
        self.config = config['conviction']
        self.spread_filter = self.config.get('spread_filter', {})
        self.depth_usd_min = self.config.get('depth_usd_min', 50000)
        self.slippage_impact_max = self.config.get('slippage_impact_max', 0.001)
        self.obi_buy_min = self.config['obi_usd'].get('buy_min', 0.80)
        self.obi_sell_max = self.config['obi_usd'].get('sell_max', 1.20)
        self.require_all_gates = self.config.get('require_all_gates', True)
    
    def check_all_gates(self, symbol: str, orderbook: Dict, 
                        direction: str, size_usd: float) -> Tuple[bool, str]:
        """
        Check all conviction gates.
        Returns: (conviction_pass, reason_string)
        Default: reject on missing data
        """
        
        if not orderbook:
            return False, "GATE_FAIL: No orderbook data"
        
        # Gate 1: Spread filter
        spread_pass, spread_msg = self._check_spread(symbol, orderbook)
        if not spread_pass:
            return False, spread_msg
        
        # Gate 2: Depth in USD
        depth_pass, depth_msg = self._check_depth(orderbook, size_usd)
        if not depth_pass:
            return False, depth_msg
        
        # Gate 3: Slippage impact
        impact_pass, impact_msg = self._check_slippage_impact(
            orderbook, direction, size_usd
        )
        if not impact_pass:
            return False, impact_msg
        
        # Gate 4: OBI in USD
        obi_pass, obi_msg = self._check_obi(orderbook, direction)
        if not obi_pass:
            return False, obi_msg
        
        return True, "CONVICTION_PASS: All gates OK"
    
    def _check_spread(self, symbol: str, orderbook: Dict) -> Tuple[bool, str]:
        """Gate 1: Spread filter (in %)"""
        bid = orderbook.get('bid')
        ask = orderbook.get('ask')
        
        if not bid or not ask:
            return False, "GATE_FAIL: No bid/ask"
        
        spread_pct = (ask - bid) / bid
        
        # Get threshold for symbol
        threshold = self.spread_filter.get(
            symbol.replace('USDT', ''),
            self.spread_filter.get('default', 0.0006)
        )
        
        if spread_pct > threshold:
            return False, f"GATE_FAIL: Spread {spread_pct*10000:.1f}bps > {threshold*10000:.1f}bps"
        
        return True, f"Spread OK: {spread_pct*10000:.1f}bps"
    
    def _check_depth(self, orderbook: Dict, size_usd: float) -> Tuple[bool, str]:
        """Gate 2: USD depth on both sides"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return False, "GATE_FAIL: No bids/asks"
        
        # Calculate USD depth on bid side
        bid_usd = sum(price * amount for price, amount in bids[:10])
        
        # Calculate USD depth on ask side
        ask_usd = sum(price * amount for price, amount in asks[:10])
        
        if bid_usd < self.depth_usd_min or ask_usd < self.depth_usd_min:
            return False, f"GATE_FAIL: Depth {bid_usd:.0f}/{ask_usd:.0f} USD < {self.depth_usd_min:.0f}"
        
        if ask_usd < size_usd or bid_usd < size_usd:
            return False, f"GATE_FAIL: Not enough depth for size {size_usd:.0f} USD"
        
        return True, f"Depth OK: {bid_usd:.0f}/{ask_usd:.0f} USD"
    
    def _check_slippage_impact(self, orderbook: Dict, direction: str, 
                               size_usd: float) -> Tuple[bool, str]:
        """Gate 3: Estimate slippage impact if filling across levels"""
        orders = orderbook.get('asks' if direction == 'buy' else 'bids', [])
        
        if not orders:
            return False, "GATE_FAIL: No orders on target side"
        
        remaining = size_usd
        total_cost = 0
        
        for price, amount in orders:
            level_usd = price * amount
            
            if remaining <= level_usd:
                total_cost += remaining
                break
            else:
                total_cost += level_usd
                remaining -= level_usd
        
        if remaining > 0:
            return False, "GATE_FAIL: Insufficient liquidity"
        
        avg_price = total_cost / size_usd if size_usd > 0 else 0
        mid_price = (orderbook.get('bid', 0) + orderbook.get('ask', 0)) / 2
        
        if mid_price == 0:
            return False, "GATE_FAIL: Invalid price"
        
        impact = abs(avg_price - mid_price) / mid_price
        
        if impact > self.slippage_impact_max:
            return False, f"GATE_FAIL: Slippage impact {impact*100:.2f}% > {self.slippage_impact_max*100:.2f}%"
        
        return True, f"Impact OK: {impact*100:.3f}%"
    
    def _check_obi(self, orderbook: Dict, direction: str) -> Tuple[bool, str]:
        """Gate 4: Order Book Imbalance (OBI) in USD"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return False, "GATE_FAIL: No bids/asks for OBI"
        
        # Calculate USD totals on top levels (e.g., top 5)
        bid_usd = sum(price * amount for price, amount in bids[:5])
        ask_usd = sum(price * amount for price, amount in asks[:5])
        
        if ask_usd == 0:
            return False, "GATE_FAIL: No ask depth"
        
        obi_ratio = bid_usd / ask_usd
        
        if direction == 'buy':
            # For buy, we want bid side stronger
            if obi_ratio < self.obi_buy_min:
                return False, f"GATE_FAIL: OBI {obi_ratio:.2f} < {self.obi_buy_min:.2f} (need stronger bid)"
        else:  # sell
            # For sell, we want ask side stronger
            if obi_ratio > self.obi_sell_max:
                return False, f"GATE_FAIL: OBI {obi_ratio:.2f} > {self.obi_sell_max:.2f} (need stronger ask)"
        
        return True, f"OBI OK: {obi_ratio:.2f}"
