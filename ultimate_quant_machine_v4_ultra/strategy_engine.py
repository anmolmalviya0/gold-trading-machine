"""
Strategy Engine - Score-Based Signal Generation
(Not if/else, but 0-100 score with threshold)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Score-based signal engine.
    Generates trade cards with:
    - Direction (BUY/SELL)
    - Entry zone
    - TP1, TP2, SL
    - Reason
    - Conviction pass/fail
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.scoring = config['signal_scoring']
        self.trade_threshold = self.scoring['trade_threshold']
        self.risk_mode = config['risk'][config['risk']['mode']]
        self.atr_mult = self.risk_mode['sl_ratio']
    
    def generate_signal(self, symbol: str, tf: str, 
                       close_prices: pd.Series,
                       high_prices: pd.Series,
                       low_prices: pd.Series,
                       rsi: pd.Series,
                       atr: pd.Series,
                       ema50: pd.Series,
                       ema200: pd.Series,
                       adx: pd.Series,
                       bb_upper: pd.Series,
                       bb_lower: pd.Series,
                       bb_width: pd.Series,
                       divergences: list,
                       liquidity_sweeps: list,
                       timestamp: int) -> Optional[Dict]:
        """
        Generate complete signal with score.
        Returns trade card dict or None if score < threshold.
        """
        
        current_idx = len(close_prices) - 1
        if current_idx < 50:
            return None
        
        current_price = close_prices.iloc[current_idx]
        current_rsi = rsi.iloc[current_idx]
        current_atr = atr.iloc[current_idx]
        current_adx = adx.iloc[current_idx]
        current_ema50 = ema50.iloc[current_idx]
        current_ema200 = ema200.iloc[current_idx]
        current_bb_width = bb_width.iloc[current_idx]
        
        # ===== SCORING =====
        score = 0.0
        reasons = []
        
        # 1. Divergence confirmed
        divergence_found = None
        for div in divergences:
            if div.get('confirmed'):
                score += self.scoring['divergence_confirmed']
                divergence_found = div
                reasons.append(f"Div:{div['type'][:3]}")
                break
        
        # 2. Trend alignment (EMA200 + EMA50)
        if current_ema50 > current_ema200:
            # Uptrend
            score += self.scoring['trend_alignment']
            reasons.append("UpTrend")
        elif current_ema50 < current_ema200:
            # Downtrend
            score += self.scoring['trend_alignment']
            reasons.append("DnTrend")
        
        # 3. Liquidity sweep confirmation
        for sweep in liquidity_sweeps:
            score += self.scoring['liquidity_sweep']
            reasons.append(f"Sweep:{sweep['type'][:4]}")
            break
        
        # 4. Regime filter (ADX or BB Width)
        if current_adx > self.config['indicators']['adx']['threshold']:
            score += self.scoring['regime_good']
            reasons.append(f"ADX:{current_adx:.1f}")
        elif current_bb_width > current_bb_width * 0.5:  # Volatility
            score += self.scoring['regime_good']
            reasons.append("Volatile")
        
        # ===== DIRECTION & ENTRY LOGIC =====
        direction = None
        entry_zone_low = None
        entry_zone_high = None
        
        if divergences and divergences[0].get('type') == 'bullish':
            direction = 'BUY'
            
            # Entry: below current low
            entry_zone_low = low_prices.iloc[current_idx] * 0.999
            entry_zone_high = current_price
            
        elif divergences and divergences[0].get('type') == 'bearish':
            direction = 'SELL'
            
            # Entry: above current high
            entry_zone_high = high_prices.iloc[current_idx] * 1.001
            entry_zone_low = current_price
        
        # Fallback to RSI oversold/overbought
        if not direction:
            if current_rsi < 30 and current_price < bb_lower.iloc[current_idx]:
                direction = 'BUY'
                entry_zone_low = low_prices.iloc[current_idx] * 0.999
                entry_zone_high = current_price
                score += 10  # Bonus for RSI
                
            elif current_rsi > 70 and current_price > bb_upper.iloc[current_idx]:
                direction = 'SELL'
                entry_zone_high = high_prices.iloc[current_idx] * 1.001
                entry_zone_low = current_price
                score += 10
        
        if not direction:
            return None
        
        # ===== RISK MANAGEMENT =====
        if direction == 'BUY':
            sl = current_price - (current_atr * self.atr_mult)
            risk = current_price - sl
            tp1 = current_price + (risk * self.risk_mode['tp_ratio'])
            tp2 = current_price + (risk * self.risk_mode['tp_ratio'] * 1.5)
        else:  # SELL
            sl = current_price + (current_atr * self.atr_mult)
            risk = sl - current_price
            tp1 = current_price - (risk * self.risk_mode['tp_ratio'])
            tp2 = current_price - (risk * self.risk_mode['tp_ratio'] * 1.5)
        
        # ===== BUILD TRADE CARD =====
        trade_card = {
            'symbol': symbol,
            'timeframe': tf,
            'timestamp': timestamp,
            'direction': direction,
            'entry_zone': (entry_zone_low, entry_zone_high),
            'tp1': tp1,
            'tp2': tp2,
            'sl': sl,
            'score': score,
            'reasons': " | ".join(reasons),
            'regime': self._get_regime_label(current_adx, current_bb_width),
            'divergence_type': divergence_found['type'] if divergence_found else None,
            'data': {
                'rsi': current_rsi,
                'atr': current_atr,
                'adx': current_adx,
                'price': current_price,
            }
        }
        
        # Check threshold
        if score >= self.trade_threshold:
            return trade_card
        
        return None
    
    def _get_regime_label(self, adx: float, bb_width: float) -> str:
        """Classify regime"""
        if adx > 40:
            return "Strong Trend"
        elif adx > 25:
            return "Trend"
        elif bb_width < 0.005:  # Narrow
            return "Squeeze"
        else:
            return "Chop"
    
    @staticmethod
    def generate_dedupe_key(symbol: str, tf: str, direction: str, 
                           entry_zone: Tuple, timestamp: int) -> str:
        """
        Generate deterministic key for signal deduplication.
        Same signal within 1 candle = dedupe.
        """
        key_str = f"{symbol}_{tf}_{direction}_{entry_zone[0]:.2f}_{entry_zone[1]:.2f}_{timestamp // 60000}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def format_trade_card(self, trade_card: Dict) -> str:
        """Format trade card for display/alerts"""
        card = f"""
╔════════════════════════════════════════════════════════════════════╗
║ TRADE CARD - V4 PRO MAX ULTRA                                      ║
╠════════════════════════════════════════════════════════════════════╣
║ Symbol:     {trade_card['symbol']:<55} ║
║ TF:         {trade_card['timeframe']:<55} ║
║ Direction:  {trade_card['direction']:<55} ║
╠════════════════════════════════════════════════════════════════════╣
║ Entry Zone: {trade_card['entry_zone'][0]:>10.2f} - {trade_card['entry_zone'][1]:<10.2f}         ║
║ TP1:        {trade_card['tp1']:>10.2f}                             ║
║ TP2:        {trade_card['tp2']:>10.2f}                             ║
║ SL:         {trade_card['sl']:>10.2f}                              ║
╠════════════════════════════════════════════════════════════════════╣
║ Score:      {trade_card['score']:>6.1f}/100                           ║
║ Regime:     {trade_card['regime']:<55} ║
║ Reason:     {trade_card['reasons']:<55} ║
╚════════════════════════════════════════════════════════════════════╝
        """
        return card
