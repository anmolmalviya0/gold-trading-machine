import pandas as pd
import numpy as np
from datetime import datetime, timezone
from .features import add_features

def get_market_regime(df, config):
    """
    Determine if we are in Trending or Ranging market using ADX.
    """
    last = df.iloc[-1]
    adx = last.get('adx', 0)
    
    if adx > 25:
        return 'TRENDING'
    else:
        return 'RANGING'

class SignalEngine:
    def __init__(self, config):
        self.config = config
        self.min_confirmations = config['signals']['min_confirmations']
        self.threshold = config['signals']['threshold']

    def generate_signal(self, df, symbol, timeframe):
        """
        [STRATEGY ROUTER]
        Dispatches the appropriate strategy based on the Asset Class.
        """
        # Ensure features
        if 'rsi' not in df.columns:
            df = add_features(df, self.config)
            
        if len(df) < 100: return None
        
        # Dispatcher
        if 'SOL' in symbol:
            return self.strategy_sol_trend_sniper(df, symbol, timeframe)
        elif 'BTC' in symbol:
            return self.strategy_btc_mean_reversion(df, symbol, timeframe)
        elif 'PAXG' in symbol or 'GOLD' in symbol:
            return self.strategy_gold_volatility_breakout(df, symbol, timeframe)
        elif 'BNB' in symbol:
            return self.strategy_bnb_range_bound(df, symbol, timeframe)
        elif 'ETH' in symbol:
            return self.strategy_eth_momentum(df, symbol, timeframe)
        else:
            # Default fallback (Trend Only)
            return self.strategy_sol_trend_sniper(df, symbol, timeframe)

    def strategy_sol_trend_sniper(self, df, symbol, timeframe):
        """
        Original 'Vintage' Strategy (62% WR on SOL).
        Logic: Daily Trend + SuperTrend + ADX + HA Green
        """
        last = df.iloc[-1]
        
        # 1. Macro Filter
        if last['close'] < last.get('trend_ema', 0): return None
        
        # 2. SuperTrend Filter
        if last.get('st_trend', 0) != 1: return None
        
        # 3. ADX Filter
        if last.get('adx', 0) < 25: return None
        
        # 4. HA Confirmation
        if last.get('ha_close', 0) <= last.get('ha_open', 0): return None
        
        return self._build_signal_payload(last, symbol, timeframe, "SOL Trend Sniper", sl_type='supertrend', tp_mult=4.0)

    def strategy_btc_mean_reversion(self, df, symbol, timeframe):
        """
        Bitcoin Mean Reversion Strategy.
        Logic: Buy the Dip in Bull Market (RSI < 30)
        """
        last = df.iloc[-1]
        
        # 1. Macro Filter (Must be in Bull Market)
        if last['close'] < last.get('trend_ema', 0): return None
        
        # 2. Entry Trigger: Deep Oversold
        rsi = last.get('rsi', 50)
        if rsi >= 30: return None # Strictly wait for < 30
        
        # 3. Confirmation: Close > Open (Green Candle)
        if last['close'] <= last['open']: return None
        
        return self._build_signal_payload(last, symbol, timeframe, "BTC Mean Reversion", sl_type='atr_wide', tp_mult=2.0)

    def strategy_gold_volatility_breakout(self, df, symbol, timeframe):
        """
        Gold (PAXG) Volatility Breakout.
        Logic: Bollinger Squeeze -> Breakout confirming Volume
        """
        last = df.iloc[-1]
        
        # 1. Squeeze Filter: Bandwidth must be low (consolidation)
        # Using vol_ratio as proxy for squeeze (low vs avg)
        vol_ratio = last.get('vol_ratio', 1.0)
        if vol_ratio > 0.8: return None # Wait for squeeze (< 0.8)
        
        # 2. Breakout Trigger: Price > Upper BB
        if last['close'] <= last.get('bb_upper', 0): return None
        
        # 3. Volume Confirmation
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] < 1.2 * vol_ma: return None # Need 20% Volume Spike
        
        return self._build_signal_payload(last, symbol, timeframe, "Gold Vol Breakout", sl_type='atr_tight', tp_mult=1.5)

    def strategy_bnb_range_bound(self, df, symbol, timeframe):
        """
        BNB Range Bound Strategy.
        Logic: Buy Lower BB Support in Sideways Market
        """
        last = df.iloc[-1]
        
        # 1. Regime Filter: ADX must be LOW (< 20) for Range Bound
        if last.get('adx', 0) > 25: return None # Trending, skip
        
        # 2. Support Entry: Price < Lower BB
        if last['close'] >= last.get('bb_lower', 0): return None
        
        # 3. Reversal Candle
        if last['close'] <= last['open']: return None
        
        return self._build_signal_payload(last, symbol, timeframe, "BNB Range Scalar", sl_type='atr_med', tp_mult=1.0)

    def strategy_eth_momentum(self, df, symbol, timeframe):
        """
        ETH Momentum Strategy (Similar to SOL but looser)
        """
        last = df.iloc[-1]
        if last['close'] < last.get('trend_ema', 0): return None
        if last.get('st_trend', 0) != 1: return None
        
        return self._build_signal_payload(last, symbol, timeframe, "ETH Momentum", sl_type='supertrend', tp_mult=3.0)

    def _build_signal_payload(self, last, symbol, timeframe, reason, sl_type='atr_med', tp_mult=2.0):
        entry = float(last['close'])
        atr = float(last.get('atr', 0))
        
        if sl_type == 'supertrend':
            st = float(last.get('supertrend', 0))
            if st > 0 and not pd.isna(st):
                sl = st
            else:
                sl = entry - (2.0 * atr)
        elif sl_type == 'atr_wide':
            sl = entry - (2.5 * atr)
        elif sl_type == 'atr_tight':
            sl = entry - (1.0 * atr)
        else: # atr_med
            sl = entry - (1.5 * atr)
            
        tp = entry + (tp_mult * atr)
        
        return {
            'timestamp': str(last.name) if hasattr(last, 'name') else datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'BUY',
            'score': 100,
            'side': 'BUY',
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'reasons': [reason],
            'signals': {reason: 100},
            'rsi': float(last.get('rsi', 50)),
            'atr': atr,
            'supertrend': float(last.get('supertrend', 0))
        }
