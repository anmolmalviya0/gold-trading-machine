"""
Multi-Timeframe Consensus Engine
Aggregate signals from 5m/15m/30m/1h and produce final decision
"""
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """Multi-timeframe signal consensus"""
    
    def __init__(self):
        self.timeframes = ['5m', '15m', '30m', '1h']
        self.weights = {
            '5m': 0.15,
            '15m': 0.25,
            '30m': 0.30,
            '1h': 0.30
        }
    
    def aggregate_signals(self, signals: Dict[str, Dict]) -> Dict:
        """
        Aggregate multi-timeframe signals
        
        Args:
            signals: {
                '5m': {'signal': 'BUY', 'confidence': 75, ...},
                '15m': {'signal': 'BUY', 'confidence': 80, ...},
                ...
            }
            
        Returns:
            {
                'final_signal': 'BUY' | 'SELL' | 'NO-TRADE',
                'bias_strength': 'STRONG' | 'MODERATE' | 'WEAK',
                'confidence': 0-100,
                'alignment_count': int,
                'timeframe_votes': {...},
                'reason_codes': [...]
            }
        """
        # Count votes
        buy_count = 0
        sell_count = 0
        no_trade_count = 0
        
        buy_confidence = []
        sell_confidence = []
        
        for tf, sig in signals.items():
            if sig['signal'] == 'BUY':
                buy_count += 1
                buy_confidence.append(sig['confidence'] * self.weights.get(tf, 0.25))
            elif sig['signal'] == 'SELL':
                sell_count += 1
                sell_confidence.append(sig['confidence'] * self.weights.get(tf, 0.25))
            else:
                no_trade_count += 1
        
        # Determine final signal
        total_tfs = len(signals)
        reasons = []
        
        # STRONG alignment (3+ agree)
        if buy_count >= 3:
            final_signal = 'BUY'
            bias_strength = 'STRONG' if buy_count == 4 else 'MODERATE'
            confidence = int(sum(buy_confidence))
            reasons.append(f'{buy_count}_timeframes_bullish')
        
        elif sell_count >= 3:
            final_signal = 'SELL'
            bias_strength = 'STRONG' if sell_count == 4 else 'MODERATE'
            confidence = int(sum(sell_confidence))
            reasons.append(f'{sell_count}_timeframes_bearish')
        
        # MODERATE alignment (2 agree, no strong opposition)
        elif buy_count == 2 and sell_count <= 1:
            final_signal = 'BUY'
            bias_strength = 'MODERATE'
            confidence = int(sum(buy_confidence))
            reasons.append('partial_bullish_alignment')
        
        elif sell_count == 2 and buy_count <= 1:
            final_signal = 'SELL'
            bias_strength = 'MODERATE'
            confidence = int(sum(sell_confidence))
            reasons.append('partial_bearish_alignment')
        
        # Conflict or no clear direction
        else:
            final_signal = 'NO-TRADE'
            bias_strength = 'WEAK'
            confidence = 0
            reasons.append('timeframe_conflict')
            reasons.append('insufficient_alignment')
        
        # Filter low confidence
        if confidence < 60:
            final_signal = 'NO-TRADE'
            bias_strength = 'WEAK'
            reasons.append('confidence_below_threshold')
        
        return {
            'final_signal': final_signal,
            'bias_strength': bias_strength,
            'confidence': confidence,
            'alignment_count': max(buy_count, sell_count),
            'timeframe_votes': {
                'BUY': buy_count,
                'SELL': sell_count,
                'NO-TRADE': no_trade_count
            },
            'reason_codes': reasons
        }


# Demo
if __name__ == "__main__":
    # Sample signals
    signals = {
        '5m': {'signal': 'BUY', 'confidence': 70},
        '15m': {'signal': 'BUY', 'confidence': 80},
        '30m': {'signal': 'BUY', 'confidence': 75},
        '1h': {'signal': 'SELL', 'confidence': 65}  # One dissenter
    }
    
    consensus = ConsensusEngine()
    result = consensus.aggregate_signals(signals)
    
    print("📊 Multi-Timeframe Consensus:")
    print(f"   Final Signal: {result['final_signal']}")
    print(f"   Bias Strength: {result['bias_strength']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Alignment: {result['alignment_count']}/4")
    print(f"   Votes: {result['timeframe_votes']}")
    print(f"   Reasons: {', '.join(result['reason_codes'])}")
