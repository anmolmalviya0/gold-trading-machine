"""
Ensemble System - Combine all 3 engines
Vote aggregation and confidence scoring
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging
from .engine_a_trend import TrendEngine
from .engine_b_meanrev import MeanReversionEngine
from .engine_c_regime import RegimeEngine

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble voting system"""
    
    def __init__(self):
        self.engine_a = TrendEngine()
        self.engine_b = MeanReversionEngine()
        self.engine_c = RegimeEngine()
        
        # Ensemble weights (can be tuned)
        self.weights = {
            'trend': 0.4,
            'meanrev': 0.3,
            'regime': 0.3
        }
    
    def train_all(self, X: pd.DataFrame, labels: Dict[str, np.ndarray]):
        """
        Train all engines
        
        Args:
            X: Feature DataFrame
            labels: {
                'trend': np.ndarray (0=DOWN, 1=FLAT, 2=UP),
                'meanrev': np.ndarray (binary),
                'regime': np.ndarray (0=CHOP, 1=TREND, 2=MEANREV)
            }
        """
        logger.info("🔧 Training ensemble engines...")
        
        # Train each engine
        self.engine_a.train(X, labels['trend'])
        self.engine_b.train(X, labels['meanrev'])
        self.engine_c.train(X, labels['regime'])
        
        logger.info("✅ All engines trained")
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Get ensemble prediction
        
        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'NO-TRADE',
                'confidence': float (0-100),
                'bias_strength': 'STRONG' | 'MODERATE' | 'WEAK',
                'regime': str,
                'engine_votes': {...},
                'reason_codes': [...]
            }
        """
        # Get individual predictions
        trend_pred = self.engine_a.predict(X)
        meanrev_pred = self.engine_b.predict(X)
        regime_pred = self.engine_c.predict(X)
        
        # Ensemble logic
        signal, confidence, reasons = self._aggregate_votes(
            trend_pred, meanrev_pred, regime_pred
        )
        
        # Bias strength
        if confidence >= 80:
            bias_strength = 'STRONG'
        elif confidence >= 60:
            bias_strength = 'MODERATE'
        else:
            bias_strength = 'WEAK'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'bias_strength': bias_strength,
            'regime': regime_pred['regime'],
            'engine_votes': {
                'trend': trend_pred['direction'],
                'meanrev': meanrev_pred['signal'],
                'regime': regime_pred['regime']
            },
            'reason_codes': reasons
        }
    
    def _aggregate_votes(self, trend, meanrev, regime) -> tuple:
        """
        Aggregate engine votes into final signal
        
        Returns:
            (signal, confidence, reason_codes)
        """
        reasons = []
        
        # Filter CHOP regime
        if regime['regime'] == 'CHOP' and regime['confidence'] > 0.7:
            return 'NO-TRADE', 0, ['chop_regime', 'low_tradability']
        
        # Initialize score
        bullish_score = 0
        bearish_score = 0
        
        # Engine A: Trend
        if trend['direction'] == 'UP':
            bullish_score += self.weights['trend'] * trend['confidence']
            reasons.append('trend_bullish')
        elif trend['direction'] == 'DOWN':
            bearish_score += self.weights['trend'] * trend['confidence']
            reasons.append('trend_bearish')
        
        # Engine B: Mean Reversion
        if meanrev['signal'] == 'REVERT':
            # In MEANREV regime, this strengthens opposite signal
            if regime['regime'] == 'MEANREV':
                # If trending up but overextended, expect reversion down
                if trend['direction'] == 'UP':
                    bearish_score += self.weights['meanrev'] * meanrev['confidence']
                    reasons.append('overextended_reversion')
                elif trend['direction'] == 'DOWN':
                    bullish_score += self.weights['meanrev'] * meanrev['confidence']
                    reasons.append('oversold_reversion')
        
        # Engine C: Regime boost
        if regime['regime'] == 'TREND':
            # Boost trending signals
            if trend['direction'] == 'UP':
                bullish_score *= 1.2
                reasons.append('regime_trend_aligned')
            elif trend['direction'] == 'DOWN':
                bearish_score *= 1.2
                reasons.append('regime_trend_aligned')
        
        # Final decision
        score_diff = abs(bullish_score - bearish_score)
        max_score = max(bullish_score, bearish_score)
        
        if score_diff < 0.15:  # Too close, no clear signal
            return 'NO-TRADE', 0, reasons + ['conflicting_signals']
        
        if bullish_score > bearish_score:
            signal = 'BUY'
            confidence = int(min(max_score * 100, 95))
        else:
            signal = 'SELL'
            confidence = int(min(max_score * 100, 95))
        
        # Filter low confidence
        if confidence < 60:
            return 'NO-TRADE', confidence, reasons + ['low_confidence']
        
        return signal, confidence, reasons
    
    def save_all(self, base_path: str):
        """Save all models"""
        self.engine_a.save(f"{base_path}_trend.pkl")
        self.engine_b.save(f"{base_path}_meanrev.pkl")
        self.engine_c.save(f"{base_path}_regime")
        
        logger.info(f"Ensemble saved to {base_path}")
    
    def load_all(self, base_path: str):
        """Load all models"""
        self.engine_a.load(f"{base_path}_trend.pkl")
        self.engine_b.load(f"{base_path}_meanrev.pkl")
        self.engine_c.load(f"{base_path}_regime")
        
        logger.info(f"Ensemble loaded from {base_path}")
