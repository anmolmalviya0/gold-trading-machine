"""
Engine C: Regime Filter
GRU-based regime classification (TREND/CHOP/MEANREV)
"""
import numpy as np
import pandas as pd
from typing import Dict
import joblib
import logging

logger = logging.getLogger(__name__)


class RegimeEngine:
    """
    GRU-based regime classifier
    Falls back to rule-based if TensorFlow not available
    """
    
    def __init__(self, sequence_length: int = 60):
        self.model = None
        self.sequence_length = sequence_length
        self.feature_names = None
        self.is_trained = False
        self.use_deep_learning = False
        
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            self.tf = tf
            self.use_deep_learning = True
            logger.info("TensorFlow available, using GRU model")
        except ImportError:
            logger.warning("TensorFlow not available, using rule-based regime detection")
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train regime classifier
        
        Args:
            X: Feature DataFrame
            y: Labels (0=CHOP, 1=TREND, 2=MEANREV)
        """
        if not self.use_deep_learning:
            logger.info("Using rule-based regime detector (no training needed)")
            self.is_trained = True
            return
        
        logger.info(f"Training Regime Engine (GRU) on {len(X)} samples")
        
        self.feature_names = list(X.columns)
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X.values, y)
        
        # Build GRU model
        self.model = self._build_gru_model(X_seq.shape[1], X_seq.shape[2])
        
        # Train
        self.model.fit(
            X_seq, y_seq,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        logger.info("✅ Regime Engine (GRU) trained")
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        """Prepare sequences for GRU"""
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_gru_model(self, sequence_length: int, n_features: int):
        """Build GRU architecture"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        
        model = Sequential([
            GRU(64, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(0.2),
            GRU(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Predict regime
        
        Returns:
            {
                'regime': 'TREND' | 'CHOP' | 'MEANREV',
                'confidence': float,
                'probabilities': {...}
            }
        """
        if not self.is_trained:
            return {
                'regime': 'CHOP',
                'confidence': 0.0,
                'probabilities': {'TREND': 0.33, 'CHOP': 0.34, 'MEANREV': 0.33}
            }
        
        # Rule-based fallback
        if not self.use_deep_learning or self.model is None:
            return self._rule_based_regime(X)
        
        # GRU prediction
        X_seq = X[self.feature_names].values[-self.sequence_length:]
        X_seq = X_seq.reshape(1, self.sequence_length, -1)
        
        probs = self.model.predict(X_seq, verbose=0)[0]
        
        regime_idx = np.argmax(probs)
        regime = ['CHOP', 'TREND', 'MEANREV'][regime_idx]
        
        return {
            'regime': regime,
            'confidence': float(probs[regime_idx]),
            'probabilities': {
                'TREND': float(probs[1]),
                'CHOP': float(probs[0]),
                'MEANREV': float(probs[2])
            }
        }
    
    def _rule_based_regime(self, X: pd.DataFrame) -> Dict:
        """Rule-based regime classification (fallback)"""
        row = X.iloc[-1]
        
        adx = row.get('adx', 25)
        bb_width = row.get('bb_width', 0.1)
        
        # Simple rules
        if adx < 20 and bb_width < 0.08:
            regime = 'CHOP'
            confidence = 0.7
        elif adx > 30:
            regime = 'TREND'
            confidence = 0.8
        elif bb_width > 0.12:
            regime = 'MEANREV'
            confidence = 0.75
        else:
            regime = 'CHOP'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'probabilities': {
                'TREND': 0.6 if regime == 'TREND' else 0.2,
                'CHOP': 0.6 if regime == 'CHOP' else 0.2,
                'MEANREV': 0.6 if regime == 'MEANREV' else 0.2
            }
        }
    
    def save(self, path: str):
        """Save model"""
        if not self.is_trained:
            return
        
        if self.use_deep_learning and self.model:
            self.model.save(path + '_gru.h5')
        
        joblib.dump({
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'use_deep_learning': self.use_deep_learning
        }, path + '_meta.pkl')
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        try:
            meta = joblib.load(path + '_meta.pkl')
            self.feature_names = meta['feature_names']
            self.sequence_length = meta['sequence_length']
            self.use_deep_learning = meta['use_deep_learning']
            
            if self.use_deep_learning:
                from tensorflow.keras.models import load_model
                self.model = load_model(path + '_gru.h5')
            
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
