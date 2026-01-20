"""
Engine B: Mean Reversion Predictor
LightGBM for detecting stretch/reversion probability
"""
import numpy as np
import pandas as pd
from typing import Dict
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class MeanReversionEngine:
    """LightGBM-based mean reversion predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # LightGBM parameters
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 150
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train mean reversion model
        
        Args:
            X: Feature DataFrame
            y: Binary labels (1=reversion expected, 0=continuation)
        """
        logger.info(f"Training Mean Reversion Engine on {len(X)} samples")
        
        self.feature_names = list(X.columns)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train LightGBM
        train_data = lgb.Dataset(X_scaled, label=y)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params['n_estimators'],
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        self.is_trained = True
        logger.info("✅ Mean Reversion Engine trained")
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Predict reversion probability
        
        Returns:
            {
                'reversion_probability': float (0-1),
                'signal': 'REVERT' | 'CONTINUE',
                'confidence': float
            }
        """
        if not self.is_trained:
            return {
                'reversion_probability': 0.5,
                'signal': 'CONTINUE',
                'confidence': 0.0
            }
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prob = self.model.predict(X_scaled)[0]
        
        signal = 'REVERT' if prob > 0.6 else 'CONTINUE'
        confidence = abs(prob - 0.5) * 2  # 0-1 scale
        
        return {
            'reversion_probability': float(prob),
            'signal': signal,
            'confidence': float(confidence)
        }
    
    def save(self, path: str):
        """Save model"""
        if not self.is_trained:
            return
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_trained = True
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
