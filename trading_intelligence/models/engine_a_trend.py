"""
Engine A: Trend Direction Predictor
XGBoost classifier for UP/DOWN/FLAT predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class TrendEngine:
    """XGBoost-based trend direction predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Model hyperparameters
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # UP, FLAT, DOWN
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train the trend model
        
        Args:
            X: Feature DataFrame
            y: Labels (0=DOWN, 1=FLAT, 2=UP)
        """
        logger.info(f"Training Trend Engine on {len(X)} samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False
        )
        
        self.is_trained = True
        logger.info("✅ Trend Engine trained")
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Predict trend direction
        
        Args:
            X: Feature DataFrame (single row or multiple)
            
        Returns:
            {
                'direction': 'UP' | 'FLAT' | 'DOWN',
                'confidence': float (0-1),
                'probabilities': {'UP': float, 'FLAT': float, 'DOWN': float}
            }
        """
        if not self.is_trained:
            return {
                'direction': 'FLAT',
                'confidence': 0.0,
                'probabilities': {'UP': 0.33, 'FLAT': 0.34, 'DOWN': 0.33}
            }
        
        # Ensure features match training
        X = X[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        
        # Map to labels
        pred_class = np.argmax(probs)
        direction = ['DOWN', 'FLAT', 'UP'][pred_class]
        
        # Confidence is the max probability
        confidence = probs[pred_class]
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probabilities': {
                'UP': float(probs[2]),
                'FLAT': float(probs[1]),
                'DOWN': float(probs[0])
            }
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances"""
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def save(self, path: str):
        """Save model to disk"""
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_trained = True
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
