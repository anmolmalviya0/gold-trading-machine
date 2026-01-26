"""
AEGIS V21 - LightGBM Adapter
============================
Adapter logic for LightGBM model integration.
"""
import joblib
import numpy as np
import pandas as pd
import os

class LightGBMPredictor:
    """Wrapper to make LightGBM compatible with Executor"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load()
        
    def load(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
                
            self.model = joblib.load(self.model_path)
            print(f"✅ LightGBM Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
            
    def predict(self, features: np.ndarray, threshold: float = 0.65) -> dict:
        """
        Make prediction (compatible with Executor interface)
        
        Args:
            features: Input array. Can be (seq_len, n_feat) or (n_feat,)
            threshold: Confidence threshold for BUY signal
            
        Returns:
            {'signal': 'BUY'/'HOLD', 'confidence': float, 'approved': bool}
        """
        if self.model is None:
            return {'signal': 'ERROR', 'confidence': 0.0, 'approved': False}
            
        # Handle input shape - LightGBM expects 2D array (n_samples, n_features)
        # If input is sequence from LSTM data prep (seq, feat), take last row
        if len(features.shape) == 2:
            features = features[-1, :]  # Take last timestep
            
        # Reshape for single prediction
        features_2d = features.reshape(1, -1)
        
        # Ensure we have correct number of features (13) by trimming or padding if needed
        # Robustness check in case raw data shape varies
        expected_feats = 13
        if features_2d.shape[1] > expected_feats:
            features_2d = features_2d[:, :expected_feats]
            
        try:
            # Get probability of class 1 (BUY)
            prob = float(self.model.predict_proba(features_2d)[0, 1])
            
            # THE GLASS BOX: Explain why (Anti-Gravity Protocol V2)
            try:
                importance = self.model.feature_importances_
                # Feature names matching the training logic
                feature_names = [
                    'RETURNS', 'LOG_RETURNS', 'HIGH_LOW_RATIO', 'CLOSE_OPEN_RATIO',
                    'SMA_RATIO_5', 'SMA_RATIO_10', 'SMA_RATIO_20', 'SMA_RATIO_50',
                    'RSI_NORM', 'MACD_HIST', 'BB_POSITION', 'ATR_RATIO', 'VOLUME_RATIO'
                ]
                
                # Trim feature names to match importance array length if they differ
                # (Safety for cases where model has fewer features)
                feature_names = feature_names[:len(importance)]
                
                top_idx = np.argmax(importance)
                reason = feature_names[top_idx] if top_idx < len(feature_names) else "UNKNOWN"
            except:
                reason = "ENSEMBLE_WEIGHT"

            return {
                'signal': 'BUY' if prob >= threshold else 'HOLD',
                'confidence': prob,
                'approved': prob >= threshold,
                'reason': reason
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'signal': 'ERROR', 'confidence': 0.0, 'approved': False, 'reason': "FAILURE"}
