"""
TERMINAL - LightGBM Adapter
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
            
    def predict(self, features: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Make prediction (compatible with Executor interface)
        
        Args:
            features: Input array. Can be (seq_len, n_feat) or (n_feat,)
            threshold: Confidence threshold for BUY/SELL signals
            
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
        
        # Ensure we have correct number of features (15)
        expected_feats = 15
        feature_names = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio',
            'adx', 'vol_z_score'
        ]
        
        if features_2d.shape[1] > expected_feats:
            features_2d = features_2d[:, :expected_feats]
            
        # Convert to DataFrame to avoid feature name warnings
        df_feats = pd.DataFrame(features_2d, columns=feature_names[:features_2d.shape[1]])
        
        try:
            # Get probability of class 1 (BUY)
            prob = float(self.model.predict_proba(df_feats)[0, 1])
            
            # THE GLASS BOX: Explain why
            try:
                importance = self.model.feature_importances_
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
