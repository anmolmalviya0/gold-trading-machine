"""
TERMINAL - LightGBM Backtest Adapter
======================================
Adapts backtest_engine.py to work with LightGBM model.
"""
import pandas as pd
import numpy as np
import joblib
import os


class LightGBMPredictor:
    """Wrapper to make LightGBM compatible with backtest engine"""
    
    def __init__(self, model_path: str = '/Users/anmol/Desktop/gold/terminal_alpha/models/terminal_lgbm.pkl'):
        self.model = joblib.load(model_path)
        self.model_path = model_path
        print(f"✅ LightGBM Model loaded from {model_path}")
    
    def predict(self, features: np.ndarray, threshold: float = 0.65) -> dict:
        """
        Make prediction (compatible with LSTM interface)
        
        Args:
            features: Array of shape (seq_len, num_features) - we only use last row
            threshold: Confidence threshold
            
        Returns:
            {'signal': 'BUY'/'HOLD', 'confidence': float, 'approved': bool}
        """
        # LightGBM doesn't need sequences - just use last row
        if len(features.shape) == 2:
            features = features[-1, :]  # Last timestep
        
        # Ensure correct shape and feature count
        if len(features) > 13:
            features = features[:13]  # Trim to 13 features
        elif len(features) < 13:
            features = np.pad(features, (0, 13 - len(features)), 'constant')
        
        features = features.reshape(1, -1)
        
        # Get probability
        prob = self.model.predict_proba(features)[0, 1]
        
        return {
            'signal': 'BUY' if prob >= threshold else 'HOLD',
            'confidence': float(prob),
            'approved': prob >= threshold
        }


if __name__ == '__main__':
    # Import and run standard backtest
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backtest_engine import BacktestEngine
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          TERMINAL - LIGHTGBM BACKTEST                   ║
    ║              The Great Filter Validation                 ║\n    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Create engine with LightGBM predictor
    engine = BacktestEngine(
        model_path='/Users/anmol/Desktop/gold/terminal_alpha/models/terminal_lgbm.pkl',
        initial_capital=10000,
        risk_per_trade=0.02
    )
    
    # Replace LSTM predictor with LightGBM
    engine.predictor = LightGBMPredictor()
    
    # Load data
    data_path = '/Users/anmol/Desktop/gold/market_data/PAXGUSDT_5m.csv'
    df = pd.read_csv(data_path)
    
    # Run backtest
    results = engine.run(df)
    
    print(f"\n📁 Results saved to logs/backtest_results.txt")
