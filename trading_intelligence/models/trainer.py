"""
Model Trainer - Walk-Forward Training Pipeline
ATR-normalized labels, time-series splits, non-overfit validation
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_store import FeatureStore
from models.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Production training pipeline"""
    
    def __init__(self):
        self.feature_store = FeatureStore()
        self.ensemble = EnsemblePredictor()
    
    def create_labels(self, df: pd.DataFrame, atr_threshold: float = 1.5, horizon: int = 10) -> Dict[str, np.ndarray]:
        """
        Create ATR-normalized labels (NON-OVERFIT)
        
        Args:
            df: DataFrame with OHLCV and ATR
            atr_threshold: ATR multiples for significant move
            horizon: Bars forward to look
            
        Returns:
            {
                'trend': labels for Engine A,
                'meanrev': labels for Engine B,
                'regime': labels for Engine C
            }
        """
        logger.info(f"Creating labels for {len(df)} samples")
        
        close = df['close'].values
        atr = df['atr_14'].values
        
        # Trend labels (ATR-normalized returns)
        trend_labels = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - horizon):
            # Future return
            future_price = close[i + horizon]
            current_price = close[i]
            move = (future_price - current_price) / atr[i]
            
            # Classify
            if move > atr_threshold:
                trend_labels[i] = 2  # UP
            elif move < -atr_threshold:
                trend_labels[i] = 0  # DOWN
            else:
                trend_labels[i] = 1  # FLAT
        
        # Mean reversion labels (detect stretch followed by reversion)
        meanrev_labels = np.zeros(len(df), dtype=int)
        
        if 'bb_position' in df.columns:
            bb_pos = df['bb_position'].values
            
            for i in range(len(df) - horizon):
                # If currently stretched
                if bb_pos[i] > 0.9 or bb_pos[i] < 0.1:
                    # Check if reverts
                    future_bb = bb_pos[i + horizon]
                    if abs(future_bb - 0.5) < abs(bb_pos[i] - 0.5):
                        meanrev_labels[i] = 1  # Reversion
        
        # Regime labels (use actual regime from features)
        regime_labels = np.zeros(len(df), dtype=int)
        
        if 'regime_numeric' in df.columns:
            regime_map = {-1: 0, 0: 1, 1: 2}  # CHOP:0, MEANREV:1, TREND:2
            for i in range(len(df)):
                regime_val = df['regime_numeric'].iloc[i]
                regime_labels[i] = regime_map.get(regime_val, 0)
        
        return {
            'trend': trend_labels,
            'meanrev': meanrev_labels,
            'regime': regime_labels
        }
    
    def walk_forward_train(self, df: pd.DataFrame, n_splits: int = 5) -> Dict:
        """
        Walk-forward validation
        
        Args:
            df: Complete dataset with features
            n_splits: Number of time-series splits
            
        Returns:
            Performance metrics
        """
        logger.info(f"🔄 Walk-forward training with {n_splits} splits")
        
        # Engineer features
        df_features = self.feature_store.engineer_features(df)
        
        # Create labels
        labels = self.create_labels(df_features)
        
        # Get feature columns
        feature_cols = self.feature_store.feature_names
        X = df_features[feature_cols].fillna(0)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = {k: v[train_idx] for k, v in labels.items()}
            y_val = {k: v[val_idx] for k, v in labels.items()}
            
            # Train ensemble
            self.ensemble.train_all(X_train, y_train)
            
            # Validate
            score = self._evaluate_fold(X_val, y_val)
            fold_scores.append(score)
            
            logger.info(f"Fold {fold + 1} - Accuracy: {score['accuracy']:.2%}")
        
        # Final train on all data
        logger.info("Training final model on all data...")
        self.ensemble.train_all(X, labels)
        
        # Average metrics
        avg_metrics = {
            'avg_accuracy': np.mean([s['accuracy'] for s in fold_scores]),
            'std_accuracy': np.std([s['accuracy'] for s in fold_scores]),
            'fold_scores': fold_scores
        }
        
        logger.info(f"✅ Training complete - Avg Accuracy: {avg_metrics['avg_accuracy']:.2%}")
        
        return avg_metrics
    
    def _evaluate_fold(self, X_val: pd.DataFrame, y_val: Dict) -> Dict:
        """Evaluate single fold"""
        correct = 0
        total = 0
        
        for i in range(len(X_val)):
            pred = self.ensemble.predict(X_val.iloc[[i]])
            
            # Compare trend prediction
            trend_true = y_val['trend'][i]
            trend_pred_map = {'DOWN': 0, 'FLAT': 1, 'UP': 2}
            trend_pred = trend_pred_map.get(pred['engine_votes']['trend'], 1)
            
            if trend_pred == trend_true:
                correct += 1
            total += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0
        }
    
    def save_pipeline(self, base_path: str = "trading_intelligence/models/saved"):
        """Save complete pipeline"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save ensemble
        self.ensemble.save_all(f"{base_path}/ensemble")
        
        # Save feature store metadata
        import joblib
        joblib.dump({
            'feature_names': self.feature_store.feature_names
        }, f"{base_path}/feature_store.pkl")
        
        logger.info(f"Pipeline saved to {base_path}")
    
    def load_pipeline(self, base_path: str = "trading_intelligence/models/saved"):
        """Load complete pipeline"""
        import joblib
        
        # Load ensemble
        self.ensemble.load_all(f"{base_path}/ensemble")
        
        # Load feature store
        meta = joblib.load(f"{base_path}/feature_store.pkl")
        self.feature_store.feature_names = meta['feature_names']
        
        logger.info(f"Pipeline loaded from {base_path}")
