"""
PHASE 2: ENSEMBLE TRAINING WITH EXPLAINABILITY
===============================================
LightGBM + RandomForest ensemble with SHAP explainability.

Features:
- LightGBM baseline (fast, robust)
- RandomForest for ensemble diversity
- SHAP for feature attribution
- Probability calibration
- Model versioning

Usage:
    python train_ensemble.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'datasets'
MODEL_DIR = BASE_DIR / 'models' / 'ensemble'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_state': 42,
}

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'n_jobs': -1,
    'random_state': 42,
}

FEATURE_COLS = [
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
    'rsi', 'macd_hist',
    'dist_sma10', 'dist_sma20', 'dist_sma50', 'dist_sma100',
    'trend_10_20', 'trend_20_50', 'trend_50_100',
    'atr_pct', 'vol_20', 'vol_rank',
    'bb_width', 'bb_position',
    'vol_ratio', 'vol_zscore',
    'roc_5', 'roc_10', 'roc_20',
]


# === DATA PREPARATION ===

def prepare_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train/val/test splits (time-based, no shuffle)"""
    
    # Clean data
    valid_cols = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=valid_cols + ['label']).copy()
    
    n = len(df_clean)
    train_end = int(n * TRAIN_CONFIG['train_ratio'])
    val_end = int(n * (TRAIN_CONFIG['train_ratio'] + TRAIN_CONFIG['val_ratio']))
    
    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]
    
    return train_df, val_df, test_df


# === MODEL TRAINING ===

class EnsembleModel:
    """
    Ensemble of LightGBM + RandomForest with SHAP explainability.
    """
    
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.scaler = RobustScaler()
        self.lgbm_model = None
        self.rf_model = None
        self.lgbm_weight = 0.6
        self.rf_weight = 0.4
        self.shap_explainer = None
        self.feature_importance = {}
        self.metadata = {}
    
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train ensemble models"""
        print("\n🤖 Training Ensemble...")
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df['label'].astype(int).values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Validation set
        if val_df is not None:
            X_val = self.scaler.transform(val_df[self.feature_cols].values)
            y_val = val_df['label'].astype(int).values
        
        # Train LightGBM
        print("   Training LightGBM...")
        self.lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        
        if val_df is not None:
            self.lgbm_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            self.lgbm_model.fit(X_train_scaled, y_train)
        
        # Train RandomForest
        print("   Training RandomForest...")
        self.rf_model = RandomForestClassifier(**RF_PARAMS)
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Feature importance (from LightGBM)
        self.feature_importance = dict(zip(
            self.feature_cols, 
            self.lgbm_model.feature_importances_
        ))
        
        # SHAP explainer
        if SHAP_AVAILABLE:
            print("   Creating SHAP explainer...")
            self.shap_explainer = shap.TreeExplainer(self.lgbm_model)
        
        # Metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'train_samples': len(train_df),
            'val_samples': len(val_df) if val_df is not None else 0,
            'features': self.feature_cols,
            'lgbm_params': LGBM_PARAMS,
            'rf_params': RF_PARAMS,
        }
        
        print("   ✅ Training complete")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions"""
        X_scaled = self.scaler.transform(X)
        
        lgbm_proba = self.lgbm_model.predict_proba(X_scaled)[:, 1]
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (
            self.lgbm_weight * lgbm_proba + 
            self.rf_weight * rf_proba
        )
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def explain(self, X: np.ndarray, n_samples: int = 100) -> Dict:
        """Get SHAP explanations for predictions"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {'error': 'SHAP not available'}
        
        X_scaled = self.scaler.transform(X[:n_samples])
        shap_values = self.shap_explainer.shap_values(X_scaled)
        
        # Handle binary classification SHAP output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_shap = dict(zip(self.feature_cols, mean_shap))
        
        # Top features
        top_features = sorted(feature_shap.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'top_features': top_features,
            'all_features': feature_shap
        }
    
    def get_signal_reasons(self, X: np.ndarray) -> List[str]:
        """Get top 3 reason codes for each prediction"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return ['SHAP not available']
        
        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        shap_values = self.shap_explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        reasons = []
        for i in range(len(X_scaled)):
            # Get top 3 features by absolute SHAP value
            feature_impacts = list(zip(self.feature_cols, shap_values[i]))
            top_3 = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)[:3]
            
            reason_codes = []
            for feat, val in top_3:
                direction = "+" if val > 0 else "-"
                reason_codes.append(f"{feat}{direction}")
            
            reasons.append(", ".join(reason_codes))
        
        return reasons
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate model on test set"""
        X_test = test_df[self.feature_cols].values
        y_test = test_df['label'].astype(int).values
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics
    
    def save(self, path: Path, symbol: str):
        """Save model and metadata"""
        model_data = {
            'scaler': self.scaler,
            'lgbm_model': self.lgbm_model,
            'rf_model': self.rf_model,
            'lgbm_weight': self.lgbm_weight,
            'rf_weight': self.rf_weight,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'metadata': self.metadata,
        }
        
        model_path = path / f"{symbol}_ensemble.pkl"
        joblib.dump(model_data, model_path)
        
        # Save metadata JSON
        meta_path = path / f"{symbol}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"   💾 Saved: {model_path.name}")
        return model_path
    
    @classmethod
    def load(cls, path: Path, symbol: str):
        """Load saved model"""
        model_path = path / f"{symbol}_ensemble.pkl"
        data = joblib.load(model_path)
        
        model = cls(data['feature_cols'])
        model.scaler = data['scaler']
        model.lgbm_model = data['lgbm_model']
        model.rf_model = data['rf_model']
        model.lgbm_weight = data['lgbm_weight']
        model.rf_weight = data['rf_weight']
        model.feature_importance = data['feature_importance']
        model.metadata = data['metadata']
        
        # Recreate SHAP explainer
        if SHAP_AVAILABLE:
            model.shap_explainer = shap.TreeExplainer(model.lgbm_model)
        
        return model


# === MAIN ===

if __name__ == "__main__":
    print("="*70)
    print("🧠 PHASE 2: ENSEMBLE TRAINING WITH EXPLAINABILITY")
    print("="*70)
    
    symbols = ['BTCUSDT', 'PAXGUSDT']
    
    for symbol in symbols:
        print(f"\n🔧 Training {symbol}...")
        
        # Load data
        data_path = DATASET_DIR / f"{symbol}_1h_labeled.parquet"
        
        if not data_path.exists():
            print(f"   ⚠️ No labeled data. Run label_and_features.py first.")
            continue
        
        df = pd.read_parquet(data_path)
        print(f"   Loaded: {len(df):,} rows")
        
        # Prepare splits
        train_df, val_df, test_df = prepare_data(df, FEATURE_COLS)
        print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        
        # Train ensemble
        model = EnsembleModel(FEATURE_COLS)
        model.fit(train_df, val_df)
        
        # Evaluate
        metrics = model.evaluate(test_df)
        print(f"\n   📊 TEST METRICS:")
        print(f"      Accuracy:  {metrics['accuracy']*100:.1f}%")
        print(f"      Precision: {metrics['precision']*100:.1f}%")
        print(f"      Recall:    {metrics['recall']*100:.1f}%")
        print(f"      F1:        {metrics['f1']*100:.1f}%")
        print(f"      ROC-AUC:   {metrics['roc_auc']*100:.1f}%")
        
        # Feature importance
        top_features = sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   🔑 TOP FEATURES:")
        for feat, imp in top_features:
            print(f"      {feat}: {imp:.2f}")
        
        # SHAP explanation
        if SHAP_AVAILABLE:
            explanations = model.explain(test_df[FEATURE_COLS].values)
            print(f"\n   🧪 SHAP TOP FEATURES:")
            for feat, shap_val in explanations['top_features'][:5]:
                print(f"      {feat}: {shap_val:.4f}")
        
        # Save model
        model.metadata['test_metrics'] = metrics
        model.save(MODEL_DIR, symbol)
    
    print("\n" + "="*70)
    print("✅ ENSEMBLE TRAINING COMPLETE")
    print("="*70)
    print(f"   Models saved to: {MODEL_DIR}")
