"""
INSTITUTIONAL QUANT MODEL
=========================
Implements De Prado standards:
- Triple Barrier Labeling
- Meta-Labeling Ensemble (Primary + Secondary models)
- Purged K-Fold Cross Validation

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TRIPLE BARRIER LABELING
# =============================================================================

def apply_triple_barrier(df: pd.DataFrame,
                          pt_sl: Tuple[float, float] = (2.0, 1.0),
                          max_holding: int = 10,
                          min_return: float = 0.0) -> pd.DataFrame:
    """
    Apply Triple Barrier Method for labeling.
    
    The triple barrier consists of:
    1. Upper horizontal barrier (Take Profit)
    2. Lower horizontal barrier (Stop Loss)
    3. Vertical barrier (Max holding period)
    
    Parameters:
    -----------
    df : DataFrame with columns ['c', 'atr'] (close price and ATR)
    pt_sl : Tuple of (profit_take_mult, stop_loss_mult) in ATR units
    max_holding : Maximum bars to hold position
    min_return : Minimum return threshold
    
    Returns:
    --------
    DataFrame with 'barrier_label', 'barrier_ret', 'barrier_touch'
    """
    df = df.copy()
    pt_mult, sl_mult = pt_sl
    
    labels = []
    returns = []
    touches = []
    
    for i in range(len(df) - max_holding):
        entry_price = df['c'].iloc[i]
        atr = df['atr'].iloc[i] if 'atr' in df.columns else entry_price * 0.01
        
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01
        
        # Barriers
        upper = entry_price + atr * pt_mult  # Take profit
        lower = entry_price - atr * sl_mult  # Stop loss
        
        label = 0
        ret = 0
        touch = 'timeout'
        
        for j in range(1, max_holding + 1):
            if i + j >= len(df):
                break
            
            high = df['h'].iloc[i + j] if 'h' in df.columns else df['c'].iloc[i + j] * 1.01
            low = df['l'].iloc[i + j] if 'l' in df.columns else df['c'].iloc[i + j] * 0.99
            close = df['c'].iloc[i + j]
            
            # Check upper barrier (win)
            if high >= upper:
                label = 1
                ret = (upper - entry_price) / entry_price
                touch = 'upper'
                break
            
            # Check lower barrier (loss)
            if low <= lower:
                label = 0
                ret = (lower - entry_price) / entry_price
                touch = 'lower'
                break
        
        # Vertical barrier (timeout)
        if touch == 'timeout':
            final_price = df['c'].iloc[i + max_holding] if i + max_holding < len(df) else entry_price
            ret = (final_price - entry_price) / entry_price
            label = 1 if ret > min_return else 0
        
        labels.append(label)
        returns.append(ret)
        touches.append(touch)
    
    # Pad end
    labels.extend([np.nan] * max_holding)
    returns.extend([np.nan] * max_holding)
    touches.extend([None] * max_holding)
    
    df['barrier_label'] = labels
    df['barrier_ret'] = returns  
    df['barrier_touch'] = touches
    
    return df


def get_side_labels(df: pd.DataFrame, 
                     threshold: float = 0.0) -> pd.Series:
    """
    Get side labels: 1 = Long, -1 = Short, 0 = No trade
    
    Uses momentum and trend signals to determine primary direction.
    """
    # Combine multiple signals for direction
    signals = pd.Series(0, index=df.index)
    
    # Momentum
    if 'ret_5' in df.columns:
        signals += np.sign(df['ret_5'].fillna(0))
    
    # Trend
    if 'trend_20_50' in df.columns:
        signals += np.sign(df['trend_20_50'].fillna(0))
    
    # RSI extreme
    if 'rsi' in df.columns:
        signals += np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
    
    # Normalize to -1, 0, 1
    return np.sign(signals)


# =============================================================================
# PURGED K-FOLD CROSS VALIDATION
# =============================================================================

class PurgedKFold:
    """
    Purged K-Fold Cross Validation for time series.
    
    Ensures no data leakage by:
    1. Purging: Removing observations from training that could leak to test
    2. Embargo: Adding gap between train and test sets
    
    Parameters:
    -----------
    n_splits : Number of CV folds
    embargo_pct : Percentage of training set to embargo after test set
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate purged train/test indices.
        
        Yields:
        -------
        train_indices, test_indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Fold size
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Test set: i-th fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Train set: everything before test (with embargo) and after
            train_before = indices[:max(0, test_start - embargo_size)]
            train_after = indices[test_end + embargo_size:]
            
            train_indices = np.concatenate([train_before, train_after])
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


def purged_cross_val_score(model, X: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5, 
                            embargo_pct: float = 0.01,
                            scoring: str = 'accuracy') -> dict:
    """
    Perform purged cross-validation and return scores.
    
    Parameters:
    -----------
    model : Sklearn-compatible model
    X : Features
    y : Labels
    n_splits : Number of CV folds
    embargo_pct : Embargo percentage
    scoring : Metric to use
    
    Returns:
    --------
    Dict with scores per fold and mean
    """
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    
    scores = []
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Score
        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'precision':
            score = precision_score(y_test, y_pred, zero_division=0)
        elif scoring == 'recall':
            score = recall_score(y_test, y_pred, zero_division=0)
        elif scoring == 'f1':
            score = f1_score(y_test, y_pred, zero_division=0)
        elif scoring == 'log_loss' and y_proba is not None:
            score = -log_loss(y_test, y_proba)
        else:
            score = accuracy_score(y_test, y_pred)
        
        scores.append(score)
        fold_details.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score
        })
    
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'folds': fold_details
    }


# =============================================================================
# META-LABELING ENSEMBLE
# =============================================================================

class MetaLabelingEnsemble:
    """
    Meta-Labeling Ensemble following De Prado methodology.
    
    Architecture:
    1. Primary Model: Predicts SIDE (Long/Short direction)
    2. Meta Model: Predicts SIZE (probability that primary is correct)
    
    Only trades when:
    - Primary signals a direction (Long or Short)
    - Meta model confidence > threshold (e.g., 70%)
    
    This separates "what to trade" from "how much to bet".
    """
    
    def __init__(self, 
                 primary_model=None,
                 meta_model=None,
                 meta_threshold: float = 0.70):
        """
        Parameters:
        -----------
        primary_model : Model for side prediction (default: RandomForest)
        meta_model : Model for bet sizing (default: GradientBoosting)
        meta_threshold : Minimum meta confidence to trade
        """
        self.primary_model = primary_model or RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=30,
            random_state=42,
            n_jobs=-1
        )
        
        self.meta_model = meta_model or GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.meta_threshold = meta_threshold
        self.is_fitted = False
        self.feature_cols = None
    
    def fit(self, X: pd.DataFrame, y_side: pd.Series, y_label: pd.Series,
            feature_cols: List[str] = None):
        """
        Train both primary and meta models.
        
        Parameters:
        -----------
        X : Features
        y_side : Side labels (-1, 0, 1) for direction
        y_label : Binary labels (0, 1) for win/loss
        feature_cols : Feature column names
        """
        self.feature_cols = feature_cols or X.columns.tolist()
        
        # Clean data
        X_clean = X[self.feature_cols].copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        valid_mask = ~X_clean.isna().any(axis=1) & ~y_side.isna() & ~y_label.isna()
        
        X_clean = X_clean[valid_mask]
        y_side_clean = y_side[valid_mask]
        y_label_clean = y_label[valid_mask]
        
        # Clip extremes
        for col in X_clean.columns:
            q1, q99 = X_clean[col].quantile(0.01), X_clean[col].quantile(0.99)
            X_clean[col] = X_clean[col].clip(q1, q99)
        
        print(f"   Training on {len(X_clean)} samples...")
        
        # 1. Train Primary Model (Side Prediction)
        # Convert side to binary: 1 = Long, 0 = Short/Neutral
        y_primary = (y_side_clean > 0).astype(int)
        self.primary_model.fit(X_clean, y_primary)
        print(f"   ✅ Primary model trained")
        
        # 2. Get primary predictions for meta-labeling
        primary_pred = self.primary_model.predict(X_clean)
        
        # 3. Meta labels: 1 if primary was correct, 0 otherwise
        # For longs (primary_pred=1): correct if y_label_clean=1
        # For shorts (primary_pred=0): correct if y_label_clean=0
        y_meta = ((primary_pred == 1) & (y_label_clean == 1)) | \
                 ((primary_pred == 0) & (y_label_clean == 0))
        y_meta = y_meta.astype(int)
        
        # Add primary prediction as feature for meta model
        X_meta = X_clean.copy()
        X_meta['primary_pred'] = primary_pred
        X_meta['primary_proba'] = self.primary_model.predict_proba(X_clean)[:, 1]
        
        self.meta_model.fit(X_meta, y_meta)
        print(f"   ✅ Meta model trained")
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> dict:
        """
        Generate trading signal with meta-labeled confidence.
        
        Returns:
        --------
        Dict with 'side', 'confidence', 'should_trade', 'bet_size'
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_clean = X[self.feature_cols].copy()
        X_clean = X_clean.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Primary prediction
        primary_pred = self.primary_model.predict(X_clean)[0]
        primary_proba = self.primary_model.predict_proba(X_clean)[0]
        
        side = 'LONG' if primary_pred == 1 else 'SHORT'
        side_confidence = primary_proba[1] if primary_pred == 1 else primary_proba[0]
        
        # Meta prediction
        X_meta = X_clean.copy()
        X_meta['primary_pred'] = primary_pred
        X_meta['primary_proba'] = primary_proba[1]
        
        meta_proba = self.meta_model.predict_proba(X_meta)[0][1]
        
        # Decision
        should_trade = meta_proba >= self.meta_threshold
        
        # Bet size (Kelly-inspired, capped at 1.0)
        if should_trade:
            edge = 2 * meta_proba - 1  # Convert prob to edge
            bet_size = min(max(edge, 0), 1.0)
        else:
            bet_size = 0.0
        
        return {
            'side': side,
            'side_confidence': side_confidence,
            'meta_confidence': meta_proba,
            'should_trade': should_trade,
            'bet_size': bet_size,
            'signal': f"{side} ({meta_proba*100:.0f}%)" if should_trade else 'NO TRADE'
        }
    
    def evaluate(self, X: pd.DataFrame, y_side: pd.Series, y_label: pd.Series) -> dict:
        """
        Evaluate model using purged cross-validation.
        """
        print("\n📊 Running Purged K-Fold Cross-Validation...")
        
        # Evaluate primary model
        y_primary = (y_side > 0).astype(int)
        primary_cv = purged_cross_val_score(
            self.primary_model, X[self.feature_cols], y_primary,
            n_splits=5, embargo_pct=0.02, scoring='accuracy'
        )
        
        print(f"\n   Primary Model (Side):")
        print(f"   Mean Accuracy: {primary_cv['mean']*100:.1f}% ± {primary_cv['std']*100:.1f}%")
        
        for fold in primary_cv['folds']:
            print(f"   Fold {fold['fold']}: {fold['score']*100:.1f}%")
        
        return {
            'primary_mean': primary_cv['mean'],
            'primary_std': primary_cv['std'],
            'primary_folds': primary_cv['folds']
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🧠 INSTITUTIONAL QUANT MODEL - Testing")
    print("="*70)
    
    # Import data processor
    from data_processor import InstitutionalDataPipeline, create_institutional_features
    
    # Load data
    df = pd.read_csv('/Users/anmol/Desktop/gold/market_data/BTCUSDT_1h.csv')
    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    print(f"\n✅ Loaded {len(df)} candles")
    
    # Process with institutional pipeline
    pipeline = InstitutionalDataPipeline(dollar_threshold=5_000_000)
    df_processed = pipeline.process_ohlcv_data(df)
    
    # Apply triple barrier
    print("\n🏷️ Applying Triple Barrier Labeling...")
    df_labeled = apply_triple_barrier(df_processed, pt_sl=(2.0, 1.0), max_holding=10)
    
    barrier_stats = df_labeled['barrier_touch'].value_counts()
    print(f"   Upper (TP): {barrier_stats.get('upper', 0)}")
    print(f"   Lower (SL): {barrier_stats.get('lower', 0)}")
    print(f"   Timeout:    {barrier_stats.get('timeout', 0)}")
    
    # Get side labels
    df_labeled['side'] = get_side_labels(df_labeled)
    
    # Clean for modeling
    feature_cols = pipeline.get_feature_columns()
    df_clean = df_labeled.dropna(subset=['barrier_label'] + feature_cols)
    
    print(f"\n📊 Clean samples: {len(df_clean)}")
    
    # Train Meta-Labeling Ensemble
    print("\n🤖 Training Meta-Labeling Ensemble...")
    
    X = df_clean[feature_cols]
    y_side = df_clean['side']
    y_label = df_clean['barrier_label']
    
    model = MetaLabelingEnsemble(meta_threshold=0.70)
    model.fit(X, y_side, y_label, feature_cols)
    
    # Evaluate
    eval_results = model.evaluate(X, y_side, y_label)
    
    # Test single prediction
    print("\n🔮 Test Prediction:")
    sample = X.iloc[-1:].copy()
    prediction = model.predict(sample)
    print(f"   Side: {prediction['side']}")
    print(f"   Side Confidence: {prediction['side_confidence']*100:.1f}%")
    print(f"   Meta Confidence: {prediction['meta_confidence']*100:.1f}%")
    print(f"   Should Trade: {prediction['should_trade']}")
    print(f"   Bet Size: {prediction['bet_size']:.2f}")
    print(f"   Signal: {prediction['signal']}")
    
    print("\n" + "="*70)
