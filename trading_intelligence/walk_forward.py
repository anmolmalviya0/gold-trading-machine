"""
PHASE 1: WALK-FORWARD VALIDATION & BACKTEST
============================================
Walk-forward validation harness with realistic backtest.

Features:
- Rolling window expanding training
- Purged test sets (no leakage)
- Realistic TP/SL simulation
- Fees + slippage
- Comprehensive metrics

Usage:
    python walk_forward.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Callable
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'backtest_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WFV_CONFIG = {
    'train_pct': 0.6,       # Initial training window
    'val_pct': 0.1,         # Validation window
    'test_pct': 0.1,        # Test window per fold
    'step_pct': 0.1,        # Step forward each fold
    'min_train_samples': 1000,
    'purge_bars': 10,       # Gap between train and test (avoid leakage)
}

BACKTEST_CONFIG = {
    'fee_pct': 0.001,       # 0.1% per trade
    'slippage_pct': 0.0005, # 0.05% slippage
    'position_size': 1000,  # USD per trade
}


# === DATA CLASSES ===

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    side: str
    sl: float
    tp: float
    exit_time: datetime = None
    exit_price: float = None
    exit_reason: str = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    def to_dict(self):
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'side': self.side,
            'sl': self.sl,
            'tp': self.tp,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct
        }


@dataclass
class FoldResult:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    profit_factor: float
    expectancy: float
    win_rate: float
    total_pnl: float
    trades: List[Trade]


# === WALK-FORWARD SPLITS ===

def create_wfv_splits(df: pd.DataFrame, 
                      train_pct: float = 0.6,
                      test_pct: float = 0.1,
                      step_pct: float = 0.1,
                      purge_bars: int = 10) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward validation splits.
    
    Returns list of (train_df, test_df) tuples.
    """
    n = len(df)
    train_size = int(n * train_pct)
    test_size = int(n * test_pct)
    step_size = int(n * step_pct)
    
    splits = []
    start_idx = 0
    
    while True:
        train_end = start_idx + train_size
        test_start = train_end + purge_bars  # Purge gap
        test_end = test_start + test_size
        
        if test_end > n:
            break
        
        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        if len(train_df) >= WFV_CONFIG['min_train_samples']:
            splits.append((train_df, test_df))
        
        start_idx += step_size
    
    return splits


# === BACKTEST ENGINE ===

def simulate_trades(predictions: pd.DataFrame, 
                    prices: pd.DataFrame,
                    fee_pct: float = 0.001,
                    slippage_pct: float = 0.0005) -> List[Trade]:
    """
    Simulate trades with realistic TP/SL execution.
    
    Parameters:
    -----------
    predictions : DataFrame with columns ['time', 'pred', 'entry', 'sl', 'tp']
    prices : DataFrame with columns ['time', 'o', 'h', 'l', 'c']
    """
    trades = []
    in_position = False
    current_trade = None
    
    pred_idx = 0
    
    for i, row in prices.iterrows():
        time = row['time'] if 'time' in row else i
        high = row['h']
        low = row['l']
        close = row['c']
        
        # Check if we have open position
        if in_position and current_trade:
            # Check exit conditions
            if current_trade.side == 'BUY':
                # TP hit
                if high >= current_trade.tp:
                    current_trade.exit_time = time
                    current_trade.exit_price = current_trade.tp * (1 - slippage_pct)
                    current_trade.exit_reason = 'TP'
                    current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * \
                                       (BACKTEST_CONFIG['position_size'] / current_trade.entry_price)
                    current_trade.pnl -= BACKTEST_CONFIG['position_size'] * fee_pct * 2  # Entry + exit fees
                    current_trade.pnl_pct = current_trade.pnl / BACKTEST_CONFIG['position_size'] * 100
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                # SL hit
                elif low <= current_trade.sl:
                    current_trade.exit_time = time
                    current_trade.exit_price = current_trade.sl * (1 - slippage_pct)
                    current_trade.exit_reason = 'SL'
                    current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * \
                                       (BACKTEST_CONFIG['position_size'] / current_trade.entry_price)
                    current_trade.pnl -= BACKTEST_CONFIG['position_size'] * fee_pct * 2
                    current_trade.pnl_pct = current_trade.pnl / BACKTEST_CONFIG['position_size'] * 100
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
            
            else:  # SELL
                # TP hit (lower price)
                if low <= current_trade.tp:
                    current_trade.exit_time = time
                    current_trade.exit_price = current_trade.tp * (1 + slippage_pct)
                    current_trade.exit_reason = 'TP'
                    current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * \
                                       (BACKTEST_CONFIG['position_size'] / current_trade.entry_price)
                    current_trade.pnl -= BACKTEST_CONFIG['position_size'] * fee_pct * 2
                    current_trade.pnl_pct = current_trade.pnl / BACKTEST_CONFIG['position_size'] * 100
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                # SL hit (higher price)
                elif high >= current_trade.sl:
                    current_trade.exit_time = time
                    current_trade.exit_price = current_trade.sl * (1 + slippage_pct)
                    current_trade.exit_reason = 'SL'
                    current_trade.pnl = (current_trade.entry_price - current_trade.exit_price) * \
                                       (BACKTEST_CONFIG['position_size'] / current_trade.entry_price)
                    current_trade.pnl -= BACKTEST_CONFIG['position_size'] * fee_pct * 2
                    current_trade.pnl_pct = current_trade.pnl / BACKTEST_CONFIG['position_size'] * 100
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
        
        # Check for new signal
        if not in_position and pred_idx < len(predictions):
            pred_row = predictions.iloc[pred_idx]
            
            if pred_row.get('pred', 0) != 0:  # Has signal
                side = 'BUY' if pred_row['pred'] == 1 else 'SELL'
                entry_price = close * (1 + slippage_pct) if side == 'BUY' else close * (1 - slippage_pct)
                
                current_trade = Trade(
                    entry_time=time,
                    entry_price=entry_price,
                    side=side,
                    sl=pred_row['sl'],
                    tp=pred_row['tp']
                )
                in_position = True
            
            pred_idx += 1
    
    return trades


def calculate_metrics(trades: List[Trade]) -> Dict:
    """Calculate backtest metrics"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe': 0
        }
    
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    
    total_pnl = sum(t.pnl for t in trades)
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    
    win_rate = len(wins) / len(trades) * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    
    expectancy = avg_win * (len(wins)/len(trades)) + avg_loss * (len(losses)/len(trades))
    
    # Drawdown
    cumsum = np.cumsum([t.pnl for t in trades])
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Sharpe (simplified)
    returns = [t.pnl_pct for t in trades]
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe
    }


# === WALK-FORWARD VALIDATION ===

def run_walk_forward(df: pd.DataFrame, 
                     feature_cols: List[str],
                     model_factory: Callable = None) -> List[FoldResult]:
    """
    Run complete walk-forward validation.
    """
    if model_factory is None:
        model_factory = lambda: Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42))
        ])
    
    # Create splits
    splits = create_wfv_splits(df, 
                               train_pct=WFV_CONFIG['train_pct'],
                               test_pct=WFV_CONFIG['test_pct'],
                               step_pct=WFV_CONFIG['step_pct'],
                               purge_bars=WFV_CONFIG['purge_bars'])
    
    print(f"   Created {len(splits)} WFV folds")
    
    results = []
    
    for fold, (train_df, test_df) in enumerate(splits):
        print(f"\n   Fold {fold+1}/{len(splits)}...", end=' ')
        
        # Prepare data
        train_clean = train_df.dropna(subset=feature_cols + ['label'])
        test_clean = test_df.dropna(subset=feature_cols + ['label'])
        
        if len(train_clean) < 100 or len(test_clean) < 10:
            print("Skip (insufficient data)")
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['label'].astype(int)
        X_test = test_clean[feature_cols]
        y_test = test_clean['label'].astype(int)
        
        # Train model
        model = model_factory()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Classification metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Prepare predictions for backtest
        predictions = test_clean[['time', 'c', 'atr']].copy()
        predictions['pred'] = y_pred
        
        # SL/TP based on ATR
        predictions['sl'] = np.where(
            predictions['pred'] == 1,
            predictions['c'] - predictions['atr'] * 1.0,
            predictions['c'] + predictions['atr'] * 1.0
        )
        predictions['tp'] = np.where(
            predictions['pred'] == 1,
            predictions['c'] + predictions['atr'] * 2.0,
            predictions['c'] - predictions['atr'] * 2.0
        )
        
        # Simulate trades
        trades = simulate_trades(predictions, test_clean)
        metrics = calculate_metrics(trades)
        
        print(f"Acc={acc*100:.1f}%, WR={metrics['win_rate']:.1f}%, PF={metrics['profit_factor']:.2f}")
        
        result = FoldResult(
            fold=fold+1,
            train_start=train_clean['time'].iloc[0] if 'time' in train_clean.columns else None,
            train_end=train_clean['time'].iloc[-1] if 'time' in train_clean.columns else None,
            test_start=test_clean['time'].iloc[0] if 'time' in test_clean.columns else None,
            test_end=test_clean['time'].iloc[-1] if 'time' in test_clean.columns else None,
            train_size=len(train_clean),
            test_size=len(test_clean),
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            profit_factor=metrics['profit_factor'],
            expectancy=metrics['expectancy'],
            win_rate=metrics['win_rate'],
            total_pnl=metrics['total_pnl'],
            trades=trades
        )
        results.append(result)
    
    return results


def summarize_results(results: List[FoldResult]) -> Dict:
    """Aggregate results across all folds"""
    if not results:
        return {}
    
    return {
        'total_folds': len(results),
        'avg_accuracy': np.mean([r.accuracy for r in results]),
        'avg_precision': np.mean([r.precision for r in results]),
        'avg_recall': np.mean([r.recall for r in results]),
        'avg_f1': np.mean([r.f1 for r in results]),
        'avg_win_rate': np.mean([r.win_rate for r in results]),
        'avg_profit_factor': np.mean([r.profit_factor for r in results]),
        'avg_expectancy': np.mean([r.expectancy for r in results]),
        'total_pnl': sum(r.total_pnl for r in results),
        'total_trades': sum(len(r.trades) for r in results)
    }


# === MAIN ===

if __name__ == "__main__":
    print("="*70)
    print("📈 PHASE 1: WALK-FORWARD VALIDATION & BACKTEST")
    print("="*70)
    
    # Feature columns (must match label_and_features.py)
    feature_cols = [
        'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
        'rsi', 'macd_hist',
        'dist_sma10', 'dist_sma20', 'dist_sma50', 'dist_sma100',
        'trend_10_20', 'trend_20_50', 'trend_50_100',
        'atr_pct', 'vol_20', 'vol_rank',
        'bb_width', 'bb_position',
        'vol_ratio'
    ]
    
    symbols = ['BTCUSDT', 'PAXGUSDT']
    
    for symbol in symbols:
        print(f"\n🔍 Running WFV for {symbol}...")
        
        # Load labeled data
        data_path = DATASET_DIR / f"{symbol}_1h_labeled.parquet"
        
        if not data_path.exists():
            print(f"   ⚠️ No labeled data found. Run label_and_features.py first.")
            continue
        
        df = pd.read_parquet(data_path)
        print(f"   Loaded: {len(df):,} rows")
        
        # Run WFV
        results = run_walk_forward(df, feature_cols)
        
        # Summary
        summary = summarize_results(results)
        
        print(f"\n   📊 SUMMARY for {symbol}:")
        print(f"      Folds: {summary.get('total_folds', 0)}")
        print(f"      Avg Accuracy: {summary.get('avg_accuracy', 0)*100:.1f}%")
        print(f"      Avg Win Rate: {summary.get('avg_win_rate', 0):.1f}%")
        print(f"      Avg Profit Factor: {summary.get('avg_profit_factor', 0):.2f}")
        print(f"      Total PnL: ${summary.get('total_pnl', 0):.2f}")
        print(f"      Total Trades: {summary.get('total_trades', 0)}")
        
        # Save results
        results_df = pd.DataFrame([r.__dict__ for r in results])
        results_df = results_df.drop(columns=['trades'])  # Remove nested trades for CSV
        results_path = OUTPUT_DIR / f"{symbol}_wfv_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"   💾 Saved: {results_path.name}")
    
    print("\n" + "="*70)
    print("✅ WALK-FORWARD VALIDATION COMPLETE")
    print("="*70)
