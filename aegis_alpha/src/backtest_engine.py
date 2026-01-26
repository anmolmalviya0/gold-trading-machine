"""
TERMINAL - Backtest Engine
============================
Validates LSTM/LightGBM models against historical data.
Implements the "Great Filter" criteria.
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_model import TerminalLSTM, SignalPredictor


class LightGBMPredictor:
    """Wrapper for LightGBM model predictions"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load()
    
    def load(self):
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            print(f"✅ LightGBM model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load LightGBM model: {e}")
            self.model = None
    
    def predict(self, features: np.ndarray, threshold: float = 0.5) -> dict:
        """Predict using LightGBM model"""
        if self.model is None:
            return {'signal': 'HOLD', 'confidence': 0.0, 'approved': False}
        
        try:
            # LightGBM expects 2D array with shape (1, n_features)
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            elif len(features.shape) == 2 and features.shape[0] > 1:
                # Take the last row if multiple rows provided
                features = features[-1:, :]
            
            # Get probability prediction
            proba = self.model.predict_proba(features)[0]
            probability = proba[1] if len(proba) > 1 else proba[0]
            
            # Signal thresholds
            buy_threshold = 0.60
            sell_threshold = 0.30
            
            if probability >= buy_threshold:
                return {'signal': 'BUY', 'confidence': probability, 'approved': True}
            elif probability <= sell_threshold:
                return {'signal': 'SELL', 'confidence': 1.0 - probability, 'approved': True}
            else:
                return {'signal': 'HOLD', 'confidence': probability, 'approved': False}
                
        except Exception as e:
            print(f"❌ LightGBM prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'approved': False}


@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    outcome: str  # 'WIN' or 'LOSS'


class BacktestEngine:
    """
    Model Backtester with Great Filter Validation
    
    Criteria:
    - Win Rate > 60%
    - Profit Factor > 1.5
    - Max Drawdown < 15%
    - Net Return > 20%
    """
    
    def __init__(
        self,
        model_path: str = '../terminal_alpha/models/terminal_lstm.pth',
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        fee_rate: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Auto-detect model type from file extension
        self.model_type = 'lgbm' if model_path.endswith('.pkl') else 'lstm'
        
        self.predictor = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
    
    def load_model(self) -> bool:
        """Load trained model (auto-detects LSTM vs LightGBM)"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        if self.model_type == 'lgbm':
            print(f"🗡️ Loading LightGBM Switchblade: {self.model_path}")
            self.predictor = LightGBMPredictor(self.model_path)
            return self.predictor.model is not None
        else:
            print(f"🧠 Loading LSTM Neural Network: {self.model_path}")
            self.predictor = SignalPredictor(self.model_path)
            return self.predictor.model is not None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features matching training pipeline"""
        df = df.copy()
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Map short column names to standard names
        column_map = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'time': 'timestamp', 'date': 'timestamp'
        }
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = df['rsi'] / 100
        
        # MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df.dropna()
    
    def run(self, df: pd.DataFrame, seq_len: int = 60) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: OHLCV DataFrame
            seq_len: Lookback window for LSTM
            
        Returns:
            Backtest metrics
        """
        print("\n" + "="*60)
        print("🔬 TERMINAL BACKTEST ENGINE")
        print("="*60)
        
        if not self.load_model():
            return self._empty_results()
        
        # Prepare data
        df = self.prepare_features(df)
        
        feature_cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
        ]
        
        # Pad features
        while len(feature_cols) < 20:
            feature_cols.append(feature_cols[-1])
        feature_cols = feature_cols[:20]
        
        features = df[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Initialize tracking
        capital = self.initial_capital
        self.equity_curve = [capital]
        self.trades = []
        
        in_position = False
        entry_price = 0
        entry_idx = 0
        
        print(f"📊 Testing {len(df) - seq_len} potential trades...")
        
        # Limit data for faster backtest (use last 10,000 samples)
        max_samples = 10000
        if len(df) > max_samples:
            print(f"   ⚡ Sampling last {max_samples} candles for speed...")
            df = df.tail(max_samples).reset_index(drop=True)
            features = features[-max_samples:] if len(features) > max_samples else features
        
        # Simulate trading
        for i in range(seq_len, len(df) - 5):
            # Get prediction
            window = features[i-seq_len:i]
            prediction = self.predictor.predict(window, threshold=0.40)  # Calibrated to triple-barrier labels
            
            if not in_position and prediction['approved']:
                # Enter LONG trade (aligned with triple-barrier training)
                in_position = True
                entry_price = df['close'].iloc[i] * (1 + self.slippage)
                entry_idx = i
            
            elif in_position:
                # Check exit (hold for 5 bars or stop loss)
                current_price = df['close'].iloc[i]
                bars_held = i - entry_idx
                
                # ATR-based stop loss (matches training labels)
                atr = df['atr'].iloc[i]
                stop_loss = entry_price - 1.5 * atr  # 1.5 ATR stop
                take_profit = entry_price + 2.0 * atr  # 2.0 ATR target
                
                should_exit = (
                    bars_held >= 20 or  # Match training horizon
                    current_price <= stop_loss or
                    current_price >= take_profit
                )
                
                if should_exit:
                    # Exit LONG trade
                    exit_price = current_price * (1 - self.slippage)
                    
                    # Calculate P&L for LONG
                    position_size = capital * self.risk_per_trade / (1.5 * atr) if atr > 0 else 0
                    pnl = (exit_price - entry_price) * position_size
                    fees = (entry_price + exit_price) * position_size * self.fee_rate
                    net_pnl = pnl - fees
                    pnl_pct = net_pnl / capital
                    
                    # Record trade
                    trade = Trade(
                        entry_time=df.index[entry_idx] if hasattr(df.index[0], 'strftime') else datetime.now(),
                        exit_time=df.index[i] if hasattr(df.index[0], 'strftime') else datetime.now(),
                        direction='LONG',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        outcome='WIN' if net_pnl > 0 else 'LOSS'
                    )
                    self.trades.append(trade)
                    
                    # Update capital
                    capital += net_pnl
                    self.equity_curve.append(capital)
                    
                    in_position = False
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return self._empty_results()
        
        # Basic stats
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.outcome == 'WIN')
        losses = total_trades - wins
        
        win_rate = wins / total_trades * 100
        
        # Profit metrics
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        net_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Drawdown
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Great Filter validation
        passed_win_rate = win_rate > 60
        passed_pf = profit_factor > 1.5
        passed_dd = max_dd < 15
        passed_return = net_return > 20
        
        great_filter_passed = all([passed_win_rate, passed_pf, passed_dd, passed_return])
        
        results = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_return': net_return,
            'max_drawdown': max_dd,
            'final_equity': self.equity_curve[-1],
            'great_filter': {
                'passed': great_filter_passed,
                'win_rate_check': passed_win_rate,
                'profit_factor_check': passed_pf,
                'max_dd_check': passed_dd,
                'net_return_check': passed_return
            }
        }
        
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "-"*60)
        print("📊 BACKTEST RESULTS")
        print("-"*60)
        
        print(f"\n💼 Trade Statistics:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Wins: {results['wins']} | Losses: {results['losses']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        
        print(f"\n💰 Financial Metrics:")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Net Return: {results['net_return']:.1f}%")
        print(f"   Max Drawdown: {results['max_drawdown']:.1f}%")
        print(f"   Final Equity: ${results['final_equity']:,.2f}")
        
        print("\n" + "="*60)
        print("🔍 THE GREAT FILTER")
        print("="*60)
        
        gf = results['great_filter']
        
        checks = [
            (f"Win Rate > 60%", gf['win_rate_check'], f"{results['win_rate']:.1f}%"),
            (f"Profit Factor > 1.5", gf['profit_factor_check'], f"{results['profit_factor']:.2f}"),
            (f"Max Drawdown < 15%", gf['max_dd_check'], f"{results['max_drawdown']:.1f}%"),
            (f"Net Return > 20%", gf['net_return_check'], f"{results['net_return']:.1f}%"),
        ]
        
        for name, passed, value in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} | {name} → {value}")
        
        print("\n" + "="*60)
        if gf['passed']:
            print("🏆 GREAT FILTER: PASSED ✅")
            print("   → System approved for Phase 4 (Daemon Activation)")
        else:
            print("⚠️ GREAT FILTER: FAILED ❌")
            print("   → Run 'python3 src/strategy_optimizer.py' before deployment")
        print("="*60)
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'net_return': 0,
            'max_drawdown': 0,
            'final_equity': self.initial_capital,
            'great_filter': {
                'passed': False,
                'win_rate_check': False,
                'profit_factor_check': False,
                'max_dd_check': False,
                'net_return_check': False
            }
        }


def generate_test_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic test data"""
    np.random.seed(123)
    
    returns = np.random.randn(n_samples) * 0.015
    close = 2000 * np.cumprod(1 + returns)
    
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.008))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.008))
    open_price = low + (high - low) * np.random.rand(n_samples)
    volume = np.random.exponential(500000, n_samples)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


if __name__ == '__main__':
    import argparse
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          TERMINAL - BACKTEST ENGINE                     ║
    ║              The Great Filter Validation                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='TERMINAL Backtest Engine')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', 
                        help='Asset symbol to backtest (e.g., BTC/USDT, ETH/USDT)')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization charts')
    args = parser.parse_args()
    
    # Map symbol to model path (check LightGBM first, then LSTM)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    symbol_clean = args.symbol.replace('/', '').replace('USDT', '')
    
    # Try LightGBM first (SWITCHBLADE PROTOCOL)
    lgbm_path = os.path.join(BASE_DIR, 'models', f'{symbol_clean.upper()}_lgbm.pkl')
    lstm_path = os.path.join(BASE_DIR, 'models', f'lstm_{symbol_clean.lower()}.pth')
    
    if os.path.exists(lgbm_path):
        model_path = lgbm_path
        print(f"🗡️ SWITCHBLADE MODE: Using LightGBM")
    elif os.path.exists(lstm_path):
        model_path = lstm_path
        print(f"🧠 LSTM MODE: Using Neural Network")
    else:
        model_path = lgbm_path  # Will error, but show which file we're looking for
        print(f"⚠️ No model found, checking: {lgbm_path}")
    
    # Map symbol to data file
    data_symbol = args.symbol.replace('/', '')  # BTC/USDT -> BTCUSDT
    data_paths = [
        os.path.join(BASE_DIR, '..', 'market_data'),
        os.path.join(BASE_DIR, 'data'),
    ]
    
    print(f"🎯 SYMBOL: {args.symbol}")
    print(f"🧠 MODEL: {model_path}")
    
    # Initialize engine
    engine = BacktestEngine(
        model_path=model_path,
        initial_capital=10000,
        risk_per_trade=0.02
    )
    
    # Load test data for specific symbol
    print("📂 Loading test data...")
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            import glob
            # Prioritize 1h data (matching training timeframe), then other intervals
            patterns = [
                f"{path}/{data_symbol}_1h.csv",      # Preferred: 1h matches training
                f"{path}/{data_symbol}_1H.csv",
                f"{path}/{data_symbol}*.csv",        # Fallback: any interval
            ]
            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    # Sort to get most recent or preferred file
                    files.sort(reverse=True)
                    for f_path in files:
                        try:
                            df = pd.read_csv(f_path)
                            df.columns = df.columns.str.lower()
                            print(f"   ✓ Loaded: {f_path} ({len(df)} rows)")
                            break
                        except Exception as e:
                            print(f"   ⚠️ Failed to load {f_path}: {e}")
                    if df is not None:
                        break
            if df is not None:
                break
    
    if df is None:
        # Try fetching from yfinance as fallback
        try:
            import yfinance as yf
            print(f"   ⚠️ No local CSV. Fetching {args.symbol} from Yahoo Finance...")
            yf_symbol = args.symbol.replace('/', '').replace('USDT', '-USD')
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="6mo", interval="1h")
            if len(df) > 100:
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                print(f"   ✅ Fetched {len(df)} rows from YFinance")
        except Exception as e:
            print(f"   ❌ YFinance fallback failed: {e}")
    
    if df is None:
        print("   ❌ Using synthetic test data (NOT RELIABLE)")
        df = generate_test_data()
    
    # Run backtest
    results = engine.run(df)
    
    # Save results
    results_file = os.path.join(BASE_DIR, 'logs', f'backtest_{symbol_clean}_results.txt')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"TERMINAL Backtest Results - {args.symbol}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"="*50 + "\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n📁 Results saved to: {results_file}")

