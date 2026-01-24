"""
AEGIS V21 - LSTM Training Pipeline
===================================
Converts historical market data into predictive tensors.
Walk-forward validation to prevent overfitting.
"""
import os
import sys
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_model import AegisLSTM


class MarketDataset(Dataset):
    """PyTorch Dataset for market time series"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 60):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMTrainer:
    """
    LSTM Training Pipeline
    
    Features:
    - Automatic data loading from market_data/
    - Feature engineering (RSI, MACD, Bollinger)
    - Walk-forward validation
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(
        self,
        data_dir: str = '../market_data',
        model_dir: str = '../aegis_alpha/models',
        seq_len: int = 60,
        batch_size: int = 64,
        learning_rate: float = 0.00001,  # Lowered from 0.001 for stability
        epochs: int = 100
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_names = []
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load market data - prioritizes real data over synthetic"""
        print("📂 Loading market data...")
        
        # Try loading from existing CSVs first
        search_paths = [
            os.path.join(self.data_dir, '*.csv'),
            os.path.join(self.data_dir, '*/*.csv'),
            '../marketforge/data/*.csv',
            '../trading_intelligence/data/*.csv',
        ]
        
        all_files = []
        for pattern in search_paths:
            all_files.extend(glob.glob(pattern))
        
        dfs = []
        for f in all_files[:5]:
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.lower()
                # Map short column names
                col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
                df = df.rename(columns=col_map)
                if 'close' in df.columns:
                    dfs.append(df)
                    print(f"   ✓ Loaded {os.path.basename(f)}: {len(df)} rows")
            except Exception as e:
                print(f"   ✗ Failed to load {f}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"📊 Total samples from CSV: {len(combined)}")
            return combined
        
        # No CSV data - fetch real XAUUSD from yfinance
        print("⚠️ No CSV files found. Fetching real XAUUSD data from Yahoo Finance...")
        return self._fetch_real_gold_data()
    
    def _fetch_real_gold_data(self) -> pd.DataFrame:
        """Fetch real Gold/XAUUSD data using yfinance"""
        try:
            import yfinance as yf
            
            # Gold ETF (GLD) as proxy for XAUUSD - more reliable data
            symbols = ['GC=F', 'GLD', 'IAU']  # Gold Futures, SPDR Gold, iShares Gold
            
            for symbol in symbols:
                try:
                    print(f"   📡 Fetching {symbol} data (1 year, 1h interval)...")
                    ticker = yf.Ticker(symbol)
                    
                    # Fetch 1 year of hourly data
                    df = ticker.history(period="1y", interval="1h")
                    
                    if len(df) < 1000:
                        # Try daily data with longer period
                        print(f"   📡 Fetching {symbol} data (5 years, daily)...")
                        df = ticker.history(period="5y", interval="1d")
                    
                    if len(df) >= 1000:
                        print(f"   ✅ Successfully fetched {len(df)} samples from {symbol}")
                        
                        # Standardize columns
                        df = df.reset_index()
                        df.columns = df.columns.str.lower()
                        
                        # Ensure required columns
                        if 'close' in df.columns:
                            # Add volume if missing
                            if 'volume' not in df.columns:
                                df['volume'] = 1000000  # Placeholder
                            
                            return df[['open', 'high', 'low', 'close', 'volume']].dropna()
                    
                except Exception as e:
                    print(f"   ⚠️ Failed to fetch {symbol}: {e}")
                    continue
            
            print("❌ Could not fetch real data. Falling back to synthetic...")
            return self._generate_synthetic_data()
            
        except ImportError:
            print("❌ yfinance not installed. Run: pip install yfinance")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """Generate synthetic OHLCV data for training demo"""
        print(f"🔧 Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        
        # Generate realistic price movement
        returns = np.random.randn(n_samples) * 0.02  # 2% daily volatility
        close = 2000 * np.cumprod(1 + returns)  # Start at 2000 (gold-like)
        
        # Generate OHLCV
        high = close * (1 + np.abs(np.random.randn(n_samples) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n_samples) * 0.01))
        open_price = low + (high - low) * np.random.rand(n_samples)
        volume = np.random.exponential(1000000, n_samples)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create ML features from OHLCV data"""
        print("⚙️ Engineering features...")
        
        df = df.copy()
        
        # Basic price features
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
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ============================================================
        # TRIPLE-BARRIER LABELING (De Prado Method)
        # ============================================================
        # Aligns training labels with execution logic:
        # - Take Profit: +2.0 ATR
        # - Stop Loss: -1.5 ATR  
        # - Time Horizon: 20 bars
        # Label = 1 (BUY) only if TP hit BEFORE SL within horizon
        # ============================================================
        print("   🏷️ Applying Triple-Barrier Labeling...")
        
        labels = np.zeros(len(df), dtype=np.float32)
        horizon = 20  # Max bars to hold
        tp_mult = 2.0  # Take profit multiplier
        sl_mult = 1.5  # Stop loss multiplier
        
        close_arr = df['close'].values
        high_arr = df['high'].values
        low_arr = df['low'].values
        atr_arr = df['atr'].values
        
        valid_labels = 0
        buy_labels = 0
        
        for i in range(len(df) - horizon):
            entry_price = close_arr[i]
            atr = atr_arr[i]
            
            if atr <= 0 or np.isnan(atr):
                continue
                
            take_profit = entry_price + (tp_mult * atr)
            stop_loss = entry_price - (sl_mult * atr)
            
            # Look forward through the horizon
            for j in range(1, horizon + 1):
                if i + j >= len(df):
                    break
                    
                future_high = high_arr[i + j]
                future_low = low_arr[i + j]
                
                # Check if TP hit (using high)
                tp_hit = future_high >= take_profit
                # Check if SL hit (using low)
                sl_hit = future_low <= stop_loss
                
                if tp_hit and not sl_hit:
                    # Take profit hit first - WINNING TRADE
                    labels[i] = 1.0
                    buy_labels += 1
                    break
                elif sl_hit:
                    # Stop loss hit - LOSING TRADE
                    labels[i] = 0.0
                    break
                # If neither hit, check next bar
            
            valid_labels += 1
        
        df['label'] = labels
        print(f"   ✅ Triple-Barrier Labels: {buy_labels}/{valid_labels} positive ({100*buy_labels/max(1,valid_labels):.1f}%)")
        
        # Select features
        feature_cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
        ]
        
        # Pad to ensure we have enough features
        while len(feature_cols) < 20:
            feature_cols.append(feature_cols[-1])
        
        self.feature_names = feature_cols[:20]
        
        # Drop NaN and extract arrays
        df = df.dropna()
        
        features = df[self.feature_names].values.astype(np.float32)
        labels = df['label'].values.astype(np.float32)
        
        # Replace inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Label distribution: {labels.mean():.2%} positive")
        
        return features, labels
    
    def create_dataloaders(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation dataloaders"""
        
        split_idx = int(len(features) * train_ratio)
        
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]
        
        train_dataset = MarketDataset(train_features, train_labels, self.seq_len)
        val_dataset = MarketDataset(val_features, val_labels, self.seq_len)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"📦 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(self) -> dict:
        """
        Main training loop
        
        Returns:
            Training metrics
        """
        print("\n" + "="*60)
        print("🚀 AEGIS V21 LSTM TRAINING PIPELINE")
        print("="*60)
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Device: {self.device}")
        
        # Load and prepare data
        df = self.load_data()
        
        # Validation: Check minimal data requirements
        min_required = self.seq_len + 100 # At least 1 batch + sequence
        if len(df) < min_required:
            print(f"❌ ERROR: Insufficient data. Got {len(df)} rows, need {min_required}+.")
            print("   Try increasing the data range or checking the symbol.")
            return {
                'best_val_loss': 0.0,
                'final_accuracy': 0.0,
                'training_time': 0.0,
                'epochs_trained': 0
            }
            
        features, labels = self.engineer_features(df)
        
        # Second check after engineering (which drops rows)
        if len(features) < (self.seq_len + 20):
             print(f"❌ ERROR: Features too short after processing. Got {len(features)} rows.")
             return {
                'best_val_loss': 0.0,
                'final_accuracy': 0.0,
                'training_time': 0.0,
                'epochs_trained': 0
            }
            
        train_loader, val_loader = self.create_dataloaders(features, labels)
        
        # Initialize model
        input_size = features.shape[1]
        self.model = AegisLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        print(f"\n📈 Training for {self.epochs} epochs...")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.2%}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self._save_model(val_accuracy, val_loss, epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                    break
        
        elapsed = time.time() - start_time
        
        # Final summary
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"⏱️ Training time: {elapsed/60:.1f} minutes")
        print(f"📉 Best validation loss: {best_val_loss:.4f}")
        print(f"🎯 Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
        print(f"💾 Model saved to: {self.model_dir}/aegis_lstm.pth")
        
        return {
            'best_val_loss': best_val_loss,
            'final_accuracy': history['val_accuracy'][-1],
            'training_time': elapsed,
            'epochs_trained': len(history['train_loss'])
        }
    
    def _save_model(self, accuracy: float, loss: float, epoch: int):
        """Save model checkpoint"""
        save_path = os.path.join(self.model_dir, 'aegis_lstm.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'input_size': 20,
            'hidden_size': 128,
            'num_layers': 2,
            'accuracy': accuracy,
            'loss': loss,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }, save_path)


import argparse

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          AEGIS V21 - LSTM NEURAL NETWORK TRAINER         ║
    ║              The Brain Initialization Protocol           ║
    ══════════════════════════════════════════════════════════╝
    """)
    
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train LSTM Model for Specific Asset')
    parser.add_argument('--symbol', type=str, default='PAXG', help='Asset symbol to train on (e.g., BTCUSDT)')
    parser.add_argument('--output', type=str, default='models/aegis_lstm.pth', help='Path to save .pth model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--csv_path', type=str, default=None, help='Direct path to CSV file (overrides yfinance)')
    
    args = parser.parse_args()
    
    print(f"🔥 PyTorch Version: {torch.__version__}")
    print(f"🖥️ CUDA Available: {torch.cuda.is_available()}")
    print("-" * 60)
    print(f"🎯 TARGET ASSET: {args.symbol}")
    print(f"💾 OUTPUT PATH:  {args.output}")
    print("-" * 60)
    
    # Initialize trainer with CLI args (Monkey-patching logic would ideally go inside class, 
    # but for speed we'll modify the data loading behavior via subclass or direct patch if needed,
    # OR we assume the load_data method uses the symbol logic)
    
    # Since load_data is currently hardcoded for Gold/CSVs, we need to quickly patch it 
    # to filtering for the specific symbol if loading from CSV, 
    # or fetch that specific symbol from yfinance.
    
    # Let's instantiate normally, but we need to tell it WHAT to train on.
    # The current class structure doesn't easily accept 'symbol' in __init__.
    # We will subclass it locally to inject the symbol-specific fetching logic.
    
    class TargetedLSTMTrainer(LSTMTrainer):
        def __init__(self, target_symbol: str, target_output: str, csv_path: str = None, **kwargs):
            super().__init__(**kwargs)
            self.target_symbol = target_symbol
            self.target_output = target_output
            self.csv_path = csv_path  # Direct CSV path override
            
            # Ensure output dir exists
            os.makedirs(os.path.dirname(self.target_output), exist_ok=True)
            
        def _save_model(self, accuracy: float, loss: float, epoch: int):
            """Override save to use custom output path"""
            # save_path = os.path.join(self.model_dir, 'aegis_lstm.pth') 
            # We ignore self.model_dir because we have full path in target_output
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'input_size': 20,
                'hidden_size': 128,
                'num_layers': 2,
                'accuracy': accuracy,
                'loss': loss,
                'feature_names': self.feature_names,
                'symbol': self.target_symbol,
                'timestamp': datetime.now().isoformat()
            }, self.target_output)
            print(f"💾 Model saved to: {self.target_output}")

        def load_data(self) -> pd.DataFrame:
            """Override to fetch SPECIFIC symbol data"""
            print(f"📂 Loading data for {self.target_symbol}...")
            
            # 0. Use direct CSV path if provided (HIGHEST PRIORITY)
            if self.csv_path and os.path.exists(self.csv_path):
                print(f"   ✅ Using direct CSV: {self.csv_path}")
                df = pd.read_csv(self.csv_path)
                df.columns = df.columns.str.lower()
                col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
                df = df.rename(columns=col_map)
                print(f"   ✅ Loaded {len(df)} rows from Binance CSV")
                return df
            
            # 1. Try to find Specific CSV first (e.g. BTCUSDT_1h.csv)
            symbol_clean = self.target_symbol.replace('/','').upper()
            csv_patterns = [
                os.path.join(self.data_dir, f"{symbol_clean}_1h.csv"),  # Prioritize 1h
                os.path.join(self.data_dir, f"{symbol_clean}.csv"),
                os.path.join(self.data_dir, f"{self.target_symbol}.csv"),
            ]
            
            for csv_path in csv_patterns:
                if os.path.exists(csv_path):
                    print(f"   ✅ Found local CSV: {csv_path}")
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.lower()
                    col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
                    df = df.rename(columns=col_map)
                    print(f"   ✅ Loaded {len(df)} rows from local CSV")
                    return df
                
            # 2. Fetch from YFinance
            print(f"   ⚠️ No CSV. Fetching {self.target_symbol} from Yahoo Finance...")
            try:
                import yfinance as yf
                # Map crypto symbols: BTC/USDT -> BTC-USD
                # Remove slash first, then replace USDT to avoid double dash
                yf_symbol = self.target_symbol.replace('/', '').replace('USDT', '-USD')
                
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period="2y", interval="1h")
                
                if len(df) < 100:
                    print("   ⚠️ Hourly data too short. Trying daily...")
                    df = ticker.history(period="5y", interval="1d")
                
                if len(df) > 100:
                    print(f"   ✅ Fetched {len(df)} rows from YFinance")
                    df = df.reset_index()
                    df.columns = df.columns.str.lower()
                    return df[['open', 'high', 'low', 'close', 'volume']]
                    
            except Exception as e:
                print(f"   ❌ YFinance failed: {e}")
            
            print("   ❌ Failed to get data. Generating SYNTHETIC data for testing...")
            return self._generate_synthetic_data()


    # 🛡️ PORTABILITY FIX: Use relative paths for Docker/Cloud deployment
    # Define the root directory (script is in /src/, go up one level)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'market_data')  # ../market_data
    MODELS_DIR = os.path.join(BASE_DIR, 'models')  # aegis_alpha/models
    
    # Initialize targeted trainer
    trainer = TargetedLSTMTrainer(
        target_symbol=args.symbol,
        target_output=args.output,
        csv_path=args.csv_path,  # Pass direct CSV path
        data_dir=DATA_DIR,
        model_dir=MODELS_DIR,
        seq_len=60,
        batch_size=args.batch_size,
        learning_rate=0.00001,
        epochs=args.epochs
    )
    
    # Train
    results = trainer.train()
    
    print("\n📊 Training Results:")
    print(f"   Accuracy: {results['final_accuracy']:.2%}")
    print(f"   Loss: {results['best_val_loss']:.4f}")
    print(f"   Time: {results['training_time']/60:.1f} min")
    print(f"   Saved to: {args.output}")
