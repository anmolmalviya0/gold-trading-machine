"""
AEGIS V21 - LightGBM Backtest Engine
=====================================
Dedicated backtest engine for LightGBM model.
Implements Triple-Barrier Logic with 0.65 Confidence Threshold.
"""
import pandas as pd
import numpy as np
import joblib
import os

class LightGBMBacktester:
    def __init__(self, data_path='/Users/anmol/Desktop/gold/market_data/PAXGUSDT_5m.csv', model_path='/Users/anmol/Desktop/gold/aegis_alpha/models/aegis_lgbm.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = None # None, 'BUY', 'SELL'
        self.trades = []
        
        # RISK SETTINGS (Triple-Barrier)
        self.tp_atr = 2.0
        self.sl_atr = 1.5
        self.confidence_threshold = 0.65  # THE GOLDEN PARAMETER
        
    def load_resources(self):
        print(f"📦 Loading Model from {self.model_path}...")
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return False
            
        print(f"📂 Loading Data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        # Normalize columns
        self.df.columns = self.df.columns.str.lower()
        column_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        self.df = self.df.rename(columns=column_map)
        
        print(f"✅ Loaded {len(self.df)} rows")
        return True

    def engineer_features(self):
        """Re-create features identical to training"""
        print("⚙️ Engineering features...")
        df = self.df.copy()
        
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
        
        # Prepare final dataframe with features and ATR
        # We need ATR for sizing, and features for prediction
        self.df = df.dropna().reset_index(drop=True)
        
        # Ensure 'ATR' column exists (uppercase alias for logic)
        self.df['ATR'] = self.df['atr']
        
        print(f"✅ Features ready: {len(self.df)} clean rows")

    def run(self):
        print("\n" + "="*60)
        print("⚡ STARTING LIGHTGBM BACKTEST")
        print("="*60)
        
        # 1. PREPARE FEATURES FOR PREDICTION
        feature_cols = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
            'rsi_norm', 'macd_hist', 'bb_position', 'atr_ratio', 'volume_ratio'
        ]
        
        X = self.df[feature_cols]
        
        # 2. BATCH PREDICTION
        print("🔮 Generating Predictions...")
        probs = self.model.predict_proba(X)[:, 1] # Probability of Class 1
        self.df['probability'] = probs
        
        print(f"   Signals > {self.confidence_threshold}: {(probs > self.confidence_threshold).sum()}")
        
        # 3. SIMULATION LOOP
        print("🔄 Simulating Market...")
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            current_price = row['close']
            current_atr = row.get('ATR', current_price * 0.01)
            
            # CHECK EXITS
            if self.position:
                self._check_exit(row, current_price)
                
            # CHECK ENTRY (If no position)
            if self.position is None:
                if row['probability'] >= self.confidence_threshold:
                    self._open_position(current_price, current_atr, i)

        self._generate_report()

    def _open_position(self, price, atr, index):
        # Position Sizing: Risk 2% of equity
        risk_per_share = atr * self.sl_atr
        if risk_per_share == 0: return
        
        risk_amount = self.balance * 0.02
        shares = risk_amount / risk_per_share
        
        self.position = {
            'entry_price': price,
            'shares': shares,
            'stop_loss': price - (atr * self.sl_atr),
            'take_profit': price + (atr * self.tp_atr),
            'entry_index': index
        }

    def _check_exit(self, row, current_price):
        pos = self.position
        
        # HIT TAKE PROFIT
        if current_price >= pos['take_profit']:
            profit = (pos['take_profit'] - pos['entry_price']) * pos['shares']
            self.balance += profit
            self._log_trade("WIN", profit)
            self.position = None
            
        # HIT STOP LOSS
        elif current_price <= pos['stop_loss']:
            loss = (pos['stop_loss'] - pos['entry_price']) * pos['shares']
            self.balance += loss
            self._log_trade("LOSS", loss)
            self.position = None
            
        # Time-based exit (optional, max 20 bars)
        elif (row.name - pos['entry_index']) >= 20: 
            # row.name is index because we reset_index
             pnl = (current_price - pos['entry_price']) * pos['shares']
             res = "WIN" if pnl > 0 else "LOSS"
             self.balance += pnl
             self._log_trade(res, pnl)
             self.position = None

    def _log_trade(self, result, amount):
        self.trades.append({
            'result': result,
            'pnl': amount,
            'balance': self.balance
        })

    def _generate_report(self):
        if not self.trades:
            print("💀 NO TRADES TRIGGERED.")
            return

        df_t = pd.DataFrame(self.trades)
        wins = df_t[df_t['result'] == 'WIN']
        losses = df_t[df_t['result'] == 'LOSS']
        
        total_trades = len(df_t)
        if total_trades == 0: return
        
        win_rate = len(wins) / total_trades
        gross_profit = wins['pnl'].sum() if not wins.empty else 0
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        net_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        print("\n" + "="*60)
        print("📊 === LIGHTGBM BACKTEST REPORT ===")
        print("="*60)
        print(f"Threshold: >{self.confidence_threshold}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate:     {win_rate*100:.1f}%  (Target: >60%)")
        print(f"Profit Factor: {profit_factor:.2f}  (Target: >1.5)")
        print(f"Final Equity: ${self.balance:.2f}")
        print(f"Net Return:   {net_return:.1f}%")
        
        print("\n" + "="*60)
        if profit_factor > 1.5 and win_rate > 0.6:
             print("🚀 DECISION: GO FOR DEPLOYMENT")
        else:
             print("🛑 DECISION: RECALIBRATE")
        print("="*60)

if __name__ == "__main__":
    bt = LightGBMBacktester()
    if bt.load_resources():
        bt.engineer_features()
        bt.run()
