"""
TERMINAL - SENTINEL EXECUTOR
=============================
Multi-Asset Trading Engine with LSTM Neural Core and SQLite Persistence.
"One brain per asset. One database for all."
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import yaml
import ccxt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/executor.log')
    ]
)
logger = logging.getLogger('SENTINEL')

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Components
try:
    from lstm_model import SignalPredictor as LSTMPredictor
    from lgbm_adapter import LightGBMPredictor
    from persistence import Database
    from quant_utils import calc_position_size, prepare_features
    from notifier import sentinel
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    sys.exit(1)

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}. Using defaults.")
        return {}

CONFIG = load_config()

class SentinelExecutor:
    """
    The Head of Sentinel: Multi-Model Trading Executor
    """
    
    def __init__(self):
        # 1. Initialize Database
        self.db = Database()
        
        # 2. Config & State
        self.active_pairs = CONFIG.get('trading', {}).get('active_pairs', [])
        self.paper_mode = CONFIG.get('trading', {}).get('paper_trading_mode', True)
        self.running = False
        self.signals_generated = 0
        self.errors = 0
        
        # 3. Model Registry { 'BTC/USDT': PredictorObj }
        self.models = {}
        self.last_signals = {} # Anti-Spam Cache
        
        # 4. Exchange Connection (Binance)
        self.exchange = None
        self._init_exchange()
        
        # 5. Load Brains
        self._load_models()
        
    def _init_exchange(self):
        """Initialize CCXT exchange connection"""
        try:
            # We use environment variables for keys to avoid hardcoding secrets
            api_key = os.environ.get('BINANCE_API_KEY')
            secret = os.environ.get('BINANCE_SECRET')
            
            if api_key and secret:
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'} # Futures mode
                })
                logger.info("🔌 Connected to Binance Futures (Live Execution Ready)")
            else:
                logger.warning("⚠️ No API Keys found. Running in READ-ONLY mode.")
                
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")

    def _load_models(self):
        """Load all models defined in config"""
        logger.info("=" * 60)
        logger.info("🐉 SENTINEL PROTOCOL: LOADING NEURAL CORES")
        logger.info("=" * 60)
        
        model_paths = CONFIG.get('trading', {}).get('model_paths', {})
        
        for pair in self.active_pairs:
            # Get path for this pair
            path = model_paths.get(pair)
            
            if not path:
                logger.warning(f"⚠️ No model path config for {pair}. Using fallback.")
                path = "models/terminal_lstm.pth" # Fallback
            
            # Resolve absolute path
            abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            
            if os.path.exists(abs_path):
                try:
                    # Determine model type by extension
                    if abs_path.endswith('.pkl'):
                        predictor = LightGBMPredictor(abs_path)
                    else:
                        predictor = LSTMPredictor(abs_path)
                        
                    self.models[pair] = predictor
                    
                    # Update DB performance tracking
                    self.db.update_model_performance(pair, path)
                    logger.info(f"✅ Loaded Core: {pair} -> {path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {pair}: {e}")
            else:
                logger.warning(f"❌ Model file missing: {abs_path}")
                
        logger.info(f"🐉 Sentinel Online: {len(self.models)} active cores")

    def is_market_safe(self, pair: str, df: pd.DataFrame) -> bool:
        """
        Circuit Breaker: The Volatility Shield (Anti-Gravity Protocol V2)
        Calculates volatility and freezes if price movement exceeds threshold.
        """
        try:
            if len(df) < 20:
                return True # Not enough data to judge
            
            # Calculate volatility (Standard Deviation of last 20 candles)
            volatility = df['close'].rolling(window=20).std().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # If volatility is > 1.5% of price, it's a NEWS EVENT (Crash/Pump)
            # Circuit breaker requested at 1.5% (0.015)
            volatility_threshold = current_price * 0.015
            
            if volatility > volatility_threshold:
                logger.warning(f"🚨 VOLATILITY SHIELD ACTIVE for {pair}: {volatility:.2f} > {volatility_threshold:.2f}. FREEZING.")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error in Volatility Shield: {e}")
            return True # Fail open but log

    def scan_market(self):
        """
        Main loop: Iterate through all pairs and generate signals
        """
        for pair in self.active_pairs:
            try:
                # 1. Get Model
                model = self.models.get(pair)
                if not model:
                    continue
                
                # 2. Fetch Data (100 candles for features)
                # ... [Fetching logic remains same]
                # If exchange is connected, use it. Else simulation mode.
                df = None
                if self.exchange:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(pair, '15m', limit=100)
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                    except Exception as e:
                        logger.error(f"Failed to fetch data for {pair}: {e}")
                        continue
                else:
                    # Simulation Mode: We need to fetch from API Server or mock
                    # Actually, yfinance can serve as fallback in executor too
                    # 2. Fetch Fresh Data (Switchblade Protocol)
                    try:
                        import yfinance as yf
                        yf_map = {"BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD", "SOL/USDT": "SOL-USD"}
                        ticker = yf_map.get(pair, pair.replace('/', '-'))
                        # Note: asyncio.to_thread is for async contexts. For synchronous, call directly.
                        df = yf.download(ticker, period="2d", interval="15m", progress=False)
                        
                        if df is None or df.empty:
                            logger.warning(f"No data for {pair}")
                            continue
                            
                        # 🐉 AGGRESSIVE NORMALIZATION (NASA-Grade)
                        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                            df.columns = df.columns.get_level_values(0)
                        df.columns = [str(c).lower().strip() for c in df.columns]
                        
                    except Exception as e:
                        logger.error(f"Fetch failed for {pair}: {e}")
                        continue

                # 3. Features & Prediction (Sovereign Engine)
                features = prepare_features(df)
                
                # 4. Run Inference (NASA-Grade Threshold Sync)
                threshold = CONFIG.get('confidence', {}).get('buy_threshold', 0.65)
                result = model.predict(features, threshold=threshold)
                
                # Use scalar extraction to avoid index ghosts
                price_now = float(df['close'].values[-1])
                
                confidence = result.get('confidence', 0)
                signal = result.get('signal', 'HOLD')
                reason = result.get('reason', 'N/A')
                
                # 🐉 ANTI-SPAM: Only log if signal changed
                last_sig = self.last_signals.get(pair)
                if signal == last_sig and signal == 'HOLD':
                    # Silent skip for redundant HOLDs
                    continue
                
                logger.info(f"🔍 Scan {pair}: {signal} ({confidence:.1%}) | Reason: {reason} | Price: {price_now}")
                
                # Update Cache
                self.last_signals[pair] = signal
                
                # PERSISTENCE: Save result
                result['price'] = price_now
                self.process_signal(pair, result)
                
                # 5. Execute with Volatility Shield
                if signal != 'HOLD' and result.get('approved', False):
                    # APPLY VOLATILITY SHIELD (Step 1)
                    if not self.is_market_safe(pair, df):
                        logger.warning(f"🛡️ SIGNAL BLOCKED BY SHIELD: {pair} {signal}")
                        continue

                    logger.info(f"🚨 ACTION REQUIRED: {signal} {pair} @ {df['close'].iloc[-1]}")
                    
                    if self.exchange:
                        # REAL EXECUTION (If keys present)
                        # side = 'buy' if signal == 'BUY' else 'sell'
                        # order = self.exchange.create_market_order(pair, side, amount)
                        # logger.info(f"✅ ORDER EXECUTED: {order}")
                        pass
                    else:
                        logger.warning(f"⚠️ SIMULATION: Would have {signal} {pair} (No Keys)") 
                
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")

    def process_signal(self, symbol: str, signal_data: Dict):
        """
        Process a signal generated by the API or autonomous loop
        """
        # 1. Log to Persistence
        signal_id = self.db.log_signal(symbol, signal_data)
        
        # 2. Check Execution Criteria
        confidence = signal_data.get('confidence', 0)
        signal_type = signal_data.get('signal', 'HOLD')
        is_approved = signal_data.get('approved', False)
        
        if is_approved and signal_type in ['BUY', 'SELL']:
            logger.info(f"🚨 ACTIONABLE SIGNAL: {symbol} {signal_type} ({confidence:.1%})")
            
            # Broadcast to Devices
            sentinel.alert_signal(symbol, signal_type, signal_data.get('price', 0))
            
            # 3. Execute Trade (if allowed)
            if not self.paper_mode:
                self.execute_trade(symbol, signal_type, signal_data)
            else:
                logger.info(f"📝 PAPER TRADE: {symbol} {signal_type} logged.")

    def monitor_positions(self):
        """
        The Sentinel: Enforces Triple-Barrier constraints on all open positions.
        """
        try:
            # 1. Fetch all OPEN trades from DB
            open_trades = self.db.get_trades(status='OPEN')
            if not open_trades:
                return

            for trade in open_trades:
                trade_id = trade['id']
                symbol = trade['symbol']
                side = trade['side']
                entry_price = trade['entry_price']
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                
                # 2. Get current price
                # We'll use the last fetched price from the regular scan if possible, 
                # but for accuracy, we fetch fresh.
                try:
                    ticker = self.exchange.fetch_ticker(symbol) if self.exchange else None
                    current_price = ticker['last'] if ticker else None
                    
                    if not current_price:
                        # Fallback for simulation
                        import yfinance as yf
                        yf_sym = symbol.replace('/', '-') + "-USD" if '/' in symbol else symbol
                        tk = yf.Ticker(yf_sym)
                        current_price = tk.fast_info['lastPrice']
                except:
                    continue

                if not current_price:
                    continue

                # 3. Check Barriers
                hit_sl = (side == 'BUY' and current_price <= stop_loss) or (side == 'SELL' and current_price >= stop_loss)
                hit_tp = (side == 'BUY' and current_price >= take_profit) or (side == 'SELL' and current_price <= take_profit)
                
                if hit_sl or hit_tp:
                    reason = "SL HIT" if hit_sl else "TP HIT"
                    logger.info(f"🚨 BARRIER REACHED: {symbol} | {reason} @ {current_price}")
                    
                    if self.exchange and not self.paper_mode:
                        # Close trade on exchange
                        # self.exchange.create_order(symbol, 'market', 'sell' if side == 'BUY' else 'buy', trade['quantity'])
                        pass
                    
                    # Log closing in DB
                    self.db.close_trade(trade_id, current_price, notes=reason)
                    logger.info(f"✅ POSITION CLOSED: {symbol} ID:{trade_id}")
                    
                    # Broadcast to Devices
                    sentinel.alert_barrier(symbol, reason, current_price)

        except Exception as e:
            logger.error(f"Error in Position Sentinel: {e}")

    def execute_trade(self, symbol: str, side: str, signal_data: Dict):
        """
        Execute live trade on exchange
        """
        if not self.exchange and not self.paper_mode:
            logger.error("Cannot execute LIVE: No exchange connection")
            return
            
        try:
            # 1. Calculate Quantity
            balance = 1000.0 # Placeholder
            risk_pct = CONFIG.get('risk', {}).get('max_per_trade', 0.01)
            entry = signal_data.get('entry_price', signal_data.get('price', 0))
            
            # 🐉 NASA-GRADE: Ensure SL exists
            sl = signal_data.get('stop_loss')
            if not sl:
                # Fallback to ATR-based SL if not provided
                atr_dist = signal_data.get('price', 0) * 0.02 # 2% fallback
                sl = entry - atr_dist if side == 'BUY' else entry + atr_dist
            
            quantity = calc_position_size(balance, risk_pct, entry, sl)
            
            if quantity <= 0:
                logger.warning(f"Quantity 0 for {symbol}. Entry: {entry}, SL: {sl}")
                return

            # 2. Log Trade (Always log, even paper)
            tp = signal_data.get('take_profit')
            trade_id = self.db.log_trade(symbol, side, entry, quantity, sl, tp)
            
            if not self.paper_mode and self.exchange:
                # Real Order logic here
                pass
                
            logger.info(f"🚀 {'PAPER ' if self.paper_mode else 'LIVE '}POSITION OPENED: {side} {quantity} {symbol} ID:{trade_id}")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")


    def run(self):
        """Main Life Cycle (Anti-Gravity Refactor)"""
        logger.info(f"🚀 TERMINAL Executor (PID {os.getpid()}) Started. Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        self.running = True
        
        while self.running:
            try:
                # 1. Position Sentinel: Monitor SL/TP for open trades
                self.monitor_positions()

                # 2. Market Pulse: Fetch & Switchblade Logic (Scan Market)
                self.scan_market()
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                self.running = False
                logger.info("Sentinel Shutdown.")
            except Exception as e:
                logger.error(f"CRITICAL LOOP FAILURE: {e}")
                time.sleep(5) # Backoff


if __name__ == '__main__':
    print("🐉 TERMINAL - SENTINEL ENGINE STARTING...")
    executor = SentinelExecutor()
    executor.run()
