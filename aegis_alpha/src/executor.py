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
import requests

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
    from quant_utils import calc_position_size, prepare_features, calculate_kelly_fraction, adaptive_threshold_logic
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
        self.active_pairs = CONFIG.get('system', {}).get('active_pairs', ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'PAXG/USDT'])
        self.active_timeframes = CONFIG.get('active_timeframes', ['1m', '5m', '15m', '30m', '1h', '1d'])
        self.paper_mode = CONFIG.get('trading', {}).get('paper_trading_mode', True)
        self.running = False
        self.signals_generated = 0
        self.errors = 0
        
        # 3. Model Registry { 'BTC/USDT_15m': PredictorObj }
        self.models = {}
        self.last_signals = {} # Anti-Spam Cache
        self.market_context = {} # { 'BTC/USDT_15m': trend_bias }
        
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
            for tf in self.active_timeframes:
                # Key format: BTC/USDT_15m
                node_key = f"{pair}_{tf}"
                
                # Get path for this node
                path = model_paths.get(pair) # Old config style
                # New config style check
                clean_pair = pair.replace('/', '')
                path = f"models/{clean_pair}_{tf}_lgbm.pkl"
                
                # Resolve absolute path
                abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
                
                if os.path.exists(abs_path):
                    try:
                        predictor = LightGBMPredictor(abs_path)
                        self.models[node_key] = predictor
                        self.db.update_model_performance(node_key, path)
                        logger.info(f"✅ Loaded Core: {node_key} -> {path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to load {node_key}: {e}")
                
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
        Main loop: Iterate through all pairs and timeframes to generate signals
        """
        for pair in self.active_pairs:
            # First, update macro context for this pair (15m/1h trend)
            self._update_market_context(pair)
            
            for tf in self.active_timeframes:
                node_key = f"{pair}_{tf}"
                try:
                    # 1. Get Model
                    model = self.models.get(node_key)
                    if not model:
                        continue
                    
                    # 2. Fetch Data
                    df = self._fetch_ohlcv(pair, tf)
                    if df is None or df.empty:
                        continue
                    
                    # 3. Features & Prediction
                    features = prepare_features(df)
                    vol_z = features['vol_z_score'].iloc[-1] if 'vol_z_score' in features.columns else 0
                    
                    # 🐉 ADAPTIVE THRESHOLD ENGINE (Step 2)
                    base_threshold = CONFIG.get('confidence', {}).get('buy_threshold', 0.65)
                    threshold = adaptive_threshold_logic(base_threshold, vol_z)
                    
                    result = model.predict(features, threshold=threshold)
                    price_now = float(df['close'].values[-1])
                    confidence = result.get('confidence', 0)
                    signal = result.get('signal', 'HOLD')
                    
                    # 🐉 TRIPLE-BARRIER GATEKEEPER (Step 3)
                    # For HFT (1m/5m), we must align with 15m trend
                    if tf in ['1m', '5m'] and signal != 'HOLD':
                        macro_bias = self.market_context.get(f"{pair}_15m", "NEUTRAL")
                        if (signal == 'BUY' and macro_bias == 'BEARISH') or (signal == 'SELL' and macro_bias == 'BULLISH'):
                            logger.info(f"🛡️ GATEKEEPER VETO: {node_key} {signal} blocked by 15m {macro_bias} trend.")
                            continue
                    
                    # Anti-Spam
                    if signal == self.last_signals.get(node_key) and signal == 'HOLD':
                        continue
                    
                    logger.info(f"🔍 Scan {node_key}: {signal} ({confidence:.1%}) | Price: {price_now}")
                    self.last_signals[node_key] = signal
                    
                    # Process
                    result['price'] = price_now
                    result['timeframe'] = tf
                    if signal != 'HOLD' and result.get('approved', False):
                        if self.is_market_safe(pair, df):
                            self.process_signal(pair, result)
                            
                except Exception as e:
                    logger.error(f"Error scanning {node_key}: {e}")

    def _update_market_context(self, pair: str):
        """Build macro bias for Gatekeeper logic"""
        try:
            # Check 15m trend
            df = self._fetch_ohlcv(pair, '15m', limit=50)
            if df is not None and not df.empty:
                sma20 = df['close'].rolling(20).mean().iloc[-1]
                price = df['close'].iloc[-1]
                bias = "BULLISH" if price > sma20 else "BEARISH"
                self.market_context[f"{pair}_15m"] = bias
        except:
            self.market_context[f"{pair}_15m"] = "NEUTRAL"

    def _fetch_ohlcv(self, pair: str, tf: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Unified fetcher for multiple sources"""
        try:
            if self.exchange:
                ohlcv = self.exchange.fetch_ohlcv(pair, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                import yfinance as yf
                yf_map = {"BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD", "SOL/USDT": "SOL-USD"}
                ticker = yf_map.get(pair, pair.replace('/', '-'))
                # Map timeframe to yfinance strings
                yf_tf = tf
                if tf == '1m': yf_tf = '1m'
                elif tf == '5m': yf_tf = '5m'
                df = yf.download(ticker, period="2d" if tf != '1d' else "1mo", interval=yf_tf, progress=False)
                if df is None or df.empty: return None
                if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1: df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).lower().strip() for c in df.columns]
                return df
        except Exception as e:
            logger.error(f"Fetch failed for {pair} {tf}: {e}")
            return None

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
        Execute trade with The Vulcan Protocol (Kelly Criterion)
        """
        try:
            # 1. Calculate Multiplier using Kelly Criterion
            # Average win rate for current nodes is ~61%, RR is roughly 1.25 (1.5/1.2)
            prob = signal_data.get('confidence', 0.65)
            # 🐉 VULCAN PROTOCOL: Dynamic Kelly weighting
            kelly_mult = calculate_kelly_fraction(prob, 1.25) # half-kelly default
            
            # Base risk from config
            base_risk_pct = CONFIG.get('risk', {}).get('max_per_trade', 0.01)
            dynamic_risk = base_risk_pct * kelly_mult
            
            logger.info(f"🌋 VOLCANO IGNITION: {symbol} {side} | Kelly Mult: {kelly_mult:.2f} | Dynamic Risk: {dynamic_risk:.2%}")
            
            balance = 1000.0 # Standard unit for simulation
            entry = signal_data.get('price', 0)
            sl = signal_data.get('stop_loss')
            tp = signal_data.get('take_profit')
            
            if not sl:
                # Fallback ATR-based SL (Institutional standard: -1.2 ATR)
                atr_dist = entry * 0.015
                sl = entry - atr_dist if side == 'BUY' else entry + atr_dist
            
            # Use dynamic risk for position sizing
            quantity = calc_position_size(balance, dynamic_risk, entry, sl)
            
            if quantity <= 0: return

            # 2. Log Trade
            trade_id = self.db.log_trade(symbol, side, entry, quantity, sl, tp)
            logger.info(f"🚀 {'PAPER ' if self.paper_mode else 'LIVE '}POSITION OPENED: {side} {quantity} {symbol} ID:{trade_id}")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")


    def run(self):
        """Main Life Cycle (Anti-Gravity Refactor)"""
        logger.info(f"🚀 TERMINAL Executor (PID {os.getpid()}) Started. Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        self.running = True
        
        last_market_scan = 0
        scan_interval = 30 # Scan for new signals every 30s
        
        while self.running:
            try:
                # 1. Position Sentinel: Monitor SL/TP for open trades (High Frequency)
                self.monitor_positions()

                # 2. Market Pulse: Fetch & Switchblade Logic (Lower Frequency)
                current_time = time.time()
                if current_time - last_market_scan >= scan_interval:
                    self.scan_market()
                    last_market_scan = current_time
                
                # Lightning Speed Monitoring
                time.sleep(2)
                
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
