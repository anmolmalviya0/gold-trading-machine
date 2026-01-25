"""
TERMINAL I - HYDRA EXECUTOR
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
logger = logging.getLogger('HYDRA')

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Components
try:
    from lstm_model import SignalPredictor as LSTMPredictor
    from lgbm_adapter import LightGBMPredictor
    from persistence import Database
    from quant_utils import calc_position_size
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

class HydraExecutor:
    """
    The Head of Hydra: Multi-Model Trading Executor
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
        logger.info("🐉 HYDRA PROTOCOL: LOADING NEURAL CORES")
        logger.info("=" * 60)
        
        model_paths = CONFIG.get('trading', {}).get('model_paths', {})
        
        for pair in self.active_pairs:
            # Get path for this pair
            path = model_paths.get(pair)
            
            if not path:
                logger.warning(f"⚠️ No model path config for {pair}. Using fallback.")
                path = "models/aegis_lstm.pth" # Fallback
            
            # Resolve absolute path
            abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            
            if os.path.exists(abs_path):
                try:
                    # Initialize LSTM Predictor
                    predictor = LSTMPredictor(abs_path)
                    self.models[pair] = predictor
                    
                    # Update DB performance tracking
                    self.db.update_model_performance(pair, path)
                    logger.info(f"✅ Loaded Core: {pair} -> {path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {pair}: {e}")
            else:
                logger.warning(f"❌ Model file missing: {abs_path}")
                
        logger.info(f"🐉 Hydra Online: {len(self.models)} active cores")

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
                
                # 2. Fetch Data (Mock for now, would be CCXT history in prod)
                # In real execution, we need:
                # ohlcv = self.exchange.fetch_ohlcv(pair, '15m', limit=100)
                # For now, we rely on the API server to push data or simulation
                
                # NOTE: This executor loop is a placeholder for the autonomous agent.
                # In the current architecture, the API Server actually triggers inferences
                # via user request. But this loop prepares for fully autonomous mode.
                pass 
                
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
            
            # 3. Execute Trade (if allowed)
            if not self.paper_mode:
                self.execute_trade(symbol, signal_type, signal_data)
            else:
                logger.info(f"📝 PAPER TRADE: {symbol} {signal_type} logged.")

    def execute_trade(self, symbol: str, side: str, signal_data: Dict):
        """
        Execute live trade on exchange
        """
        if not self.exchange:
            logger.error("Cannot execute: No exchange connection")
            return
            
        try:
            # 1. Calculate Quantity
            balance = 1000.0 # Placeholder, should fetch from exchange
            risk_pct = CONFIG['risk']['max_per_trade']
            entry = signal_data['entry_price']
            sl = signal_data['stop_loss']
            
            quantity = calc_position_size(balance, risk_pct, entry, sl)
            
            if quantity <= 0:
                logger.warning(f"Calculated 0 quantity for {symbol}")
                return

            # 2. Place Order
            # order = self.exchange.create_order(symbol, 'market', side, quantity)
            
            # 3. Log Trade to DB
            self.db.log_trade(symbol, side, entry, quantity, sl, signal_data['take_profit'])
            logger.info(f"🚀 ORDER SENT: {side} {quantity} {symbol}")
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")

    def run(self):
        """Main Life Cycle"""
        logger.info("Starting Hydra Executor loop...")
        self.running = True
        
        while self.running:
            try:
                # In V2, this loop would fetch live candles via CCXT
                # and pass them to self.models[pair].predict(features)
                # For now, we keep the process alive to serve as the system anchor
                time.sleep(10)
                
            except KeyboardInterrupt:
                self.running = False
                logger.info("Hydra Shutdown.")

if __name__ == '__main__':
    print("🐉 TERMINAL I - HYDRA ENGINE STARTING...")
    executor = HydraExecutor()
    executor.run()
