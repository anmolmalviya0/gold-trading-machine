"""
Trading Intelligence System - Configuration
"""

# Asset Specifications
ASSETS = {
    'BTC': {
        'binance_symbol': 'BTCUSDT',
        'broker_symbol': None,
        'point_value': 1.0,
        'pip_decimals': 0,
        'type': 'crypto'
    },
    'PAXG': {
        'binance_symbol': 'PAXGUSDT',
        'broker_symbol': None,
        'point_value': 1.0,
        'pip_decimals': 2,
        'type': 'crypto'
    },
    'XAU': {
        'binance_symbol': None,  # Fallback to PAXG if unavailable
        'broker_symbol': 'XAUUSD',
        'point_value': 0.01,  # $0.01 per pip
        'pip_decimals': 2,
        'type': 'forex'
    }
}

# Timeframes
TIMEFRAMES = ['5m', '15m', '30m', '1h']

# Binance Configuration
BINANCE_CONFIG = {
    'ws_base_url': 'wss://stream.binance.com:9443/ws',
    'rest_base_url': 'https://api.binance.com',
    'reconnect_delay': 5,  # seconds
    'max_reconnect_attempts': 10
}

# Broker Configuration
BROKER_CONFIG = {
    'type': 'MT5',  # MT5, IBKR, or OANDA
    'timeout': 10,
    'max_retries': 3
}

# Model Configuration
MODEL_CONFIG = {
    'engine_a': {
        'type': 'xgboost',
        'lookback': 100,
        'forecast_bars': 10
    },
    'engine_b': {
        'type': 'lightgbm',
        'lookback': 50,
        'threshold_atr': 1.5
    },
    'engine_c': {
        'type': 'gru',
        'sequence_length': 60,
        'hidden_size': 128
    }
}

# Signal Configuration
SIGNAL_CONFIG = {
    'min_confidence': 60,  # Minimum confidence to show signal
    'high_confidence': 80,  # Threshold for Telegram alerts
    'r_multiple_min': 1.5,  # Minimum R:R to consider trade
    'atr_stop_multiplier': 2.0,
    'targets': [1.0, 2.0, 3.0]  # R-multiples for TP1/TP2/TP3
}

# Multi-Timeframe Consensus
CONSENSUS_CONFIG = {
    'strong_alignment': 3,  # 3+ timeframes agree
    'moderate_alignment': 2,  # 2 timeframes agree
    'conflict_threshold': 2  # Max opposing signals
}

# Reliability
RELIABILITY_CONFIG = {
    'watchdog_interval': 30,  # seconds
    'stale_data_threshold': 120,  # seconds
    'max_memory_mb': 2048,
    'log_level': 'INFO'
}

# UI
UI_CONFIG = {
    'streamlit_port': 8501,
    'refresh_interval': 1,  # seconds
    'telegram_enabled': True
}
