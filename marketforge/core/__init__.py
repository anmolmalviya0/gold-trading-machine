"""
FORGE TRADING SYSTEM - Core Module
"""
from .data import HistoricalData, LiveDataFeed, load_config
from .features import add_features, get_market_regime
from .signals import SignalEngine
from .conviction import ConvictionFilter, create_trade_card
from .backtest import Backtester, WalkForwardValidator
from .database import Database
from .drift import DriftDetector, DriftState
from .news import NewsEngine, NewsImpact
from .supervisor import Supervisor, EngineState
from .mutation_lab import MutationLab

__all__ = [
    'HistoricalData',
    'LiveDataFeed',
    'load_config',
    'add_features',
    'get_market_regime',
    'SignalEngine',
    'ConvictionFilter',
    'create_trade_card',
    'Backtester',
    'WalkForwardValidator',
    'Database',
    'DriftDetector',
    'DriftState',
    'NewsEngine',
    'NewsImpact',
    'Supervisor',
    'EngineState',
    'MutationLab'
]
