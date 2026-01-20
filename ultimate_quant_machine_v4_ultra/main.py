#!/usr/bin/env python3
"""
ULTIMATE QUANT MACHINE V4 PRO MAX ULTRA - Main Runner

Usage:
    python main.py backtest --symbol PAXGUSDT --tf 15m
    python main.py live --symbol PAXGUSDT
    python main.py download
"""

import argparse
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Import modules
from indicators import Indicators
from conviction_monitor import ConvictionMonitor
from strategy_engine import StrategyEngine
from backtester import BacktestEngine
from walkforward import WalkForwardValidator
from live_ws import BinanceWebSocketFeed, OrderBookSnapshot
from alerts import AlertManager
from db import SignalDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
CONFIG_FILE = Path(__file__).parent / "config.yaml"
with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)


class QuantMachineV4:
    """Main engine"""
    
    def __init__(self):
        self.config = CONFIG
        self.db = SignalDatabase()
        self.strategy = StrategyEngine(CONFIG)
        self.conviction = ConvictionMonitor(CONFIG)
        self.backtest = BacktestEngine(CONFIG)
        self.walkforward = WalkForwardValidator(CONFIG)
        self.alerts = AlertManager(CONFIG)
        self.ob = OrderBookSnapshot(CONFIG)
        self.indicators = Indicators()
    
    def backtest_symbol(self, symbol: str, tf: str):
        """Run backtest on a symbol/timeframe"""
        
        logger.info(f"Starting backtest: {symbol} {tf}")
        
        # Load CSV
        csv_path = Path(__file__).parent / 'data' / f'{symbol}_{tf}.csv'
        if not csv_path.exists():
            logger.error(f"Data file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['open_time'])
        
        # Calculate indicators
        logger.info("Calculating indicators...")
        df['rsi'] = self.indicators.rsi_wilder(df['close'], 14)
        df['atr'] = self.indicators.atr_wilder(df['high'], df['low'], df['close'], 14)
        df['ema50'] = self.indicators.ema(df['close'], 50)
        df['ema200'] = self.indicators.ema(df['close'], 200)
        df['adx'] = self.indicators.adx(df['high'], df['low'], df['close'], 14)
        bb_u, bb_m, bb_l, bb_w = self.indicators.bollinger_bands(df['close'], 20, 2.0)
        df['bb_upper'] = bb_u
        df['bb_middle'] = bb_m
        df['bb_lower'] = bb_l
        df['bb_width'] = bb_w
        
        # Detect pivots
        logger.info("Detecting pivots...")
        ph_idx, pl_idx = self.indicators.detect_pivots(df['high'], df['low'], 5, 5)
        
        # Generate signals
        logger.info("Generating signals...")
        signals = []
        
        for idx in range(50, len(df)):
            # Get slices
            close_slice = df['close'].iloc[:idx+1]
            high_slice = df['high'].iloc[:idx+1]
            low_slice = df['low'].iloc[:idx+1]
            rsi_slice = df['rsi'].iloc[:idx+1]
            atr_slice = df['atr'].iloc[:idx+1]
            ema50_slice = df['ema50'].iloc[:idx+1]
            ema200_slice = df['ema200'].iloc[:idx+1]
            adx_slice = df['adx'].iloc[:idx+1]
            bb_u_slice = df['bb_upper'].iloc[:idx+1]
            bb_l_slice = df['bb_lower'].iloc[:idx+1]
            bb_w_slice = df['bb_width'].iloc[:idx+1]
            
            # Detect divergence
            divs = self.indicators.detect_divergence_rsi_pivot(
                close_slice, high_slice, low_slice, rsi_slice, ph_idx, pl_idx
            )
            
            # Detect sweeps
            sweeps = self.indicators.detect_liquidity_sweep(
                high_slice, low_slice, close_slice, ph_idx, pl_idx
            )
            
            # Generate signal
            timestamp = int(df['timestamp'].iloc[idx].timestamp() * 1000)
            
            signal = self.strategy.generate_signal(
                symbol, tf,
                close_slice, high_slice, low_slice, rsi_slice, atr_slice,
                ema50_slice, ema200_slice, adx_slice,
                bb_u_slice, bb_l_slice, bb_w_slice,
                divs, sweeps, timestamp
            )
            
            if signal:
                signals.append(signal)
                logger.info(f"Signal {idx}: {signal['direction']} @ {signal['score']:.0f}")
        
        logger.info(f"Generated {len(signals)} signals")
        
        # Run backtest
        logger.info("Running backtest...")
        stats = self.backtest.run_backtest(symbol, tf, df, signals)
        
        # Print results
        self._print_backtest_results(stats)
        
        return stats
    
    def backtest_walkforward(self, symbol: str, tf: str):
        """Run walk-forward backtest"""
        
        logger.info(f"Starting walk-forward: {symbol} {tf}")
        
        # Load CSV
        csv_path = Path(__file__).parent / 'data' / f'{symbol}_{tf}.csv'
        if not csv_path.exists():
            logger.error(f"Data file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['open_time'])
        
        # Create splits
        splits = self.walkforward.create_splits(df, symbol)
        
        all_results = {}
        
        for split in splits:
            logger.info(f"Processing {split['name']} ({split['stage']})...")
            
            split_df = split['df'].reset_index(drop=True)
            
            # Calculate indicators on split
            split_df['rsi'] = self.indicators.rsi_wilder(split_df['close'], 14)
            split_df['atr'] = self.indicators.atr_wilder(split_df['high'], split_df['low'], split_df['close'], 14)
            split_df['ema50'] = self.indicators.ema(split_df['close'], 50)
            split_df['ema200'] = self.indicators.ema(split_df['close'], 200)
            split_df['adx'] = self.indicators.adx(split_df['high'], split_df['low'], split_df['close'], 14)
            bb_u, bb_m, bb_l, bb_w = self.indicators.bollinger_bands(split_df['close'], 20, 2.0)
            split_df['bb_upper'] = bb_u
            split_df['bb_lower'] = bb_l
            split_df['bb_width'] = bb_w
            
            # Detect pivots
            ph_idx, pl_idx = self.indicators.detect_pivots(split_df['high'], split_df['low'])
            
            # Generate signals
            signals = []
            for idx in range(50, len(split_df)):
                close_slice = split_df['close'].iloc[:idx+1]
                high_slice = split_df['high'].iloc[:idx+1]
                low_slice = split_df['low'].iloc[:idx+1]
                rsi_slice = split_df['rsi'].iloc[:idx+1]
                atr_slice = split_df['atr'].iloc[:idx+1]
                ema50_slice = split_df['ema50'].iloc[:idx+1]
                ema200_slice = split_df['ema200'].iloc[:idx+1]
                adx_slice = split_df['adx'].iloc[:idx+1]
                bb_u_slice = split_df['bb_upper'].iloc[:idx+1]
                bb_l_slice = split_df['bb_lower'].iloc[:idx+1]
                bb_w_slice = split_df['bb_width'].iloc[:idx+1]
                
                divs = self.indicators.detect_divergence_rsi_pivot(
                    close_slice, high_slice, low_slice, rsi_slice, ph_idx, pl_idx
                )
                sweeps = self.indicators.detect_liquidity_sweep(
                    high_slice, low_slice, close_slice, ph_idx, pl_idx
                )
                
                timestamp = int(split_df['timestamp'].iloc[idx].timestamp() * 1000)
                signal = self.strategy.generate_signal(
                    symbol, tf,
                    close_slice, high_slice, low_slice, rsi_slice, atr_slice,
                    ema50_slice, ema200_slice, adx_slice,
                    bb_u_slice, bb_l_slice, bb_w_slice,
                    divs, sweeps, timestamp
                )
                
                if signal:
                    signals.append(signal)
            
            # Backtest on split
            stats = self.backtest.run_backtest(symbol, tf, split_df, signals)
            all_results[split['name']] = stats
        
        # Validate results
        validation = self.walkforward.validate_results({
            name: {
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor']
            }
            for name, stats in all_results.items()
        })
        
        self.walkforward.print_validation_report(validation)
        
        return all_results, validation
    
    def _print_backtest_results(self, stats: dict):
        """Pretty print backtest results"""
        
        result = f"""
╔════════════════════════════════════════════════════════════════════╗
║ BACKTEST RESULTS - {stats['symbol']} {stats['timeframe']}
╠════════════════════════════════════════════════════════════════════╣
║ Total Trades:       {stats['total_trades']:<50} ║
║ Win Rate:           {stats['win_rate']:>6.2f}%                           ║
║ Profit Factor:      {stats['profit_factor']:>6.2f}                           ║
║ Total PnL:          ${stats['total_pnl']:>10.2f}                        ║
║ Max Drawdown:       ${stats['max_drawdown']:>10.2f}                      ║
║ Expectancy:         ${stats['expectancy']:>10.2f}                       ║
╚════════════════════════════════════════════════════════════════════╝
        """
        print(result)


def main():
    parser = argparse.ArgumentParser(description="V4 Pro Max Ultra Engine")
    subparsers = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest')
    backtest_parser.add_argument('--symbol', default='PAXGUSDT')
    backtest_parser.add_argument('--tf', default='15m')
    backtest_parser.add_argument('--walkforward', action='store_true')
    
    # Live command
    live_parser = subparsers.add_parser('live')
    live_parser.add_argument('--symbol', default='PAXGUSDT')
    
    # Download command
    download_parser = subparsers.add_parser('download')
    
    args = parser.parse_args()
    
    engine = QuantMachineV4()
    
    if args.command == 'backtest':
        if args.walkforward:
            engine.backtest_walkforward(args.symbol, args.tf)
        else:
            engine.backtest_symbol(args.symbol, args.tf)
    
    elif args.command == 'live':
        logger.info(f"Live mode: {args.symbol}")
        # TODO: Implement live mode
        logger.info("Live mode not yet implemented")
    
    elif args.command == 'download':
        from download_history import download_all_symbols
        download_all_symbols(
            ['PAXGUSDT', 'BTCUSDT'],
            ['5m', '15m', '30m', '1h'],
            start_date='2021-01-01'
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
