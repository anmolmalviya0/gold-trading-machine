"""
Download historical data from Binance
"""

import pandas as pd
import logging
from binance.client import Client
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def download_klines(symbol: str, interval: str, start_date: str = "2021-01-01", 
                   end_date: str = None, save_dir: str = "./data") -> pd.DataFrame:
    """
    Download historical klines from Binance.
    
    Args:
        symbol: 'PAXGUSDT', 'BTCUSDT', etc.
        interval: '5m', '15m', '30m', '1h'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string, default today
        save_dir: Where to save CSV
    
    Returns:
        DataFrame with OHLCV
    """
    
    if end_date is None:
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
    
    client = Client()
    
    # Convert interval
    interval_map = {
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
    }
    
    bnc_interval = interval_map.get(interval)
    if not bnc_interval:
        raise ValueError(f"Unknown interval: {interval}")
    
    logger.info(f"Downloading {symbol} {interval} from {start_date} to {end_date}")
    
    klines = client.get_historical_klines(
        symbol,
        bnc_interval,
        start_str=start_date,
        end_str=end_date
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df['timestamp'] = df['open_time']
    
    # Save CSV
    Path(save_dir).mkdir(exist_ok=True)
    csv_path = Path(save_dir) / f"{symbol}_{interval}.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved {len(df)} candles to {csv_path}")
    
    return df


def download_all_symbols(symbols: list, intervals: list, 
                        start_date: str = "2021-01-01",
                        save_dir: str = "./data"):
    """Download all symbol/interval combinations"""
    
    for symbol in symbols:
        for interval in intervals:
            try:
                download_klines(symbol, interval, start_date, save_dir=save_dir)
            except Exception as e:
                logger.error(f"Failed to download {symbol} {interval}: {e}")


if __name__ == "__main__":
    # Download PAXG data from 2021
    download_all_symbols(
        symbols=['PAXGUSDT', 'BTCUSDT'],
        intervals=['5m', '15m', '30m', '1h'],
        start_date="2021-01-01"
    )
