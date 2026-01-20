"""
Alerts Module - Telegram + Local logging
"""

import logging
import json
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage signal alerts via Telegram + local logs"""
    
    def __init__(self, config: Dict):
        self.config = config['alerts']
        self.telegram_enabled = self.config.get('telegram', {}).get('enabled', False)
        self.local_log = self.config.get('local_log', True)
        self.log_file = self.config.get('log_file', './logs/signals.log')
        
        # Setup local logging
        if self.local_log:
            self._setup_local_logger()
        
        # Setup Telegram if enabled
        if self.telegram_enabled:
            self._setup_telegram()
    
    def _setup_local_logger(self):
        """Setup file logger"""
        from pathlib import Path
        Path(self.log_file).parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - SIGNAL - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def _setup_telegram(self):
        """Setup Telegram bot"""
        try:
            from telegram import Bot
            token = self.config['telegram'].get('token')
            if not token:
                logger.warning("Telegram token not configured")
                self.telegram_enabled = False
            else:
                self.telegram_bot = Bot(token=token)
        except ImportError:
            logger.warning("python-telegram-bot not installed")
            self.telegram_enabled = False
    
    def send_trade_alert(self, trade_card: Dict):
        """Send trade card alert"""
        
        message = self._format_alert(trade_card)
        
        # Local log
        if self.local_log:
            logger.info(message)
        
        # Telegram
        if self.telegram_enabled:
            self._send_telegram(message, trade_card)
    
    def _format_alert(self, trade_card: Dict) -> str:
        """Format alert message"""
        
        msg = f"""
TRADE SIGNAL - {trade_card['symbol']} {trade_card['timeframe']}
Direction: {trade_card['direction']}
Entry Zone: {trade_card['entry_zone'][0]:.2f} - {trade_card['entry_zone'][1]:.2f}
TP1: {trade_card['tp1']:.2f}
TP2: {trade_card['tp2']:.2f}
SL: {trade_card['sl']:.2f}
Score: {trade_card['score']:.0f}/100
Regime: {trade_card['regime']}
Reason: {trade_card['reasons']}
"""
        return msg
    
    def _send_telegram(self, message: str, trade_card: Dict):
        """Send via Telegram"""
        try:
            chat_id = self.config['telegram'].get('chat_id')
            if not chat_id:
                logger.warning("Telegram chat_id not configured")
                return
            
            self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message
            )
            logger.info("Telegram alert sent")
        
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
    
    def log_candle(self, symbol: str, tf: str, ohlcv: Dict):
        """Log candle data"""
        logger.debug(f"{symbol} {tf}: O={ohlcv['open']:.2f} H={ohlcv['high']:.2f} L={ohlcv['low']:.2f} C={ohlcv['close']:.2f}")
    
    def log_error(self, error: str):
        """Log error"""
        logger.error(error)
