"""
TELEGRAM ALERTS MODULE
Send high-confidence trading signals to Telegram
"""
import asyncio
import aiohttp
from datetime import datetime

class TelegramAlerts:
    """Send trading alerts to Telegram"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram alerts.
        
        To get your bot token and chat ID:
        1. Message @BotFather on Telegram to create a bot
        2. Copy the token (e.g., 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
        3. Start a chat with your bot
        4. Get your chat_id from https://api.telegram.org/bot<TOKEN>/getUpdates
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self.last_alert = {}  # Track last alert per symbol to avoid spam
        self.min_interval = 300  # Minimum 5 minutes between alerts per symbol
    
    async def send_message(self, text: str) -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            print(f"📱 [Telegram disabled] {text[:50]}...")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        print(f"✅ Telegram alert sent")
                        return True
                    else:
                        print(f"❌ Telegram error: {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    async def send_signal_alert(self, symbol: str, signal: str, confidence: float,
                                 entry: float, sl: float, tp: float, reason: str = ""):
        """
        Send a formatted trading signal alert.
        Only sends for high-confidence signals (>= 55%).
        Respects rate limiting to avoid spam.
        """
        # Only send for actionable signals
        if signal not in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']:
            return False
        
        # Only high confidence
        if confidence < 55:
            return False
        
        # Rate limiting
        now = datetime.now().timestamp()
        last = self.last_alert.get(symbol, 0)
        if now - last < self.min_interval:
            return False
        
        self.last_alert[symbol] = now
        
        # Format message
        emoji = "🟢" if "BUY" in signal else "🔴"
        direction = "⬆️ LONG" if "BUY" in signal else "⬇️ SHORT"
        
        message = f"""
{emoji} *{signal}* - {symbol}

{direction} Signal
━━━━━━━━━━━━━━━━
📊 *Confidence:* {confidence:.0f}%
💰 *Entry:* ${entry:,.2f}
🛑 *Stop Loss:* ${sl:,.2f}
🎯 *Take Profit:* ${tp:,.2f}
━━━━━━━━━━━━━━━━
📝 {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        return await self.send_message(message)
    
    async def send_trade_closed(self, symbol: str, side: str, pnl: float, 
                                  reason: str, win: bool):
        """Send alert when a trade closes"""
        emoji = "✅" if win else "❌"
        pnl_emoji = "💰" if pnl > 0 else "💸"
        
        message = f"""
{emoji} *Trade Closed* - {symbol}

{pnl_emoji} PnL: ${pnl:+.2f}
📍 Side: {side}
📋 Reason: {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        return await self.send_message(message)


# Configuration instructions
CONFIG_INSTRUCTIONS = """
=== TELEGRAM SETUP INSTRUCTIONS ===

1. Open Telegram and search for @BotFather
2. Send /newbot and follow the prompts
3. Copy the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
4. Start a chat with your new bot (click the link BotFather gives you)
5. Send any message to the bot
6. Get your chat_id by visiting:
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   Look for "chat":{"id":<YOUR_CHAT_ID>}

7. Add to your config:
   TELEGRAM_BOT_TOKEN = "your_token_here"
   TELEGRAM_CHAT_ID = "your_chat_id_here"

=====================================
"""


# Test function
async def test_telegram(bot_token: str, chat_id: str):
    """Test Telegram connection"""
    alerts = TelegramAlerts(bot_token, chat_id)
    
    # Test basic message
    success = await alerts.send_message("🔔 *Test Alert*\n\nTelegram alerts are working!")
    
    if success:
        print("\n✅ Telegram test successful!")
        
        # Test signal alert
        await alerts.send_signal_alert(
            symbol="BTCUSDT",
            signal="STRONG BUY",
            confidence=65.5,
            entry=95000.00,
            sl=93500.00,
            tp=97500.00,
            reason="ML: 65% | Model: 88%"
        )
    else:
        print("\n❌ Telegram test failed. Check your token and chat_id.")
        print(CONFIG_INSTRUCTIONS)


if __name__ == "__main__":
    print("="*50)
    print("🔔 TELEGRAM ALERTS MODULE")
    print("="*50)
    print(CONFIG_INSTRUCTIONS)
    
    # Example usage (replace with your credentials)
    # asyncio.run(test_telegram("YOUR_BOT_TOKEN", "YOUR_CHAT_ID"))
