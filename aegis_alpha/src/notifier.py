import os
import requests
import logging

logger = logging.getLogger('SENTINEL_NOTIFIER')

class NotificationSentinel:
    """
    TERMINAL - OMNI-CHANNEL SENTINEL
    ================================
    Bridges the machine's intent to the Commander's physical devices.
    Supports: macOS Native, iOS (via ntfy), and System Logs.
    """
    
    def __init__(self, ntfy_topic="terminal_alpha_pulse"):
        self.topic = ntfy_topic
        self.base_url = f"https://ntfy.sh/{self.topic}"
        logger.info(f"🛰️ Sentinel initialized on topic: {self.topic}")

    def notify(self, title, message, tags=None, priority=3):
        """
        Broadcast alert to all registered nodes.
        priority: 1 (min) to 5 (max)
        """
        # 1. macOS Alert (Only if running on local Mac)
        try:
            os.system(f"osascript -e 'display notification \"{message}\" with title \"🐉 {title}\" sound name \"Hero\"'")
        except:
            pass
            
        # 2. iPhone/Mobile Alert (ntfy.sh)
        try:
            # We add a slight delay to prevent rate limits during bursts
            payload = message.encode("utf-8")
            headers = {
                "Title": f"TERMINAL: {title}",
                "Priority": str(priority),
                "Tags": tags if tags else "zap,chart_with_upwards_trend"
            }
            res = requests.post(self.base_url, data=payload, headers=headers, timeout=5)
            if res.status_code == 200:
                logger.info(f"📡 Pulse broadcasted to {self.topic}")
        except Exception as e:
            logger.error(f"⚠️ Pulse failed: {e}")

    def alert_signal(self, symbol, signal, price):
        """High-Priority Signal Alert"""
        emoji = "🚀" if signal == "BUY" else "🔻"
        tags = "moneybag,rocket" if signal == "BUY" else "warning,small_red_triangle_down"
        self.notify(
            title=f"SIGNAL: {symbol}",
            message=f"{emoji} {signal} @ {price}",
            tags=tags,
            priority=5
        )

    def alert_barrier(self, symbol, reason, price):
        """Exit Barrier Alert (SL/TP)"""
        self.notify(
            title=f"BARRIER: {symbol}",
            message=f"🏁 {reason} @ {price}",
            tags="checkered_flag,shield",
            priority=4
        )

# Singleton Instance
sentinel = NotificationSentinel()
