"""
PHASE 4/5: MONITORING & OPS
===========================
Production monitoring, healthchecks, and kill switch.

Features:
- Healthcheck endpoint
- Watchdog process
- Kill switch
- Telegram alerts
- Performance dashboard

Usage:
    python ops_monitor.py
"""
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import aiohttp
from typing import Optional

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'paper_trading.db'
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

OPS_CONFIG = {
    # Telegram (set via environment)
    'telegram_token': os.environ.get('TELEGRAM_TOKEN'),
    'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID'),
    
    # Thresholds
    'max_daily_loss': 100,
    'max_drawdown': 200,
    'heartbeat_interval': 60,  # seconds
    
    # Watchdog
    'process_names': ['live_terminal', 'paper_trading'],
}


# === TELEGRAM ALERTS ===

class TelegramAlerter:
    """Send alerts via Telegram"""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or OPS_CONFIG['telegram_token']
        self.chat_id = chat_id or OPS_CONFIG['telegram_chat_id']
        self.enabled = bool(self.token and self.chat_id)
        
        if not self.enabled:
            print("⚠️ Telegram not configured. Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
    
    async def send(self, message: str, parse_mode: str = 'HTML'):
        """Send message to Telegram"""
        if not self.enabled:
            print(f"[ALERT] {message}")
            return False
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False
    
    async def send_signal(self, symbol: str, side: str, entry: float, 
                          sl: float, tp: float, confidence: float):
        """Send signal alert"""
        emoji = "🟢" if side == "BUY" else "🔴"
        message = f"""
{emoji} <b>SIGNAL: {side} {symbol}</b>

💰 Entry: ${entry:.2f}
🛑 Stop Loss: ${sl:.2f}
🎯 Take Profit: ${tp:.2f}
📊 Confidence: {confidence:.1f}%

<i>Generated at {datetime.now().strftime('%H:%M:%S')}</i>
"""
        await self.send(message)
    
    async def send_trade_closed(self, symbol: str, side: str, 
                                pnl: float, exit_reason: str):
        """Send trade closed alert"""
        emoji = "✅" if pnl > 0 else "❌"
        message = f"""
{emoji} <b>CLOSED: {side} {symbol}</b>

Exit: {exit_reason}
PnL: ${pnl:.2f}

<i>Closed at {datetime.now().strftime('%H:%M:%S')}</i>
"""
        await self.send(message)
    
    async def send_circuit_breaker(self, reason: str):
        """Send circuit breaker alert"""
        message = f"""
🚨 <b>CIRCUIT BREAKER ACTIVATED</b>

Reason: {reason}

⛔ All trading halted until manual reset.

<i>Triggered at {datetime.now().strftime('%H:%M:%S')}</i>
"""
        await self.send(message)


# === HEALTH CHECKS ===

class HealthChecker:
    """Monitor system health"""
    
    def __init__(self):
        self.last_heartbeat = datetime.now()
        self.alerts = TelegramAlerter()
    
    def check_database(self) -> bool:
        """Check database connectivity"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except:
            return False
    
    def check_processes(self) -> dict:
        """Check if required processes are running"""
        import subprocess
        results = {}
        
        for proc_name in OPS_CONFIG['process_names']:
            result = subprocess.run(
                ['pgrep', '-f', proc_name], 
                capture_output=True
            )
            results[proc_name] = result.returncode == 0
        
        return results
    
    def check_drawdown(self) -> tuple:
        """Check current drawdown"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get cumulative PnL
        cursor.execute("""
            SELECT SUM(pnl) FROM paper_trades 
            WHERE status = 'CLOSED' AND date(exit_time) = date('now')
        """)
        daily_pnl = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(pnl) FROM paper_trades WHERE status = 'CLOSED'")
        total_pnl = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return daily_pnl, total_pnl
    
    async def run_healthcheck(self) -> dict:
        """Run all health checks"""
        daily_pnl, total_pnl = self.check_drawdown()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'database': self.check_database(),
            'processes': self.check_processes(),
            'daily_pnl': daily_pnl,
            'total_pnl': total_pnl,
            'circuit_breaker': daily_pnl < -OPS_CONFIG['max_daily_loss']
        }
        
        # Alert on circuit breaker
        if result['circuit_breaker']:
            await self.alerts.send_circuit_breaker(
                f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${OPS_CONFIG['max_daily_loss']}"
            )
        
        # Alert on process down
        for proc, running in result['processes'].items():
            if not running:
                await self.alerts.send(f"⚠️ Process <b>{proc}</b> is not running!")
        
        return result
    
    def write_heartbeat(self):
        """Write heartbeat to disk"""
        heartbeat_file = LOG_DIR / 'heartbeat.json'
        data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'alive'
        }
        with open(heartbeat_file, 'w') as f:
            json.dump(data, f)
        
        self.last_heartbeat = datetime.now()


# === KILL SWITCH ===

class KillSwitch:
    """Emergency kill switch for all trading"""
    
    def __init__(self):
        self.switch_file = LOG_DIR / 'kill_switch.json'
        self.active = self._read_state()
    
    def _read_state(self) -> bool:
        """Read kill switch state from disk"""
        if self.switch_file.exists():
            try:
                with open(self.switch_file) as f:
                    data = json.load(f)
                    return data.get('active', False)
            except:
                pass
        return False
    
    def activate(self, reason: str = "Manual"):
        """Activate kill switch"""
        self.active = True
        data = {
            'active': True,
            'reason': reason,
            'activated_at': datetime.now().isoformat()
        }
        with open(self.switch_file, 'w') as f:
            json.dump(data, f)
        print(f"🛑 KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate(self):
        """Deactivate kill switch"""
        self.active = False
        if self.switch_file.exists():
            self.switch_file.unlink()
        print("✅ Kill switch deactivated")
    
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self._read_state()


# === WATCHDOG ===

async def watchdog_loop(interval: int = 60):
    """Main watchdog loop"""
    print("🐕 Watchdog started...")
    
    health = HealthChecker()
    kill_switch = KillSwitch()
    
    while True:
        try:
            # Check kill switch
            if kill_switch.is_active():
                print("   ⛔ Kill switch is active")
            
            # Run health checks
            result = await health.run_healthcheck()
            
            # Write heartbeat
            health.write_heartbeat()
            
            # Log status
            print(f"   ❤️ Heartbeat | Daily PnL: ${result['daily_pnl']:.2f} | " +
                  f"DB: {'✅' if result['database'] else '❌'}")
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            print(f"   ❌ Watchdog error: {e}")
            await asyncio.sleep(interval)


# === CLI COMMANDS ===

def cmd_status():
    """Print current status"""
    health = HealthChecker()
    kill_switch = KillSwitch()
    
    print("\n" + "="*50)
    print("📊 SYSTEM STATUS")
    print("="*50)
    
    # Database
    print(f"   Database: {'✅ OK' if health.check_database() else '❌ ERROR'}")
    
    # Processes
    procs = health.check_processes()
    for proc, running in procs.items():
        print(f"   {proc}: {'✅ Running' if running else '⏹️ Stopped'}")
    
    # Kill switch
    print(f"   Kill Switch: {'⛔ ACTIVE' if kill_switch.is_active() else '✅ Off'}")
    
    # PnL
    daily, total = health.check_drawdown()
    print(f"   Daily PnL: ${daily:.2f}")
    print(f"   Total PnL: ${total:.2f}")
    
    print("="*50)


def cmd_kill(reason: str = "Manual kill"):
    """Activate kill switch"""
    ks = KillSwitch()
    ks.activate(reason)


def cmd_resume():
    """Deactivate kill switch"""
    ks = KillSwitch()
    ks.deactivate()


# === MAIN ===

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("🔧 PHASE 4/5: MONITORING & OPS")
    print("="*70)
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'status':
            cmd_status()
        elif cmd == 'kill':
            reason = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else "Manual"
            cmd_kill(reason)
        elif cmd == 'resume':
            cmd_resume()
        elif cmd == 'watch':
            asyncio.run(watchdog_loop())
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python ops_monitor.py [status|kill|resume|watch]")
    else:
        # Default: show status
        cmd_status()
        
        print("\n📖 Available commands:")
        print("   python ops_monitor.py status  - Show system status")
        print("   python ops_monitor.py kill    - Activate kill switch")
        print("   python ops_monitor.py resume  - Deactivate kill switch")
        print("   python ops_monitor.py watch   - Start watchdog loop")
