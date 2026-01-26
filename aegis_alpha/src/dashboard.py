"""
TERMINAL - COMMAND CENTER (DASHBOARD)
=======================================
Real-time terminal dashboard for monitoring the bot.
Run with: python3 src/dashboard.py
"""
import os
import time
import sys
import subprocess
from datetime import datetime

def get_process_status():
    try:
        # Check if process is running
        result = subprocess.run(['pgrep', '-f', 'executor.py'], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p]
        
        if pids:
            return "🟢 ONLINE", pids[0]
        else:
            return "🔴 OFFLINE", "N/A"
    except:
        return "⚪ UNKNOWN", "N/A"

def read_last_logs(n=10):
    log_path = 'logs/executor.log'
    if not os.path.exists(log_path):
        return ["Waiting for logs..."]
    
    try:
        # Use tail command for efficiency
        result = subprocess.run(['tail', '-n', str(n), log_path], capture_output=True, text=True)
        return result.stdout.strip().split('\n')
    except:
        return ["Error reading logs"]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Configuration (Static for now, matches executor)
    STRATEGY = "LightGBM Switchblade"
    CONFIDENCE = 0.65
    RISK_LIMIT = "3.0%"
    
    print("Launching TERMINAL Command Center...")
    time.sleep(1)
    
    while True:
        try:
            status, pid = get_process_status()
            logs = read_last_logs(15)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            clear_screen()
            print(f"╔══════════════════════════════════════════════════════════════════════════╗")
            print(f"║                   🦅 TERMINAL COMMAND CENTER                            ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════╣")
            print(f"║  TIME:   {now:<54}  ║")
            print(f"║  STATUS: {status:<10} (PID: {pid:<7})                                 ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════╣")
            print(f"║  🧠 BRAIN:   {STRATEGY:<48}  ║")
            print(f"║  🎯 TARGET:  Conf > {CONFIDENCE}                                             ║")
            print(f"║  🛡️ RISK:    Max {RISK_LIMIT} / Trade                                       ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════╣")
            print(f"║  📡 LIVE LOGS                                                            ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════╣")
            
            for log in logs:
                # Truncate log if too long to avoid wrapping mess
                clean_log = log[:74] 
                print(f"║  {clean_log:<74}  ║")
                
            print(f"╚══════════════════════════════════════════════════════════════════════════╝")
            print(f" Press Ctrl+C to Exit Dashboard (Bot continues running)")
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nDashboard closed.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    # Ensure we are in the right directory or can find files
    if not os.path.exists('src'):
        # Try to find the project root if run from src
        if os.path.exists('../src'):
            os.chdir('..')
            
    main()
