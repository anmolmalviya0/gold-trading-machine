#!/bin/bash
# 🐉 TERMINAL: ETERNAL GUARD (V19)
# ==============================
# Fusing Caffeinate & Process Monitoring for 24/7 Sovereign Uptime.
# [NASA-GRADE STATION HARDENING]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEGIS_HOME="$SCRIPT_DIR"
LOG_FILE="$AEGIS_HOME/logs/eternal_guard.log"
DAEMON_SH="$AEGIS_HOME/daemon.sh"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$AEGIS_HOME/logs"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}" >> "$LOG_FILE"
echo -e "${BLUE}║${NC}          ${GREEN}TERMINAL - ETERNAL GUARD V19${NC}                  ${BLUE}║${NC}" >> "$LOG_FILE"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}" >> "$LOG_FILE"

# 1. OPTIMIZATION: Start Caffeinate (Option 1) in background
echo -e "$(date) | ⚡ IGNITING CAFFEINATE VISOR..." >> "$LOG_FILE"
nohup caffeinate -dis > /dev/null 2>&1 & 
CAFF_PID=$!
echo -e "$(date) | ✅ CAFFEINATE ACTIVE [PID: $CAFF_PID]" >> "$LOG_FILE"

# SELF-CLEANUP: Kill caffeinate when guard is stopped
trap "echo -e '$(date) | 🛑 SHUTTING DOWN GUARD. Neutralizing caffeinate...' >> '$LOG_FILE'; kill $CAFF_PID 2>/dev/null; exit" SIGINT SIGTERM

# 2. DEBUG & MONITOR LOOP
# User requested "Return it for 50-100 seconds" - we will run checks in this cadence.
CHECK_INTERVAL=60 # Seconds

monitor_cycle() {
    while true; do
        echo -e "$(date) | 🔍 FORENSIC HEALTH INTERROGATION..." >> "$LOG_FILE"
        
        # Check if Daemon is running
        if "$DAEMON_SH" status | grep -q "RUNNING"; then
            echo -e "$(date) | 🟢 STATION STABLE." >> "$LOG_FILE"
        else
            echo -e "$(date) | ⚠️ STATION FRACTURE DETECTED. RE-IGNITING..." >> "$LOG_FILE"
            "$DAEMON_SH" restart >> "$LOG_FILE" 2>&1
        fi
        
        # Check Caffeinate
        if ! ps -p "$CAFF_PID" > /dev/null; then
            echo -e "$(date) | ⚠️ CAFFEINATE CRASHED. RESTORING..." >> "$LOG_FILE"
            nohup caffeinate -dis &
            CAFF_PID=$!
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# 3. EXECUTION
monitor_cycle
