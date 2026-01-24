#!/bin/bash
#
# AEGIS V21 - Daemon Control Script
# ==================================
# Manages 24/7 trading bot operation
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEGIS_HOME="$SCRIPT_DIR"
PID_FILE="$AEGIS_HOME/aegis.pid"
LOG_FILE="$AEGIS_HOME/logs/aegis.log"
ERROR_LOG="$AEGIS_HOME/logs/aegis_error.log"
PLIST_NAME="com.aegis.trading"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"

# Ensure log directory exists
mkdir -p "$AEGIS_HOME/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}          ${GREEN}AEGIS V21 - DAEMON CONTROLLER${NC}                  ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check if running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Start the bot
start() {
    print_banner
    
    if is_running; then
        echo -e "${YELLOW}⚠️  AEGIS is already running (PID: $(cat $PID_FILE))${NC}"
        return 1
    fi
    
    echo -e "${BLUE}🚀 Starting AEGIS Trading Bot...${NC}"
    
    # Check model exists
    if [ ! -f "$AEGIS_HOME/models/aegis_lstm.pth" ]; then
        echo -e "${YELLOW}⚠️  Warning: LSTM model not found at $AEGIS_HOME/models/aegis_lstm.pth${NC}"
    fi
    
    # Use specific python path that has dependencies
    PYTHON_EXEC="/Applications/anaconda3/bin/python3"
    
    # Start API Server
    echo -e "   Starting API Server..."
    cd "$AEGIS_HOME"
    nohup $PYTHON_EXEC src/api_server.py >> "$AEGIS_HOME/logs/api_server.log" 2>&1 &
    API_PID=$!
    
    # Start Executor
    echo -e "   Starting Executor..."
    nohup $PYTHON_EXEC src/executor.py >> "$LOG_FILE" 2>> "$ERROR_LOG" &
    EXEC_PID=$!
    
    # Start Dashboard
    echo -e "   Starting Dashboard (Terminal I)..."
    cd "$AEGIS_HOME/aegis-ui"
    nohup npm run dev >> "$AEGIS_HOME/logs/nextjs.log" 2>&1 &
    DASH_PID=$!
    cd "$AEGIS_HOME"
    
    # Save PIDs (Executor is main PID)
    echo $EXEC_PID > "$PID_FILE"
    echo $API_PID > "$AEGIS_HOME/api.pid"
    echo $DASH_PID > "$AEGIS_HOME/dashboard.pid"
    
    sleep 2
    
    if is_running; then
        echo -e "${GREEN}✅ AEGIS started successfully${NC}"
        echo -e "   Executor PID: $EXEC_PID"
        echo -e "   API PID:      $API_PID"
        echo -e "   Logs: $LOG_FILE"
    else
        echo -e "${RED}❌ Failed to start AEGIS${NC}"
        echo -e "   Check: $ERROR_LOG"
        return 1
    fi
}

# Stop the bot
stop() {
    print_banner
    
    if ! is_running; then
        echo -e "${YELLOW}⚠️  AEGIS is not running${NC}"
        rm -f "$PID_FILE"
        rm -f "$AEGIS_HOME/api.pid"
        return 0
    fi
    
    PID=$(cat "$PID_FILE")
    echo -e "${BLUE}🛑 Stopping AEGIS (PID: $PID)...${NC}"
    
    kill "$PID" 2>/dev/null
    
    # Kill API server if exists
    if [ -f "$AEGIS_HOME/api.pid" ]; then
        API_PID=$(cat "$AEGIS_HOME/api.pid")
        kill "$API_PID" 2>/dev/null
        rm -f "$AEGIS_HOME/api.pid"
    fi
    
    # Kill Dashboard if exists
    if [ -f "$AEGIS_HOME/dashboard.pid" ]; then
        DASH_PID=$(cat "$AEGIS_HOME/dashboard.pid")
        kill "$DASH_PID" 2>/dev/null
        rm -f "$AEGIS_HOME/dashboard.pid"
    fi
    
    # Force kill any rogue dashboard processes
    pkill -f "next-server"
    pkill -f "next dev"
    # Force kill any rogue api_server processes
    pkill -f "src/api_server.py"
    
    # Wait for graceful shutdown
    for i in {1..5}; do
        if ! is_running; then
            break
        fi
        sleep 1
    done
    
    # Force kill if needed
    if is_running; then
        kill -9 "$PID" 2>/dev/null
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ AEGIS stopped${NC}"
}

# Restart
restart() {
    stop
    sleep 2
    start
}

# Check status
status() {
    print_banner
    
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}✅ AEGIS is RUNNING${NC}"
        echo -e "   PID: $PID"
        echo -e "   Uptime: $(ps -o etime= -p $PID 2>/dev/null || echo 'unknown')"
        
        # Show last log entries
        echo ""
        echo -e "${BLUE}📋 Recent Activity:${NC}"
        tail -5 "$LOG_FILE" 2>/dev/null || echo "   No logs available"
    else
        echo -e "${RED}❌ AEGIS is STOPPED${NC}"
    fi
}

# View logs
logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No logs found at $LOG_FILE"
    fi
}

# View error logs
errors() {
    if [ -f "$ERROR_LOG" ]; then
        tail -50 "$ERROR_LOG"
    else
        echo "No error logs found"
    fi
}

# Install as LaunchAgent (macOS)
install() {
    print_banner
    echo -e "${BLUE}📦 Installing AEGIS as LaunchAgent...${NC}"
    
    # Create LaunchAgent plist
    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$AEGIS_HOME/src/executor.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$AEGIS_HOME</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$ERROR_LOG</string>
</dict>
</plist>
EOF
    
    # Load the agent
    launchctl load "$PLIST_PATH" 2>/dev/null
    
    echo -e "${GREEN}✅ Installed as LaunchAgent${NC}"
    echo -e "   Plist: $PLIST_PATH"
    echo -e "   AEGIS will start automatically on login"
}

# Uninstall LaunchAgent
uninstall() {
    print_banner
    echo -e "${BLUE}🗑️  Uninstalling AEGIS LaunchAgent...${NC}"
    
    launchctl unload "$PLIST_PATH" 2>/dev/null
    rm -f "$PLIST_PATH"
    
    stop
    
    echo -e "${GREEN}✅ LaunchAgent uninstalled${NC}"
}

# Health check
health() {
    print_banner
    echo -e "${BLUE}🏥 AEGIS Health Check${NC}"
    echo ""
    
    # Check Python
    echo -n "   Python 3.8+: "
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌${NC}"
    fi
    
    # Check PyTorch
    echo -n "   PyTorch: "
    if python3 -c "import torch" 2>/dev/null; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌ (Run: pip install torch)${NC}"
    fi
    
    # Check trained model
    echo -n "   Trained Model: "
    if [ -f "$AEGIS_HOME/models/aegis_lstm.pth" ]; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌ (Run: python3 src/lstm_trainer.py)${NC}"
    fi
    
    # Check running
    echo -n "   Daemon Status: "
    if is_running; then
        echo -e "${GREEN}✅ Running${NC}"
    else
        echo -e "${YELLOW}⚠️ Stopped${NC}"
    fi
    
    echo ""
}

# Show usage
usage() {
    print_banner
    echo "Usage: $0 {command}"
    echo ""
    echo "Commands:"
    echo "   start      Start the trading bot"
    echo "   stop       Stop the trading bot"
    echo "   restart    Restart the trading bot"
    echo "   status     Check if bot is running"
    echo "   logs       View live logs"
    echo "   errors     View error logs"
    echo "   health     Run health check"
    echo "   install    Install as macOS LaunchAgent (24/7)"
    echo "   uninstall  Remove LaunchAgent"
    echo ""
}

# Main
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    errors)
        errors
        ;;
    health)
        health
        ;;
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    *)
        usage
        exit 1
        ;;
esac
