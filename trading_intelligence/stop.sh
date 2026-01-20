#!/bin/bash
# Stop all trading processes

echo "🛑 Stopping all trading processes..."

# Kill terminal
pkill -f "live_terminal" 2>/dev/null || true

# Kill watchdog
pkill -f "ops_monitor" 2>/dev/null || true

echo "✅ All processes stopped"