#!/bin/bash
echo "🚀 INITIALIZING INSTITUTIONAL TRADING SUITE..."

# Kill old processes to ensure clean state
echo "🧹 Cleaning up old processes..."
pkill -f live_terminal.py
pkill -f tradingview_webhook.py
pkill -f strategy_marketplace.py

# Start services
echo "Starting Terminal (Port 8000)..."
python3 live_terminal.py > terminal.log 2>&1 &
echo $! > terminal.pid

echo "Starting Webhook Server (Port 8001)..."
python3 tradingview_webhook.py > webhook.log 2>&1 &
echo $! > webhook.pid

echo "Starting Strategy Marketplace (Port 8002)..."
python3 strategy_marketplace.py > marketplace.log 2>&1 &
echo $! > marketplace.pid

# Wait a moment for startup
sleep 2

echo "✅ ALL SYSTEMS ONLINE"
echo "   📊 Terminal:    http://localhost:8000"
echo "   🔗 Webhooks:    http://localhost:8001"
echo "   🛒 Marketplace: http://localhost:8002"
echo ""
echo "Monitor logs with: tail -f terminal.log"
