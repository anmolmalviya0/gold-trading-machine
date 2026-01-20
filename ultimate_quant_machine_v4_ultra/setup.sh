#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ V4 PRO MAX ULTRA SETUP                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Create directories
mkdir -p data logs models

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download data (optional)
echo ""
echo "Ready to run:"
echo ""
echo "  # Download historical data"
echo "  python main.py download"
echo ""
echo "  # Backtest single year"
echo "  python main.py backtest --symbol PAXGUSDT --tf 15m"
echo ""
echo "  # Walk-forward validation"
echo "  python main.py backtest --symbol PAXGUSDT --tf 15m --walkforward"
echo ""
echo "  # Live mode (paper trading)"
echo "  python main.py live --symbol PAXGUSDT"
echo ""

echo "Setup complete!"
