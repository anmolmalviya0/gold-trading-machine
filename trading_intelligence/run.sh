#!/bin/bash
# Run the trading system

set -e

cd "$(dirname "$0")"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for required files
if [ ! -d "models/ensemble" ]; then
    echo "❌ Models not found. Run training first:"
    echo "   python train_ensemble.py"
    exit 1
fi

# Start the terminal
echo "🚀 Starting Institutional Trading Terminal..."
python live_terminal.py