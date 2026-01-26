#!/bin/bash

# ============================================================
# TERMINAL - PROTOCOL OMEGA CLOUD INSTALLER
# ============================================================
# TARGET: Ubuntu 22.04/24.04 LTS
# ROLE: Stratospheric Migration

set -e # Exit on error

echo "🐉 INITIATING TERMINAL CLOUD SETUP..."

# 1. SYSTEM UPDATE
echo "🔄 Updating Apt Repositories..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git htop screen curl sqlite3

# 2. DIRECTORY ARCHITECTURE
echo "📂 Creating Fortress Directory..."
mkdir -p ~/gold/aegis_alpha/logs
mkdir -p ~/gold/aegis_alpha/data
mkdir -p ~/gold/aegis_alpha/models

cd ~/gold/aegis_alpha

# 3. PYTHON SANDBOX
echo "🐍 Initializing Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

# 4. NEURO-LINK (DEPENDENCIES)
echo "📦 Installing Intelligence Stack..."
pip install --upgrade pip
pip install numpy pandas lightgbm scikit-learn
pip install ccxt websocket-client pyyaml requests schedule joblib
pip install fastapi uvicorn httpx python-multipart # API Stack

# 5. NODE.JS (FOR DASHBOARD)
echo "🌐 Installing Node.js for TERMINAL Display..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 6. PERMISSIONS
echo "⚖️ Setting Execution Rights..."
# We assume the user will scp daemon.sh shortly
touch daemon.sh
chmod +x daemon.sh

echo ""
echo "✅ CLOUD RECEPTACLE READY."
echo "------------------------------------------------------------"
echo "NEXT STEPS (Execute on your LOCAL MacBook):"
echo "1. scp -r src/ ubuntu@<SERVER_IP>:~/gold/aegis_alpha/"
echo "2. scp -r models/ ubuntu@<SERVER_IP>:~/gold/aegis_alpha/"
echo "3. scp config.yaml ubuntu@<SERVER_IP>:~/gold/aegis_alpha/"
echo "4. CREATE YOUR .env FILE MANUALLY ON THE SERVER."
echo "------------------------------------------------------------"
echo "The Interceptor is waiting for its brain."
