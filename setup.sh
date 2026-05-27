#!/bin/bash
echo "==================================================="
echo "Embrained Setup Script (Mac/Linux)"
echo "==================================================="

echo "Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed. Please install Python 3.12."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "[ERROR] node is not installed. Please install Node.js."
    exit 1
fi

echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo ""
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setting up Node.js frontend..."
cd cognitive-engine/frontend || exit
npm install
echo "Building frontend for production..."
npm run build
cd ../..

echo ""
echo "==================================================="
echo "Setup Complete!"
echo "You can now run the app using: ./start.sh"
echo "==================================================="
