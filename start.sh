#!/bin/bash
echo "==================================================="
echo "Starting Embrained App"
echo "==================================================="

if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

echo ""
echo "Select Engine Mode:"
echo "[1] Discrete Markov (Default)"
echo "[2] Continuous Action Chunking"
read -p "Enter choice (1 or 2): " mode

ENGINE_FLAG=""
if [ "$mode" == "2" ]; then
    ENGINE_FLAG="--continuous"
    echo "Launching Continuous Action Chunking engine..."
else
    echo "Launching Discrete Markov engine..."
fi

source venv/bin/activate

echo ""
echo "Server starting on port 8080..."
echo "Please navigate to http://localhost:8080 in your default web browser."
echo ""

python3 cognitive-engine/app.py --port 8080 --plexus $ENGINE_FLAG
