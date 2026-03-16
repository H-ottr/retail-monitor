#!/bin/bash
# ============================================================
# RetailGuard AI — Quick Start Script
# ============================================================
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║      RetailGuard AI Monitoring System    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[✗] Python3 not found. Please install Python 3.9+"
    exit 1
fi

echo "[1/3] Installing requirements..."
pip install -r requirements.txt -q

echo "[2/3] Setting up directories..."
mkdir -p app/static/uploads/incidents
mkdir -p app/static/processed
mkdir -p data/images/train
mkdir -p data/images/val
mkdir -p data/labels/train
mkdir -p data/labels/val

echo "[3/3] Starting RetailGuard AI..."
echo ""
echo "  🌐 URL:      http://localhost:5000"
echo "  👤 Admin:    admin / admin123"
echo "  👤 Staff:    staff / staff123"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

cd app && python3 app.py
