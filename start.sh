#!/usr/bin/env bash
# RAG Assistant — Linux/macOS one-click launcher
set -euo pipefail
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[setup] Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies (skip if marker exists)
if [ ! -f ".deps_installed" ]; then
    echo "[setup] Installing Python dependencies (first run may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch .deps_installed
fi

# Build frontend if missing
if [ ! -f "src/frontend/dist/index.html" ]; then
    echo "[setup] Building frontend (first run)..."
    bash scripts/build_frontend.sh || echo "[warning] Frontend build failed; UI will not load."
fi

# Launch
echo "[startup] Launching RAG Assistant..."
python -m src.main
