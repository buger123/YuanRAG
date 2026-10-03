#!/usr/bin/env bash
# Build the React frontend into src/frontend/dist
set -e

cd "$(dirname "$0")/../src/frontend"
export NPM_CONFIG_CACHE="$PWD/.npm-cache"

if [ ! -d node_modules ]; then
    echo "[build] Installing frontend dependencies..."
    npm install --no-audit --no-fund --loglevel=error
fi

echo "[build] Building frontend..."
npm run build --silent

echo "[build] Frontend built successfully."
