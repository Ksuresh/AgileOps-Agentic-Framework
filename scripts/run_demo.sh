#!/usr/bin/env bash

set -e

echo "========================================"
echo "AAF Demo Runner"
echo "========================================"

if [ ! -d ".venv" ]; then
  echo "[1/4] Creating virtual environment..."
  python -m venv .venv
else
  echo "[1/4] Virtual environment already exists."
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

echo "[4/4] Running reproducibility experiment..."
python run_experiments.py

echo ""
echo "Launching Gradio UI..."
echo "Open the local URL shown below in your browser."
python ui/gradio_app.py
