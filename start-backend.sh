#!/usr/bin/env bash
set -euo pipefail

# Resolve python executable
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Error: python3/python not found in PATH." >&2
  exit 1
fi

echo "Creating venv if missing..."
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv || true
fi

# If activation script still missing, retry with a clean venv and fallbacks
if [ ! -f .venv/bin/activate ]; then
  echo "Recreating venv..."
  rm -rf .venv
  "$PYTHON" -m venv .venv || true
fi

if [ ! -f .venv/bin/activate ]; then
  echo "venv module may be unavailable. Trying virtualenv fallback..."
  "$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || true
  "$PYTHON" -m pip install virtualenv >/dev/null 2>&1
  "$PYTHON" -m virtualenv .venv
fi

echo "Activating venv..."
. .venv/bin/activate

echo "Installing backend requirements..."
pip install -r backend/requirements.txt

echo "Starting backend on http://127.0.0.1:8080 ..."
export FLASK_APP=backend.app.main
python -m flask run --host=0.0.0.0 --port=8080
