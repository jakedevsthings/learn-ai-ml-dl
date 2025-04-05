#!/bin/bash

echo "🧹 CLEAN.SH STARTING..."

# Check if venv is active and warn
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "⚠️  VIRTUAL_ENV is active: $VIRTUAL_ENV"
    echo "ℹ️  Please deactivate manually before running this script, if needed."
else
    echo "✅ No virtual environment active — continuing."
fi

echo "🧨 Removing .venv directory..."
rm -rf .venv

echo "🧼 Removing Python cache dirs..."
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache .ipynb_checkpoints .vscode-server

echo "🧹 Removing Jupyter kernel (learn-ai-ml-dl)..."
jupyter kernelspec remove -f learn-ai-ml-dl

echo "✅ CLEAN COMPLETE."
