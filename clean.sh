#!/bin/bash

echo "🧹 [CLEAN] Removing .venv..."
rm -rf .venv

echo "🧼 [CLEAN] Removing Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache .ipynb_checkpoints .vscode-server

echo "🚮 [CLEAN] Unregistering Jupyter kernel..."
jupyter kernelspec remove -f learn-ai-ml-dl

echo "✅ [DONE] Project cleaned!"
