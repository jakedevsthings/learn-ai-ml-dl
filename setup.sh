#!/bin/bash

echo "⚙️  [SETUP] Creating virtual environment..."
python3 -m venv .venv

echo "📦 [SETUP] Activating and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "📓 [Jupyter] Registering kernel..."
python -m ipykernel install --user --name=learn-ai-ml-dl --display-name "Python (learn-ai-ml-dl)"

echo "✅ [DONE] Environment ready!"
echo ""
echo "🚀 Activating your environment now..."
echo ""

# Keep user inside the activated shell
$SHELL
