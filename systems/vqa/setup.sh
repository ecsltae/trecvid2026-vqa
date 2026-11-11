#!/usr/bin/env bash
set -euo pipefail
VENV=venv
source "$VENV/bin/activate"

echo "Installing VQA dependencies..."
pip install yt-dlp bert-score sacrebleu nltk --quiet

echo "Downloading NLTK data for METEOR..."
python3 -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True)"

echo "Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi
echo "OK: ffmpeg $(ffmpeg -version 2>&1 | head -1)"

echo "Pulling LLaVA via Ollama (~4.7GB)..."
ollama pull llava:13b

echo ""
echo "VQA sandbox ready."
echo "Download topics CSV from: https://www-nlpir.nist.gov/projects/tv2025/vqa.html"
echo "Place it at: data/vqa/topics.csv"
