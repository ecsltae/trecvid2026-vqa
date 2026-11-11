#!/usr/bin/env bash
# Guide for downloading TRECVID 2025 VQA data.
# Videos must be fetched via yt-dlp from YouTube.

set -euo pipefail

DATADIR=data/vqa
mkdir -p "$DATADIR/videos" "$DATADIR/frames"

echo "=== TRECVID 2025 VQA Data Setup ==="
echo ""
echo "1. Download the topics CSV from:"
echo "   https://www-nlpir.nist.gov/projects/tv2025/vqa.html"
echo "   Save to: $DATADIR/topics.csv"
echo ""
echo "2. Once you have topics.csv, download videos:"
echo "   source "${VENV:-.venv}/bin/activate""
echo "   python systems/vqa/download_videos.py \\"
echo "     --topics $DATADIR/topics.csv \\"
echo "     --out $DATADIR/videos/"
echo ""
echo "3. Extract frames:"
echo "   python systems/vqa/extract_frames.py \\"
echo "     --videos $DATADIR/videos/ \\"
echo "     --out $DATADIR/frames/ \\"
echo "     --mode keyframe --max-frames 8"
echo ""
echo "NOTE: YouTube availability varies — expect ~5-15% of videos to fail."
echo "      Failed video IDs are logged to $DATADIR/download_failures.txt"
