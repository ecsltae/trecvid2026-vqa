#!/usr/bin/env bash
# Fetch the TRECVID VQA 2026 testing dataset (Task 1 / AG).
#
# The files sit behind HTTP Basic auth (realm "trec2026") using your TREC
# ACTIVE PARTICIPANTS username/password — the same ones used for the tracks
# page. They are NOT stored in this repo.
#
# Provide them either as environment variables:
#     export TREC_USER=xxx TREC_PASS=yyy
#     bash scripts/download_vqa_2026.sh
# or interactively:
#     bash scripts/download_vqa_2026.sh          # prompts for both
#
# NOTE: NIST will not score any run unless the signed data agreement has been
# emailed to george.awad@nist.gov with the team name in the message.

set -euo pipefail
cd .

OUT=data/vqa/2026
BASE=https://trec.nist.gov/act_part/tracks/vqa/2026/testing.dataset
mkdir -p "$OUT"

if [ -z "${TREC_USER:-}" ]; then read -rp "TREC participant username: " TREC_USER; fi
if [ -z "${TREC_PASS:-}" ]; then read -rsp "TREC participant password: " TREC_PASS; echo; fi

fetch () {  # fetch <url> <dest>
    echo "→ $(basename "$2")"
    if ! curl -fSL --user "$TREC_USER:$TREC_PASS" --max-time 3600 -o "$2" "$1"; then
        echo "FAILED: $1 (401 = wrong credentials; 404 = URL moved)" >&2
        return 1
    fi
}

fetch "$BASE/vqa.task1.readme.txt"                       "$OUT/vqa.task1.readme.txt"
fetch "$BASE/task1/vqa.2026.testingDataset.Task1.json"   "$OUT/vqa.2026.testingDataset.Task1.json"
fetch "$BASE/videos.zip"                                 "$OUT/videos.zip"

echo ""
echo "=== Downloaded to $OUT ==="
ls -la "$OUT"

if [ -s "$OUT/videos.zip" ]; then
    echo ""
    echo "Unpacking provided clips..."
    unzip -q -o "$OUT/videos.zip" -d "$OUT/videos/"
    echo "Clips: $(find "$OUT/videos" -type f | wc -l)"
fi

echo ""
echo "Queries: $(python3 -c "import json;print(len(json.load(open('$OUT/vqa.2026.testingDataset.Task1.json'))))" 2>/dev/null || echo '?')"
echo ""
echo "Next: download YouTube videos + extract frames, then run the v3 pipeline."
