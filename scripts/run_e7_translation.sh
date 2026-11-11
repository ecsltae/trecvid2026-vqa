#!/usr/bin/env bash
# Does REAL translation convert to accuracy?
#
# --task translate was silently ignored by large-v3-turbo, which has no
# translation head: 478 of 1189 non-empty dev transcripts stayed in Cyrillic,
# Hangul, Devanagari or Han. large-v3 does translate -- 478 -> 6.
#
# This matters because the transcript effect is moderated by LANGUAGE, not by
# frames: English gated items measured +28.57pp (p=0.0215) while non-English
# measured +10.0 / 0.0 / -5.0pp across three baseline pairings. 472 videos just
# became readable, so the prediction is a gain concentrated on exactly those.
#
# Both arms are routed and differ ONLY in the transcript file, so anything that
# moves is the translation.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
B="--recipe v3 --no-draft --parse-repair --stage2-predict 1024 --num-ctx 16384 \
   --model qwen3-vl:8b-instruct --adaptive --transcript-mode routed"
OLLAMA_HOST=127.0.0.1:11435 python systems/vqa/run_vqa_mc_v5.py $B \
  --transcripts /tmp/dev8_translate_full.jsonl \
  --topics /tmp/dev8full_topics.json --frames data/vqa/frames_pool16_512 \
  --out /tmp/e7_turbo.csv > logs/e7_turbo.log 2>&1 &
OLLAMA_HOST=127.0.0.1:11436 python systems/vqa/run_vqa_mc_v5.py $B \
  --transcripts /tmp/xl_translate.jsonl \
  --topics /tmp/dev8full_topics.json --frames data/vqa/frames_pool16_512 \
  --out /tmp/e7_largev3.csv > logs/e7_largev3.log 2>&1 &
wait
echo "[e7] done $(date +%H:%M:%S)"
