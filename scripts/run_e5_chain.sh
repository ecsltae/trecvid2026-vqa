#!/usr/bin/env bash
# Chain: translate the remaining dev videos on GPU1, then run the 32b power arms
# there. Sequencing lives in bash, not in an agent's judgment -- a -p session
# ends when the model stops emitting, not when its background job finishes.
set -u
cd .
source "${VENV:-.venv}/bin/activate"
LOG=logs

python systems/vqa/transcribe.py --videos /tmp/dev8rest --out /tmp/dev8_translate_rest.jsonl \
  --device-index 1 --task translate --beam-size 5 --max-no-speech 0.6 --min-logprob -1.0 \
  > "$LOG/e5_translate.log" 2>&1
cat /tmp/dev8_translate.jsonl /tmp/dev8_translate_rest.jsonl > /tmp/dev8_translate_full.jsonl
echo "[e5] transcripts ready: $(wc -l < /tmp/dev8_translate_full.jsonl) videos $(date +%H:%M:%S)"

# wait for whisper to release the card before loading a 20GB model onto it
while pgrep -f "systems/vqa/transcribe.py" >/dev/null 2>&1; do sleep 20; done
exec scripts/run_e5_power.sh
