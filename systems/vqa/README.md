# TRECVID 2025 VQA Sandbox

Video Question Answering using LLaVA (vision-language model) via Ollama.

## Quick Start

```bash
source "${VENV:-.venv}/bin/activate"
cd systems/vqa
bash setup.sh
```

## Task Overview

- **Input**: ~2,000 YouTube Shorts (~30s) + questions
- **Subtask AG**: Generate ranked answers → CSV (`Q_ID,Video_ID,Rank,Answer,Time`)
- **Subtask MC**: Rank 4 choices → CSV (`Q_ID,Video_ID,Rank,Selected_Option,Time`)
- **Eval AG**: STS, METEOR, BERTScore, NDCG
- **Eval MC**: Top-1 accuracy, MRR
- **Track page**: https://www-nlpir.nist.gov/projects/tv2025/vqa.html

## Pipeline

### 1. Download topics CSV
From the TRECVID 2025 VQA task page. Place at `data/vqa/topics.csv`.

### 2. Download videos
```bash
python download_videos.py \
    --topics data/vqa/topics.csv \
    --out data/vqa/videos/ \
    --limit 100   # remove --limit for full set
```

### 3. Extract frames
```bash
python extract_frames.py \
    --videos data/vqa/videos/ \
    --out data/vqa/frames/ \
    --mode keyframe --max-frames 8
```

### 4. Run inference

**Answer Generation (AG):**
```bash
python run_vqa_ag.py \
    --topics data/vqa/topics.csv \
    --frames data/vqa/frames/ \
    --out runs/vqa/ag/TREC_VQA_llava_AG.csv
```

**Multiple Choice (MC):**
```bash
python run_vqa_mc.py \
    --topics data/vqa/topics.csv \
    --frames data/vqa/frames/ \
    --out runs/vqa/mc/TREC_VQA_llava_MC.csv
```

### 5. Evaluate
```bash
python eval/vqa/eval_vqa.py \
    --pred runs/vqa/ag/TREC_VQA_llava_AG.csv \
    --qrels data/vqa/qrels.csv \
    --task ag
```

## Smoke Test (no real videos needed)
```bash
python run_vqa_ag.py --topics tests/fixtures/topics_toy.csv --dry-run --limit 2
python -m pytest tests/test_smoke.py -v
```

## Notes

- `llava:13b` is pulled by `setup.sh` (~9 GB)
- Frame extraction requires `ffmpeg`: `sudo apt install ffmpeg`
- METEOR requires `nltk` wordnet data (downloaded by `setup.sh`)
