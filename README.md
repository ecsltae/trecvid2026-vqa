# BiTeM at TRECVID 2026 VQA

Code, development-set runs and the analysis and figure scripts for the notebook paper

> **BiTeM at TRECVID 2026 VQA: What a 500-Item Development Set Could Not See.** Esteban Gaillac,
> Julien Knafou, Paul Van Rijen, Alexandre Flament, Patrick Ruch. HES-SO and SIB Swiss Institute of
> Bioinformatics, Geneva. TRECVID 2026.

The paper is `paper/main_bitem_trec_vqa.pdf`, with its source in `paper/tex/`.

## What is here

| path | content |
|---|---|
| `systems/vqa/` | the pipelines: frame extraction, answer generation by sampling and clustering (`run_vqa_v4.py`, `run_ag_v2.py`), multiple-choice ranking (`run_vqa_mc_v5.py`, `run_mc_v2.py`), few-shot retrieval, transcription and routing, run fusion (`fuse_runs.py`) |
| `eval/vqa/` | scoring (`eval_mc.py`, `eval_ag_official.py`, `official_sts.py`), development slices, router and selection analyses, submission builders |
| `scripts/` | drivers for the experiments in the paper |
| `runs/dev_mc/` | the ten multiple-choice development runs behind the paper's replicate and model-size figures, on the 497-item development set |
| `runs/dev_ag/` | the 128 sampled answers per development question behind the selection figure |
| `paper/` | the paper, its LaTeX source, figures, and the scripts that draw them (`paper/figures/`) |

## What is not here

- **Videos, frames and the collection's questions and answers.** These come from the TRECVID VQA
  collections and are distributed by NIST to participants. The development runs therefore carry
  ids, our own outputs and scores only: a multiple-choice run row is `q_id, video_id, rank,
  option_index, score`, where `option_index` is the option's position (1 to 8) in our
  development-set file built by `systems/vqa/build_dev8.py`; the answer-generation file keeps the
  question and video ids, our sampled answers and their ranking, without the question text.
- **The official scores**, which NIST sent to participants.

The figure scripts and scorers expect those inputs under `data/` and `submissions/`, laid out as
the paths in the scripts show.

## Requirements

Python 3.10 with `torch`, `transformers`, `sentence-transformers`, `faster-whisper`, `numpy`,
`scipy`, `matplotlib`, `pandas` and `ollama`; models served by Ollama (Qwen3-VL 8B, 30B-A3B and
32B, Qwen3.5 122B). Set `VENV` to your virtual environment for the shell drivers.

## Licence

No licence has been chosen yet; until one is, contact the authors before reusing the code.
