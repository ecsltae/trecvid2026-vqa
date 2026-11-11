"""
TRECVID VQA Answer Generation (AG) v2 — Two-Stage Caption+Reason Pipeline.

Stage 1 (vision model, llava:13b): Video frames → rich visual description + temporal narrative.
Stage 2 (text model, qwen3:32b):  Description + question [+ few-shot] → 10 ranked answers.

Decoupling perception from reasoning lets us leverage qwen3:32b's superior language
reasoning on top of the visual grounding produced by the VLM.

Output CSV (no header):
    Q_ID,Video_ID,Rank,Answer,Time

Usage:
    python run_ag_v2.py \\
        --topics data/vqa/topics/vqa_2025_ag_topics.csv \\
        --frames data/vqa/frames_full/ \\
        --out runs/vqa/ag/TREC_VQA_twostage_AG.csv \\
        [--fewshot]            # enable dynamic few-shot from training data
        [--training data/vqa/training]
        [--limit 10]
        [--dry-run]
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient

_CAPTION_SYSTEM = (
    "You are a precise video analyst. Describe what you observe in the provided video frames. "
    "Focus on: the main subject(s), their actions and movements, the setting/environment, "
    "any text or objects visible, and how the scene evolves across frames. "
    "Be factual and specific. Do not guess — only describe what is clearly visible."
)

_CAPTION_PROMPT = (
    "Describe the visual content of these video frames in detail. "
    "Organize your description as:\n"
    "1. Main subject(s) and their appearance\n"
    "2. Actions and events occurring\n"
    "3. Setting and background\n"
    "4. Temporal progression (what changes frame to frame)\n"
    "Be specific and factual."
)

_REASON_SYSTEM = (
    "You are an expert at answering questions about video content. "
    "You will be given a visual description of a video and a question. "
    "Generate ranked answers based solely on the visual evidence described."
)

_REASON_TMPL = """\
{fewshot_block}
Visual description of the video:
{caption}

Question: {question}

Based on the visual description above, provide {n} distinct answers ranked from most to least likely correct.

REQUIREMENTS:
- Each answer must be grounded in the visual description
- Answers must be genuinely distinct (not just rephrasing)
- If the visual description clearly indicates one answer, provide it first; explore alternatives for lower ranks
- Keep each answer to 1-2 sentences

FORMAT (exactly {n} numbered answers):
1. [Most likely answer]
2. [Second most likely]
...
{n}. [Least likely but plausible]

Provide your {n} ranked answers now:"""

_REASON_FEWSHOT_PREFIX = """\
Here are examples of similar video questions with their correct answers:
{examples}

Now answer this new question:
"""


def load_topics(topics_csv):
    rows = []
    with open(topics_csv, newline="", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        first_field = first.split(",")[0].strip().strip('"')
        has_header = not first_field.lstrip().isdigit()
        reader = csv.reader(f)
        if has_header:
            next(reader)
        for row in reader:
            row = [c.strip().strip('"') for c in row]
            if len(row) < 3:
                continue
            rows.append({"Q_ID": row[0], "Video_ID": row[1], "Question": row[2]})
    return rows


def get_frames(frames_dir, video_id, max_frames=8):
    vid_dir = Path(frames_dir) / video_id
    if not vid_dir.exists():
        return []
    return sorted(vid_dir.glob("frame_*.jpg"))[:max_frames]


def caption_frames(vlm_client, frame_paths, dry_run=False, cache=None, video_id=None):
    """Stage 1: produce a visual description from frames.

    If cache is provided and video_id is in it, skips llava entirely.
    """
    if cache is not None and video_id and video_id in cache:
        return cache[video_id] or "[No visual content cached]"
    if dry_run or not frame_paths:
        return "[Dry-run: no visual content available]"
    return vlm_client.generate_multimodal(
        prompt=_CAPTION_PROMPT,
        images=frame_paths,
        system=_CAPTION_SYSTEM,
    )


def generate_answers(llm_client, question, caption, num_answers=10,
                     fewshot_examples=None, dry_run=False):
    """Stage 2: reason over caption to produce ranked answers."""
    if dry_run:
        return [(f"Placeholder answer {i+1} for: {question[:30]}", 0.0) for i in range(num_answers)]

    fewshot_block = (fewshot_examples + "\n\n") if fewshot_examples else ""

    prompt = _REASON_TMPL.format(
        fewshot_block=fewshot_block,
        caption=caption,
        question=question,
        n=num_answers,
    )
    t0 = time.time()
    raw = llm_client.generate(prompt, system=_REASON_SYSTEM, think=False)
    elapsed = time.time() - t0
    return _parse_ranked_answers(raw, num_answers, elapsed)


def _parse_ranked_answers(raw, num_answers, elapsed):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    answers = []
    for line in lines:
        m = re.match(r"^(\d+)[.)]\s*(.+)", line)
        if m and len(answers) < num_answers:
            answers.append((m.group(2).strip(), elapsed))
    while len(answers) < num_answers:
        answers.append((f"Unable to determine answer #{len(answers)+1}", elapsed))
    return answers[:num_answers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="systems/vqa/config_v2.yaml")
    parser.add_argument("--training", default="data/vqa/training")
    parser.add_argument("--captions-cache", default=None, help="Path to pre-computed captions JSON (skips llava Stage 1)")
    parser.add_argument("--fewshot", action="store_true", help="Enable dynamic few-shot from training data")
    parser.add_argument("--num-answers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-written Q_IDs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    frames_dir = args.frames or cfg.get("paths", {}).get("frames_dir", "data/vqa/frames_full")
    out_path = Path(args.out or Path(cfg.get("paths", {}).get("output_dir", "runs/vqa")) / "ag" / "TREC_VQA_twostage_AG.csv")
    vision_model = cfg.get("ollama", {}).get("vision_model", "llava:13b")
    text_model = cfg.get("ollama", {}).get("stage2_model", "qwen3:32b")
    max_frames = cfg.get("frame_extraction", {}).get("max_frames", 8)
    num_answers = args.num_answers or cfg.get("ag_subtask", {}).get("num_answers", 10)

    caption_cache = None
    if args.captions_cache and Path(args.captions_cache).exists():
        import json
        caption_cache = json.loads(Path(args.captions_cache).read_text(encoding="utf-8"))
        print(f"Caption cache loaded: {len(caption_cache)} entries (Stage 1 will be skipped)")

    vlm = OllamaClient(model=vision_model, temperature=0.1)
    llm = OllamaClient(model=text_model, temperature=0.1)

    retriever = None
    if args.fewshot:
        sys.path.insert(0, str(Path(__file__).parent))
        from fewshot_retriever import FewShotRetriever
        retriever = FewShotRetriever(args.training)
        print(f"Few-shot retriever loaded: {len(retriever)} training examples")

    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[:args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-written Q_IDs
    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    done_ids.add(row[0].strip())
        print(f"Resuming: {len(done_ids)} questions already done")

    mode = "a" if args.resume and done_ids else "w"
    print(f"Processing {len(topics)} topics | vision={vision_model} | text={text_model} | fewshot={args.fewshot}")

    with open(out_path, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for i, row in enumerate(topics):
            q_id = row["Q_ID"]
            video_id = row["Video_ID"]
            question = row["Question"]

            if q_id in done_ids:
                continue

            frame_paths = get_frames(frames_dir, video_id, max_frames)
            if not frame_paths and not args.dry_run:
                print(f"  [{i+1}/{len(topics)}] {q_id} SKIP (no frames for {video_id})")
                # Write fallback answers so submission stays complete
                for rank in range(1, num_answers + 1):
                    writer.writerow([q_id, video_id, rank, "Unable to retrieve video content", "0.00"])
                continue

            t_start = time.time()
            caption = caption_frames(vlm, frame_paths, args.dry_run, caption_cache, video_id)

            fewshot_text = None
            if retriever:
                examples = retriever.retrieve(question, k=3)
                fewshot_text = _REASON_FEWSHOT_PREFIX.format(
                    examples=retriever.format_ag_fewshot(examples)
                ) if examples else None

            answers = generate_answers(llm, question, caption, num_answers, fewshot_text, args.dry_run)
            total_elapsed = time.time() - t_start

            for rank, (answer_text, _) in enumerate(answers, 1):
                writer.writerow([q_id, video_id, rank, answer_text, f"{total_elapsed:.2f}"])

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(topics)}] done — {q_id}")

    print(f"Wrote AG v2 results to {out_path}")


if __name__ == "__main__":
    main()
