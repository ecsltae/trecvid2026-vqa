"""
TRECVID VQA Multiple Choice (MC) v2 — Two-Stage Caption+Reason Pipeline.

Stage 1 (vision model, llava:13b): Video frames → rich visual description.
Stage 2 (text model, qwen3:32b):  Description + question + options [+ few-shot] → ranked options.

Output CSV (no header):
    Q_ID,Video_ID,Rank,option_X,Time

Usage:
    python run_mc_v2.py \\
        --topics data/vqa/topics/vqa_2025_mc_topics.csv \\
        --frames data/vqa/frames_full/ \\
        --out runs/vqa/mc/TREC_VQA_twostage_MC.csv \\
        [--fewshot]
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

_RANK_SYSTEM = (
    "You are an expert at answering multiple-choice questions about video content. "
    "You will be given a visual description of a video, a question, and four answer options. "
    "Rank the options from most to least likely correct based on the visual evidence described."
)

_RANK_TMPL = """\
{fewshot_block}
Visual description of the video:
{caption}

Question: {question}

Options:
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

Step 1 — Identify which elements of the visual description are relevant to the question.
Step 2 — Evaluate each option against the visual evidence:
  A) {opt_a}: [Does the visual description support this?]
  B) {opt_b}: [Does the visual description support this?]
  C) {opt_c}: [Does the visual description support this?]
  D) {opt_d}: [Does the visual description support this?]
Step 3 — Rank from most to least supported by the visual evidence.

On the LAST line write exactly: RANKING: X,X,X,X  (four letters A-D, ordered best to worst)"""

_RANK_FEWSHOT_PREFIX = """\
Here are examples of similar video questions with their correct answers:
{examples}

Now answer this new question:
"""

OPTION_KEYS = ["A", "B", "C", "D"]


def load_topics(topics_csv):
    rows = []
    with open(topics_csv, newline="", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        first_field = first.split(",")[0].strip().strip('"')
        has_header = not first_field.lstrip().isdigit()
        reader = csv.reader(f)
        if has_header:
            header = [h.strip() for h in next(reader)]
            for row in reader:
                d = dict(zip(header, [c.strip().strip('"') for c in row]))
                rows.append({
                    "Q_ID": d.get("Q_ID") or d.get("q_id", ""),
                    "Video_ID": d.get("Video_ID") or d.get("video_id", ""),
                    "Question": d.get("Question") or d.get("question", ""),
                    "A": d.get("Choice_A") or d.get("A", ""),
                    "B": d.get("Choice_B") or d.get("B", ""),
                    "C": d.get("Choice_C") or d.get("C", ""),
                    "D": d.get("Choice_D") or d.get("D", ""),
                })
        else:
            for row in reader:
                row = [c.strip().strip('"') for c in row]
                if len(row) < 7:
                    continue
                rows.append({
                    "Q_ID": row[0], "Video_ID": row[1], "Question": row[2],
                    "A": row[3], "B": row[4], "C": row[5], "D": row[6],
                })
    return rows


def get_frames(frames_dir, video_id, max_frames=8):
    vid_dir = Path(frames_dir) / video_id
    if not vid_dir.exists():
        return []
    return sorted(vid_dir.glob("frame_*.jpg"))[:max_frames]


def caption_frames(vlm_client, frame_paths, dry_run=False, cache=None, video_id=None):
    if cache is not None and video_id and video_id in cache:
        return cache[video_id] or "[No visual content cached]"
    if dry_run or not frame_paths:
        return "[Dry-run: no visual content available]"
    return vlm_client.generate_multimodal(
        prompt=_CAPTION_PROMPT,
        images=frame_paths,
        system=_CAPTION_SYSTEM,
    )


def parse_ranking(response):
    """Extract letter ranking from model response."""
    for line in reversed(response.upper().split("\n")):
        if "RANKING:" in line:
            letters = re.findall(r"\b([A-D])\b", line)
            seen = []
            for l in letters:
                if l not in seen:
                    seen.append(l)
            if len(seen) >= 2:
                break
    else:
        letters = re.findall(r"\b([A-D])\b", response.upper())
        seen = []
        for l in letters:
            if l not in seen:
                seen.append(l)
    for l in OPTION_KEYS:
        if l not in seen:
            seen.append(l)
    return seen[:4]


def rank_options(llm_client, question, options, caption, fewshot_text=None, dry_run=False):
    if dry_run:
        return OPTION_KEYS[:]

    fewshot_block = ""
    if fewshot_text:
        fewshot_block = fewshot_text + "\n\n"

    prompt = _RANK_TMPL.format(
        fewshot_block=fewshot_block,
        caption=caption,
        question=question,
        opt_a=options.get("A", ""),
        opt_b=options.get("B", ""),
        opt_c=options.get("C", ""),
        opt_d=options.get("D", ""),
    )
    response = llm_client.generate(prompt, system=_RANK_SYSTEM, think=True)
    return parse_ranking(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="systems/vqa/config_v2.yaml")
    parser.add_argument("--training", default="data/vqa/training")
    parser.add_argument("--captions-cache", default=None, help="Path to pre-computed captions JSON (skips llava Stage 1)")
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    frames_dir = args.frames or cfg.get("paths", {}).get("frames_dir", "data/vqa/frames_full")
    out_path = Path(args.out or Path(cfg.get("paths", {}).get("output_dir", "runs/vqa")) / "mc" / "TREC_VQA_twostage_MC.csv")
    vision_model = cfg.get("ollama", {}).get("vision_model", "llava:13b")
    text_model = cfg.get("ollama", {}).get("stage2_model", "qwen3:32b")
    max_frames = cfg.get("frame_extraction", {}).get("max_frames", 8)

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
            options = {k: row.get(k, "") for k in "ABCD"}

            if q_id in done_ids:
                continue

            frame_paths = get_frames(frames_dir, video_id, max_frames)
            if not frame_paths and not args.dry_run:
                print(f"  [{i+1}/{len(topics)}] {q_id} SKIP (no frames for {video_id})")
                for rank, letter in enumerate(OPTION_KEYS, 1):
                    writer.writerow([q_id, video_id, rank, letter, "0.00"])
                continue

            t_start = time.time()
            caption = caption_frames(vlm, frame_paths, args.dry_run, caption_cache, video_id)

            fewshot_text = None
            if retriever:
                examples = retriever.retrieve(question, k=3)
                fewshot_text = _RANK_FEWSHOT_PREFIX.format(
                    examples=retriever.format_mc_fewshot(examples)
                ) if examples else None

            ranking = rank_options(llm, question, options, caption, fewshot_text, args.dry_run)
            total_elapsed = time.time() - t_start

            for rank, letter in enumerate(ranking, 1):
                writer.writerow([q_id, video_id, rank, letter, f"{total_elapsed:.2f}"])

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(topics)}] done — {q_id}")

    print(f"Wrote MC v2 results to {out_path}")


if __name__ == "__main__":
    main()
