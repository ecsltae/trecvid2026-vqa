"""
TRECVID VQA Answer Generation (AG) subtask.

Feeds video frames + question to LLaVA via Ollama, produces ranked CSV.

Output CSV (no header):
    Q_ID,Video_ID,Rank,Answer,Time

Usage:
    python run_vqa_ag.py \\
        --topics /path/to/topics.csv \\
        --frames /path/to/frames/ \\
        --out /path/to/runs/vqa/ag/run.csv \\
        [--limit 10] \\
        [--dry-run]
"""
import argparse
import base64
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient

_SYSTEM = (
    "You are a video understanding assistant. "
    "Answer questions based only on what is visible in the provided video frames. "
    "Be concise and factual."
)

_MULTI_ANSWER_TMPL = """\
Question: {question}

You MUST provide EXACTLY {n} different possible answers, ranked from most likely to least likely.

INSTRUCTIONS:
- Generate {n} DISTINCT answers (not just one answer rephrased)
- Each answer should offer a different perspective or interpretation
- If multiple interpretations are possible, explore them
- If only one clear answer exists, provide increasingly speculative alternatives for lower ranks
- Format: "N. Answer text" where N is 1 to {n}
- Keep each answer to 1-2 sentences maximum

REQUIRED OUTPUT FORMAT (YOU MUST PROVIDE ALL {n} ANSWERS):
1. [Most likely answer based on visible evidence]
2. [Second possibility or alternative interpretation]
3. [Third possibility or speculative answer]
...continue until...
{n}. [Least likely but still plausible answer]

Now provide your {n} ranked answers:\
"""


def load_topics(topics_csv):
    """Load AG topics. Supports:
    - No-header format (real TRECVID 2025): Q_ID, Video_ID, Question
    - Header format (toy fixtures): Q_ID, Video_ID, Question
    """
    rows = []
    with open(topics_csv, newline="", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        # Detect header by checking if first field looks numeric
        first_field = first.split(",")[0].strip().strip('"')
        has_header = not first_field.lstrip().isdigit()
        reader = csv.reader(f)
        if has_header:
            header = [h.strip() for h in next(reader)]
        for row in reader:
            row = [c.strip().strip('"') for c in row]
            if len(row) < 3:
                continue
            rows.append({
                "Q_ID": row[0],
                "Video_ID": row[1],
                "Question": row[2],
            })
    return rows


def get_frames(frames_dir, video_id, max_frames=8):
    """Return sorted list of frame paths for a video."""
    vid_dir = Path(frames_dir) / video_id
    if not vid_dir.exists():
        return []
    frames = sorted(vid_dir.glob("frame_*.jpg"))[:max_frames]
    return frames


def generate_ag_answers(client, question, frame_paths, num_answers=5, dry_run=False):
    """Return list of (answer_str, elapsed_seconds) for top num_answers.

    Uses a single model call requesting N ranked answers, which is faster and
    more coherent than N separate calls.
    """
    if dry_run or not frame_paths:
        return [(f"Placeholder answer {i+1} for: {question[:30]}", 0.0) for i in range(num_answers)]

    t0 = time.time()
    prompt = _MULTI_ANSWER_TMPL.format(question=question, n=num_answers)
    raw = client.generate_multimodal(
        prompt=prompt,
        images=frame_paths,
        system=_SYSTEM,
    )
    elapsed = time.time() - t0
    answers = _parse_ranked_answers(raw, num_answers, elapsed)
    return answers


def _parse_ranked_answers(raw, num_answers, elapsed):
    """Parse numbered list output into (text, time) pairs."""
    import re
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    answers = []
    for line in lines:
        m = re.match(r"^(\d+)[.)]\s*(.+)", line)
        if m and len(answers) < num_answers:
            answers.append((m.group(2).strip(), elapsed))
    # Pad with fallback if model returned fewer than expected
    while len(answers) < num_answers:
        answers.append((f"Unable to determine answer #{len(answers)+1}", elapsed))
    return answers[:num_answers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="systems/vqa/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    frames_dir = args.frames or cfg.get("paths", {}).get("frames_dir", "data/vqa/frames")
    out_path = args.out or Path(cfg.get("paths", {}).get("output_dir", "runs/vqa")) / "ag" / f"{cfg.get('run_id','run')}_AG.csv"
    run_id = cfg.get("run_id", "TREC_VQA_baseline")
    max_frames = cfg.get("frame_extraction", {}).get("max_frames", 8)
    num_answers = cfg.get("ag_subtask", {}).get("num_answers", 5)
    model = cfg.get("ollama", {}).get("vision_model", "llava:13b")

    client = OllamaClient(model=model, temperature=0.1)
    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[:args.limit]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(topics)} topics (dry_run={args.dry_run})...")

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for row in topics:
            q_id = row.get("Q_ID") or row.get("q_id")
            video_id = row.get("Video_ID") or row.get("video_id")
            question = row.get("Question") or row.get("question", "")

            frame_paths = get_frames(frames_dir, video_id, max_frames)
            answers = generate_ag_answers(client, question, frame_paths, num_answers, args.dry_run)

            for rank, (answer_text, elapsed) in enumerate(answers, 1):
                writer.writerow([q_id, video_id, rank, answer_text, f"{elapsed:.2f}"])

    print(f"Wrote AG results to {out_path}")


if __name__ == "__main__":
    main()
