"""
TRECVID VQA Multiple Choice (MC) subtask.

Uses LLaVA + CoT to rank 4 answer options.

Output CSV (no header):
    Q_ID,Video_ID,Rank,Selected_Option,Time

Usage:
    python run_vqa_mc.py \\
        --topics /path/to/topics.csv \\
        --frames /path/to/frames/ \\
        --out /path/to/runs/vqa/mc/run.csv \\
        [--limit 10] \\
        [--dry-run]
"""
import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient

_MC_SYSTEM = (
    "You are a video understanding assistant. "
    "Given video frames and a question with 4 answer options, "
    "rank all options from most likely correct to least likely. "
    "Reply with ONLY the 4 option letters in order, comma-separated, e.g.: B,A,D,C"
)

_MC_TMPL = (
    "Question: {question}\n"
    "A) {opt_a}\n"
    "B) {opt_b}\n"
    "C) {opt_c}\n"
    "D) {opt_d}\n\n"
    "Rank all 4 options from most to least likely correct. Reply with letters only, e.g.: B,A,D,C"
)

_MC_COT_SYSTEM = (
    "You are a video understanding assistant. Reason carefully about what you see in the "
    "video frames before answering. Consider each option based on the visual evidence."
)

_MC_COT_TMPL = """\
Question: {question}

Options:
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

First, describe the key visual information from the frames relevant to this question.
Then, briefly evaluate each option.
Finally, on the LAST line, write: RANKING: <four letters A-D separated by commas, ordered from best to worst>

Example format: RANKING: B,A,D,C\
"""

OPTION_KEYS = ["A", "B", "C", "D"]


def load_topics(topics_csv):
    """Load MC topics. Supports:
    - No-header format (real TRECVID 2025): Q_ID, Video_ID, Question, A, B, C, D
    - Header format (toy fixtures): Q_ID, Video_ID, Question, Choice_A/B/C/D
    """
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


def parse_mc_ranking(response):
    """Extract letter ranking from model response. Returns list of letters."""
    # For CoT responses: look for RANKING: line first
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
        # Fall back: scan whole response for letter sequence
        letters = re.findall(r"\b([A-D])\b", response.upper())
        seen = []
        for l in letters:
            if l not in seen:
                seen.append(l)
    # Fill in any missing letters
    for l in OPTION_KEYS:
        if l not in seen:
            seen.append(l)
    return seen[:4]


def rank_options(client, question, options, frame_paths, dry_run=False,
                 prompt_style="direct", shuffle_choices=False, seed=None):
    """Return list of option letters ranked best-first.

    prompt_style: 'direct' (fast) | 'cot' (chain-of-thought, slower but more accurate)
    shuffle_choices: if True, randomize presentation order to test positional bias
    seed: optional random seed for reproducible shuffling
    """
    if dry_run or not frame_paths:
        return OPTION_KEYS[:]

    # Shuffle option mapping if requested
    if shuffle_choices:
        if seed is not None:
            random.seed(seed)
        shuffled_keys = OPTION_KEYS[:]
        random.shuffle(shuffled_keys)
        # Map: original_label -> shuffled_position_label
        # E.g., if shuffled_keys = ['C', 'A', 'D', 'B'], then:
        #   original A → shown at position A (value from original C)
        #   original B → shown at position B (value from original A)
        shuffle_map = {orig: shuf for orig, shuf in zip(OPTION_KEYS, shuffled_keys)}
        reverse_map = {shuf: orig for orig, shuf in shuffle_map.items()}

        opt_a = options.get(shuffle_map["A"], "")
        opt_b = options.get(shuffle_map["B"], "")
        opt_c = options.get(shuffle_map["C"], "")
        opt_d = options.get(shuffle_map["D"], "")
    else:
        opt_a = options.get("A", "")
        opt_b = options.get("B", "")
        opt_c = options.get("C", "")
        opt_d = options.get("D", "")

    if prompt_style == "cot":
        prompt = _MC_COT_TMPL.format(
            question=question, opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d
        )
        response = client.generate_multimodal(
            prompt=prompt, images=frame_paths, system=_MC_COT_SYSTEM
        )
    else:
        prompt = _MC_TMPL.format(
            question=question, opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d
        )
        response = client.generate_multimodal(
            prompt=prompt, images=frame_paths, system=_MC_SYSTEM
        )

    ranking = parse_mc_ranking(response)

    # If shuffled, map model's ranking back to original labels
    if shuffle_choices:
        ranking = [shuffle_map[letter] for letter in ranking]

    return ranking


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
    out_path = args.out or Path(cfg.get("paths", {}).get("output_dir", "runs/vqa")) / "mc" / f"{cfg.get('run_id','run')}_MC.csv"
    model = cfg.get("ollama", {}).get("vision_model", "llava:13b")
    max_frames = cfg.get("frame_extraction", {}).get("max_frames", 8)
    temperature = cfg.get("ollama", {}).get("temperature", 0.1)

    client = OllamaClient(model=model, temperature=temperature)
    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[:args.limit]

    shuffle_choices = cfg.get("mc_subtask", {}).get("shuffle_choices", True)  # Default True: eliminates A-bias

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(topics)} topics (dry_run={args.dry_run}, shuffle={shuffle_choices})...")

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for row in topics:
            q_id = row.get("Q_ID") or row.get("q_id")
            video_id = row.get("Video_ID") or row.get("video_id")
            question = row.get("Question") or row.get("question", "")
            # Keys are already normalized to A/B/C/D by load_topics()
            options = {k: row.get(k, "") for k in "ABCD"}

            t0 = time.time()
            frame_paths = get_frames(frames_dir, video_id, max_frames)
            prompt_style = cfg.get("mc_subtask", {}).get("prompt_style", "direct")
            # Use q_id as seed for reproducible shuffling per question
            seed = hash(str(q_id)) % (2**32) if shuffle_choices else None
            ranking = rank_options(client, question, options, frame_paths,
                                   args.dry_run, prompt_style, shuffle_choices, seed)
            elapsed = time.time() - t0

            for rank, letter in enumerate(ranking, 1):
                writer.writerow([q_id, video_id, rank, letter, f"{elapsed:.2f}"])

    print(f"Wrote MC results to {out_path}")


if __name__ == "__main__":
    main()
