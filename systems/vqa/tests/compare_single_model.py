#!/usr/bin/env python3
"""
Test a single model on a single question.

Usage: python test_single_model.py --model qwen3:32b --question Q1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient
from systems.vqa.run_vqa_mc import _MC_COT_SYSTEM, _MC_COT_TMPL, parse_mc_ranking

# Test questions
TEST_QUESTIONS = {
    "Q1": {
        "question": "What is the main object visible in the video?",
        "opt_a": "A red bicycle",
        "opt_b": "A blue car",
        "opt_c": "A green tree",
        "opt_d": "A yellow dog",
        "context": (
            "The video shows a parked vehicle on a street. "
            "The vehicle has four wheels, windows, doors, and appears to be blue in color. "
            "There are trees visible in the background."
        ),
        "expected": "B,C,A,D",
    },
    "Q2": {
        "question": "What action is being performed in the video?",
        "opt_a": "Swimming",
        "opt_b": "Running",
        "opt_c": "Cooking",
        "opt_d": "Reading",
        "context": (
            "A person is standing in a kitchen. They are holding a wooden spoon and "
            "stirring something in a large pot on the stove. Steam is visible rising from the pot. "
            "Various cooking utensils and ingredients are visible on the counter."
        ),
        "expected": "C,B,D,A",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., qwen3:32b)")
    parser.add_argument("--question", required=True, help="Question ID (Q1, Q2)")
    args = parser.parse_args()

    test_q = TEST_QUESTIONS[args.question]

    print(f"\nModel: {args.model}")
    print(f"Question: {test_q['question']}")
    print(f"Expected: {test_q['expected']}\n")

    client = OllamaClient(model=args.model, temperature=0.0)

    prompt = _MC_COT_TMPL.format(
        question=test_q["question"],
        opt_a=test_q["opt_a"],
        opt_b=test_q["opt_b"],
        opt_c=test_q["opt_c"],
        opt_d=test_q["opt_d"],
    )
    full_prompt = f"{test_q['context']}\n\n{prompt}"

    t0 = time.time()
    response = client.generate_with_retry(
        full_prompt, system=_MC_COT_SYSTEM, temperature=0.0, think=False
    )
    elapsed = time.time() - t0

    ranking = parse_mc_ranking(response)

    print(f"Ranking: {ranking}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Correct: {'✓' if ','.join(ranking) == test_q['expected'] else '✗'}")
    print(f"\nFull response:\n{response}\n")


if __name__ == "__main__":
    main()
