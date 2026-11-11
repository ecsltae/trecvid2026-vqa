#!/usr/bin/env python3
"""
Compare qwen3:32b vs qwen3.5:122b for VQA MC tasks.

Tests both models on the same question to evaluate quality improvement
with the larger model.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient
from systems.vqa.run_vqa_mc import _MC_COT_SYSTEM, _MC_COT_TMPL, parse_mc_ranking

# Test questions (diverse set)
TEST_QUESTIONS = [
    {
        "id": "Q1",
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
        "rationale": "Blue car is central, trees in background, bicycle not present, dog not present",
    },
    {
        "id": "Q2",
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
        "rationale": "Cooking is clearly happening, running not visible, reading not shown, swimming impossible in kitchen",
    },
    {
        "id": "Q3",
        "question": "What weather condition is shown in the video?",
        "opt_a": "Snowing",
        "opt_b": "Sunny",
        "opt_c": "Raining",
        "opt_d": "Foggy",
        "context": (
            "The video shows an outdoor scene. Water droplets are falling from the sky. "
            "People are carrying umbrellas. The ground is wet and there are puddles visible. "
            "The sky is overcast with dark clouds."
        ),
        "expected": "C,D,A,B",
        "rationale": "Rain is clearly shown, fog could explain overcast, snow less likely given umbrellas, sunny contradicts clouds",
    },
]


def test_model(model_name, test_q):
    """Test a single model on a single question."""
    client = OllamaClient(model=model_name, temperature=0.0)

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

    return {
        "response": response,
        "ranking": ranking,
        "elapsed": elapsed,
    }


def compare_models():
    """Compare qwen3:32b vs qwen3.5:122b on all test questions."""
    models = ["qwen3:32b", "qwen3.5:122b"]

    print("\n" + "=" * 80)
    print("VQA MODEL COMPARISON: qwen3:32b vs qwen3.5:122b")
    print("=" * 80)

    results = {model: [] for model in models}

    for test_q in TEST_QUESTIONS:
        print(f"\n{'=' * 80}")
        print(f"Question {test_q['id']}: {test_q['question']}")
        print(f"Context: {test_q['context'][:100]}...")
        print(f"Expected ranking: {test_q['expected']}")
        print(f"Rationale: {test_q['rationale']}")
        print(f"{'=' * 80}\n")

        for model_name in models:
            print(f"\nTesting {model_name}...")
            result = test_model(model_name, test_q)
            results[model_name].append(result)

            print(f"  Ranking: {result['ranking']}")
            print(f"  Time: {result['elapsed']:.2f}s")
            print(f"  Correct: {'✓' if result['ranking'] == test_q['expected'] else '✗'}")
            print(f"\n  Response preview (first 300 chars):")
            print(f"  {result['response'][:300]}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_name in models:
        correct = sum(
            1 for i, r in enumerate(results[model_name])
            if r["ranking"] == TEST_QUESTIONS[i]["expected"]
        )
        avg_time = sum(r["elapsed"] for r in results[model_name]) / len(results[model_name])

        print(f"\n{model_name}:")
        print(f"  Accuracy: {correct}/{len(TEST_QUESTIONS)} ({100*correct/len(TEST_QUESTIONS):.1f}%)")
        print(f"  Avg time: {avg_time:.2f}s")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_models()
