#!/usr/bin/env python3
"""
Test few-shot MC prompting vs zero-shot CoT prompting.

Since we don't have real video frames yet, this tests the prompt quality
by comparing response structure and reasoning quality on synthetic examples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient
from systems.vqa.run_vqa_mc import _MC_COT_SYSTEM, _MC_COT_TMPL, parse_mc_ranking
from systems.vqa.prompts_fewshot import MC_FEWSHOT_SYSTEM, MC_FEWSHOT_TMPL

# Test question
TEST_Q = {
    "question": "What is the main object visible in the video?",
    "opt_a": "A red bicycle",
    "opt_b": "A blue car",
    "opt_c": "A green tree",
    "opt_d": "A yellow dog",
}

# Simulated visual context (would normally come from frames)
VISUAL_CONTEXT = (
    "The video shows a parked vehicle on a street. "
    "The vehicle has four wheels, windows, doors, and appears to be blue in color. "
    "There are trees visible in the background."
)


def test_zeroshot_cot(client):
    """Test zero-shot CoT prompt (current baseline)."""
    print("=" * 80)
    print("ZERO-SHOT CoT (Baseline)")
    print("=" * 80)

    prompt = _MC_COT_TMPL.format(**TEST_Q)
    full_prompt = f"{VISUAL_CONTEXT}\n\n{prompt}"

    print(f"\nSystem: {_MC_COT_SYSTEM}")
    print(f"\nPrompt:\n{full_prompt}\n")

    response = client.generate_with_retry(
        full_prompt, system=_MC_COT_SYSTEM, temperature=0.0, think=False
    )

    print(f"Response:\n{response}\n")

    ranking = parse_mc_ranking(response)
    print(f"Parsed ranking: {ranking}")
    print()


def test_fewshot(client):
    """Test few-shot prompt (improved)."""
    print("=" * 80)
    print("FEW-SHOT (Improved)")
    print("=" * 80)

    prompt = MC_FEWSHOT_TMPL.format(**TEST_Q)
    full_prompt = f"{VISUAL_CONTEXT}\n\n{prompt}"

    print(f"\nSystem: {MC_FEWSHOT_SYSTEM}")
    print(f"\nPrompt length: {len(full_prompt)} chars")
    print(f"\n[Showing last 500 chars of prompt]")
    print(f"{full_prompt[-500:]}\n")

    response = client.generate_with_retry(
        full_prompt, system=MC_FEWSHOT_SYSTEM, temperature=0.0, think=False
    )

    print(f"Response:\n{response}\n")

    ranking = parse_mc_ranking(response)
    print(f"Parsed ranking: {ranking}")
    print()


def main():
    print("\n" + "=" * 80)
    print("VQA MC FEW-SHOT PROMPT TEST")
    print("=" * 80)
    print(f"\nTest question: {TEST_Q['question']}")
    print(f"Visual context: {VISUAL_CONTEXT}")
    print()

    # Use a smaller model for quick testing
    client = OllamaClient(model="qwen3:32b", temperature=0.0)

    # Test both approaches
    test_zeroshot_cot(client)
    test_fewshot(client)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Expected ranking: B (blue car) > C (tree) > A (bicycle) > D (dog)")
    print("Few-shot should show more structured reasoning and clearer ranking.")
    print()


if __name__ == "__main__":
    main()
