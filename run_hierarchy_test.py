#!/usr/bin/env python3
"""Test hierarchical architecture: cosmos → body → text."""

import time
from cognitive_gen.generator import CognitiveGenerator
from cognitive_gen.detector import AIDetector
from cognitive_gen.hierarchical_context import (
    generate_hierarchical_context,
    generate_compressed_hierarchy,
    generate_grounded_cosmos,
    generate_social_to_body,
    generate_species_body,
    generate_cascading_context,
)
from cognitive_gen.refined_context import generate_minimal_animal_context


def test_hierarchy(samples: int = 5):
    """Test hierarchical variants against the previous best."""

    print("=" * 70)
    print("HIERARCHICAL ARCHITECTURE TEST")
    print("Cosmos → Existence → Species → Tribe → Self → Body → Moment → Text")
    print("=" * 70)

    generator = CognitiveGenerator()
    detector = AIDetector()

    variants = {
        # Previous best for comparison
        "minimal_animal (baseline)": generate_minimal_animal_context,

        # Hierarchical variants
        "full_hierarchy": generate_hierarchical_context,
        "compressed (cosmos→species→body→moment)": generate_compressed_hierarchy,
        "grounded_cosmos (cosmos→existence→body→moment)": generate_grounded_cosmos,
        "social_to_body (tribe→self→body→moment)": generate_social_to_body,
        "species_body (species→body→moment)": generate_species_body,
        "cascade (explicit pressure flow)": generate_cascading_context,
    }

    results = {}

    for name, gen_func in variants.items():
        print(f"\n--- {name.upper()} ---")
        scores = []
        best_text = ""
        best_score = 1.0
        best_context = None

        for i in range(samples):
            context = gen_func()
            result = generator.generate_with_context(
                "creative_fiction",
                context=context,
                include_inner_monologue=True,
            )
            score = detector.get_ai_score(result.surface_text)
            scores.append(score)
            print(f"  {i+1}: AI prob = {score:.3f}")

            if score < best_score:
                best_score = score
                best_text = result.surface_text
                best_context = context

            time.sleep(0.3)

        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        results[name] = {
            "mean": mean,
            "std": std,
            "min": min(scores),
            "max": max(scores),
            "best_text": best_text,
            "best_context": str(best_context.to_prompt())[:200] if best_context else "",
        }
        print(f"  Mean: {mean:.3f} (±{std:.3f}) | Best: {min(scores):.3f}")

    # Rank results
    ranked = sorted(results.items(), key=lambda x: x[1]["mean"])

    print("\n" + "=" * 70)
    print("RANKINGS (by mean)")
    print("=" * 70)

    for i, (name, data) in enumerate(ranked):
        indicator = "***" if data["min"] < 0.05 else "**" if data["min"] < 0.1 else "*" if data["min"] < 0.2 else ""
        print(f"\n{i+1}. {name}")
        print(f"   Mean: {data['mean']:.3f} (±{data['std']:.3f}) | Min: {data['min']:.3f} | Max: {data['max']:.3f} {indicator}")

    # Show best samples from top 3
    print("\n" + "=" * 70)
    print("BEST SAMPLES")
    print("=" * 70)

    # Find overall best
    best_overall = min(results.items(), key=lambda x: x[1]["min"])
    print(f"\n### BEST OVERALL: {best_overall[0]} (score: {best_overall[1]['min']:.3f})")
    print(f"\n{best_overall[1]['best_text'][:800]}...")

    # Show runner up if different
    sorted_by_min = sorted(results.items(), key=lambda x: x[1]["min"])
    if sorted_by_min[1][0] != best_overall[0]:
        runner_up = sorted_by_min[1]
        print(f"\n### RUNNER UP: {runner_up[0]} (score: {runner_up[1]['min']:.3f})")
        print(f"\n{runner_up[1]['best_text'][:600]}...")

    return results, ranked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-n", type=int, default=5)
    args = parser.parse_args()

    test_hierarchy(samples=args.samples)
