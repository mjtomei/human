#!/usr/bin/env python3
"""Test the refined animal-based architecture."""

import time
from cognitive_gen.generator import CognitiveGenerator
from cognitive_gen.detector import AIDetector
from cognitive_gen.refined_context import (
    generate_animal_context,
    generate_minimal_animal_context,
    generate_predator_context,
    generate_prey_context,
    generate_somatic_context,
)


def test_refined_architecture(samples: int = 5):
    """Test all variants of the refined architecture."""

    print("=" * 70)
    print("TESTING REFINED ANIMAL ARCHITECTURE")
    print("=" * 70)

    generator = CognitiveGenerator()
    detector = AIDetector()

    variants = {
        "full_animal": generate_animal_context,
        "minimal_animal": generate_minimal_animal_context,
        "predator": generate_predator_context,
        "prey": generate_prey_context,
        "somatic": generate_somatic_context,
    }

    results = {}

    for name, gen_func in variants.items():
        print(f"\n--- {name.upper()} ---")
        scores = []
        best_text = ""
        best_score = 1.0

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

            time.sleep(0.3)

        mean = sum(scores) / len(scores)
        results[name] = {
            "mean": mean,
            "min": min(scores),
            "max": max(scores),
            "best_text": best_text,
        }
        print(f"  Mean: {mean:.3f} | Best: {min(scores):.3f}")

    # Rank results
    ranked = sorted(results.items(), key=lambda x: x[1]["mean"])

    print("\n" + "=" * 70)
    print("RANKINGS")
    print("=" * 70)

    for i, (name, data) in enumerate(ranked):
        print(f"\n{i+1}. {name}")
        print(f"   Mean: {data['mean']:.3f} | Min: {data['min']:.3f} | Max: {data['max']:.3f}")

    # Show best overall sample
    best_overall = min(results.items(), key=lambda x: x[1]["min"])
    print("\n" + "=" * 70)
    print(f"BEST SAMPLE ({best_overall[0]}, score: {best_overall[1]['min']:.3f})")
    print("=" * 70)
    print(f"\n{best_overall[1]['best_text']}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-n", type=int, default=5)
    args = parser.parse_args()

    test_refined_architecture(samples=args.samples)
