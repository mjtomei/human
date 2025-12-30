#!/usr/bin/env python3
"""Experiment with deep cognitive contexts - searching for what minimizes detection."""

import json
import time
from datetime import datetime
from pathlib import Path

from scipy import stats

from cognitive_gen.generator import CognitiveGenerator, WRITING_PROMPTS
from cognitive_gen.detector import AIDetector
from cognitive_gen.context import generate_random_context
from cognitive_gen.deep_context import generate_deep_context, generate_minimal_deep_context


def run_deep_experiment(
    samples_per_condition: int = 5,
    text_type: str = "creative_fiction",
    verbose: bool = True,
):
    """
    Compare baseline, standard cognitive, and deep cognitive contexts.
    """
    if verbose:
        print("=" * 70)
        print("DEEP COGNITIVE CONTEXT EXPERIMENT")
        print("Searching the depths of subconsciousness...")
        print("=" * 70)

    generator = CognitiveGenerator()
    detector = AIDetector()

    results = {
        "baseline": [],
        "cognitive": [],
        "deep": [],
        "abyss_only": [],  # Minimal - just primal/existential, no surface
    }

    prompt = WRITING_PROMPTS[text_type]

    # Baseline
    if verbose:
        print(f"\n--- BASELINE (no context) ---")
    for i in range(samples_per_condition):
        result = generator.generate_baseline(text_type)
        score = detector.get_ai_score(result.surface_text)
        results["baseline"].append({
            "score": score,
            "text": result.surface_text,
        })
        if verbose:
            print(f"  {i+1}: AI prob = {score:.3f}")
        time.sleep(0.3)

    # Standard cognitive context
    if verbose:
        print(f"\n--- STANDARD COGNITIVE ---")
    for i in range(samples_per_condition):
        context = generate_random_context(text_type)
        result = generator.generate_with_context(text_type, context=context)
        score = detector.get_ai_score(result.surface_text)
        results["cognitive"].append({
            "score": score,
            "text": result.surface_text,
            "context": str(context.explicit_goals) + " | " + context.emotional_state,
        })
        if verbose:
            print(f"  {i+1}: AI prob = {score:.3f}")
        time.sleep(0.3)

    # Deep cognitive context (all layers)
    if verbose:
        print(f"\n--- DEEP COGNITIVE (full subconscious) ---")
    for i in range(samples_per_condition):
        context = generate_deep_context(text_type)
        result = generator.generate_with_context(text_type, context=context)
        score = detector.get_ai_score(result.surface_text)
        results["deep"].append({
            "score": score,
            "text": result.surface_text,
            "primal": context.primal_drives[:2],
            "death": context.death_awareness[:50],
            "abyss": context.existential_substrate[:50],
        })
        if verbose:
            print(f"  {i+1}: AI prob = {score:.3f}")
            print(f"       Primal: {context.primal_drives[0][:40]}...")
        time.sleep(0.3)

    # Abyss only - minimal surface, maximum depth
    if verbose:
        print(f"\n--- ABYSS ONLY (pure subconscious) ---")
    for i in range(samples_per_condition):
        context = generate_minimal_deep_context(text_type)
        result = generator.generate_with_context(text_type, context=context)
        score = detector.get_ai_score(result.surface_text)
        results["abyss_only"].append({
            "score": score,
            "text": result.surface_text,
            "existential": context.existential_substrate,
            "feral": context.feral_self,
        })
        if verbose:
            print(f"  {i+1}: AI prob = {score:.3f}")
        time.sleep(0.3)

    # Analysis
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

    summary = {}
    for condition, samples in results.items():
        scores = [s["score"] for s in samples]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        summary[condition] = {"mean": mean, "std": std, "scores": scores}

        if verbose:
            print(f"\n{condition.upper()}:")
            print(f"  Mean AI prob: {mean:.3f} (+/- {std:.3f})")
            print(f"  Range: {min(scores):.3f} - {max(scores):.3f}")

    # Statistical comparisons
    if verbose:
        print("\n--- STATISTICAL COMPARISONS (vs baseline) ---")

    for condition in ["cognitive", "deep", "abyss_only"]:
        t, p = stats.ttest_ind(
            summary["baseline"]["scores"],
            summary[condition]["scores"]
        )
        diff = summary["baseline"]["mean"] - summary[condition]["mean"]
        if verbose:
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"{condition}: diff={diff:+.3f}, p={p:.4f} {sig}")

    # Find best and worst samples across all conditions
    all_samples = []
    for condition, samples in results.items():
        for s in samples:
            s["condition"] = condition
            all_samples.append(s)

    all_samples.sort(key=lambda x: x["score"])

    if verbose:
        print("\n" + "=" * 70)
        print("BEST SAMPLES (lowest AI detection)")
        print("=" * 70)

        for s in all_samples[:3]:
            print(f"\n[{s['condition'].upper()}] AI prob: {s['score']:.3f}")
            print(f"Text: {s['text'][:400]}...")
            if "primal" in s:
                print(f"Primal drives: {s['primal']}")
            if "existential" in s:
                print(f"Existential: {s['existential'][:60]}...")

        print("\n" + "=" * 70)
        print("WORST SAMPLES (highest AI detection)")
        print("=" * 70)

        for s in all_samples[-3:]:
            print(f"\n[{s['condition'].upper()}] AI prob: {s['score']:.3f}")
            print(f"Text: {s['text'][:300]}...")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    with open(f"results/deep_experiment_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "text_type": text_type,
            "samples_per_condition": samples_per_condition,
            "summary": {k: {"mean": v["mean"], "std": v["std"]} for k, v in summary.items()},
            "results": results,
        }, f, indent=2)

    if verbose:
        print(f"\nResults saved to results/deep_experiment_{timestamp}.json")

    return results, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-n", type=int, default=5)
    parser.add_argument("--type", "-t", default="creative_fiction")
    args = parser.parse_args()

    run_deep_experiment(
        samples_per_condition=args.samples,
        text_type=args.type,
    )
