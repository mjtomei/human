#!/usr/bin/env python3
"""Search for the cognitive architectures that minimize detection."""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from cognitive_gen.generator import CognitiveGenerator, WRITING_PROMPTS
from cognitive_gen.detector import AIDetector
from cognitive_gen.deep_context import DeepCognitiveContext


# Curated combinations to test - going deeper
ABYSS_CONFIGURATIONS = {
    "death_and_body": {
        "name": "Mortality + Somatic",
        "primal_drives": [
            "The body already decaying while still alive",
            "Revulsion at the flesh that contains consciousness",
        ],
        "death_awareness": "Every word written against the certainty of becoming nothing",
        "somatic_knowledge": "The body knows the truth the mind keeps rewriting",
        "existential_substrate": "Meaning as a desperate fiction over the void",
    },
    "abandonment_core": {
        "name": "Primal Abandonment",
        "primal_drives": [
            "The infant's scream that never stopped, only went silent",
            "Love as a debt that can never be repaid",
        ],
        "belonging_hunger": "The certainty of fundamental unlovability beneath all performance",
        "preverbal_shame": "Existing as a burden, a mistake, an imposition",
        "unnamed_pursuer": "The rejection that's always coming, has always already happened",
    },
    "feral_witness": {
        "name": "The Animal Watching",
        "primal_drives": [
            "The predator that watches from behind your eyes",
            "Hunger that politeness barely contains",
        ],
        "dissociated_observer": "The one who sees you pretending to be a person",
        "feral_self": "The animal that would tear everything down if given permission",
        "existential_substrate": "Civilization as a thin membrane over chaos",
    },
    "inherited_void": {
        "name": "Ancestral Emptiness",
        "primal_drives": [
            "Grief inherited from people who died before you were born",
            "Patterns repeating without origin or end",
        ],
        "ancestral_residue": "The dead speaking through your gestures, your fears, your loves",
        "existential_substrate": "Time as circular, not linear - everything has already happened",
        "death_awareness": "You are the temporary arrangement of matter that briefly noticed itself",
    },
    "dissolution": {
        "name": "Ego Dissolution",
        "primal_drives": [
            "The drive to merge with another until the self disappears",
            "The terror and longing of boundaries dissolving",
        ],
        "dissociated_observer": "The constructed self watching itself construct itself",
        "existential_substrate": "The 'I' as a story the brain tells to organize chaos",
        "feral_self": "What remains when the story stops",
    },
    "pure_void": {
        "name": "The Void Itself",
        "primal_drives": [],
        "existential_substrate": "The silence beneath thought. The nothing that everything floats in.",
        "death_awareness": "Consciousness as a brief flicker between two infinite darknesses",
        "unnamed_pursuer": "The ending that was always there, waiting",
        "feral_self": "The scream with no mouth",
    },
}


def create_context_from_config(config: dict) -> DeepCognitiveContext:
    """Create a DeepCognitiveContext from a configuration dict."""
    return DeepCognitiveContext(
        primal_drives=config.get("primal_drives", []),
        death_awareness=config.get("death_awareness", ""),
        belonging_hunger=config.get("belonging_hunger", ""),
        preverbal_shame=config.get("preverbal_shame", ""),
        dissociated_observer=config.get("dissociated_observer", ""),
        somatic_knowledge=config.get("somatic_knowledge", ""),
        existential_substrate=config.get("existential_substrate", ""),
        ancestral_residue=config.get("ancestral_residue", ""),
        unnamed_pursuer=config.get("unnamed_pursuer", ""),
        feral_self=config.get("feral_self", ""),
    )


def search_abyss(samples_per_config: int = 4, verbose: bool = True):
    """Search through abyss configurations for minimum detection."""

    if verbose:
        print("=" * 70)
        print("SEARCHING THE ABYSS")
        print("Testing curated configurations of primal cognitive architecture")
        print("=" * 70)

    generator = CognitiveGenerator()
    detector = AIDetector()

    results = {}

    for config_id, config in ABYSS_CONFIGURATIONS.items():
        if verbose:
            print(f"\n--- {config['name'].upper()} ---")

        context = create_context_from_config(config)
        samples = []

        for i in range(samples_per_config):
            result = generator.generate_with_context(
                "creative_fiction",
                context=context,
                include_inner_monologue=True,
            )
            score = detector.get_ai_score(result.surface_text)
            samples.append({
                "score": score,
                "text": result.surface_text,
                "inner_monologue": result.inner_monologue,
            })
            if verbose:
                print(f"  {i+1}: AI prob = {score:.3f}")
            time.sleep(0.3)

        scores = [s["score"] for s in samples]
        results[config_id] = {
            "name": config["name"],
            "config": {k: v for k, v in config.items() if k != "name"},
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "samples": samples,
        }

    # Rank by mean score
    ranked = sorted(results.items(), key=lambda x: x[1]["mean"])

    if verbose:
        print("\n" + "=" * 70)
        print("RANKINGS (lowest detection = most human)")
        print("=" * 70)

        for i, (config_id, data) in enumerate(ranked):
            print(f"\n{i+1}. {data['name']}")
            print(f"   Mean: {data['mean']:.3f} | Min: {data['min']:.3f} | Max: {data['max']:.3f}")

        # Show the best sample overall
        all_samples = []
        for config_id, data in results.items():
            for s in data["samples"]:
                s["config"] = data["name"]
                all_samples.append(s)

        all_samples.sort(key=lambda x: x["score"])

        print("\n" + "=" * 70)
        print("BEST SAMPLE FOUND")
        print("=" * 70)
        best = all_samples[0]
        print(f"\nConfig: {best['config']}")
        print(f"AI Probability: {best['score']:.3f}")
        print(f"\n{best['text']}")

        if best.get("inner_monologue"):
            print("\n--- INNER MONOLOGUE ---")
            print(best["inner_monologue"][:600])

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    output = {
        "timestamp": timestamp,
        "rankings": [(k, {"name": v["name"], "mean": v["mean"], "min": v["min"]})
                     for k, v in ranked],
        "results": {k: {**v, "samples": v["samples"][:2]} for k, v in results.items()},
    }
    with open(f"results/abyss_search_{timestamp}.json", "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\nResults saved to results/abyss_search_{timestamp}.json")

    return results, ranked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-n", type=int, default=4)
    args = parser.parse_args()

    search_abyss(samples_per_config=args.samples)
