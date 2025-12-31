#!/usr/bin/env python3
"""
Run comprehensive genetic search on Mat Tomei's essays.
Clusters essays by theme and uses all available text.
"""

import os
import sys

# Define essay clusters by theme
CLUSTERS = {
    "fractal_power": {
        "name": "Fractal Hierarchies & Power",
        "essays": ["violence", "currency", "regulators", "state", "slaves"],
        "description": "Power dynamics, fractal hierarchies, political economy"
    },
    "cosmos_theology": {
        "name": "God, Cosmos & Entropy",
        "essays": ["god", "separation", "turtles", "demons"],
        "description": "God, simulation theory, entropy, good vs evil, spiritual hierarchy"
    },
    "tech_consciousness": {
        "name": "Technology & Consciousness",
        "essays": ["horus", "water", "memes", "llms"],
        "description": "Machine learning, consciousness, chaos, information theory"
    },
    "ethics_refinement": {
        "name": "Personal Ethics & Refinement",
        "essays": ["shame", "tantra", "help", "selfhelp", "strength", "honesty"],
        "description": "Self-improvement, morality, psychological transformation"
    },
    "short_pieces": {
        "name": "Shorter Reflections",
        "essays": ["void", "transplants", "food", "reproduction", "rape", "right", "communes", "whores", "reason"],
        "description": "Various shorter essays on specific topics"
    }
}

def load_essays(essay_dir="/tmp/mtomei_essays"):
    """Load all essays from disk."""
    essays = {}
    for filename in os.listdir(essay_dir):
        if filename.endswith('.txt'):
            name = filename[:-4]
            with open(os.path.join(essay_dir, filename), 'r') as f:
                essays[name] = f.read().strip()
    return essays

def create_cluster_samples(essays, clusters):
    """Create writing samples from essay clusters."""
    samples = []

    for cluster_id, cluster_info in clusters.items():
        cluster_text = []
        for essay_name in cluster_info["essays"]:
            if essay_name in essays:
                cluster_text.append(essays[essay_name])

        if cluster_text:
            # Combine essays in this cluster
            combined = "\n\n".join(cluster_text)
            samples.append(combined)
            print(f"  {cluster_info['name']}: {len(combined)} chars, {len(combined.split())} words")

    return samples

def main():
    print("=" * 70)
    print("COMPREHENSIVE GENETIC SEARCH: Mat Tomei")
    print("=" * 70)

    # Load essays
    print("\nLoading essays...")
    essays = load_essays()
    print(f"Loaded {len(essays)} essays")

    # Create cluster samples
    print("\nCreating thematic clusters:")
    samples = create_cluster_samples(essays, CLUSTERS)

    total_words = sum(len(s.split()) for s in samples)
    print(f"\nTotal: {len(samples)} clusters, {total_words} words")

    # Update reverse_inference.py with new samples
    print("\nUpdating reverse_inference.py with full samples...")

    # Import and modify
    sys.path.insert(0, '/home/matt/human')

    from cognitive_gen.reverse_inference import PUBLIC_FIGURES
    from cognitive_gen.genetic_search import run_genetic_search

    # Create a custom entry with full samples
    mtomei_full = {
        "name": "Mat Tomei (Full Corpus)",
        "source": "All essays from mtomei.com clustered by theme",
        "samples": samples
    }

    # Run the genetic search
    print("\n" + "=" * 70)
    print("Starting genetic search with full corpus...")
    print("=" * 70)

    result = run_genetic_search(
        writing_samples=samples,
        figure_name="Mat Tomei (Full Corpus)",
        state_version="v2",
        population_size=30,
        max_generations=30,  # More generations for larger corpus
        model_name="mistralai/Mistral-7B-v0.3",
        progress_file="progress_full.txt"
    )

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Baseline perplexity: {result['baseline_perplexity']:.3f}")
    print(f"Best perplexity: {result['best_perplexity']:.3f}")
    print(f"Improvement: {result['improvement']*100:.1f}%")

    print("\n" + "=" * 70)
    print("BEST HYPOTHESIS")
    print("=" * 70)
    for dim, value in result['best_hypothesis'].items():
        if value:
            print(f"  {dim}: {value}")

    # Save results
    import json
    with open("results/genetic_v2_mtomei_full.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to results/genetic_v2_mtomei_full.json")

if __name__ == "__main__":
    main()
