#!/usr/bin/env python3
"""
Run genetic search for cognitive hypotheses.

This uses a genetic algorithm with:
- Linkage learning (dimensions that work together)
- Semantic mutation (Claude-based meaningful variations)
- Adaptive exploration (more randomness when stuck)
- Fitness sharing (maintain population diversity)

Example usage:
    python run_genetic_search.py grief_post --version v1 --generations 15
    python run_genetic_search.py hemingway --version v2 --population 30
"""

import argparse
import json
import os

from cognitive_gen.genetic_search import run_genetic_search, ALL_WRITERS


def main():
    parser = argparse.ArgumentParser(
        description="Genetic search for cognitive hypotheses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s grief_post                    # Run with defaults (v1, 20 pop, 15 gen)
  %(prog)s hemingway --version v2        # Use expanded cognitive architecture
  %(prog)s career_post -p 30 -g 25       # Larger search
  %(prog)s orwell -m gpt2-xl             # Use smaller/faster model
        """,
    )

    parser.add_argument(
        "figure",
        choices=list(ALL_WRITERS.keys()),
        help="Writer to analyze",
    )

    parser.add_argument(
        "--version", "-v",
        choices=["v1", "v2"],
        default="v1",
        help="Cognitive state architecture: v1 (20 dims) or v2 (30+ dims). Default: v1",
    )

    parser.add_argument(
        "--population", "-p",
        type=int,
        default=20,
        help="Population size (default: 20)",
    )

    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=15,
        help="Maximum generations (default: 15)",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Perplexity model (default: mistralai/Mistral-7B-v0.3)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: results/genetic_{version}_{figure}.json)",
    )

    parser.add_argument(
        "--progress",
        type=str,
        default="progress.txt",
        help="Progress file for live monitoring with 'watch cat progress.txt' (default: progress.txt)",
    )

    args = parser.parse_args()

    print(f"\nGenetic Search for Cognitive Hypotheses")
    print(f"=" * 50)
    print(f"Figure: {args.figure}")
    print(f"Architecture: {args.version} ({'20 dimensions' if args.version == 'v1' else '30+ dimensions'})")
    print(f"Population: {args.population}")
    print(f"Max generations: {args.generations}")
    print(f"Perplexity model: {args.model}")
    print(f"Progress file: {args.progress}")
    print(f"=" * 50)
    print(f"\nMonitor progress in another terminal with:")
    print(f"  watch -n 1 cat {args.progress}")
    print(f"=" * 50 + "\n")

    result = run_genetic_search(
        args.figure,
        state_version=args.version,
        population_size=args.population,
        max_generations=args.generations,
        model_name=args.model,
        progress_file=args.progress,
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = args.output or f"results/genetic_{args.version}_{args.figure}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Best improvement: {result['best_improvement']:.1f}%")
    print(f"Generations run: {result['generations']}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
