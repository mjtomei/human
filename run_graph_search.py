#!/usr/bin/env python3
"""
Run graph-based cognitive hypothesis search.

Usage:
    python run_graph_search.py [--generations N] [--population N] [--neighbors N]

Resume from checkpoint:
    python run_graph_search.py --resume [--checkpoint PATH]

List checkpoints:
    python run_graph_search.py --list-checkpoints

Monitor progress:
    watch -n 2 cat /tmp/graph_search_live.txt
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Graph-based cognitive hypothesis search"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=100,
        help="Number of generations (default: 100)"
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=15,
        help="Population per location (default: 15)"
    )
    parser.add_argument(
        "--neighbors", "-n",
        type=int,
        default=3,
        help="Number of neighbors for sparse graph (default: 3)"
    )
    parser.add_argument(
        "--epoch-length", "-e",
        type=int,
        default=5,
        help="Generations between migration epochs (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (default: results/graph_search_<timestamp>.json)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/tmp/graph_search_checkpoints",
        help="Directory for checkpoints (default: /tmp/graph_search_checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N generations (default: 1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file to resume from"
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit"
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Limit number of locations (for testing)"
    )

    args = parser.parse_args()

    # Handle list-checkpoints
    if args.list_checkpoints:
        from cognitive_gen.graph_search.checkpoint import CheckpointManager
        manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
        checkpoints = manager.list_checkpoints()

        if not checkpoints:
            print("No checkpoints found.")
            return 0

        print("Available checkpoints:")
        print("-" * 70)
        for cp in checkpoints:
            print(f"  Gen {cp['generation']:4d} | {cp['timestamp']} | "
                  f"best={cp['best_fitness']*100:5.1f}% | locs={cp['n_locations']}")
            print(f"         {cp['path']}")
        print("-" * 70)
        return 0

    # Print header
    print("=" * 70)
    print("GRAPH-BASED COGNITIVE HYPOTHESIS SEARCH")
    print("=" * 70)
    print()
    print(f"Generations: {args.generations}")
    print(f"Population per location: {args.population}")
    print(f"Neighbors: {args.neighbors}")
    print(f"Epoch length: {args.epoch_length}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    if args.resume or args.checkpoint:
        print(f"Resuming from: {args.checkpoint or 'latest checkpoint'}")
    print()
    print("Monitor progress: watch -n 2 cat /tmp/graph_search_live.txt")
    print()
    print("=" * 70)

    # Import here to delay model loading until after arg parsing
    from cognitive_gen.graph_search.graph_search import run_graph_search

    # Run search
    result = run_graph_search(
        total_generations=args.generations,
        population_per_location=args.population,
        n_neighbors=args.neighbors,
        epoch_length=args.epoch_length,
        verbose=not args.quiet,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume or (args.checkpoint is not None),
        checkpoint_path=args.checkpoint,
        max_locations=args.max_locations,
    )

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Best location: {result.best_location}")
    print(f"Best fitness: {result.best_fitness * 100:.2f}% improvement")
    print(f"Baseline perplexity: {result.baseline_ppl:.3f}")
    print(f"Final locations: {result.final_n_locations} ({result.final_n_meta_locations} meta)")
    print(f"Meta-locations created: {len(result.meta_locations_created)}")
    print()

    if result.best_hypothesis:
        print("Best hypothesis dimensions:")
        for dim in result.best_hypothesis.active_dimensions:
            value = result.best_hypothesis.values.get(dim, "")
            print(f"  {dim}:")
            print(f"    {value}")
        print()

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"graph_search_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
