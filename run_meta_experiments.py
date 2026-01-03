#!/usr/bin/env python3
"""Run Meta Search experiments: Meta V1Mode and Complete Meta Search."""

import json
import sys
from datetime import datetime

from cognitive_gen.meta_search import MetaSearch, MetaScorer, MetaHypothesisGenerator

WRITERS = ["kafka", "nietzsche", "hemingway"]
RUNS_PER_WRITER = 3
GENERATIONS = 25
POPULATION_SIZE = 30


def run_experiments(method: str):
    """Run experiments for a given method.

    Args:
        method: Either 'meta_v1mode' or 'meta_complete'
    """
    use_v1_mode = method == "meta_v1mode"
    use_v1_format = use_v1_mode  # Scorer format matches search mode

    print(f"\n{'='*70}")
    print(f"RUNNING: {method.upper()}")
    print(f"use_v1_mode={use_v1_mode}, use_v1_format={use_v1_format}")
    print(f"{'='*70}\n")

    # Initialize scorer and generator once
    scorer = MetaScorer(model_name='mistralai/Mistral-7B-v0.3', use_v1_format=use_v1_format)
    generator = MetaHypothesisGenerator(use_local=True)

    all_results = {}

    for writer in WRITERS:
        writer_results = []

        for run in range(1, RUNS_PER_WRITER + 1):
            print(f"\n{'='*70}")
            print(f"{method.upper()} - {writer.upper()} - RUN {run}/{RUNS_PER_WRITER}")
            print(f"{'='*70}")

            search = MetaSearch(
                scorer=scorer,
                generator=generator,
                population_size=POPULATION_SIZE,
                use_v1_mode=use_v1_mode
            )

            res = search.search(writer, generations=GENERATIONS)

            improvement = (res.baseline_perplexity - res.best_perplexity) / res.baseline_perplexity * 100

            result = {
                'run': run,
                'method': method,
                'writer': writer,
                'baseline_ppl': res.baseline_perplexity,
                'best_ppl': res.best_perplexity,
                'improvement': improvement,
                'timestamp': datetime.now().isoformat()
            }
            writer_results.append(result)

            print(f"\n>>> Run {run} Result: {improvement:.1f}% improvement")
            print(json.dumps(result, indent=2))

        mean_improvement = sum(r['improvement'] for r in writer_results) / len(writer_results)
        std_improvement = (sum((r['improvement'] - mean_improvement)**2 for r in writer_results) / len(writer_results)) ** 0.5

        all_results[writer] = {
            'runs': writer_results,
            'mean': mean_improvement,
            'std': std_improvement
        }

        print(f"\n{'='*70}")
        print(f"{writer.upper()} {method.upper()} - COMPLETE")
        print(f"Mean: {mean_improvement:.1f}% (±{std_improvement:.1f}%)")
        print(f"{'='*70}")

    return all_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_meta_experiments.py <meta_v1mode|meta_complete|both>")
        sys.exit(1)

    mode = sys.argv[1]

    results = {}

    if mode in ['meta_v1mode', 'both']:
        results['meta_v1mode'] = run_experiments('meta_v1mode')

    if mode in ['meta_complete', 'both']:
        results['meta_complete'] = run_experiments('meta_complete')

    # Save results
    output_file = f"/home/matt/human/meta_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")

    # Print summary table
    print("\n## Summary Table\n")
    print("| Writer | Meta V1Mode | Meta Complete |")
    print("|--------|-------------|---------------|")

    for writer in WRITERS:
        v1_str = "N/A"
        complete_str = "N/A"

        if 'meta_v1mode' in results and writer in results['meta_v1mode']:
            r = results['meta_v1mode'][writer]
            v1_str = f"{r['mean']:.1f}% (±{r['std']:.1f}%)"

        if 'meta_complete' in results and writer in results['meta_complete']:
            r = results['meta_complete'][writer]
            complete_str = f"{r['mean']:.1f}% (±{r['std']:.1f}%)"

        print(f"| {writer.capitalize()} | {v1_str} | {complete_str} |")


if __name__ == "__main__":
    main()
