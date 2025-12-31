#!/usr/bin/env python3
"""
Run reverse inference experiment on public figures.

This attempts to infer the cognitive substrate from observed writing.
"""

from cognitive_gen.reverse_inference import (
    run_inference_experiment,
    PUBLIC_FIGURES,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Infer cognitive states from public figures' writing"
    )
    parser.add_argument(
        "figure",
        choices=list(PUBLIC_FIGURES.keys()),
        help="Which public figure to analyze"
    )
    parser.add_argument(
        "--hypotheses", "-n",
        type=int,
        default=30,
        help="Number of hypotheses to test (default: 30)"
    )
    args = parser.parse_args()

    print(f"\nAvailable figures: {', '.join(PUBLIC_FIGURES.keys())}")
    print(f"Analyzing: {args.figure}\n")

    result = run_inference_experiment(args.figure, args.hypotheses)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: results/reverse_{args.figure}.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
