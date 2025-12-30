#!/usr/bin/env python3
"""CLI entry point for running the cognitive context experiment."""

import argparse
import sys

from cognitive_gen import run_experiment
from cognitive_gen.experiment import print_sample_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Test whether cognitive context conditioning reduces AI detection scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 2 samples per condition
  python run_experiment.py --samples 2

  # Full experiment with 10 samples
  python run_experiment.py --samples 10

  # Test only personal essays
  python run_experiment.py --samples 5 --types personal_essay

  # Higher temperature for more variation
  python run_experiment.py --samples 5 --temperature 1.2
        """,
    )

    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=5,
        help="Number of samples per condition per text type (default: 5)",
    )

    parser.add_argument(
        "--types",
        "-t",
        nargs="+",
        choices=["personal_essay", "creative_fiction", "email", "message"],
        default=None,
        help="Text types to test (default: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--show-samples",
        "-s",
        action="store_true",
        help="Print example samples after experiment",
    )

    args = parser.parse_args()

    try:
        results = run_experiment(
            samples_per_condition=args.samples,
            text_types=args.types,
            output_dir=args.output,
            temperature=args.temperature,
            verbose=not args.quiet,
        )

        if args.show_samples:
            for text_type in results.text_types:
                print_sample_comparison(results, text_type)

        # Return exit code based on significance
        # 0 = significant result, 1 = not significant
        sys.exit(0 if results.p_value < 0.05 else 1)

    except KeyboardInterrupt:
        print("\nExperiment interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
