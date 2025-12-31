#!/usr/bin/env python3
"""
Run reverse inference using perplexity scoring.

This measures how well each cognitive hypothesis predicts the target text
using perplexity (how "surprised" the model is by the text).

Lower perplexity = hypothesis better explains the writing.
"""

from cognitive_gen.reverse_inference_perplexity import (
    run_perplexity_experiment,
    PUBLIC_FIGURES,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Infer cognitive states using perplexity scoring"
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
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt2-medium",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Model size (default: gpt2-medium)"
    )
    args = parser.parse_args()

    print(f"\nAvailable figures: {', '.join(PUBLIC_FIGURES.keys())}")
    print(f"Analyzing: {args.figure}")
    print(f"Model: {args.model}\n")

    result = run_perplexity_experiment(
        args.figure,
        args.hypotheses,
        args.model,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Baseline perplexity: {result['baseline_perplexity']:.2f}")
    print(f"Best perplexity: {result['best_perplexity']:.2f}")
    print(f"Best improvement: {result['best_improvement']:.1f}%")
    print(f"Results saved to: results/reverse_ppl_{args.figure}.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
