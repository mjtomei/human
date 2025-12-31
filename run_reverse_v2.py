#!/usr/bin/env python3
"""
Run iterative reverse inference (v2).

Key improvements over v1:
1. Hypotheses generated from the text itself (not fixed pools)
2. Broader cognitive dimensions (intellectual, moral, not just primal)
3. Iterative refinement based on what works
"""

from cognitive_gen.reverse_inference_v2 import (
    run_iterative_inference,
    ALL_WRITERS,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Iterative reverse inference with generated hypotheses"
    )
    parser.add_argument(
        "figure",
        choices=list(ALL_WRITERS.keys()),
        help="Which writer to analyze (famous or anonymous)"
    )
    parser.add_argument(
        "--initial", "-i",
        type=int,
        default=10,
        help="Initial hypotheses to generate (default: 10)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=3,
        help="Number of refinement iterations (default: 3)"
    )
    parser.add_argument(
        "--refine", "-r",
        type=int,
        default=5,
        help="Hypotheses to generate per refinement (default: 5)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt2-xl",
        help="Perplexity model (default: gpt2-xl, try: microsoft/phi-2)"
    )
    args = parser.parse_args()

    print(f"\nIterative Reverse Inference v2")
    print(f"Figure: {args.figure}")
    print(f"Initial hypotheses: {args.initial}")
    print(f"Refinement iterations: {args.iterations}")
    print(f"Hypotheses per refinement: {args.refine}")
    print(f"Perplexity model: {args.model}\n")

    result = run_iterative_inference(
        args.figure,
        n_initial=args.initial,
        n_iterations=args.iterations,
        n_refine_per_iter=args.refine,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
