"""
Reverse inference using perplexity as the scoring metric.

Instead of asking Claude to judge similarity, we measure how well
a hypothesis helps predict the target text using perplexity.

Lower perplexity = the model is less "surprised" by the text
= the hypothesis better explains/predicts the writing.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_gen.reverse_inference import (
    CognitiveHypothesis,
    generate_random_hypothesis,
    PRIMARY_DRIVE_POOL,
    HIDDEN_FEAR_POOL,
    UNSPOKEN_DESIRE_POOL,
    SELF_DECEPTION_POOL,
    PUBLIC_FIGURES,
)


class PerplexityScorer:
    """
    Score hypotheses by measuring perplexity of target text
    when conditioned on hypothesis + context.
    """

    def __init__(self, model_name: str = "gpt2-medium"):
        """
        Initialize with a causal language model.

        Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
        Larger = better quality but slower
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def compute_perplexity(
        self,
        context: str,
        target: str,
        max_context_tokens: int = 800,
    ) -> float:
        """
        Compute perplexity of target text given context.

        Perplexity = exp(average negative log likelihood per token)
        Lower = model predicts target better given context
        """
        # Combine context and target
        full_text = context + "\n\n" + target

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        input_ids = encodings.input_ids.to(self.device)

        # Find where target starts
        context_encodings = self.tokenizer(
            context + "\n\n",
            return_tensors="pt",
            truncation=True,
            max_length=max_context_tokens,
        )
        context_length = context_encodings.input_ids.shape[1]

        # Compute loss only on target tokens
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)

            # Get per-token losses
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Only count losses for target tokens (after context)
            if context_length < len(losses):
                target_losses = losses[context_length-1:]
                avg_loss = target_losses.mean().item()
            else:
                avg_loss = losses.mean().item()

            perplexity = math.exp(avg_loss)

        return perplexity

    def score_hypothesis(
        self,
        hypothesis: CognitiveHypothesis,
        context_samples: list[str],
        target_text: str,
    ) -> float:
        """
        Score a hypothesis by measuring perplexity of target.

        Returns negative perplexity (so higher = better, for consistency).
        """
        # Build conditioning context
        hyp_prompt = hypothesis.to_prompt()
        context_text = "\n\n---\n\n".join(context_samples)

        full_context = f"""{hyp_prompt}

The following are samples of writing from someone experiencing this cognitive state:

{context_text}

Here is another sample of their writing:"""

        perplexity = self.compute_perplexity(full_context, target_text)

        # Return negative so higher = better
        return -perplexity

    def compute_baseline_perplexity(
        self,
        context_samples: list[str],
        target_text: str,
    ) -> float:
        """Compute perplexity without any hypothesis (baseline)."""
        context_text = "\n\n---\n\n".join(context_samples)

        full_context = f"""The following are samples of writing from an author:

{context_text}

Here is another sample of their writing:"""

        return self.compute_perplexity(full_context, target_text)


def run_perplexity_experiment(
    figure_key: str,
    n_hypotheses: int = 30,
    model_name: str = "gpt2-medium",
) -> dict:
    """
    Run reverse inference using perplexity scoring.
    """
    if figure_key not in PUBLIC_FIGURES:
        raise ValueError(f"Unknown figure: {figure_key}")

    figure = PUBLIC_FIGURES[figure_key]
    samples = figure["samples"]

    print("=" * 70)
    print(f"REVERSE INFERENCE (PERPLEXITY): {figure['name']}")
    print(f"Source: {figure['source']}")
    print(f"Model: {model_name}")
    print("=" * 70)

    scorer = PerplexityScorer(model_name)

    # Split samples
    context_samples = samples[:-1]
    target_sample = samples[-1]

    # Compute baseline perplexity
    baseline_ppl = scorer.compute_baseline_perplexity(context_samples, target_sample)
    print(f"\nBaseline perplexity (no hypothesis): {baseline_ppl:.2f}")

    # Test hypotheses
    results = []

    for i in range(n_hypotheses):
        hypothesis = generate_random_hypothesis()

        neg_ppl = scorer.score_hypothesis(
            hypothesis,
            context_samples,
            target_sample,
        )
        perplexity = -neg_ppl

        # Improvement over baseline (lower is better for perplexity)
        improvement = (baseline_ppl - perplexity) / baseline_ppl * 100

        results.append({
            "hypothesis": hypothesis,
            "perplexity": perplexity,
            "improvement": improvement,
        })

        marker = "↓" if perplexity < baseline_ppl else "↑"
        print(f"Hypothesis {i+1}/{n_hypotheses}: ppl = {perplexity:.2f} ({marker} {abs(improvement):.1f}%)")

    # Sort by perplexity (lower is better)
    results.sort(key=lambda x: x["perplexity"])

    # Build profile from top hypotheses
    top_k = max(1, n_hypotheses // 5)
    top_results = results[:top_k]

    print(f"\n" + "=" * 70)
    print(f"TOP {top_k} HYPOTHESES (lowest perplexity)")
    print("=" * 70)

    for i, r in enumerate(top_results):
        h = r["hypothesis"]
        print(f"\n#{i+1} - Perplexity: {r['perplexity']:.2f} (↓{r['improvement']:.1f}% vs baseline)")
        if h.primary_drive:
            print(f"  Primary drive: {h.primary_drive}")
        if h.hidden_fear:
            print(f"  Hidden fear: {h.hidden_fear}")
        if h.body_state:
            print(f"  Body state: {h.body_state}")
        if h.predator_aspect:
            print(f"  Predator: {h.predator_aspect}")
        if h.prey_aspect:
            print(f"  Prey: {h.prey_aspect}")
        if h.self_deception:
            print(f"  Self-deception: {h.self_deception}")

    # Aggregate profile
    profile = {
        "primary_drives": {},
        "hidden_fears": {},
        "body_states": {},
        "predator_aspects": {},
        "prey_aspects": {},
        "self_deceptions": {},
    }

    for r in top_results:
        h = r["hypothesis"]
        # Weight by improvement (higher improvement = better)
        weight = max(0.01, r["improvement"])

        if h.primary_drive:
            profile["primary_drives"][h.primary_drive] = \
                profile["primary_drives"].get(h.primary_drive, 0) + weight
        if h.hidden_fear:
            profile["hidden_fears"][h.hidden_fear] = \
                profile["hidden_fears"].get(h.hidden_fear, 0) + weight
        if h.body_state:
            profile["body_states"][h.body_state] = \
                profile["body_states"].get(h.body_state, 0) + weight
        if h.predator_aspect:
            profile["predator_aspects"][h.predator_aspect] = \
                profile["predator_aspects"].get(h.predator_aspect, 0) + weight
        if h.prey_aspect:
            profile["prey_aspects"][h.prey_aspect] = \
                profile["prey_aspects"].get(h.prey_aspect, 0) + weight
        if h.self_deception:
            profile["self_deceptions"][h.self_deception] = \
                profile["self_deceptions"].get(h.self_deception, 0) + weight

    # Normalize
    for key in profile:
        if profile[key]:
            total = sum(profile[key].values())
            profile[key] = {
                k: v/total for k, v in
                sorted(profile[key].items(), key=lambda x: x[1], reverse=True)
            }

    print(f"\n" + "=" * 70)
    print("AGGREGATED COGNITIVE PROFILE")
    print("=" * 70)

    for category, distribution in profile.items():
        if distribution:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item, prob in list(distribution.items())[:3]:
                print(f"  {prob:.1%}: {item}")

    return {
        "figure": figure["name"],
        "baseline_perplexity": baseline_ppl,
        "best_perplexity": results[0]["perplexity"],
        "best_improvement": results[0]["improvement"],
        "profile": profile,
        "top_hypotheses": [
            {
                "perplexity": r["perplexity"],
                "improvement": r["improvement"],
                "primary_drive": r["hypothesis"].primary_drive,
                "hidden_fear": r["hypothesis"].hidden_fear,
                "body_state": r["hypothesis"].body_state,
            }
            for r in top_results
        ],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=list(PUBLIC_FIGURES.keys()))
    parser.add_argument("--hypotheses", "-n", type=int, default=30)
    parser.add_argument("--model", "-m", type=str, default="gpt2-medium",
                       help="Model: gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    args = parser.parse_args()

    result = run_perplexity_experiment(
        args.figure,
        args.hypotheses,
        args.model,
    )

    # Save results
    import os
    os.makedirs("results", exist_ok=True)

    # Convert to JSON-serializable format
    output = {
        "figure": result["figure"],
        "baseline_perplexity": result["baseline_perplexity"],
        "best_perplexity": result["best_perplexity"],
        "best_improvement": result["best_improvement"],
        "profile": result["profile"],
        "top_hypotheses": result["top_hypotheses"],
    }

    with open(f"results/reverse_ppl_{args.figure}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/reverse_ppl_{args.figure}.json")
