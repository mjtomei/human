"""
Reverse inference v2: Iterative hypothesis generation and refinement.

Key improvements:
1. Generate hypotheses from the text itself (not fixed pools)
2. Broader cognitive dimensions (intellectual, moral, political, not just primal)
3. Iterative refinement based on what works
4. Use more recent model for perplexity
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import anthropic

from cognitive_gen.reverse_inference import PUBLIC_FIGURES, REGULAR_WRITERS

# Combine all writers
ALL_WRITERS = {**PUBLIC_FIGURES, **REGULAR_WRITERS}


@dataclass
class CognitiveState:
    """
    A richer cognitive state with multiple dimensions.
    Not just primal - includes intellectual, moral, social, aesthetic.
    """
    # === EMBODIED (from before) ===
    body_state: str = ""
    preverbal_feeling: str = ""

    # === INTELLECTUAL ===
    core_belief: str = ""  # What they hold to be fundamentally true
    intellectual_stance: str = ""  # How they approach ideas
    what_they_notice: str = ""  # What draws their attention

    # === MORAL/ETHICAL ===
    moral_framework: str = ""  # How they judge right/wrong
    what_outrages_them: str = ""  # What they can't tolerate
    what_they_protect: str = ""  # What they feel responsible for

    # === RELATIONAL ===
    stance_toward_reader: str = ""  # How they position vis-a-vis audience
    who_they_write_for: str = ""  # Real or imagined audience
    what_they_want_reader_to_feel: str = ""

    # === TEMPORAL ===
    relationship_to_past: str = ""  # How history weighs on them
    relationship_to_future: str = ""  # Hope, dread, indifference
    sense_of_urgency: str = ""

    # === AESTHETIC ===
    what_they_find_beautiful: str = ""
    what_they_find_ugly: str = ""
    relationship_to_language: str = ""  # Tool, art, weapon, window

    # === HIDDEN ===
    what_they_cant_say_directly: str = ""
    the_wound: str = ""  # The injury that shapes the voice
    the_compensation: str = ""  # How they cope with the wound

    def to_prompt(self) -> str:
        sections = ["=== COGNITIVE STATE ===", "Write as if experiencing:", ""]

        for field_name, value in self.__dict__.items():
            if value:
                label = field_name.replace("_", " ").title()
                sections.append(f"{label}: {value}")

        sections.extend(["", "Let these shape the voice without naming them.", "=== END ==="])
        return "\n".join(sections)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


class HypothesisGenerator:
    """Generate cognitive hypotheses from observed text using Claude."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_initial_hypotheses(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
    ) -> list[CognitiveState]:
        """Generate initial hypotheses by analyzing the writing."""

        samples_text = "\n\n---\n\n".join(writing_samples)

        prompt = f"""Analyze these writing samples and generate {n_hypotheses} different hypotheses about the writer's cognitive/psychological state.

For each hypothesis, consider MULTIPLE dimensions - not just emotions or fears, but also:
- Intellectual: What do they believe? How do they think?
- Moral: What do they find intolerable? What are they protecting?
- Relational: How do they position themselves toward the reader?
- Temporal: How does time (past/future) weigh on them?
- Aesthetic: What is their relationship to language and beauty?
- Hidden: What can't they say directly? What wound shapes them?

Writing samples:
{samples_text}

Generate {n_hypotheses} distinct cognitive state hypotheses. For each, provide values for these fields (leave blank if not applicable):

- body_state: Physical sensation
- preverbal_feeling: Feeling before words
- core_belief: Fundamental truth they hold
- intellectual_stance: How they approach ideas
- what_they_notice: What draws their attention
- moral_framework: How they judge right/wrong
- what_outrages_them: What they can't tolerate
- what_they_protect: What they feel responsible for
- stance_toward_reader: How they position vis-a-vis audience
- who_they_write_for: Real or imagined audience
- what_they_want_reader_to_feel: Intended effect
- relationship_to_past: How history weighs on them
- relationship_to_future: Hope, dread, indifference
- sense_of_urgency: Time pressure
- what_they_find_beautiful: Aesthetic values
- what_they_find_ugly: Aesthetic aversions
- relationship_to_language: Tool, art, weapon, window
- what_they_cant_say_directly: The unspeakable
- the_wound: The injury that shapes the voice
- the_compensation: How they cope with the wound

Return as JSON array of objects. Be specific and grounded in the actual text.
Each hypothesis should emphasize different aspects - don't just vary the same dimension."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON from response
        text = response.content[0].text

        # Find JSON array in response
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            print("Warning: Could not parse JSON from response")
            return []

        try:
            hypotheses_data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error: {e}")
            return []

        # Convert to CognitiveState objects
        hypotheses = []
        for h in hypotheses_data:
            state = CognitiveState(
                body_state=h.get("body_state", ""),
                preverbal_feeling=h.get("preverbal_feeling", ""),
                core_belief=h.get("core_belief", ""),
                intellectual_stance=h.get("intellectual_stance", ""),
                what_they_notice=h.get("what_they_notice", ""),
                moral_framework=h.get("moral_framework", ""),
                what_outrages_them=h.get("what_outrages_them", ""),
                what_they_protect=h.get("what_they_protect", ""),
                stance_toward_reader=h.get("stance_toward_reader", ""),
                who_they_write_for=h.get("who_they_write_for", ""),
                what_they_want_reader_to_feel=h.get("what_they_want_reader_to_feel", ""),
                relationship_to_past=h.get("relationship_to_past", ""),
                relationship_to_future=h.get("relationship_to_future", ""),
                sense_of_urgency=h.get("sense_of_urgency", ""),
                what_they_find_beautiful=h.get("what_they_find_beautiful", ""),
                what_they_find_ugly=h.get("what_they_find_ugly", ""),
                relationship_to_language=h.get("relationship_to_language", ""),
                what_they_cant_say_directly=h.get("what_they_cant_say_directly", ""),
                the_wound=h.get("the_wound", ""),
                the_compensation=h.get("the_compensation", ""),
            )
            hypotheses.append(state)

        return hypotheses

    def refine_hypotheses(
        self,
        writing_samples: list[str],
        top_hypotheses: list[tuple[CognitiveState, float]],
        n_new: int = 5,
    ) -> list[CognitiveState]:
        """Generate refined hypotheses based on what worked."""

        samples_text = "\n\n---\n\n".join(writing_samples)

        # Format top hypotheses
        top_descriptions = []
        for i, item in enumerate(top_hypotheses[:3]):
            hyp, ppl, improvement = item
            desc = f"Hypothesis {i+1} (perplexity: {ppl:.2f}, improvement: {improvement:.1f}%):\n"
            for k, v in hyp.to_dict().items():
                desc += f"  {k}: {v}\n"
            top_descriptions.append(desc)

        prompt = f"""Based on analyzing writing samples, these cognitive state hypotheses scored best at predicting the writer's voice:

{chr(10).join(top_descriptions)}

Writing samples for reference:
{samples_text}

Generate {n_new} NEW hypotheses that:
1. Build on what worked in the top hypotheses
2. Explore variations and combinations of successful elements
3. Try to be even more specific and accurate

Return as JSON array with the same fields as before."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        start = text.find('[')
        end = text.rfind(']') + 1

        if start == -1 or end == 0:
            return []

        try:
            hypotheses_data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []

        hypotheses = []
        for h in hypotheses_data:
            state = CognitiveState(**{k: h.get(k, "") for k in CognitiveState.__dataclass_fields__})
            hypotheses.append(state)

        return hypotheses


class PerplexityScorerV2:
    """Score hypotheses using a more recent model."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.3"):
        """
        Initialize with a causal language model.

        Options:
        - mistralai/Mistral-7B-v0.3 (7B, very capable, default)
        - meta-llama/Llama-3.1-8B (8B, latest Llama)
        - microsoft/phi-2 (2.7B, smaller but decent)
        - gpt2-xl (1.5B, older, fast fallback)
        """
        import os
        # Ensure CUDA libraries are found
        if 'LD_LIBRARY_PATH' not in os.environ or '/usr/local/cuda' not in os.environ.get('LD_LIBRARY_PATH', ''):
            os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

        print(f"Loading {model_name}...")
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Use GPU if available
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cuda",
            )
            self.device = "cuda"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.device = "cpu"
            self.model.to(self.device)

        self.model.eval()

        # Handle models without pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def compute_perplexity(self, context: str, target: str) -> float:
        """Compute perplexity of target given context."""

        full_text = context + "\n\n" + target

        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encodings.input_ids.to(self.device)

        # Find where target starts
        context_enc = self.tokenizer(context + "\n\n", return_tensors="pt", truncation=True, max_length=1500)
        context_length = context_enc.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            if context_length < len(losses):
                target_losses = losses[context_length-1:]
                avg_loss = target_losses.mean().item()
            else:
                avg_loss = losses.mean().item()

            perplexity = math.exp(avg_loss)

        return perplexity

    def compute_perplexity_batch(
        self,
        contexts: list[str],
        target: str,
        batch_size: int = 8,
    ) -> list[float]:
        """
        Compute perplexity for multiple contexts in parallel.

        All contexts are evaluated against the same target text.
        Returns list of perplexities, one per context.
        """
        all_perplexities = []

        # Process in batches to avoid OOM
        for batch_start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[batch_start:batch_start + batch_size]
            batch_perplexities = self._compute_batch(batch_contexts, target)
            all_perplexities.extend(batch_perplexities)

        return all_perplexities

    def _compute_batch(self, contexts: list[str], target: str) -> list[float]:
        """Process a single batch of contexts."""

        # Build full texts
        full_texts = [ctx + "\n\n" + target for ctx in contexts]

        # Tokenize with padding
        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)

        # Get context lengths for each sequence
        context_lengths = []
        for ctx in contexts:
            ctx_enc = self.tokenizer(ctx + "\n\n", truncation=True, max_length=1500)
            context_lengths.append(len(ctx_enc.input_ids))

        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Shift for next-token prediction
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Compute per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # Shape: (batch_size, seq_len-1)
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch_size, -1)

            # Compute perplexity for each sequence, only on target tokens
            perplexities = []
            for i in range(batch_size):
                ctx_len = context_lengths[i]
                seq_mask = shift_mask[i]  # 1 for real tokens, 0 for padding

                # Create mask for target tokens only (after context, excluding padding)
                target_mask = torch.zeros_like(seq_mask)
                if ctx_len - 1 < seq_len - 1:
                    target_mask[ctx_len - 1:] = 1
                target_mask = target_mask * seq_mask  # Also exclude padding

                if target_mask.sum() > 0:
                    masked_losses = losses[i] * target_mask
                    avg_loss = masked_losses.sum() / target_mask.sum()
                    ppl = math.exp(avg_loss.item())
                else:
                    # Fallback if no target tokens
                    ppl = math.exp(losses[i][seq_mask.bool()].mean().item())

                perplexities.append(ppl)

        return perplexities

    def score_hypothesis(
        self,
        hypothesis: CognitiveState,
        context_samples: list[str],
        target_text: str,
    ) -> float:
        """Score hypothesis by perplexity improvement."""

        hyp_prompt = hypothesis.to_prompt()
        context_text = "\n\n---\n\n".join(context_samples)

        full_context = f"""{hyp_prompt}

Samples of writing from someone in this cognitive state:

{context_text}

Another sample of their writing:"""

        return self.compute_perplexity(full_context, target_text)

    def score_hypotheses_batch(
        self,
        hypotheses: list[CognitiveState],
        context_samples: list[str],
        target_text: str,
        batch_size: int = 8,
    ) -> list[float]:
        """Score multiple hypotheses in parallel."""

        context_text = "\n\n---\n\n".join(context_samples)

        # Build all contexts
        contexts = []
        for hyp in hypotheses:
            hyp_prompt = hyp.to_prompt()
            full_context = f"""{hyp_prompt}

Samples of writing from someone in this cognitive state:

{context_text}

Another sample of their writing:"""
            contexts.append(full_context)

        return self.compute_perplexity_batch(contexts, target_text, batch_size)


def run_iterative_inference(
    figure_key: str,
    n_initial: int = 10,
    n_iterations: int = 3,
    n_refine_per_iter: int = 5,
    model_name: str = "gpt2-xl",  # Default to gpt2-xl for speed; use phi-2 for quality
) -> dict:
    """
    Run iterative reverse inference.

    1. Generate initial hypotheses from text
    2. Score them
    3. Refine based on top performers
    4. Repeat
    """

    if figure_key not in ALL_WRITERS:
        raise ValueError(f"Unknown figure: {figure_key}")

    figure = ALL_WRITERS[figure_key]
    samples = figure["samples"]

    print("=" * 70)
    print(f"ITERATIVE REVERSE INFERENCE: {figure['name']}")
    print(f"Model: {model_name}")
    print("=" * 70)

    # Initialize
    generator = HypothesisGenerator()
    scorer = PerplexityScorerV2(model_name)

    context_samples = samples[:-1]
    target_sample = samples[-1]

    # Baseline
    baseline_context = "Samples of writing from an author:\n\n" + "\n\n---\n\n".join(context_samples) + "\n\nAnother sample:"
    baseline_ppl = scorer.compute_perplexity(baseline_context, target_sample)
    print(f"\nBaseline perplexity: {baseline_ppl:.2f}")

    all_results = []

    for iteration in range(n_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print("=" * 70)

        if iteration == 0:
            # Generate initial hypotheses
            print(f"\nGenerating {n_initial} initial hypotheses from text...")
            hypotheses = generator.generate_initial_hypotheses(samples, n_initial)
            print(f"Generated {len(hypotheses)} hypotheses")
        else:
            # Refine based on top performers
            top_n = min(3, len(all_results))
            top_hypotheses = all_results[:top_n]
            print(f"\nRefining based on top {top_n} hypotheses...")
            hypotheses = generator.refine_hypotheses(samples, top_hypotheses, n_refine_per_iter)
            print(f"Generated {len(hypotheses)} refined hypotheses")

        # Score hypotheses in batch
        if hypotheses:
            perplexities = scorer.score_hypotheses_batch(
                hypotheses, context_samples, target_sample, batch_size=8
            )

            for i, (hyp, ppl) in enumerate(zip(hypotheses, perplexities)):
                improvement = (baseline_ppl - ppl) / baseline_ppl * 100
                all_results.append((hyp, ppl, improvement))

                marker = "↓" if ppl < baseline_ppl else "↑"
                print(f"  Hypothesis {i+1}: ppl = {ppl:.2f} ({marker} {abs(improvement):.1f}%)")

        # Sort all results
        all_results.sort(key=lambda x: x[1])

    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nBaseline perplexity: {baseline_ppl:.2f}")
    print(f"Best perplexity: {all_results[0][1]:.2f}")
    print(f"Best improvement: {all_results[0][2]:.1f}%")

    print(f"\n{'='*70}")
    print("TOP 3 COGNITIVE STATE HYPOTHESES")
    print("=" * 70)

    for i, (hyp, ppl, improvement) in enumerate(all_results[:3]):
        print(f"\n#{i+1} - Perplexity: {ppl:.2f} ({improvement:+.1f}% vs baseline)")
        for k, v in hyp.to_dict().items():
            print(f"  {k}: {v}")

    return {
        "figure": figure["name"],
        "baseline_perplexity": baseline_ppl,
        "best_perplexity": all_results[0][1],
        "best_improvement": all_results[0][2],
        "top_hypotheses": [
            {"perplexity": ppl, "improvement": imp, "state": hyp.to_dict()}
            for hyp, ppl, imp in all_results[:5]
        ],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=list(ALL_WRITERS.keys()))
    parser.add_argument("--initial", "-i", type=int, default=10)
    parser.add_argument("--iterations", "-n", type=int, default=3)
    parser.add_argument("--refine", "-r", type=int, default=5)
    parser.add_argument("--model", "-m", type=str, default="gpt2-xl",
                       help="Model: gpt2-xl, microsoft/phi-2, etc.")
    args = parser.parse_args()

    result = run_iterative_inference(
        args.figure,
        n_initial=args.initial,
        n_iterations=args.iterations,
        n_refine_per_iter=args.refine,
        model_name=args.model,
    )

    # Save
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/reverse_v2_{args.figure}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to results/reverse_v2_{args.figure}.json")
