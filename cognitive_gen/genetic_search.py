"""
Genetic algorithm for cognitive hypothesis search.

Features:
- Linkage learning (discover dimension dependencies)
- Semantic mutation (Claude-based meaningful variations)
- Adaptive exploration (increase randomness when stuck)
- Fitness sharing (maintain diversity)
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import fields
from itertools import combinations
from typing import Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_gen.cognitive_state import (
    CognitiveState,
    CognitiveStateV1,
    CognitiveStateV2,
    get_state_class,
)
from cognitive_gen.semantic_mutation import SemanticMutator, HypothesisGenerator
from cognitive_gen.reverse_inference import PUBLIC_FIGURES, REGULAR_WRITERS

ALL_WRITERS = {**PUBLIC_FIGURES, **REGULAR_WRITERS}


class PerplexityScorer:
    """Score hypotheses using perplexity measurement."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.3"):
        import os
        if 'LD_LIBRARY_PATH' not in os.environ or '/usr/local/cuda' not in os.environ.get('LD_LIBRARY_PATH', ''):
            os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

        print(f"Loading {model_name}...")
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
            )
            self.device = "cpu"
            self.model.to(self.device)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def compute_baseline(self, context_samples: list[str], target: str) -> float:
        """Compute baseline perplexity without any hypothesis."""
        context_text = "\n\n---\n\n".join(context_samples)
        context = f"Samples of writing from an author:\n\n{context_text}\n\nAnother sample:"
        return self._compute_perplexity(context, target)

    def score(self, hypothesis: CognitiveState, context_samples: list[str], target: str) -> float:
        """Score a single hypothesis."""
        hyp_prompt = hypothesis.to_prompt()
        context_text = "\n\n---\n\n".join(context_samples)

        full_context = f"""{hyp_prompt}

Samples of writing from someone in this cognitive state:

{context_text}

Another sample of their writing:"""

        return self._compute_perplexity(full_context, target)

    def score_batch(
        self,
        hypotheses: list[CognitiveState],
        context_samples: list[str],
        target: str,
        batch_size: int = 8,
    ) -> list[float]:
        """Score multiple hypotheses in batches."""
        context_text = "\n\n---\n\n".join(context_samples)

        contexts = []
        for hyp in hypotheses:
            hyp_prompt = hyp.to_prompt()
            full_context = f"""{hyp_prompt}

Samples of writing from someone in this cognitive state:

{context_text}

Another sample of their writing:"""
            contexts.append(full_context)

        all_perplexities = []
        for batch_start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[batch_start:batch_start + batch_size]
            batch_ppls = self._compute_batch(batch_contexts, target)
            all_perplexities.extend(batch_ppls)

        return all_perplexities

    def _compute_perplexity(self, context: str, target: str) -> float:
        """Compute perplexity of target given context."""
        full_text = context + "\n\n" + target

        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encodings.input_ids.to(self.device)

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

    def _compute_batch(self, contexts: list[str], target: str) -> list[float]:
        """Compute perplexity for a batch of contexts."""
        full_texts = [ctx + "\n\n" + target for ctx in contexts]

        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        )
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)

        context_lengths = []
        for ctx in contexts:
            ctx_enc = self.tokenizer(ctx + "\n\n", truncation=True, max_length=1500)
            context_lengths.append(len(ctx_enc.input_ids))

        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch_size, -1)

            perplexities = []
            for i in range(batch_size):
                ctx_len = context_lengths[i]
                seq_mask = shift_mask[i]

                # Account for left-padding: count padding tokens at the start
                padding_offset = (attention_mask[i] == 0).sum().item()

                target_mask = torch.zeros_like(seq_mask)
                # Start position must account for padding offset
                start_pos = int(padding_offset + ctx_len - 1)
                if start_pos < seq_len - 1:
                    target_mask[start_pos:] = 1
                target_mask = target_mask * seq_mask

                if target_mask.sum() > 0:
                    masked_losses = losses[i] * target_mask
                    avg_loss = masked_losses.sum() / target_mask.sum()
                    ppl = math.exp(avg_loss.item())
                else:
                    ppl = math.exp(losses[i][seq_mask.bool()].mean().item())

                perplexities.append(ppl)

        return perplexities


class ProgressVisualizer:
    """Write live progress to a file for monitoring with 'watch'."""

    WIDTH = 148  # Total width including borders

    def __init__(self, filepath: str = "progress.txt"):
        self.filepath = filepath
        self.start_time = None
        self.figure_name = ""
        self.state_version = ""
        self.baseline_ppl = 0.0

    def _line(self, content: str, pad_char: str = " ") -> str:
        """Create a line with proper border alignment."""
        inner_width = self.WIDTH - 4  # Account for "║  " and "  ║"
        return f"║  {content:{pad_char}<{inner_width}}  ║"

    def _header(self) -> str:
        return "╔" + "═" * (self.WIDTH - 2) + "╗"

    def _footer(self) -> str:
        return "╚" + "═" * (self.WIDTH - 2) + "╝"

    def _separator(self) -> str:
        return "╠" + "═" * (self.WIDTH - 2) + "╣"

    def _empty(self) -> str:
        return self._line("")

    def start(self, figure_name: str, state_version: str, baseline_ppl: float,
              population_size: int, max_generations: int):
        import time
        self.start_time = time.time()
        self.figure_name = figure_name
        self.state_version = state_version
        self.baseline_ppl = baseline_ppl
        self.population_size = population_size
        self.max_generations = max_generations

    def update(
        self,
        generation: int,
        perplexities: list[float],
        fitnesses: list[float],
        best_ever_ppl: float,
        best_ever_fitness: float,
        best_hypothesis: dict,
        stagnation_severity: float,
        mutation_rate: float,
        history: list[dict],
    ):
        import time

        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        # Progress bar
        progress = (generation + 1) / self.max_generations
        bar_width = 80
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Fitness history sparkline (last 40 generations)
        recent_best = [h['best_fitness'] * 100 for h in history[-40:]]
        if recent_best:
            min_f, max_f = min(recent_best), max(recent_best)
            range_f = max_f - min_f if max_f > min_f else 1
            sparkline_chars = " ▁▂▃▄▅▆▇█"
            sparkline = ""
            for f in recent_best:
                idx = int((f - min_f) / range_f * 8)
                sparkline += sparkline_chars[min(idx, 8)]
        else:
            sparkline = ""

        # Population fitness distribution (ASCII histogram)
        sorted_fits = sorted(fitnesses, reverse=True)
        histogram_width = 60
        histogram_height = 8
        if sorted_fits:
            max_fit = max(sorted_fits)
            min_fit = min(sorted_fits)
            fit_range = max_fit - min_fit if max_fit > min_fit else 1
            histogram_lines = []
            for row in range(histogram_height, 0, -1):
                threshold = min_fit + (row / histogram_height) * fit_range
                line = ""
                bucket_size = max(1, len(sorted_fits) // histogram_width)
                for i in range(0, min(histogram_width, len(sorted_fits)), bucket_size):
                    bucket = sorted_fits[i:i+bucket_size]
                    if bucket and max(bucket) >= threshold:
                        line += "█"
                    else:
                        line += " "
                histogram_lines.append(line)
        else:
            histogram_lines = [" " * histogram_width] * histogram_height

        # Build output
        lines = [
            self._header(),
            self._empty(),
            self._line(f"GENETIC SEARCH: {self.figure_name}"),
            self._empty(),
            self._line(f"Architecture: {self.state_version}        Population: {self.population_size}        Elapsed: {elapsed_str}"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line(f"GENERATION {generation + 1} / {self.max_generations}"),
            self._empty(),
            self._line(f"[{bar}]  {progress*100:5.1f}%"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("PERPLEXITY METRICS"),
            self._empty(),
            self._line(f"    Baseline:                {self.baseline_ppl:>10.3f}"),
            self._empty(),
            self._line(f"    Best ever:               {best_ever_ppl:>10.3f}      ( ↓ {(1 - best_ever_ppl/self.baseline_ppl)*100:5.1f}%  improvement )"),
            self._empty(),
            self._line(f"    Current generation:"),
            self._line(f"        Best:                {min(perplexities):>10.3f}"),
            self._line(f"        Mean:                {np.mean(perplexities):>10.3f}"),
            self._line(f"        Std:                 {np.std(perplexities):>10.3f}"),
            self._line(f"        Worst:               {max(perplexities):>10.3f}"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("SEARCH STATE"),
            self._empty(),
            self._line(f"    Best improvement:        {best_ever_fitness*100:>10.1f}%"),
            self._empty(),
            self._line(f"    Stagnation severity:     {stagnation_severity:>10.2f}      ( 0.0 = exploring,  1.0 = stuck )"),
            self._empty(),
            self._line(f"    Mutation rate:           {mutation_rate:>10.2f}      ( adapts based on stagnation )"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("FITNESS HISTORY (recent generations)"),
            self._empty(),
            self._line(f"    {sparkline}"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("POPULATION FITNESS DISTRIBUTION"),
            self._empty(),
        ]

        # Add histogram
        for i, hist_line in enumerate(histogram_lines):
            if i == 0:
                lines.append(self._line(f"    best  │{hist_line}│"))
            elif i == histogram_height - 1:
                lines.append(self._line(f"    worst │{hist_line}│"))
            else:
                lines.append(self._line(f"          │{hist_line}│"))

        lines.extend([
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("BEST HYPOTHESIS"),
            self._empty(),
        ])

        # Add hypothesis dimensions
        for dim, value in best_hypothesis.items():
            dim_str = dim.replace("_", " ").title()
            val_str = str(value)[:100]
            lines.append(self._line(f"    {dim_str:<35}  {val_str}"))

        lines.extend([
            self._empty(),
            self._footer(),
        ])

        # Write to file
        with open(self.filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    def finish(self, best_hypothesis: dict, best_fitness: float, best_ppl: float):
        import time

        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        lines = [
            self._header(),
            self._empty(),
            self._line("╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗"),
            self._line("║                                    GENETIC SEARCH COMPLETE                                         ║"),
            self._line("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝"),
            self._empty(),
            self._line(f"Figure: {self.figure_name}"),
            self._line(f"Total time: {elapsed_str}"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("FINAL RESULTS"),
            self._empty(),
            self._line(f"    Baseline perplexity:     {self.baseline_ppl:>10.3f}"),
            self._empty(),
            self._line(f"    Final best perplexity:   {best_ppl:>10.3f}"),
            self._empty(),
            self._line(f"    Total improvement:       {best_fitness*100:>10.1f}%"),
            self._empty(),
            self._separator(),
            self._empty(),
            self._line("FINAL BEST HYPOTHESIS"),
            self._empty(),
        ]

        for dim, value in best_hypothesis.items():
            dim_str = dim.replace("_", " ").title()
            val_str = str(value)[:100]
            lines.append(self._line(f"    {dim_str:<35}  {val_str}"))

        lines.extend([
            self._empty(),
            self._footer(),
        ])

        with open(self.filepath, "w") as f:
            f.write("\n".join(lines) + "\n")


class StagnationDetector:
    """Detect when the search is stuck in local optima."""

    def __init__(self, patience: int = 5, min_improvement: float = 0.005):
        self.history = []
        self.patience = patience
        self.min_improvement = min_improvement

    def update(self, best_fitness: float, population_fitness: list[float]):
        self.history.append({
            'best': best_fitness,
            'mean': np.mean(population_fitness),
            'std': np.std(population_fitness),
        })

    def is_stagnating(self) -> bool:
        if len(self.history) < self.patience:
            return False

        recent = self.history[-self.patience:]
        improvement = recent[-1]['best'] - recent[0]['best']
        return improvement < self.min_improvement

    def severity(self) -> float:
        """How stuck are we? 0 = fine, 1 = very stuck."""
        if len(self.history) < self.patience:
            return 0.0

        best = self.history[-1]['best']
        generations_stuck = 0
        for h in reversed(self.history):
            if h['best'] < best - self.min_improvement:
                break
            generations_stuck += 1

        return min(1.0, generations_stuck / (self.patience * 2))


class GeneticSearch:
    """
    Genetic algorithm for cognitive hypothesis search.

    Features linkage learning, semantic mutation, and adaptive exploration.
    """

    def __init__(
        self,
        scorer: PerplexityScorer,
        state_version: str = "v1",
        population_size: int = 20,
        elite_size: int = 2,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        claude_model: str = "claude-sonnet-4-20250514",
        progress_file: str = "progress.txt",
        use_local: bool = False,
    ):
        self.scorer = scorer
        self.state_class = get_state_class(state_version)
        self.state_version = state_version
        self.population_size = population_size
        self.elite_size = elite_size
        self.base_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.generator = HypothesisGenerator(claude_model, state_version, use_local=use_local)
        self.mutator = SemanticMutator(claude_model, use_local=use_local)
        self.stagnation = StagnationDetector()
        self.progress = ProgressVisualizer(progress_file)

        # Linkage groups (semantic priors)
        self.linkage_groups = list(self.state_class.get_dimension_groups().values())

        # Learned linkage strengths
        self.linkage_scores = {}

        # Dimension importance (updated via ablation)
        self.dimension_importance = {d: 1.0 for d in self.state_class.get_dimensions()}

    def tournament_select(
        self,
        population: list[CognitiveState],
        fitnesses: list[float],
        k: int = 3,
    ) -> CognitiveState:
        """Select one parent via tournament."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def crossover(self, parent_a: CognitiveState, parent_b: CognitiveState) -> CognitiveState:
        """Linkage-aware crossover."""
        child_dict = {}
        assigned = set()

        # Inherit linked groups as units
        for group in self.linkage_groups:
            source = random.choice([parent_a, parent_b])
            source_dict = source.to_dict()
            for dim in group:
                if dim in source_dict:
                    child_dict[dim] = source_dict[dim]
                assigned.add(dim)

        # Remaining dimensions: uniform crossover
        a_dict = parent_a.to_dict()
        b_dict = parent_b.to_dict()
        for dim in self.state_class.get_dimensions():
            if dim not in assigned:
                source = random.choice([a_dict, b_dict])
                if dim in source:
                    child_dict[dim] = source[dim]

        return self.state_class.from_dict(child_dict)

    def mutate(
        self,
        hypothesis: CognitiveState,
        mutation_rate: float,
        stagnation_severity: float,
    ) -> CognitiveState:
        """Apply semantic mutation with adaptive rate and type."""
        if random.random() > mutation_rate:
            return hypothesis

        hyp_dict = hypothesis.to_dict()

        # Choose mutation type based on stagnation
        if stagnation_severity > 0.7:
            # Very stuck: try drastic changes
            weights = {'vary': 0.1, 'intensify': 0.1, 'soften': 0.1, 'opposite': 0.4, 'linked': 0.2, 'new': 0.1}
        elif stagnation_severity > 0.3:
            # Somewhat stuck
            weights = {'vary': 0.3, 'intensify': 0.15, 'soften': 0.15, 'opposite': 0.2, 'linked': 0.15, 'new': 0.05}
        else:
            # Normal exploration
            weights = {'vary': 0.4, 'intensify': 0.2, 'soften': 0.2, 'opposite': 0.1, 'linked': 0.1, 'new': 0.0}

        mutation_type = random.choices(list(weights.keys()), weights=list(weights.values()))[0]

        if mutation_type == 'linked':
            # Mutate a whole linked group together
            group = random.choice(self.linkage_groups)
            hyp_dict = self.mutator.mutate_linked_group(hyp_dict, group, self._context_samples)
            return self.state_class.from_dict(hyp_dict)

        if mutation_type == 'new':
            # Generate completely new value
            dim = random.choice(self.state_class.get_dimensions())
            new_value = self.generator.generate_dimension_value(dim, self._context_samples)
            hyp_dict[dim] = new_value
            return self.state_class.from_dict(hyp_dict)

        # Single dimension semantic mutation
        filled_dims = [d for d, v in hyp_dict.items() if v]
        if not filled_dims:
            return hypothesis

        # Weight by importance
        dim_weights = [self.dimension_importance.get(d, 1.0) for d in filled_dims]
        dim = random.choices(filled_dims, weights=dim_weights)[0]
        old_value = hyp_dict[dim]

        try:
            new_value = getattr(self.mutator, mutation_type)(dim, old_value)
            hyp_dict[dim] = new_value
        except Exception as e:
            print(f"Mutation error: {e}")

        return self.state_class.from_dict(hyp_dict)

    def compute_shared_fitness(
        self,
        hypothesis: CognitiveState,
        fitness: float,
        population: list[CognitiveState],
        sigma: float = 0.3,
    ) -> float:
        """Apply fitness sharing to maintain diversity."""
        niche_count = 0.0

        hyp_dims = set(hypothesis.to_dict().keys())

        for other in population:
            other_dims = set(other.to_dict().keys())

            # Simple distance: Jaccard distance on active dimensions
            intersection = len(hyp_dims & other_dims)
            union = len(hyp_dims | other_dims)
            if union == 0:
                distance = 1.0
            else:
                distance = 1 - (intersection / union)

            if distance < sigma:
                sharing = 1 - (distance / sigma)
                niche_count += sharing

        return fitness / max(1, niche_count)

    def inject_diversity(
        self,
        population: list[CognitiveState],
        fitnesses: list[float],
        injection_rate: float = 0.2,
    ) -> list[CognitiveState]:
        """Replace worst individuals with fresh random ones."""
        n_inject = max(1, int(len(population) * injection_rate))

        # Keep best
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        survivors = [h for h, f in ranked[:-n_inject]]

        # Generate fresh - use context_samples to avoid target leakage
        print(f"    Injecting {n_inject} new hypotheses...")
        fresh = self.generator.generate_initial_hypotheses(self._context_samples, n_hypotheses=n_inject)

        return survivors + fresh

    def search(
        self,
        writing_samples: list[str],
        max_generations: int = 20,
        patience: int = 5,
        verbose: bool = True,
        figure_name: str = "Unknown",
    ) -> tuple[CognitiveState, float, list[dict]]:
        """
        Run genetic search.

        Returns: (best_hypothesis, best_fitness, history)
        """
        context_samples = writing_samples[:-1]
        target = writing_samples[-1]

        # Store context samples for mutation/injection methods (avoid target leakage)
        self._context_samples = context_samples

        # Baseline
        baseline_ppl = self.scorer.compute_baseline(context_samples, target)
        if verbose:
            print(f"Baseline perplexity: {baseline_ppl:.2f}")

        # Start progress visualization
        self.progress.start(
            figure_name=figure_name,
            state_version=self.state_version,
            baseline_ppl=baseline_ppl,
            population_size=self.population_size,
            max_generations=max_generations,
        )

        # Initialize population - use context_samples to avoid target leakage
        if verbose:
            print(f"Generating initial population of {self.population_size}...")
        population = self.generator.generate_initial_hypotheses(
            context_samples, n_hypotheses=self.population_size
        )

        # Retry if we didn't get enough hypotheses
        retries = 0
        while len(population) < self.population_size and retries < 3:
            retries += 1
            if verbose:
                print(f"  Got {len(population)} hypotheses, need {self.population_size}. Retrying ({retries}/3)...")
            extra = self.generator.generate_initial_hypotheses(
                context_samples, n_hypotheses=self.population_size - len(population)
            )
            population.extend(extra)

        if len(population) == 0:
            raise ValueError("Failed to generate any initial hypotheses. Check Claude API.")

        if len(population) < self.population_size:
            if verbose:
                print(f"  Warning: Only got {len(population)} hypotheses, continuing anyway.")

        best_ever = None
        best_ever_fitness = -float('inf')
        best_ever_ppl = float('inf')
        history = []

        for gen in range(max_generations):
            # Evaluate population
            perplexities = self.scorer.score_batch(population, context_samples, target)
            fitnesses = [(baseline_ppl - ppl) / baseline_ppl for ppl in perplexities]

            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_ppl = perplexities[gen_best_idx]

            if gen_best_fitness > best_ever_fitness:
                best_ever = population[gen_best_idx]
                best_ever_fitness = gen_best_fitness
                best_ever_ppl = gen_best_ppl

            # Update stagnation detector
            self.stagnation.update(max(fitnesses), fitnesses)
            severity = self.stagnation.severity()

            # Compute adaptive mutation rate
            mutation_rate = self.base_mutation_rate + (0.5 - self.base_mutation_rate) * severity

            # Record history
            history.append({
                'generation': gen,
                'best_ppl': gen_best_ppl,
                'best_fitness': gen_best_fitness,
                'mean_ppl': np.mean(perplexities),
                'stagnation': severity,
                'mutation_rate': mutation_rate,
            })

            # Update progress file
            self.progress.update(
                generation=gen,
                perplexities=perplexities,
                fitnesses=fitnesses,
                best_ever_ppl=best_ever_ppl,
                best_ever_fitness=best_ever_fitness,
                best_hypothesis=best_ever.to_dict() if best_ever else {},
                stagnation_severity=severity,
                mutation_rate=mutation_rate,
                history=history,
            )

            if verbose:
                print(f"Gen {gen:2d}: best={gen_best_ppl:.2f} (+{gen_best_fitness*100:.1f}%) "
                      f"mean={np.mean(perplexities):.2f} stag={severity:.2f}")

            # Check convergence
            if self.stagnation.is_stagnating() and gen >= patience:
                if severity > 0.8:
                    if verbose:
                        print(f"  Severe stagnation detected, injecting diversity...")
                    population = self.inject_diversity(
                        population, fitnesses, injection_rate=0.3
                    )
                    self.stagnation = StagnationDetector()  # Reset
                    continue

            # Apply fitness sharing
            shared_fitnesses = [
                self.compute_shared_fitness(h, f, population)
                for h, f in zip(population, fitnesses)
            ]

            # Create next generation
            offspring = []

            # Elitism
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            for i in elite_indices:
                offspring.append(population[i])

            # Fill with offspring
            while len(offspring) < self.population_size:
                # Selection
                parent_a = self.tournament_select(population, shared_fitnesses)
                parent_b = self.tournament_select(population, shared_fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = random.choice([parent_a, parent_b])

                # Mutation
                child = self.mutate(child, mutation_rate, severity)

                offspring.append(child)

            population = offspring

        # Finalize progress file
        self.progress.finish(
            best_hypothesis=best_ever.to_dict() if best_ever else {},
            best_fitness=best_ever_fitness,
            best_ppl=best_ever_ppl,
        )

        if verbose:
            print(f"\nSearch complete. Best perplexity: {best_ever_ppl:.2f} "
                  f"(+{best_ever_fitness*100:.1f}% vs baseline)")

        return best_ever, best_ever_fitness, history


def run_genetic_search(
    figure_key: str = None,
    state_version: str = "v1",
    population_size: int = 20,
    max_generations: int = 15,
    model_name: str = "mistralai/Mistral-7B-v0.3",
    progress_file: str = "progress.txt",
    writing_samples: list[str] = None,
    figure_name: str = None,
    use_local: bool = False,
) -> dict:
    """Run genetic search on a writer.

    Can provide either:
    - figure_key: Look up samples from ALL_WRITERS
    - writing_samples + figure_name: Use provided samples directly
    """

    if writing_samples is not None:
        samples = writing_samples
        name = figure_name or "Custom"
    elif figure_key is not None:
        if figure_key not in ALL_WRITERS:
            raise ValueError(f"Unknown figure: {figure_key}. Available: {list(ALL_WRITERS.keys())}")
        figure = ALL_WRITERS[figure_key]
        samples = figure["samples"]
        name = figure["name"]
    else:
        raise ValueError("Must provide either figure_key or writing_samples")

    print("=" * 70)
    print(f"GENETIC SEARCH: {name}")
    print(f"State version: {state_version}")
    print(f"Population: {population_size}, Generations: {max_generations}")
    print(f"Scoring model: {model_name}")
    print(f"Generation: {'Local LLM (Mistral-7B-Instruct)' if use_local else 'Claude API'}")
    print(f"Progress file: {progress_file}")
    print("=" * 70)

    scorer = PerplexityScorer(model_name)

    ga = GeneticSearch(
        scorer=scorer,
        state_version=state_version,
        population_size=population_size,
        progress_file=progress_file,
        use_local=use_local,
    )

    best_hypothesis, best_fitness, history = ga.search(
        samples,
        max_generations=max_generations,
        figure_name=name,
    )

    print("\n" + "=" * 70)
    print("BEST HYPOTHESIS")
    print("=" * 70)
    for dim, value in best_hypothesis.to_dict().items():
        print(f"  {dim}: {value}")

    return {
        "figure": name,
        "state_version": state_version,
        "baseline_ppl": history[0]['mean_ppl'] if history else None,
        "best_ppl": history[-1]['best_ppl'] if history else None,
        "best_improvement": best_fitness * 100,
        "generations": len(history),
        "best_hypothesis": best_hypothesis.to_dict(),
        "history": history,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=list(ALL_WRITERS.keys()))
    parser.add_argument("--version", "-v", choices=["v1", "v2"], default="v1")
    parser.add_argument("--population", "-p", type=int, default=20)
    parser.add_argument("--generations", "-g", type=int, default=15)
    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--progress", type=str, default="progress.txt",
                       help="Progress file for 'watch' monitoring")
    args = parser.parse_args()

    result = run_genetic_search(
        args.figure,
        state_version=args.version,
        population_size=args.population,
        max_generations=args.generations,
        model_name=args.model,
        progress_file=args.progress,
    )

    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/genetic_{args.version}_{args.figure}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to results/genetic_{args.version}_{args.figure}.json")
