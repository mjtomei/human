"""
Meta-Search: Discovers both which dimensions matter AND their values.

Instead of fixing the dimension set, this search treats dimension
inclusion as part of the optimization. Hypotheses are sparse -
they only include dimensions that help prediction.

Features:
- Large pool of ~60 candidate dimensions
- Sparse hypotheses (typically 5-15 active dimensions)
- Discovers emergent structure rather than imposing hierarchy
- V1-style search operators: semantic mutations, linkage-aware crossover,
  fitness sharing, elitism, stagnation detection
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_gen.dimension_pool import (
    ALL_DIMENSIONS,
    DIMENSION_GROUPS,
    DIMENSION_TO_GROUP,
    N_DIMENSIONS,
    V1_DIMENSIONS,
    V1_LINKAGE_GROUPS,
    V1_DIMENSION_TO_GROUP,
    get_dimension_description,
    get_random_dimensions,
)


# =============================================================================
# SPARSE HYPOTHESIS
# =============================================================================

@dataclass
class SparseHypothesis:
    """
    A hypothesis with only some dimensions active.

    values: dict mapping dimension name -> value (or None if inactive)
    """
    values: dict = field(default_factory=dict)

    @property
    def active_dimensions(self) -> list[str]:
        """Dimensions that have values."""
        return [d for d, v in self.values.items() if v is not None]

    @property
    def n_active(self) -> int:
        """Number of active dimensions."""
        return len(self.active_dimensions)

    def get_prompt_text(self, use_v1_format: bool = False) -> str:
        """Generate prompt text for active dimensions only."""
        if not self.active_dimensions:
            return ""

        if use_v1_format:
            # Exact v1 format: no sorting, no bullets, with wrapper
            sections = ["=== COGNITIVE STATE ===", "Write as if experiencing:", ""]
            for dim in self.active_dimensions:  # No sorting - preserve order
                value = self.values[dim]
                if value:
                    label = dim.replace("_", " ").title()
                    sections.append(f"{label}: {value}")
            sections.extend(["", "Let these shape the voice without naming them.", "=== END ==="])
            return "\n".join(sections)
        else:
            lines = ["[Cognitive Context]"]
            for dim in sorted(self.active_dimensions):
                value = self.values[dim]
                readable = dim.replace("_", " ").title()
                lines.append(f"- {readable}: {value}")
            return "\n".join(lines)

    def to_dict(self) -> dict:
        """Only include active dimensions."""
        return {d: v for d, v in self.values.items() if v is not None}

    def copy(self) -> "SparseHypothesis":
        return SparseHypothesis(values=dict(self.values))


# =============================================================================
# LOCAL LLM (for API-free generation)
# =============================================================================

class LocalLLM:
    """Local LLM for hypothesis generation and mutation.

    Uses Mistral-Instruct for generation (separate from scoring model).
    Supports batched generation for efficiency.
    """

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._loaded = False
        return cls._instance

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if not self._loaded:
            print(f"Loading generation model: {self._model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"  # For batch generation
            self._loaded = True
            print(f"Generation model loaded on {next(self._model.parameters()).device}")

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for instruct model."""
        try:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            return f"[INST] {prompt} [/INST]"

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Generate single response."""
        self._ensure_loaded()

        text = self._format_prompt(prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 500) -> list[str]:
        """Generate responses for multiple prompts (batched for efficiency)."""
        if not prompts:
            return []

        self._ensure_loaded()

        # Format all prompts
        texts = [self._format_prompt(p) for p in prompts]

        # Tokenize with padding
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode each response
        responses = []
        for i, output in enumerate(outputs):
            input_len = (inputs['attention_mask'][i] == 1).sum().item()
            response = self._tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses


# =============================================================================
# SEMANTIC MUTATOR (from v1)
# =============================================================================

class SemanticMutator:
    """Generate semantic mutations of dimension values."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", use_local: bool = False):
        self.use_local = use_local
        if use_local:
            self.local_llm = LocalLLM()
        else:
            self.client = anthropic.Anthropic()
            self.model = model

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate using either local or API model."""
        if self.use_local:
            return self.local_llm.generate(prompt, max_new_tokens=max_tokens)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

    def vary(self, dimension: str, value: str) -> str:
        """Generate a slight variation of a value."""
        prompt = f"""For the cognitive dimension "{dimension}", generate a slight variation of:
"{value}"

Return ONLY the new value, nothing else. Keep similar meaning but use different words or angle."""
        return self._generate(prompt, 100).strip('"')

    def intensify(self, dimension: str, value: str) -> str:
        """Make a value more extreme."""
        prompt = f"""For the cognitive dimension "{dimension}", make this more extreme/intense:
"{value}"

Return ONLY the intensified version, nothing else."""
        return self._generate(prompt, 100).strip('"')

    def soften(self, dimension: str, value: str) -> str:
        """Make a value more subtle."""
        prompt = f"""For the cognitive dimension "{dimension}", make this more subtle/muted:
"{value}"

Return ONLY the softened version, nothing else."""
        return self._generate(prompt, 100).strip('"')

    def opposite(self, dimension: str, value: str) -> str:
        """Generate the psychological opposite."""
        prompt = f"""For the cognitive dimension "{dimension}", generate the psychological opposite of:
"{value}"

Return ONLY the opposite value, nothing else."""
        return self._generate(prompt, 100).strip('"')

    def mutate_linked_group(
        self,
        hypothesis: SparseHypothesis,
        group_dims: list[str],
        writing_samples: list[str],
    ) -> SparseHypothesis:
        """Mutate linked dimensions together to maintain coherence."""
        current_values = {d: hypothesis.values.get(d) for d in group_dims
                         if hypothesis.values.get(d)}

        if not current_values:
            return hypothesis

        samples_text = "\n---\n".join(writing_samples[:2])

        prompt = f"""Given these linked cognitive dimensions for a writer:

{chr(10).join(f'{d}: {v}' for d, v in current_values.items())}

And samples of their writing:
{samples_text}

Generate a coherent variation that keeps the psychological structure but shifts the specifics.
These dimensions are linked - they should remain internally consistent.

Return as JSON object with the same keys. Return ONLY the JSON, nothing else."""

        try:
            text = self._generate(prompt, 300)
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                new_values = json.loads(text[start:end])
                result = hypothesis.copy()
                for d, v in new_values.items():
                    if d in group_dims:
                        result.values[d] = v
                return result
        except:
            pass

        return hypothesis


# =============================================================================
# HYPOTHESIS GENERATOR
# =============================================================================

class MetaHypothesisGenerator:
    """Generate sparse hypotheses with variable dimension sets."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", use_local: bool = False):
        self.use_local = use_local
        if use_local:
            self.local_llm = LocalLLM()
        else:
            self.client = anthropic.Anthropic()
            self.model = model
        self.mutator = SemanticMutator(model, use_local=use_local)

    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate using either local or API model."""
        if self.use_local:
            return self.local_llm.generate(prompt, max_new_tokens=max_tokens)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

    def generate_initial(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
        dims_per_hypothesis: int = 8,
    ) -> list[SparseHypothesis]:
        """Generate initial sparse hypotheses with random dimension subsets."""

        samples_text = "\n\n---\n\n".join(writing_samples[:3])
        results = []

        for _ in range(n_hypotheses):
            # Pick random subset of dimensions
            dims = get_random_dimensions(dims_per_hypothesis)
            dim_list = "\n".join(f"- {d}: {get_dimension_description(d)}" for d in dims)

            prompt = f"""Analyze this writing and generate values for these cognitive dimensions:

Writing:
{samples_text}

Dimensions to fill:
{dim_list}

Return as JSON object with dimension names as keys.
Be specific and grounded in the text.
Return ONLY valid JSON."""

            try:
                text = self._generate(prompt, 1000)
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    values = json.loads(text[start:end])
                    # Only keep requested dimensions
                    filtered = {d: values.get(d) for d in dims if values.get(d)}
                    results.append(SparseHypothesis(values=filtered))

            except Exception as e:
                print(f"Warning: generation error: {e}")

        return results

    def generate_dimension_value(
        self,
        dimension: str,
        writing_samples: list[str],
    ) -> str:
        """Generate a value for a new dimension."""

        samples_text = "\n---\n".join(writing_samples[:2])

        prompt = f"""For the dimension "{dimension}" ({get_dimension_description(dimension)}),
generate a value based on this writing:

{samples_text}

Return ONLY the value, nothing else. Be specific."""

        try:
            return self._generate(prompt, 100).strip('"')
        except:
            return None

    def generate_v1_seeded(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
    ) -> list[SparseHypothesis]:
        """Generate hypotheses using v1's exact dimensions with BATCHED generation."""

        samples_text = "\n\n---\n\n".join(writing_samples)

        # Create individual prompts for each hypothesis (more diverse than asking for array)
        prompts = []
        for i in range(n_hypotheses):
            prompt = f"""Analyze these writing samples and generate a hypothesis about the writer's cognitive/psychological state.

Writing samples:
{samples_text}

Generate cognitive state hypothesis #{i+1}. Provide values for 5-10 of these dimensions (leave others blank):

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

Return as JSON object. Be specific and grounded in the text.
Try a DIFFERENT angle than other hypotheses - focus on different aspects."""
            prompts.append(prompt)

        # Generate all hypotheses in batch
        print(f"  Generating {n_hypotheses} hypotheses in batch...")
        if self.use_local:
            responses = self.local_llm.generate_batch(prompts, max_new_tokens=1000)
        else:
            # Fall back to sequential for API
            responses = [self._generate(p, 1000) for p in prompts]

        # Parse responses
        results = []
        for text in responses:
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    h = json.loads(text[start:end])
                    filtered = {d: h.get(d) for d in V1_DIMENSIONS if h.get(d)}
                    if filtered:
                        results.append(SparseHypothesis(values=filtered))
            except Exception as e:
                pass  # Skip failed parses

        print(f"  Successfully parsed {len(results)} hypotheses")
        return results


# =============================================================================
# STAGNATION DETECTOR (from v1)
# =============================================================================

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

    def reset(self):
        self.history = []


# =============================================================================
# SCORER
# =============================================================================

class MetaScorer:
    """Score hypotheses using perplexity."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.3",  # Use BASE model for scoring
        use_v1_format: bool = False,
    ):
        print(f"Loading scorer model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.device = next(self.model.parameters()).device
        print(f"Scorer loaded on {self.device}")
        self.use_v1_format = use_v1_format

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_perplexity(
        self,
        context: str,
        target: str,
        hypothesis: Optional[SparseHypothesis] = None,
    ) -> float:
        """Compute perplexity."""

        parts = []
        if hypothesis:
            prompt_text = hypothesis.get_prompt_text(use_v1_format=self.use_v1_format)
            if prompt_text:
                parts.append(prompt_text)

        if self.use_v1_format:
            # v1 format: specific wrapper around samples
            # Must match genetic_search.py PerplexityScorer.score() exactly
            if context:
                parts.append(f"Samples of writing from someone in this cognitive state:\n\n{context}\n\nAnother sample of their writing:")
            context_text = "\n\n".join(parts)
            full_text = context_text + "\n\n" + target  # Two newlines before target (matching v1)
        else:
            if context:
                parts.append(f"[Writing]\n{context}")
            parts.append(target)
            full_text = "\n\n".join(parts)
            context_text = "\n\n".join(parts[:-1])

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Context length = everything before target (including trailing newlines)
        context_len = len(self.tokenizer(context_text + "\n\n").input_ids)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            if context_len > 0 and context_len < len(losses):
                target_losses = losses[context_len - 1:]
                avg_loss = target_losses.mean().item()
            else:
                avg_loss = losses.mean().item()

            return math.exp(avg_loss)

    def score(
        self,
        context: str,
        target: str,
        hypothesis: SparseHypothesis,
        baseline_ppl: float,
    ) -> tuple[float, float]:
        """
        Score a hypothesis.

        Returns:
            (fitness, raw_perplexity)

        fitness = improvement (higher = better)
        """
        ppl = self.compute_perplexity(context, target, hypothesis)
        fitness = (baseline_ppl - ppl) / baseline_ppl
        return fitness, ppl

    def baseline(self, context: str, target: str) -> float:
        return self.compute_perplexity(context, target, None)

    def score_batch(
        self,
        batch: list[tuple[str, str, SparseHypothesis, float]],
        max_batch_size: int = 64,
    ) -> list[tuple[float, float]]:
        """
        Score multiple hypotheses in batches.

        Args:
            batch: List of (context, target, hypothesis, baseline_ppl) tuples
            max_batch_size: Maximum batch size for GPU (default 64, increase for more memory)

        Returns:
            List of (fitness, perplexity) tuples
        """
        if not batch:
            return []

        results = []

        # Process in chunks
        for i in range(0, len(batch), max_batch_size):
            chunk = batch[i:i + max_batch_size]
            chunk_results = self._score_batch_chunk(chunk)
            results.extend(chunk_results)

        return results

    def _score_batch_chunk(
        self,
        chunk: list[tuple[str, str, SparseHypothesis, float]],
    ) -> list[tuple[float, float]]:
        """Score a single batch chunk."""
        # Build all texts and track context lengths
        full_texts = []
        context_texts = []

        for context, target, hypothesis, _ in chunk:
            parts = []
            if hypothesis:
                prompt_text = hypothesis.get_prompt_text(use_v1_format=self.use_v1_format)
                if prompt_text:
                    parts.append(prompt_text)

            if self.use_v1_format:
                if context:
                    parts.append(f"Samples of writing from someone in this cognitive state:\n\n{context}\n\nAnother sample of their writing:")
                context_text = "\n\n".join(parts)
                full_text = context_text + "\n\n" + target
            else:
                if context:
                    parts.append(f"[Writing]\n{context}")
                parts.append(target)
                full_text = "\n\n".join(parts)
                context_text = "\n\n".join(parts[:-1])

            full_texts.append(full_text)
            context_texts.append(context_text + "\n\n")

        # Tokenize all at once with padding
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)

        # Get context lengths for each sequence
        context_lens = []
        for ctx_text in context_texts:
            ctx_ids = self.tokenizer(ctx_text, truncation=True, max_length=2048).input_ids
            context_lens.append(len(ctx_ids))

        # Single forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()

            # Compute per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # Shape: (batch_size, seq_len)
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_logits.size(0), -1)

            # Get attention mask for proper averaging (ignore padding)
            attention_mask = inputs.attention_mask[..., 1:]  # Shift to match losses

        # Compute perplexity for each sequence
        results = []
        for idx, (_, _, _, baseline_ppl) in enumerate(chunk):
            seq_losses = losses[idx]
            seq_mask = attention_mask[idx]
            ctx_len = context_lens[idx]

            # Only count losses on target tokens (after context)
            if ctx_len > 0 and ctx_len < seq_mask.sum().item():
                # Mask out context tokens
                target_mask = seq_mask.clone()
                target_mask[:ctx_len - 1] = 0
                target_losses = seq_losses * target_mask
                n_target_tokens = target_mask.sum().item()
                if n_target_tokens > 0:
                    avg_loss = target_losses.sum().item() / n_target_tokens
                else:
                    avg_loss = seq_losses[seq_mask.bool()].mean().item()
            else:
                # Use all non-padding tokens
                avg_loss = seq_losses[seq_mask.bool()].mean().item()

            ppl = math.exp(avg_loss)
            fitness = (baseline_ppl - ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
            results.append((fitness, ppl))

        return results


# =============================================================================
# META-SEARCH ALGORITHM (with v1 operators)
# =============================================================================

@dataclass
class MetaSearchResult:
    """Result of meta-search."""
    best_hypothesis: SparseHypothesis
    best_fitness: float
    best_perplexity: float
    baseline_perplexity: float
    improvement: float
    n_dimensions: int
    dimension_frequency: dict
    history: list


class MetaSearch:
    """
    Meta-search with v1-style operators.

    Features:
    - Tournament selection
    - Linkage-aware crossover
    - Semantic mutations (vary/intensify/soften/opposite)
    - Fitness sharing
    - Elitism
    - Stagnation detection with diversity injection
    """

    def __init__(
        self,
        scorer: MetaScorer,
        generator: MetaHypothesisGenerator,
        population_size: int = 30,
        elite_size: int = 2,
        base_mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        add_dim_rate: float = 0.1,
        remove_dim_rate: float = 0.1,
        min_dims: int = 3,
        max_dims: int = 40,
        use_v1_mode: bool = False,  # Use v1's dimensions and linkage groups
    ):
        self.scorer = scorer
        self.generator = generator
        self.population_size = population_size
        self.elite_size = elite_size
        self.base_mutation_rate = base_mutation_rate
        self.crossover_rate = crossover_rate
        self.add_dim_rate = add_dim_rate
        self.remove_dim_rate = remove_dim_rate
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.use_v1_mode = use_v1_mode

        self.stagnation = StagnationDetector()

        # Use v1's linkage groups if in v1 mode
        if use_v1_mode:
            self.linkage_groups = list(V1_LINKAGE_GROUPS.values())
            self.dim_to_group = V1_DIMENSION_TO_GROUP
            self.allowed_dims = set(V1_DIMENSIONS)
        else:
            self.linkage_groups = list(DIMENSION_GROUPS.values())
            self.dim_to_group = DIMENSION_TO_GROUP
            self.allowed_dims = set(ALL_DIMENSIONS)

    def tournament_select(
        self,
        population: list[SparseHypothesis],
        fitnesses: list[float],
        k: int = 3,
    ) -> SparseHypothesis:
        """Select one parent via tournament."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def linkage_aware_crossover(
        self,
        parent_a: SparseHypothesis,
        parent_b: SparseHypothesis,
    ) -> SparseHypothesis:
        """
        Crossover that inherits dimension groups as units.

        Modified for dimension selection: randomly DROP groups to prevent bloat.
        - If both parents have a group: pick one parent's version
        - If only one parent has a group: 50% chance to inherit, 50% to drop
        """
        child_values = {}

        # Group each parent's dimensions
        a_groups = defaultdict(dict)
        for dim in parent_a.active_dimensions:
            group = self.dim_to_group.get(dim, "other")
            a_groups[group][dim] = parent_a.values[dim]

        b_groups = defaultdict(dict)
        for dim in parent_b.active_dimensions:
            group = self.dim_to_group.get(dim, "other")
            b_groups[group][dim] = parent_b.values[dim]

        # All groups present in either parent
        all_groups = set(a_groups.keys()) | set(b_groups.keys())

        for group_name in all_groups:
            a_has = group_name in a_groups and a_groups[group_name]
            b_has = group_name in b_groups and b_groups[group_name]

            if a_has and b_has:
                # Both parents have this group - pick one
                source = random.choice([a_groups[group_name], b_groups[group_name]])
                child_values.update(source)
            elif a_has:
                # Only A has it
                if self.use_v1_mode:
                    # In v1 mode, always inherit (fixed dims)
                    child_values.update(a_groups[group_name])
                elif random.random() < 0.5:
                    # In meta mode, 50% chance to inherit
                    child_values.update(a_groups[group_name])
            elif b_has:
                # Only B has it
                if self.use_v1_mode:
                    # In v1 mode, always inherit (fixed dims)
                    child_values.update(b_groups[group_name])
                elif random.random() < 0.5:
                    # In meta mode, 50% chance to inherit
                    child_values.update(b_groups[group_name])

        return SparseHypothesis(values=child_values)

    def mutate(
        self,
        hypothesis: SparseHypothesis,
        writing_samples: list[str],
        stagnation_severity: float,
    ) -> SparseHypothesis:
        """Apply mutations with adaptive rates based on stagnation."""

        result = hypothesis.copy()

        # Adaptive mutation rate
        mutation_rate = self.base_mutation_rate + (0.5 - self.base_mutation_rate) * stagnation_severity

        # Dimension addition (only if not in v1 mode, which has fixed dims)
        if not self.use_v1_mode and random.random() < self.add_dim_rate and result.n_active < self.max_dims:
            inactive = [d for d in self.allowed_dims if d not in result.active_dimensions]
            if inactive:
                new_dim = random.choice(inactive)
                value = self.generator.generate_dimension_value(new_dim, writing_samples)
                if value:
                    result.values[new_dim] = value

        # Dimension removal (only if not in v1 mode, which has fixed dims)
        if not self.use_v1_mode and random.random() < self.remove_dim_rate and result.n_active > self.min_dims:
            if result.active_dimensions:
                dim_to_remove = random.choice(result.active_dimensions)
                result.values[dim_to_remove] = None

        # Value mutation
        if random.random() < mutation_rate and result.active_dimensions:
            # Choose mutation type based on stagnation (matching v1's weights)
            if stagnation_severity > 0.7:
                weights = {'vary': 0.1, 'intensify': 0.1, 'soften': 0.1,
                          'opposite': 0.4, 'linked': 0.2, 'new': 0.1}
            elif stagnation_severity > 0.3:
                weights = {'vary': 0.3, 'intensify': 0.15, 'soften': 0.15,
                          'opposite': 0.2, 'linked': 0.15, 'new': 0.05}
            else:
                weights = {'vary': 0.4, 'intensify': 0.2, 'soften': 0.2,
                          'opposite': 0.1, 'linked': 0.1, 'new': 0.0}

            mutation_type = random.choices(
                list(weights.keys()),
                weights=list(weights.values())
            )[0]

            if mutation_type == 'linked':
                # Find a group that has active dimensions
                active_groups = defaultdict(list)
                for dim in result.active_dimensions:
                    group = self.dim_to_group.get(dim, "other")
                    active_groups[group].append(dim)

                if active_groups:
                    group_name = random.choice(list(active_groups.keys()))
                    group_dims = active_groups[group_name]
                    result = self.generator.mutator.mutate_linked_group(
                        result, group_dims, writing_samples
                    )
            elif mutation_type == 'new':
                # Generate completely new value for a dimension (v1 feature)
                dim = random.choice(result.active_dimensions)
                new_value = self.generator.generate_dimension_value(dim, writing_samples)
                if new_value:
                    result.values[dim] = new_value
            else:
                # Single dimension mutation
                dim = random.choice(result.active_dimensions)
                old_value = result.values[dim]
                try:
                    mutator = self.generator.mutator
                    new_value = getattr(mutator, mutation_type)(dim, old_value)
                    result.values[dim] = new_value
                except Exception as e:
                    print(f"Mutation error: {e}")

        return result

    def compute_shared_fitness(
        self,
        hypothesis: SparseHypothesis,
        fitness: float,
        population: list[SparseHypothesis],
        sigma: float = 0.3,
    ) -> float:
        """Apply fitness sharing to maintain diversity."""
        niche_count = 0.0
        hyp_dims = set(hypothesis.active_dimensions)

        for other in population:
            other_dims = set(other.active_dimensions)

            # Jaccard distance on active dimensions
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
        population: list[SparseHypothesis],
        fitnesses: list[float],
        injection_rate: float = 0.3,
    ) -> list[SparseHypothesis]:
        """Replace worst individuals with fresh random ones."""
        n_inject = max(1, int(len(population) * injection_rate))

        # Keep best
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        survivors = [h.copy() for h, f in ranked[:-n_inject]]

        # Generate fresh - use context_samples to avoid target leakage
        print(f"    Injecting {n_inject} new hypotheses...")
        if self.use_v1_mode:
            fresh = self.generator.generate_v1_seeded(self._context_samples, n_hypotheses=n_inject)
        else:
            fresh = self.generator.generate_initial(self._context_samples, n_hypotheses=n_inject)

        return survivors + fresh

    def search(
        self,
        writing_samples: list[str],
        generations: int = 25,
        initial_dims: int = 10,
    ) -> MetaSearchResult:
        """Run meta-search."""

        # v1 uses "---" as separator, regular uses just newlines
        context_samples = writing_samples[:-1]
        if self.use_v1_mode:
            context = "\n\n---\n\n".join(context_samples)
        else:
            context = "\n\n".join(context_samples)
        target = writing_samples[-1]
        # Store context_samples for use by inject_diversity
        self._context_samples = context_samples

        baseline_ppl = self.scorer.baseline(context, target)
        print(f"\nBaseline perplexity: {baseline_ppl:.3f}")

        # Generate initial population - use only context_samples to avoid target leakage
        print(f"Generating initial population...")
        if self.use_v1_mode:
            print(f"  Using v1 dimensions (20 fixed)")
            population = self.generator.generate_v1_seeded(
                context_samples,
                n_hypotheses=self.population_size,
            )
        else:
            population = self.generator.generate_initial(
                context_samples,
                n_hypotheses=self.population_size,
                dims_per_hypothesis=initial_dims,
            )

        # Fill if needed
        while len(population) < self.population_size and population:
            base = random.choice(population).copy()
            if base.active_dimensions:
                dim = random.choice(base.active_dimensions)
                base.values[dim] = self.generator.mutator.vary(dim, base.values[dim])
            population.append(base)

        if not population:
            print("Failed to generate initial population!")
            return None

        print(f"Generated {len(population)} initial hypotheses")

        # Track dimension frequency in top performers
        dim_frequency = {d: 0 for d in ALL_DIMENSIONS}

        # Search state
        best_ever = None
        best_ever_fitness = float('-inf')
        best_ever_ppl = float('inf')
        history = []

        for gen in range(generations):
            # Score population
            scored = []
            for h in population:
                fitness, ppl = self.scorer.score(context, target, h, baseline_ppl)
                scored.append((h, fitness, ppl))

            scored.sort(key=lambda x: -x[1])  # Higher fitness = better

            # Extract for convenience
            fitnesses = [f for _, f, _ in scored]
            perplexities = [p for _, _, p in scored]

            best_h, best_fit, best_ppl = scored[0]

            if best_fit > best_ever_fitness:
                best_ever = best_h.copy()
                best_ever_fitness = best_fit
                best_ever_ppl = best_ppl

            # Update stagnation detector
            self.stagnation.update(max(fitnesses), fitnesses)
            severity = self.stagnation.severity()

            # Track dimension frequency in top 25%
            top_quarter = scored[:len(scored)//4]
            for h, _, _ in top_quarter:
                for dim in h.active_dimensions:
                    dim_frequency[dim] += 1

            improvement = (baseline_ppl - best_ppl) / baseline_ppl * 100
            best_imp = (baseline_ppl - best_ever_ppl) / baseline_ppl * 100

            print(f"Gen {gen:2d}: best={best_ppl:.2f} ({improvement:+.1f}%) "
                  f"dims={best_h.n_active} stag={severity:.2f} | "
                  f"ever={best_ever_ppl:.2f} ({best_imp:+.1f}%)")

            history.append({
                "generation": gen,
                "best_ppl": best_ppl,
                "best_fitness": best_fit,
                "best_n_dims": best_h.n_active,
                "mean_ppl": sum(perplexities) / len(perplexities),
                "stagnation": severity,
            })

            # Check for severe stagnation
            if self.stagnation.is_stagnating() and severity > 0.8:
                print(f"    Severe stagnation detected, injecting diversity...")
                population = self.inject_diversity(
                    [h for h, _, _ in scored],
                    fitnesses,
                )
                self.stagnation.reset()
                continue

            # Compute shared fitness for selection
            pop_list = [h for h, _, _ in scored]
            shared_fitnesses = [
                self.compute_shared_fitness(h, f, pop_list)
                for h, f in zip(pop_list, fitnesses)
            ]

            # Create next generation
            next_gen = []

            # Elitism - keep top performers
            for h, _, _ in scored[:self.elite_size]:
                next_gen.append(h.copy())

            # Fill with offspring
            while len(next_gen) < self.population_size:
                # Tournament selection
                parent_a = self.tournament_select(pop_list, shared_fitnesses)
                parent_b = self.tournament_select(pop_list, shared_fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.linkage_aware_crossover(parent_a, parent_b)
                else:
                    child = random.choice([parent_a, parent_b]).copy()

                # Mutation
                child = self.mutate(child, writing_samples, severity)

                next_gen.append(child)

            population = next_gen

        # Final results
        final_improvement = (baseline_ppl - best_ever_ppl) / baseline_ppl

        print("\n" + "=" * 70)
        print("META-SEARCH COMPLETE")
        print("=" * 70)
        print(f"Baseline: {baseline_ppl:.3f}")
        print(f"Best: {best_ever_ppl:.3f} ({final_improvement*100:.1f}% improvement)")
        print(f"Active dimensions: {best_ever.n_active}")

        print("\n" + "=" * 70)
        print("DISCOVERED DIMENSIONS (ranked by value)")
        print("=" * 70)
        for dim in sorted(best_ever.active_dimensions):
            value = best_ever.values[dim]
            print(f"  {dim}: {value}")

        # Top dimensions by frequency
        print("\n" + "=" * 70)
        print("DIMENSION FREQUENCY (in top 25% of each generation)")
        print("=" * 70)
        sorted_freq = sorted(dim_frequency.items(), key=lambda x: -x[1])
        for dim, freq in sorted_freq[:15]:
            if freq > 0:
                print(f"  {dim}: {freq}")

        return MetaSearchResult(
            best_hypothesis=best_ever,
            best_fitness=best_ever_fitness,
            best_perplexity=best_ever_ppl,
            baseline_perplexity=baseline_ppl,
            improvement=final_improvement,
            n_dimensions=best_ever.n_active,
            dimension_frequency=dim_frequency,
            history=history,
        )


def meta_search(
    writing_samples: list[str],
    population_size: int = 30,
    generations: int = 25,
    initial_dims: int = 10,
    min_dims: int = 3,
    max_dims: int = 40,
    model_name: str = "mistralai/Mistral-7B-v0.3",
) -> MetaSearchResult:
    """Convenience function to run meta-search."""

    print("=" * 70)
    print("META-SEARCH: Discovering Dimensions + Values")
    print(f"Pool: {N_DIMENSIONS} candidate dimensions")
    print(f"Population: {population_size}, Generations: {generations}")
    print("=" * 70)

    scorer = MetaScorer(model_name)
    generator = MetaHypothesisGenerator()

    search = MetaSearch(
        scorer=scorer,
        generator=generator,
        population_size=population_size,
        min_dims=min_dims,
        max_dims=max_dims,
    )

    return search.search(
        writing_samples=writing_samples,
        generations=generations,
        initial_dims=initial_dims,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser()
    parser.add_argument("figure", type=str)
    parser.add_argument("--population", "-p", type=int, default=30)
    parser.add_argument("--generations", "-g", type=int, default=25)
    args = parser.parse_args()

    from cognitive_gen.reverse_inference import PUBLIC_FIGURES

    figure = PUBLIC_FIGURES[args.figure]

    result = meta_search(
        writing_samples=figure["samples"],
        population_size=args.population,
        generations=args.generations,
    )

    # Save
    os.makedirs("results", exist_ok=True)
    with open(f"results/meta_{args.figure}.json", "w") as f:
        json.dump({
            "improvement": result.improvement,
            "best_perplexity": result.best_perplexity,
            "baseline_perplexity": result.baseline_perplexity,
            "n_dimensions": result.n_dimensions,
            "dimensions": result.best_hypothesis.to_dict(),
            "dimension_frequency": result.dimension_frequency,
            "history": result.history,
        }, f, indent=2)

    print(f"\nSaved to results/meta_{args.figure}.json")
