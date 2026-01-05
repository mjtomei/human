"""
Main graph-based search algorithm.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from cognitive_gen.meta_search import (
    MetaScorer, MetaHypothesisGenerator, SparseHypothesis,
)
from cognitive_gen.dimension_pool import (
    V1_DIMENSIONS, V1_LINKAGE_GROUPS, ALL_DIMENSIONS, DIMENSION_POOL,
    DIMENSION_GROUPS, sample_dimensions_by_group, get_dimension_prompt_section,
    ARCHITECTURE_TEMPLATES, sample_from_template, get_random_template_name,
)

from .location_graph import LocationGraph, create_graph
from .location import Location
from .migration import MigrationSystem
from .meta_location import MetaLocationManager, MetaLocation
from .visualization import GraphVisualizer
from .essay_index import create_index


@dataclass
class GraphSearchResult:
    """Result of graph-based search."""
    generations_completed: int
    best_hypothesis: Optional[SparseHypothesis]
    best_location: Optional[str]
    best_fitness: float
    baseline_ppl: float
    improvement: float

    # Graph state
    final_n_locations: int
    final_n_meta_locations: int
    final_n_edges: int

    # History
    history: list[dict] = field(default_factory=list)
    meta_locations_created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'generations_completed': self.generations_completed,
            'best_location': self.best_location,
            'best_fitness': self.best_fitness,
            'baseline_ppl': self.baseline_ppl,
            'improvement': self.improvement,
            'best_hypothesis': self.best_hypothesis.to_dict() if self.best_hypothesis else None,
            'final_n_locations': self.final_n_locations,
            'final_n_meta_locations': self.final_n_meta_locations,
            'final_n_edges': self.final_n_edges,
            'meta_locations_created': self.meta_locations_created,
        }


class GraphSearch:
    """
    Graph-based cognitive hypothesis search.

    Each essay is a location with its own population. Populations evolve
    independently but can migrate and cross-breed. Strong connections
    lead to meta-location creation.
    """

    def __init__(
        self,
        scorer: MetaScorer,
        generator: MetaHypothesisGenerator,
        graph: Optional[LocationGraph] = None,
        population_per_location: int = 15,
        epoch_length: int = 5,
        elite_size: int = 2,
        base_mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        batch_size: int = 32,  # Conservative batch size to avoid OOM with 2 models loaded
        checkpoint_dir: str = "/tmp/graph_search_checkpoints",
        checkpoint_interval: int = 1,  # Save checkpoint every generation
    ):
        self.scorer = scorer
        self.generator = generator
        self.graph = graph
        self.population_per_location = population_per_location
        self.epoch_length = epoch_length
        self.elite_size = elite_size
        self.base_mutation_rate = base_mutation_rate
        self.crossover_rate = crossover_rate
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

        # Sub-systems
        self.migration = MigrationSystem()
        self.meta_manager = MetaLocationManager(population_size=population_per_location)
        self.visualizer = GraphVisualizer()

        # Checkpoint manager
        from .checkpoint import CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_interval=checkpoint_interval,
        )

        # Tracking
        self.history: list[dict] = []
        self.start_generation: int = 0  # For resuming from checkpoint

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> dict:
        """
        Resume from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses latest.

        Returns:
            Checkpoint metadata dict
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Restore the graph
        self.graph = self.checkpoint_manager.restore_graph(checkpoint)

        # Set start generation for search loop
        self.start_generation = checkpoint['metadata']['generation'] + 1

        # Restore meta_manager state if available
        meta_state = checkpoint.get('meta_manager', {})
        if meta_state.get('calibrated'):
            self.meta_manager.calibrated = True
            self.meta_manager.base_threshold = meta_state.get('base_threshold')
        self.meta_manager.created_metas = meta_state.get('meta_slugs', [])

        print(f"Resumed from checkpoint at generation {checkpoint['metadata']['generation']}")
        print(f"  Best fitness: {checkpoint['metadata']['best_fitness']:.3f}")
        print(f"  Locations: {checkpoint['metadata']['n_locations']}")
        print(f"  Will continue from generation {self.start_generation}")

        return checkpoint['metadata']

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        return self.checkpoint_manager.list_checkpoints()

    def initialize_graph(self, n_neighbors: int = 3):
        """Initialize the graph if not provided."""
        if self.graph is None:
            self.graph = create_graph(
                population_per_location=self.population_per_location,
                n_neighbors=n_neighbors,
            )

    def initialize_populations(self):
        """Generate initial population for each location using batched generation."""
        print("Initializing populations...")

        # Collect all prompts across all locations
        all_prompts = []
        prompt_map = []  # [(slug, hyp_idx), ...] to track which prompt goes where

        for slug, location in self.graph.locations.items():
            samples_text = location.context_region
            for hyp_idx in range(self.population_per_location):
                prompt = self._make_hypothesis_prompt(samples_text, hyp_idx)
                all_prompts.append(prompt)
                prompt_map.append((slug, hyp_idx))

        total_prompts = len(all_prompts)
        print(f"  Generating {total_prompts} hypotheses across {self.graph.n_locations} locations...")

        # Batch generate all hypotheses in chunks (to avoid OOM)
        gen_batch_size = self.batch_size  # Use same batch size as scoring
        responses = []

        for chunk_start in range(0, total_prompts, gen_batch_size):
            chunk_end = min(chunk_start + gen_batch_size, total_prompts)
            chunk_prompts = all_prompts[chunk_start:chunk_end]

            self.visualizer.output_path.write_text(
                f"Generating hypotheses {chunk_start+1}-{chunk_end}/{total_prompts}..."
            )

            try:
                if hasattr(self.generator, 'local_llm') and self.generator.use_local:
                    chunk_responses = self.generator.local_llm.generate_batch(
                        chunk_prompts, max_new_tokens=1000
                    )
                else:
                    chunk_responses = [self.generator._generate(p, 1000) for p in chunk_prompts]
                responses.extend(chunk_responses)
            except Exception as e:
                print(f"Warning: Batch generation failed for chunk: {e}, falling back to sequential")
                for prompt in chunk_prompts:
                    try:
                        resp = self.generator._generate(prompt, 1000)
                        responses.append(resp)
                    except Exception:
                        responses.append("{}")

        # Parse responses and distribute to locations
        import json

        # Initialize empty populations
        for location in self.graph.locations.values():
            location.population = []
            location.fitness_scores = []

        parsed_count = 0
        for (slug, hyp_idx), response in zip(prompt_map, responses):
            location = self.graph.locations[slug]
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    h = json.loads(response[start:end])
                    # Accept ANY valid dimension from the full pool
                    filtered = {d: v for d, v in h.items() if d in ALL_DIMENSIONS and v}
                    if filtered:
                        location.population.append(SparseHypothesis(values=filtered))
                        location.fitness_scores.append(0.0)
                        parsed_count += 1
                        continue
            except Exception:
                pass
            # Fallback: empty hypothesis
            location.population.append(SparseHypothesis(values={}))
            location.fitness_scores.append(0.0)

        print(f"  Successfully parsed {parsed_count}/{total_prompts} hypotheses")

        # Compute baseline perplexity for each location
        for slug, location in self.graph.locations.items():
            context, target = location.get_random_target()
            location.baseline_ppl = self.scorer.baseline(context, target)

        print(f"Initialized {self.graph.n_locations} locations")

    def _make_hypothesis_prompt(self, samples_text: str, hyp_idx: int) -> str:
        """Create a prompt for generating a single hypothesis with varied dimensions.

        50% of hypotheses use architecture templates (coherent psychological theories)
        50% use pure random sampling (unbiased exploration)
        """
        import random
        old_state = random.getstate()
        random.seed(hyp_idx * 1000 + hash(samples_text[:100]) % 10000)

        dims_per_group = 2 + (hyp_idx % 3)  # 2-4 dims per group

        # 50% from templates, 50% pure random
        use_template = (hyp_idx % 2 == 0)

        if use_template:
            # Use a template - cycle through them based on index
            template_names = list(ARCHITECTURE_TEMPLATES.keys())
            template_name = template_names[(hyp_idx // 2) % len(template_names)]
            dimensions, groups = sample_from_template(template_name, dims_per_group)
            source_info = f"(template: {template_name})"
        else:
            # Pure random sampling
            n_groups = 4 + ((hyp_idx // 2) % 4)  # 4-7 groups
            dimensions, groups = sample_dimensions_by_group(
                n_groups=n_groups,
                dims_per_group=dims_per_group,
            )
            source_info = "(random exploration)"

        random.setstate(old_state)

        # Generate prompt section with sampled dimensions
        dimension_list = get_dimension_prompt_section(dimensions)

        # Vary density instruction
        density_instructions = [
            f"Provide values for 3-5 of these dimensions that seem MOST essential:",
            f"Provide values for 5-10 of these dimensions that seem relevant:",
            f"Provide values for as many dimensions as genuinely apply:",
        ]
        density_instruction = density_instructions[hyp_idx % len(density_instructions)]

        return f"""Analyze these writing samples and generate a hypothesis about the writer's cognitive/psychological state.

Writing samples:
{samples_text}

Generate cognitive state hypothesis #{hyp_idx+1} {source_info}. {density_instruction}

{dimension_list}

Return as JSON object with dimension names as keys. Focus on dimensions where you have insight from the text."""

    def score_location(self, location: Location):
        """Score all hypotheses at a location (used for single-location scoring)."""
        # Ensure fitness_scores matches population size
        while len(location.fitness_scores) < len(location.population):
            location.fitness_scores.append(0.0)
        if len(location.fitness_scores) > len(location.population):
            location.fitness_scores = location.fitness_scores[:len(location.population)]

        # Build batch for this location
        batch = []
        for i, hyp in enumerate(location.population):
            context, target = location.get_random_target()
            batch.append((context, target, hyp, location.baseline_ppl, i))

        # Score in batch
        score_batch = [(ctx, tgt, hyp, baseline) for ctx, tgt, hyp, baseline, _ in batch]
        try:
            results = self.scorer.score_batch(score_batch, max_batch_size=self.batch_size)
            for (_, _, _, _, idx), (fitness, ppl) in zip(batch, results):
                location.fitness_scores[idx] = fitness
        except Exception as e:
            print(f"Warning: Batch scoring failed at {location.slug}: {e}")
            # Fallback to individual scoring
            for ctx, tgt, hyp, baseline, idx in batch:
                try:
                    fitness, ppl = self.scorer.score(ctx, tgt, hyp, baseline)
                    location.fitness_scores[idx] = fitness
                except Exception:
                    location.fitness_scores[idx] = 0.0

    def score_all_locations_batch(self):
        """Score all hypotheses across all locations in one big batch."""
        # Ensure fitness_scores matches population size for all locations
        for location in self.graph.locations.values():
            while len(location.fitness_scores) < len(location.population):
                location.fitness_scores.append(0.0)
            if len(location.fitness_scores) > len(location.population):
                location.fitness_scores = location.fitness_scores[:len(location.population)]

        # Build global batch
        global_batch = []  # (context, target, hyp, baseline, loc_slug, idx)

        for slug, location in self.graph.locations.items():
            compute_weight = location.get_compute_weight()
            n_to_score = max(1, int(len(location.population) * compute_weight))

            # Select which to score
            if n_to_score < len(location.population):
                if location.fitness_scores:
                    elite_indices = sorted(
                        range(len(location.fitness_scores)),
                        key=lambda i: location.fitness_scores[i],
                        reverse=True,
                    )[:self.elite_size]
                else:
                    elite_indices = []
                other_indices = [i for i in range(len(location.population)) if i not in elite_indices]
                n_others = n_to_score - len(elite_indices)
                if n_others > 0 and other_indices:
                    sampled = random.sample(other_indices, min(n_others, len(other_indices)))
                else:
                    sampled = []
                indices_to_score = list(set(elite_indices) | set(sampled))
            else:
                indices_to_score = list(range(len(location.population)))

            for idx in indices_to_score:
                hyp = location.population[idx]
                context, target = location.get_random_target()
                global_batch.append((context, target, hyp, location.baseline_ppl, slug, idx))

        if not global_batch:
            return

        # Score all at once
        score_batch = [(ctx, tgt, hyp, baseline) for ctx, tgt, hyp, baseline, _, _ in global_batch]
        print(f"  Scoring {len(score_batch)} hypotheses...", flush=True)

        try:
            results = self.scorer.score_batch(score_batch, max_batch_size=self.batch_size)

            # Distribute results back to locations
            for (_, _, _, _, slug, idx), (fitness, ppl) in zip(global_batch, results):
                self.graph.locations[slug].fitness_scores[idx] = fitness

        except Exception as e:
            print(f"Warning: Global batch scoring failed: {e}")
            # Fallback to per-location scoring
            for location in self.graph.locations.values():
                self.score_location(location)

    def evolve_location(self, location: Location, skip_scoring: bool = False):
        """Run one generation of evolution at a location.

        Args:
            location: Location to evolve
            skip_scoring: If True, skip scoring (assumes batch scoring already done)
        """
        if len(location.population) < 2:
            return

        # Score population (if not already done in batch)
        if not skip_scoring:
            self.score_location(location)

        # Update best ever
        location.update_best()

        # Update stagnation
        location.stagnation_state.update(
            max(location.fitness_scores),
            location.fitness_scores,
        )

        # Check stagnation response
        action = location.stagnation_state.handle(max(location.fitness_scores))

        # Determine mutation rate based on stagnation
        severity = location.stagnation_state.severity()
        mutation_rate = self.base_mutation_rate + (0.5 - self.base_mutation_rate) * severity

        # Handle stagnation actions
        if action == "inject_diversity":
            self._inject_diversity(location)
            self.visualizer.add_event(f"Gen {location.generation}: Injected diversity at {location.slug}")
        elif action == "deeply_stuck":
            self.visualizer.add_event(f"Gen {location.generation}: {location.slug} marked deeply stuck")

        # Elitism: keep best
        sorted_indices = sorted(
            range(len(location.fitness_scores)),
            key=lambda i: location.fitness_scores[i],
            reverse=True,
        )

        elites = [(location.population[i].copy(), location.fitness_scores[i])
                  for i in sorted_indices[:self.elite_size]]

        # Generate new population
        new_population = [e[0] for e in elites]
        new_fitness = [e[1] for e in elites]

        while len(new_population) < self.population_per_location:
            # Tournament selection
            parent_a = self._tournament_select(location)
            parent_b = self._tournament_select(location)

            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent_a, parent_b)
            else:
                child = parent_a.copy()

            # Mutation
            if random.random() < mutation_rate:
                child = self._mutate(child, location, severity)

            new_population.append(child)
            new_fitness.append(0.0)  # Will be scored next generation

        location.population = new_population[:self.population_per_location]
        location.fitness_scores = new_fitness[:self.population_per_location]
        location.generation += 1
        location.record_history()

    def _tournament_select(self, location: Location, k: int = 3) -> SparseHypothesis:
        """Tournament selection."""
        k = min(k, len(location.population))
        indices = random.sample(range(len(location.population)), k)
        best_idx = max(indices, key=lambda i: location.fitness_scores[i])
        return location.population[best_idx]

    def _crossover(self, parent_a: SparseHypothesis, parent_b: SparseHypothesis) -> SparseHypothesis:
        """Linkage-aware crossover."""
        from cognitive_gen.dimension_pool import DIMENSION_TO_GROUP

        dims_a = set(parent_a.active_dimensions)
        dims_b = set(parent_b.active_dimensions)

        # Group dimensions
        groups_a = {}
        for dim in dims_a:
            group = DIMENSION_TO_GROUP.get(dim, 'other')
            groups_a.setdefault(group, []).append(dim)

        groups_b = {}
        for dim in dims_b:
            group = DIMENSION_TO_GROUP.get(dim, 'other')
            groups_b.setdefault(group, []).append(dim)

        # Choose groups from each parent
        all_groups = set(groups_a.keys()) | set(groups_b.keys())
        offspring_values = {}

        for group in all_groups:
            has_a = group in groups_a
            has_b = group in groups_b

            if has_a and has_b:
                source = parent_a if random.random() < 0.5 else parent_b
                dims = groups_a[group] if source is parent_a else groups_b[group]
            elif has_a:
                source = parent_a
                dims = groups_a[group]
            else:
                source = parent_b
                dims = groups_b[group]

            for dim in dims:
                offspring_values[dim] = source.values.get(dim)

        return SparseHypothesis(values=offspring_values)

    def _mutate(self, hypothesis: SparseHypothesis, location: Location, severity: float) -> SparseHypothesis:
        """Apply mutation to a hypothesis."""
        result = hypothesis.copy()

        # Simple mutation: randomly modify one dimension value
        if result.active_dimensions and random.random() < 0.5:
            dim = random.choice(result.active_dimensions)
            try:
                new_value = self.generator.generate_dimension_value(
                    dim, [location.context_region]
                )
                result.values[dim] = new_value
            except Exception:
                pass  # Keep original value

        # Add dimension with low probability
        if random.random() < 0.1:
            from cognitive_gen.dimension_pool import ALL_DIMENSIONS
            inactive = [d for d in ALL_DIMENSIONS if d not in result.values or result.values[d] is None]
            if inactive:
                new_dim = random.choice(inactive)
                try:
                    value = self.generator.generate_dimension_value(
                        new_dim, [location.context_region]
                    )
                    result.values[new_dim] = value
                except Exception:
                    pass

        # Remove dimension with low probability
        if random.random() < 0.1 and result.n_active > 3:
            dim = random.choice(result.active_dimensions)
            result.values[dim] = None

        return result

    def _inject_diversity(self, location: Location):
        """Inject fresh hypotheses with varied dimensions to escape local optima."""
        n_inject = max(1, len(location.population) // 3)

        # Generate fresh hypotheses with varied dimension groups
        # Use different groups than what's currently dominant in the population
        fresh = []
        samples_text = location.context_region[:3000]  # Limit context size

        for i in range(n_inject):
            try:
                # 50% templates, 50% random for diversity injection
                if i % 2 == 0:
                    template_name = get_random_template_name()
                    dimensions, _ = sample_from_template(template_name, dims_per_group=3)
                else:
                    dimensions, _ = sample_dimensions_by_group(n_groups=5, dims_per_group=3)
                dimension_list = get_dimension_prompt_section(dimensions)

                prompt = f"""Analyze this writing and generate a hypothesis about the writer's cognitive state.

Writing:
{samples_text}

Provide values for these dimensions:

{dimension_list}

Return as JSON object."""

                response = self.generator.generate_single(prompt)
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    import json
                    h = json.loads(response[start:end])
                    filtered = {d: v for d, v in h.items() if d in ALL_DIMENSIONS and v}
                    if filtered:
                        fresh.append(SparseHypothesis(values=filtered))
            except Exception:
                pass

        if not fresh:
            return

        # Replace worst members
        if location.fitness_scores:
            sorted_indices = sorted(
                range(len(location.fitness_scores)),
                key=lambda i: location.fitness_scores[i],
            )

            for i, fresh_hyp in zip(sorted_indices[:len(fresh)], fresh):
                location.population[i] = fresh_hyp
                location.fitness_scores[i] = 0.0

    def search(
        self,
        total_generations: int = 100,
        verbose: bool = True,
    ) -> GraphSearchResult:
        """
        Run graph-based search.

        Args:
            total_generations: Maximum generations to run
            verbose: Print progress

        Returns:
            GraphSearchResult with best hypothesis and statistics
        """
        start_time = time.time()

        # Initialize (skip if resuming from checkpoint)
        if self.start_generation == 0:
            self.initialize_graph()
            self.visualizer.write_startup(self.graph, total_generations)
            self.initialize_populations()
        else:
            # Resuming - show resumed state
            self.visualizer.write_startup(self.graph, total_generations, resuming_from=self.start_generation)
            if verbose:
                print(f"Resuming from generation {self.start_generation}")

        # Track best globally - initialize from current state when resuming
        best_global_slug = None
        best_global_fitness = float('-inf')
        best_global_hyp = None

        # When resuming, find best from restored state
        for slug, loc in self.graph.locations.items():
            if loc.best_ever_fitness > best_global_fitness:
                best_global_fitness = loc.best_ever_fitness
                best_global_slug = slug
                best_global_hyp = loc.best_ever.copy() if loc.best_ever else None

        for gen in range(self.start_generation, total_generations):
            # Phase 1a: Batch score all locations (single forward pass)
            self.score_all_locations_batch()

            # Phase 1b: Local evolution at each location (scoring already done)
            for location in self.graph.locations.values():
                self.evolve_location(location, skip_scoring=True)

            # Update global best
            for slug, loc in self.graph.locations.items():
                if loc.best_ever_fitness > best_global_fitness:
                    best_global_fitness = loc.best_ever_fitness
                    best_global_slug = slug
                    best_global_hyp = loc.best_ever.copy() if loc.best_ever else None

            # Phase 2: Migration (every epoch_length generations)
            is_epoch_end = (gen + 1) % self.epoch_length == 0

            if is_epoch_end:
                # Migration and cross-breeding
                epoch_result = self.migration.run_epoch(
                    self.graph, self.scorer, gen, batch_size=self.batch_size
                )

                if verbose and epoch_result['offspring_placed'] > 0:
                    print(f"  Gen {gen}: {epoch_result['offspring_placed']} offspring placed")

                # Check for meta-location creation
                new_metas = self.meta_manager.check_and_create_meta(self.graph, gen)
                for meta in new_metas:
                    # Calculate baseline for new meta-location
                    context, target = meta.get_random_target()
                    meta.baseline_ppl = self.scorer.baseline(context, target)
                    self.visualizer.add_event(f"Gen {gen}: Created {meta.slug}")

            # Record history
            stats = self.graph.get_statistics()
            stats['generation'] = gen
            stats['best_global_fitness'] = best_global_fitness
            stats['best_global_slug'] = best_global_slug
            self.history.append(stats)

            # Update visualization
            self.visualizer.write(
                self.graph,
                gen,
                total_generations,
                self.meta_manager,
                best_global_slug,
                best_global_fitness,
            )

            # Progress output
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: {stats['n_locations']} locs, "
                      f"best={best_global_fitness:.3f} at {best_global_slug}")

            # Save checkpoint periodically
            if self.checkpoint_manager.should_save(gen):
                hyperparams = {
                    'population_per_location': self.population_per_location,
                    'epoch_length': self.epoch_length,
                    'elite_size': self.elite_size,
                    'base_mutation_rate': self.base_mutation_rate,
                    'crossover_rate': self.crossover_rate,
                    'batch_size': self.batch_size,
                }
                self.checkpoint_manager.save_checkpoint(
                    graph=self.graph,
                    generation=gen,
                    total_generations=total_generations,
                    best_fitness=best_global_fitness,
                    best_location=best_global_slug or "",
                    hyperparameters=hyperparams,
                    meta_manager_state=self.meta_manager.get_status(self.graph),
                )
                if verbose:
                    print(f"  Checkpoint saved at gen {gen}")

        # Final visualization
        runtime = time.time() - start_time
        self.visualizer.write_final(
            self.graph,
            total_generations,
            best_global_slug,
            best_global_fitness,
            runtime,
        )

        # Compile result
        baseline_ppl = self.graph.locations[best_global_slug].baseline_ppl if best_global_slug else 0

        return GraphSearchResult(
            generations_completed=total_generations,
            best_hypothesis=best_global_hyp,
            best_location=best_global_slug,
            best_fitness=best_global_fitness,
            baseline_ppl=baseline_ppl,
            improvement=best_global_fitness,
            final_n_locations=self.graph.n_locations,
            final_n_meta_locations=self.graph.n_meta_locations,
            final_n_edges=self.graph.n_edges,
            history=self.history,
            meta_locations_created=self.meta_manager.created_metas,
        )


def run_graph_search(
    total_generations: int = 100,
    population_per_location: int = 15,
    n_neighbors: int = 3,
    epoch_length: int = 5,
    model_name: str = "mistralai/Mistral-7B-v0.3",
    batch_size: int = 32,  # Conservative batch size to avoid OOM with 2 models loaded
    verbose: bool = True,
    checkpoint_dir: str = "/tmp/graph_search_checkpoints",
    checkpoint_interval: int = 1,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    max_locations: Optional[int] = None,
) -> GraphSearchResult:
    """
    Convenience function to run graph search.

    Args:
        total_generations: Number of generations
        population_per_location: Population size per location
        n_neighbors: Neighbors for sparse graph init
        epoch_length: Generations between migration epochs
        model_name: Scorer model
        batch_size: Batch size for scoring (default 32 for safety)
        verbose: Print progress
        checkpoint_dir: Directory for saving checkpoints
        checkpoint_interval: Save checkpoint every N generations
        resume: Whether to resume from checkpoint
        checkpoint_path: Specific checkpoint file to resume from

    Returns:
        GraphSearchResult
    """
    print("Loading scorer model...")
    scorer = MetaScorer(model_name=model_name, use_v1_format=True)
    print("Scorer loaded")

    print("Loading generator...")
    generator = MetaHypothesisGenerator(use_local=True)
    print("Generator loaded")

    # Create search (graph will be created or restored from checkpoint)
    search = GraphSearch(
        scorer=scorer,
        generator=generator,
        graph=None,  # Will be initialized or restored
        population_per_location=population_per_location,
        epoch_length=epoch_length,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    # Resume from checkpoint if requested
    if resume or checkpoint_path:
        try:
            search.resume_from_checkpoint(checkpoint_path)
        except FileNotFoundError as e:
            if resume and not checkpoint_path:
                print(f"No checkpoint found, starting fresh")
            else:
                raise e
    else:
        # Create fresh graph
        search.graph = create_graph(
            population_per_location=population_per_location,
            n_neighbors=n_neighbors,
            max_locations=max_locations,
        )

    # Run
    return search.search(
        total_generations=total_generations,
        verbose=verbose,
    )
