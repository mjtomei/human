"""
Migration system for hypothesis movement and cross-breeding between locations.
"""

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from cognitive_gen.meta_search import SparseHypothesis

if TYPE_CHECKING:
    from .location_graph import LocationGraph
    from .location import Location


@dataclass
class CrossBreedingResult:
    """Result of a cross-breeding attempt."""
    edge_source: str
    edge_target: str
    offspring: SparseHypothesis
    fitness_on_source: float
    fitness_on_target: float
    success: bool
    placed_at: Optional[str] = None


class MigrationSystem:
    """
    Handle migration and cross-breeding between locations.

    Migration: Top performers can move to neighboring locations
    Cross-breeding: Parents from different locations produce offspring
    """

    def __init__(
        self,
        migration_rate: float = 0.1,
        breeding_attempts_per_edge: int = 3,
        success_threshold: float = 0.0,
    ):
        """
        Args:
            migration_rate: Fraction of population that migrates per epoch
            breeding_attempts_per_edge: Cross-breeding attempts per edge per epoch
            success_threshold: Minimum fitness to count as successful crossing
        """
        self.migration_rate = migration_rate
        self.breeding_attempts_per_edge = breeding_attempts_per_edge
        self.success_threshold = success_threshold

    def select_migrants(self, location: 'Location', n: int) -> list[SparseHypothesis]:
        """
        Select hypotheses to migrate from a location.

        Prefers high-fitness hypotheses.
        """
        if not location.population or not location.fitness_scores:
            return []

        n = min(n, len(location.population))

        # Weight selection by fitness (softmax-like)
        fitness = location.fitness_scores
        min_fit = min(fitness)
        weights = [f - min_fit + 0.1 for f in fitness]  # Shift to positive
        total = sum(weights)
        probs = [w / total for w in weights]

        # Sample without replacement
        indices = []
        remaining_probs = list(probs)
        remaining_indices = list(range(len(location.population)))

        for _ in range(n):
            if not remaining_indices:
                break
            # Normalize remaining probs
            total = sum(remaining_probs)
            if total <= 0:
                break
            normalized = [p / total for p in remaining_probs]

            # Sample
            r = random.random()
            cumsum = 0
            for i, (idx, p) in enumerate(zip(remaining_indices, normalized)):
                cumsum += p
                if r <= cumsum:
                    indices.append(idx)
                    remaining_indices.pop(i)
                    remaining_probs.pop(i)
                    break

        return [location.population[i].copy() for i in indices]

    def choose_destination(
        self,
        source: 'Location',
        graph: 'LocationGraph',
    ) -> Optional[str]:
        """
        Choose migration destination weighted by edge strength.

        Returns slug of destination or None if no valid destinations.
        """
        if not source.edges:
            return None

        # Weight by edge strength
        destinations = []
        weights = []

        for target_slug, edge in source.edges.items():
            if target_slug in graph.locations:
                destinations.append(target_slug)
                weights.append(edge.weight + 0.1)  # Add base weight

        if not destinations:
            return None

        total = sum(weights)
        probs = [w / total for w in weights]

        r = random.random()
        cumsum = 0
        for dest, p in zip(destinations, probs):
            cumsum += p
            if r <= cumsum:
                return dest

        return destinations[-1]

    def migrate(
        self,
        graph: 'LocationGraph',
        generation: int,
    ) -> list[dict]:
        """
        Execute migration phase.

        Returns list of migration events.
        """
        events = []

        for source_slug, source_loc in graph.locations.items():
            if not source_loc.population:
                continue

            # Number of migrants based on population size
            n_migrants = max(1, int(len(source_loc.population) * self.migration_rate))
            migrants = self.select_migrants(source_loc, n_migrants)

            for migrant in migrants:
                dest_slug = self.choose_destination(source_loc, graph)
                if dest_slug and dest_slug in graph.locations:
                    dest_loc = graph.locations[dest_slug]
                    dest_loc.receive_migrant(migrant)

                    events.append({
                        'type': 'migration',
                        'from': source_slug,
                        'to': dest_slug,
                        'generation': generation,
                    })

        return events

    def select_parent(self, location: 'Location') -> Optional[tuple[SparseHypothesis, float]]:
        """Select a parent for breeding, weighted by fitness.

        Returns:
            Tuple of (hypothesis, fitness) or None if selection fails
        """
        if not location.population or not location.fitness_scores:
            return None

        # Ensure fitness_scores matches population size
        if len(location.fitness_scores) != len(location.population):
            return None

        # Tournament selection
        k = min(3, len(location.population))
        indices = random.sample(range(len(location.population)), k)
        best_idx = max(indices, key=lambda i: location.fitness_scores[i])

        return location.population[best_idx], location.fitness_scores[best_idx]

    def crossover(
        self,
        parent_a: SparseHypothesis,
        parent_b: SparseHypothesis,
        loc_a_slug: str = None,
        loc_b_slug: str = None,
        generation: int = 0,
    ) -> SparseHypothesis:
        """
        Create offspring from two parents.

        Uses linkage-aware crossover: inherit dimension groups as units.
        Groups are included probabilistically to maintain average parent size
        (prevents dimension inflation over generations).

        Tracks dimension origins for lineage analysis.
        """
        from cognitive_gen.dimension_pool import DIMENSION_TO_GROUP

        # Collect all active dimensions from both parents
        dims_a = set(parent_a.active_dimensions)
        dims_b = set(parent_b.active_dimensions)

        # Group dimensions by category
        groups_a = {}
        for dim in dims_a:
            group = DIMENSION_TO_GROUP.get(dim, 'other')
            if group not in groups_a:
                groups_a[group] = []
            groups_a[group].append(dim)

        groups_b = {}
        for dim in dims_b:
            group = DIMENSION_TO_GROUP.get(dim, 'other')
            if group not in groups_b:
                groups_b[group] = []
            groups_b[group].append(dim)

        # Calculate target dimensions (average of parents)
        n_dims_a = len(dims_a)
        n_dims_b = len(dims_b)
        target_dims = (n_dims_a + n_dims_b) / 2

        # Estimate total dims if we included all groups
        all_groups = set(groups_a.keys()) | set(groups_b.keys())
        total_if_all = 0
        for group in all_groups:
            if group in groups_a and group in groups_b:
                # Would pick one randomly, so use average
                total_if_all += (len(groups_a[group]) + len(groups_b[group])) / 2
            elif group in groups_a:
                total_if_all += len(groups_a[group])
            else:
                total_if_all += len(groups_b[group])

        # Probability to include each group to hit target on average
        if total_if_all > 0:
            p_include = min(1.0, target_dims / total_if_all)
        else:
            p_include = 1.0

        # Select groups probabilistically
        offspring_values = {}
        offspring_origins = {}
        included_groups = []

        for group in all_groups:
            if random.random() < p_include:
                included_groups.append(group)

                # Pick which parent's version to use
                has_a = group in groups_a
                has_b = group in groups_b

                if has_a and has_b:
                    # Both have this group - choose randomly
                    if random.random() < 0.5:
                        source = parent_a
                        source_slug = loc_a_slug
                        dims = groups_a[group]
                    else:
                        source = parent_b
                        source_slug = loc_b_slug
                        dims = groups_b[group]
                elif has_a:
                    source = parent_a
                    source_slug = loc_a_slug
                    dims = groups_a[group]
                else:
                    source = parent_b
                    source_slug = loc_b_slug
                    dims = groups_b[group]

                # Copy values and origins from chosen parent
                for dim in dims:
                    offspring_values[dim] = source.values.get(dim)
                    # Preserve existing origin if present, else set to source location
                    if dim in source.origins:
                        offspring_origins[dim] = source.origins[dim]
                    elif source_slug:
                        offspring_origins[dim] = {'location': source_slug, 'generation': generation}

        # Ensure at least some dimensions (fallback if unlucky)
        if len(offspring_values) < 2:
            # Include at least one group from each parent if possible
            for group in list(groups_a.keys())[:1] + list(groups_b.keys())[:1]:
                if group in groups_a:
                    for dim in groups_a[group]:
                        offspring_values[dim] = parent_a.values.get(dim)
                        if dim in parent_a.origins:
                            offspring_origins[dim] = parent_a.origins[dim]
                        elif loc_a_slug:
                            offspring_origins[dim] = {'location': loc_a_slug, 'generation': generation}
                elif group in groups_b:
                    for dim in groups_b[group]:
                        offspring_values[dim] = parent_b.values.get(dim)
                        if dim in parent_b.origins:
                            offspring_origins[dim] = parent_b.origins[dim]
                        elif loc_b_slug:
                            offspring_origins[dim] = {'location': loc_b_slug, 'generation': generation}

        return SparseHypothesis(values=offspring_values, origins=offspring_origins)

    def cross_breed(
        self,
        graph: 'LocationGraph',
        scorer,  # MetaScorer
        generation: int,
        batch_size: int = 200,
    ) -> list[CrossBreedingResult]:
        """
        Perform cross-breeding between connected locations using batch scoring.

        Args:
            graph: The location graph
            scorer: Scorer for evaluating offspring
            generation: Current generation number
            batch_size: Batch size for scoring (default 200 for 120GB memory)

        Returns:
            List of cross-breeding results
        """
        # Phase 1: Generate all offspring and build scoring batch
        offspring_info = []  # [(edge, offspring, loc_a, loc_b, ctx_a, tgt_a, ctx_b, tgt_b), ...]

        for edge in graph.get_all_edges():
            loc_a = graph.locations.get(edge.source)
            loc_b = graph.locations.get(edge.target)

            if not loc_a or not loc_b:
                continue

            if not loc_a.population or not loc_b.population:
                continue

            for _ in range(self.breeding_attempts_per_edge):
                # Select parents (returns tuple of hypothesis, fitness)
                parent_a_result = self.select_parent(loc_a)
                parent_b_result = self.select_parent(loc_b)

                if not parent_a_result or not parent_b_result:
                    continue

                parent_a, fitness_a = parent_a_result
                parent_b, fitness_b = parent_b_result

                # Create offspring with lineage tracking
                offspring = self.crossover(
                    parent_a, parent_b,
                    loc_a_slug=edge.source,
                    loc_b_slug=edge.target,
                    generation=generation,
                )

                # Get scoring contexts
                ctx_a, tgt_a = loc_a.get_random_target()
                ctx_b, tgt_b = loc_b.get_random_target()

                # Track better parent fitness for edge weight calculation
                better_parent_fitness = max(fitness_a, fitness_b)

                offspring_info.append((edge, offspring, loc_a, loc_b, ctx_a, tgt_a, ctx_b, tgt_b, better_parent_fitness))

        if not offspring_info:
            return []

        # Phase 2: Build batch for scoring (2 scores per offspring - on each parent location)
        score_batch = []
        for edge, offspring, loc_a, loc_b, ctx_a, tgt_a, ctx_b, tgt_b, _ in offspring_info:
            # Score on location A
            score_batch.append((ctx_a, tgt_a, offspring, loc_a.baseline_ppl))
            # Score on location B
            score_batch.append((ctx_b, tgt_b, offspring, loc_b.baseline_ppl))

        # Phase 3: Batch score all offspring
        try:
            score_results = scorer.score_batch(score_batch, max_batch_size=batch_size)
        except Exception as e:
            print(f"Warning: Batch scoring failed in cross_breed: {e}")
            # Fallback to individual scoring
            score_results = []
            for ctx, tgt, hyp, baseline in score_batch:
                try:
                    fitness, ppl = scorer.score(ctx, tgt, hyp, baseline)
                    score_results.append((fitness, ppl))
                except Exception:
                    score_results.append((0.0, float('inf')))

        # Phase 4: Process results
        results = []
        for i, (edge, offspring, loc_a, loc_b, _, _, _, _, parent_fitness) in enumerate(offspring_info):
            fit_a, _ = score_results[i * 2]      # Score on loc A
            fit_b, _ = score_results[i * 2 + 1]  # Score on loc B

            # Determine fitness
            best_fit = max(fit_a, fit_b)

            # Place offspring if better than worst in either location
            placed_at = None

            # Can join location A if better than worst
            if loc_a.fitness_scores and fit_a > min(loc_a.fitness_scores):
                worst_idx = loc_a.fitness_scores.index(min(loc_a.fitness_scores))
                loc_a.population[worst_idx] = offspring.copy()
                loc_a.fitness_scores[worst_idx] = fit_a
                placed_at = edge.source

            # Can also join location B if better than worst
            if loc_b.fitness_scores and fit_b > min(loc_b.fitness_scores):
                worst_idx = loc_b.fitness_scores.index(min(loc_b.fitness_scores))
                loc_b.population[worst_idx] = offspring.copy()
                loc_b.fitness_scores[worst_idx] = fit_b
                if placed_at:
                    placed_at = f"{placed_at}+{edge.target}"
                else:
                    placed_at = edge.target

            # Record crossing - weight increase based on % improvement over parent
            placed = placed_at is not None
            success = best_fit > parent_fitness  # Success means improving over parent

            edge.record_placement(best_fit, parent_fitness, placed)

            # Also record on reverse edge if exists
            reverse_edge = graph.get_edge(edge.target, edge.source)
            if reverse_edge:
                reverse_edge.record_placement(best_fit, parent_fitness, placed)

            results.append(CrossBreedingResult(
                edge_source=edge.source,
                edge_target=edge.target,
                offspring=offspring,
                fitness_on_source=fit_a,
                fitness_on_target=fit_b,
                success=success,
                placed_at=placed_at,
            ))

        return results

    def run_epoch(
        self,
        graph: 'LocationGraph',
        scorer,
        generation: int,
        batch_size: int = 200,
    ) -> dict:
        """
        Run a complete migration epoch (migration + cross-breeding).

        Returns summary of events.
        """
        # Migration
        migration_events = self.migrate(graph, generation)

        # Cross-breeding (with batch scoring)
        breeding_results = self.cross_breed(graph, scorer, generation, batch_size=batch_size)

        # Summarize
        successful_breedings = sum(1 for r in breeding_results if r.success)
        placements = sum(1 for r in breeding_results if r.placed_at)

        return {
            'generation': generation,
            'migrations': len(migration_events),
            'breeding_attempts': len(breeding_results),
            'successful_breedings': successful_breedings,
            'offspring_placed': placements,
            'migration_events': migration_events,
            'breeding_results': breeding_results,
        }
