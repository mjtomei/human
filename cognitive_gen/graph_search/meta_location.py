"""
Meta-location: A higher-level location spanning multiple essays.

Meta-locations are ADDITIONAL nodes - original locations remain.
Overlapping meta-locations are allowed.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

from .location import Location, LocationStagnationState
from .edge import Edge

if TYPE_CHECKING:
    from .location_graph import LocationGraph

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from cognitive_gen.meta_search import SparseHypothesis


class MetaLocation(Location):
    """
    A location spanning multiple essays.

    Original locations continue to exist - meta-locations are ADDITIONAL.
    Overlapping meta-locations are allowed.
    """

    def __init__(
        self,
        locations: list[Location],
        generation: int,
        population_size: int = 15,
    ):
        """
        Create a meta-location from component locations.

        Args:
            locations: Component locations to combine
            generation: Generation when this was created
            population_size: Target population size
        """
        self.member_slugs = [loc.slug for loc in locations]
        self.creation_generation = generation
        self.is_meta = True

        # Create slug
        slug = "meta_" + "_".join(sorted(self.member_slugs))

        # Combine essay content
        combined_content = "\n\n---\n\n".join(loc.essay_content for loc in locations)

        # Combine context regions
        context_region = "\n\n---\n\n".join(loc.context_region for loc in locations)

        # Combine target regions from all members
        target_region = "\n\n---\n\n".join(loc.target_region for loc in locations)

        # Seed population from best of each component
        population = []
        for loc in locations:
            if loc.best_ever:
                population.append(loc.best_ever.copy())
            # Also take top performers
            if loc.population and loc.fitness_scores:
                sorted_pop = sorted(
                    zip(loc.population, loc.fitness_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for hyp, _ in sorted_pop[:2]:
                    population.append(hyp.copy())

        # Trim to target size
        if len(population) > population_size:
            population = population[:population_size]

        # Initialize parent class
        super().__init__(
            slug=slug,
            essay_content=combined_content,
            context_region=context_region,
            target_region=target_region,
            population=population,
            fitness_scores=[0.0] * len(population),  # Will be scored
            edges={},
            stagnation_state=LocationStagnationState(),
            best_ever=None,
            best_ever_fitness=float('-inf'),
        )

    def to_dict(self) -> dict:
        """Serialize meta-location state."""
        base = super().to_dict()
        base['member_slugs'] = self.member_slugs
        base['creation_generation'] = self.creation_generation
        base['is_meta'] = True
        return base


class MetaLocationManager:
    """
    Manage creation of meta-locations based on edge weights and stagnation.

    The merge threshold is calibrated dynamically based on when populations
    start stagnating, then adjusted downward as more locations become
    deeply stuck.
    """

    def __init__(
        self,
        min_threshold: float = 0.2,
        stagnation_calibration_ratio: float = 0.3,
        population_size: int = 15,
    ):
        """
        Args:
            min_threshold: Minimum merge threshold
            stagnation_calibration_ratio: Calibrate when this fraction stagnated once
            population_size: Population size for new meta-locations
        """
        self.min_threshold = min_threshold
        self.stagnation_calibration_ratio = stagnation_calibration_ratio
        self.population_size = population_size

        self.base_threshold: Optional[float] = None
        self.calibrated: bool = False
        self.created_metas: list[str] = []

    def calibrate_threshold(self, graph: 'LocationGraph'):
        """
        Calibrate initial threshold based on stagnation patterns.

        Called when enough locations have stagnated at least once.
        """
        if self.calibrated:
            return

        # Check how many locations have stagnated at least once
        stagnated_once = sum(
            1 for loc in graph.locations.values()
            if loc.stagnation_state.diversity_injections > 0
        )
        stagnation_rate = stagnated_once / len(graph.locations) if graph.locations else 0

        if stagnation_rate >= self.stagnation_calibration_ratio:
            # Calibrate threshold based on current max edge weight
            edges = graph.get_all_edges()
            if edges:
                max_weight = max(e.weight for e in edges)
                self.base_threshold = max_weight * 1.5  # 50% above current max
            else:
                self.base_threshold = 1.0

            self.calibrated = True
            print(f"Calibrated merge threshold: {self.base_threshold:.3f} "
                  f"(at {stagnation_rate:.1%} stagnation rate)")

    def get_merge_threshold(self, graph: 'LocationGraph') -> float:
        """
        Get current merge threshold based on deep stagnation.

        Returns high value if not calibrated yet.
        """
        if not self.calibrated or self.base_threshold is None:
            return 999.0  # Effectively disable merging until calibrated

        # Compute deeply stuck ratio
        deeply_stuck = sum(
            1 for loc in graph.locations.values()
            if loc.stagnation_state.deeply_stuck
        )
        deeply_stuck_ratio = deeply_stuck / len(graph.locations) if graph.locations else 0

        # Lower threshold as more locations become deeply stuck
        threshold = self.base_threshold - (self.base_threshold - self.min_threshold) * deeply_stuck_ratio

        return threshold

    def check_and_create_meta(
        self,
        graph: 'LocationGraph',
        generation: int,
    ) -> list[MetaLocation]:
        """
        Check for edges exceeding threshold and create meta-locations.

        Original locations REMAIN - meta-locations are additional.
        """
        # Try to calibrate if not done yet
        self.calibrate_threshold(graph)

        threshold = self.get_merge_threshold(graph)
        created = []

        for edge in graph.get_all_edges():
            if edge.weight < threshold:
                continue

            loc_a = graph.locations.get(edge.source)
            loc_b = graph.locations.get(edge.target)

            if not loc_a or not loc_b:
                continue

            # Check if this meta already exists
            meta_slug = "meta_" + "_".join(sorted([edge.source, edge.target]))
            if meta_slug in graph.locations:
                continue

            # Create meta-location
            meta = MetaLocation(
                locations=[loc_a, loc_b],
                generation=generation,
                population_size=self.population_size,
            )

            # Add to graph
            graph.add_location(meta)
            self.created_metas.append(meta.slug)

            # Connect meta to neighbors of both components
            neighbor_slugs = set(loc_a.edges.keys()) | set(loc_b.edges.keys())
            for neighbor_slug in neighbor_slugs:
                if neighbor_slug not in [loc_a.slug, loc_b.slug]:
                    meta.edges[neighbor_slug] = Edge(
                        source=meta.slug,
                        target=neighbor_slug,
                        weight=0.1,
                    )

            # Reset deeply_stuck for components (they got help)
            loc_a.stagnation_state.reset_deeply_stuck()
            loc_b.stagnation_state.reset_deeply_stuck()

            created.append(meta)

            print(f"Created meta-location: {meta.slug} "
                  f"(edge weight {edge.weight:.3f} >= threshold {threshold:.3f})")

        return created

    def get_status(self, graph: 'LocationGraph') -> dict:
        """Get meta-location manager status."""
        return {
            'calibrated': self.calibrated,
            'base_threshold': self.base_threshold,
            'current_threshold': self.get_merge_threshold(graph) if self.calibrated else None,
            'created_metas': len(self.created_metas),
            'meta_slugs': self.created_metas.copy(),
        }
