"""
Edge class for tracking connections between locations.
"""

from dataclasses import dataclass, field


@dataclass
class Edge:
    """
    Weighted edge between two locations in the graph.

    Weight increases with successful cross-breeding and decreases with failures.
    When weight exceeds the merge threshold, locations may form a meta-location.
    """
    source: str  # Source location slug
    target: str  # Target location slug
    weight: float = 0.1  # Initial weight

    # Crossing statistics
    successful_crossings: int = 0
    failed_crossings: int = 0
    total_offspring_fitness: float = 0.0

    def record_crossing(self, offspring_fitness: float, parent_fitness: float):
        """
        Record the result of a cross-breeding attempt.

        Weight increase is based on % improvement over parent fitness.
        No penalty for failed placements.

        Args:
            offspring_fitness: Fitness of the offspring (higher = better)
            parent_fitness: Fitness of the better parent
        """
        self.total_offspring_fitness += offspring_fitness

        if offspring_fitness > parent_fitness and parent_fitness > 0:
            self.successful_crossings += 1
            # Weight increase = % improvement over parent
            improvement = (offspring_fitness - parent_fitness) / parent_fitness
            self.weight += improvement
        else:
            # Track as failed but no penalty
            self.failed_crossings += 1

    def record_placement(self, offspring_fitness: float, parent_fitness: float, placed: bool):
        """
        Record the result of a cross-breeding attempt based on actual placement.

        Tiered weight increase:
        - Placement (beating worst) = weak signal, small bonus
        - Beating parent = strong signal, bonus scales with % improvement

        Args:
            offspring_fitness: Fitness of the offspring (higher = better)
            parent_fitness: Fitness of the better parent
            placed: Whether the offspring was placed in at least one population
        """
        self.total_offspring_fitness += offspring_fitness

        if placed:
            base_bonus = 0.02  # Small bonus for any viable placement
            if offspring_fitness > parent_fitness and parent_fitness > 0:
                # Strong signal - crossover improved fitness
                self.successful_crossings += 1
                improvement = (offspring_fitness - parent_fitness) / parent_fitness
                self.weight += base_bonus + improvement
            else:
                # Weak signal - viable but didn't beat parent
                self.successful_crossings += 1
                self.weight += base_bonus
        else:
            # Not placed - no weight change
            self.failed_crossings += 1

    @property
    def total_crossings(self) -> int:
        return self.successful_crossings + self.failed_crossings

    @property
    def success_rate(self) -> float:
        """Success rate of cross-breeding attempts."""
        if self.total_crossings == 0:
            return 0.5  # Default when no data
        return self.successful_crossings / self.total_crossings

    @property
    def avg_offspring_fitness(self) -> float:
        """Average fitness of offspring from this edge."""
        if self.total_crossings == 0:
            return 0.0
        return self.total_offspring_fitness / self.total_crossings

    def to_dict(self) -> dict:
        """Serialize edge to dictionary."""
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'successful_crossings': self.successful_crossings,
            'failed_crossings': self.failed_crossings,
            'success_rate': self.success_rate,
            'avg_offspring_fitness': self.avg_offspring_fitness,
        }

    def __repr__(self) -> str:
        return (f"Edge({self.source} <-> {self.target}, "
                f"weight={self.weight:.3f}, "
                f"{self.successful_crossings}/{self.total_crossings} successful)")
