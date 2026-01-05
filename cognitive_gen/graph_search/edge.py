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

    # Weight update parameters
    success_bonus: float = 0.1
    failure_penalty: float = 0.02

    def record_crossing(self, offspring_fitness: float, success_threshold: float = 0.0):
        """
        Record the result of a cross-breeding attempt.

        Args:
            offspring_fitness: Fitness of the offspring (higher = better)
            success_threshold: Minimum fitness to count as success
        """
        self.total_offspring_fitness += offspring_fitness

        if offspring_fitness > success_threshold:
            self.successful_crossings += 1
            # Bonus scales with fitness
            self.weight += self.success_bonus * (1 + offspring_fitness)
        else:
            self.failed_crossings += 1
            self.weight = max(0, self.weight - self.failure_penalty)

    def record_placement(self, offspring_fitness: float, placed: bool):
        """
        Record the result of a cross-breeding attempt based on actual placement.

        This method uses whether the offspring was placed in a population
        (i.e., beat the worst member) as the success criterion, rather than
        just having positive fitness.

        Args:
            offspring_fitness: Fitness of the offspring (higher = better)
            placed: Whether the offspring was placed in at least one population
        """
        self.total_offspring_fitness += offspring_fitness

        if placed:
            self.successful_crossings += 1
            # Bonus scales with fitness
            self.weight += self.success_bonus * (1 + offspring_fitness)
        else:
            self.failed_crossings += 1
            self.weight = max(0, self.weight - self.failure_penalty)

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
