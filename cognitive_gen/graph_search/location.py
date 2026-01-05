"""
Location class representing a node in the graph (one essay).
"""

import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

# Import from parent package
import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from cognitive_gen.meta_search import SparseHypothesis, StagnationDetector

if TYPE_CHECKING:
    from .edge import Edge


@dataclass
class LocationStagnationState:
    """
    Track stagnation state with hierarchical response.

    Level 1: Within-location (mutation increase, diversity injection)
    Level 2: Cross-location (mark deeply stuck for merge consideration)
    """
    stagnation: StagnationDetector = field(default_factory=StagnationDetector)
    diversity_injections: int = 0
    fitness_at_injection: Optional[float] = None
    deeply_stuck: bool = False
    generations_since_injection: int = 0

    def update(self, best_fitness: float, population_fitness: list[float]):
        """Update stagnation tracking."""
        self.stagnation.update(best_fitness, population_fitness)
        self.generations_since_injection += 1

    def handle(self, current_fitness: float) -> str:
        """
        Determine action based on stagnation state.

        Returns:
            "normal" - Continue as usual
            "increase_mutation" - Increase mutation rate
            "inject_diversity" - Inject fresh hypotheses
            "deeply_stuck" - Local solutions exhausted, need graph-level help
        """
        if not self.stagnation.is_stagnating():
            return "normal"

        severity = self.stagnation.severity()

        if severity > 0.8:
            # Check if previous injection helped
            if self.fitness_at_injection is not None:
                improvement = current_fitness - self.fitness_at_injection
                if improvement < 0.01 and self.generations_since_injection > 5:
                    self.deeply_stuck = True
                    return "deeply_stuck"  # Injection didn't help

            # Try diversity injection
            self.fitness_at_injection = current_fitness
            self.generations_since_injection = 0
            self.diversity_injections += 1
            return "inject_diversity"

        return "increase_mutation"

    def reset_deeply_stuck(self):
        """Reset deeply stuck flag (called after meta-location creation)."""
        self.deeply_stuck = False
        self.fitness_at_injection = None

    def severity(self) -> float:
        """Get current stagnation severity."""
        return self.stagnation.severity()


@dataclass
class Location:
    """
    A node in the graph representing one essay.

    Each location has:
    - Its own population of hypotheses
    - Context region (first 60%, for hypothesis generation)
    - Target region (last 40%, for perplexity scoring)
    - Edges to neighboring locations
    - Stagnation tracking
    """
    slug: str
    essay_content: str
    context_region: str = ""

    # Population
    population: list[SparseHypothesis] = field(default_factory=list)
    fitness_scores: list[float] = field(default_factory=list)

    # Edges to other locations
    edges: dict[str, 'Edge'] = field(default_factory=dict)

    # Tracking
    stagnation_state: LocationStagnationState = field(default_factory=LocationStagnationState)
    best_ever: Optional[SparseHypothesis] = None
    best_ever_fitness: float = float('-inf')
    baseline_ppl: float = 0.0
    generation: int = 0

    # History
    history: list[dict] = field(default_factory=list)

    # Target region (last 40% of essay for scoring)
    target_region: str = ""

    # Config
    context_ratio: float = 0.6

    def __post_init__(self):
        """Split essay into context and target regions."""
        if self.essay_content and not self.context_region:
            self._split_essay()

    def _split_essay(self):
        """Split essay into non-overlapping context and target regions."""
        content = self.essay_content

        # Find split point
        split_point = int(len(content) * self.context_ratio)

        # Try to split at paragraph boundary
        search_start = max(0, split_point - 200)
        search_end = min(len(content), split_point + 200)

        # Look for double newline (paragraph break)
        best_split = split_point
        for i in range(search_start, search_end):
            if i < len(content) - 1 and content[i:i+2] == '\n\n':
                if abs(i - split_point) < abs(best_split - split_point):
                    best_split = i + 2

        self.context_region = content[:best_split].strip()
        self.target_region = content[best_split:].strip()

    def get_random_target(self) -> tuple[str, str]:
        """
        Get (context, target) pair for scoring.

        Returns:
            (context_region, target_region)
        """
        if not self.target_region:
            # Fallback: use full essay split
            return self.context_region, self.essay_content[len(self.context_region):]

        return self.context_region, self.target_region

    def get_compute_weight(self) -> float:
        """
        Get compute allocation weight based on health.

        Returns:
            1.0 for healthy locations, lower for stagnant ones
        """
        if self.stagnation_state.deeply_stuck:
            return 0.3  # Minimal compute, waiting for merge
        elif self.stagnation_state.severity() > 0.5:
            return 0.6  # Reduced compute
        else:
            return 1.0  # Full compute

    def update_best(self):
        """Update best_ever if current best is better."""
        if not self.fitness_scores:
            return

        best_idx = max(range(len(self.fitness_scores)),
                       key=lambda i: self.fitness_scores[i])
        best_fitness = self.fitness_scores[best_idx]

        if best_fitness > self.best_ever_fitness:
            self.best_ever = self.population[best_idx].copy()
            self.best_ever_fitness = best_fitness

    def record_history(self):
        """Record current state to history."""
        self.history.append({
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores) if self.fitness_scores else 0,
            'mean_fitness': sum(self.fitness_scores) / len(self.fitness_scores) if self.fitness_scores else 0,
            'pop_size': len(self.population),
            'stagnation_severity': self.stagnation_state.severity(),
            'deeply_stuck': self.stagnation_state.deeply_stuck,
        })

    def receive_migrant(self, hypothesis: SparseHypothesis, fitness: Optional[float] = None):
        """
        Receive a migrant hypothesis from another location.

        If fitness is provided.and better than worst, replaces worst member.
        Otherwise appends if under capacity.
        """
        if fitness is not None and self.fitness_scores:
            worst_fitness = min(self.fitness_scores)
            if fitness > worst_fitness:
                worst_idx = self.fitness_scores.index(worst_fitness)
                self.population[worst_idx] = hypothesis.copy()
                self.fitness_scores[worst_idx] = fitness
                return

        # Just append (will be trimmed during evolution if over capacity)
        self.population.append(hypothesis.copy())
        # Always append a fitness score (0.0 if not provided)
        self.fitness_scores.append(fitness if fitness is not None else 0.0)

    @property
    def n_active_dims_avg(self) -> float:
        """Average number of active dimensions in population."""
        if not self.population:
            return 0
        return sum(h.n_active for h in self.population) / len(self.population)

    def to_dict(self) -> dict:
        """Serialize location state."""
        return {
            'slug': self.slug,
            'generation': self.generation,
            'pop_size': len(self.population),
            'best_ever_fitness': self.best_ever_fitness,
            'best_ever_dims': self.best_ever.n_active if self.best_ever else 0,
            'stagnation_severity': self.stagnation_state.severity(),
            'deeply_stuck': self.stagnation_state.deeply_stuck,
            'diversity_injections': self.stagnation_state.diversity_injections,
            'n_edges': len(self.edges),
            'compute_weight': self.get_compute_weight(),
        }

    def __repr__(self) -> str:
        status = "DEEP" if self.stagnation_state.deeply_stuck else f"stag={self.stagnation_state.severity():.2f}"
        return (f"Location({self.slug}, pop={len(self.population)}, "
                f"best={self.best_ever_fitness:.3f}, {status})")
