"""
Location graph structure managing all locations and their connections.
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator
import numpy as np

from .location import Location
from .edge import Edge
from .essay_index import EssayIndex, create_index


class LocationGraph:
    """
    Graph of locations (essays) with weighted edges.

    Supports sparse initialization where each location is connected
    to its most similar neighbors.
    """

    def __init__(
        self,
        essay_index: Optional[EssayIndex] = None,
        population_per_location: int = 15,
        n_neighbors: int = 3,
    ):
        self.essay_index = essay_index
        self.population_per_location = population_per_location
        self.n_neighbors = n_neighbors

        self.locations: dict[str, Location] = {}
        self._edge_cache: Optional[list[Edge]] = None

    def initialize(self, mode: str = "sparse", max_locations: Optional[int] = None):
        """
        Create locations and edges.

        Args:
            mode: "sparse" (connect to N neighbors) or "full" (all connected)
            max_locations: Limit number of locations (for testing)
        """
        if self.essay_index is None:
            self.essay_index = create_index()

        # Create location for each essay
        all_slugs = self.essay_index.get_all_slugs()
        if max_locations is not None:
            all_slugs = all_slugs[:max_locations]

        for slug in all_slugs:
            essay_content = self.essay_index.get_essay(slug)
            if essay_content:
                location = Location(
                    slug=slug,
                    essay_content=essay_content,
                )
                self.locations[slug] = location

        print(f"Created {len(self.locations)} locations")

        # Create edges
        if mode == "sparse":
            self._init_sparse()
        elif mode == "full":
            self._init_full()
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        self._edge_cache = None  # Invalidate cache

    def _init_sparse(self):
        """Connect each location to N most similar neighbors."""
        for slug, location in self.locations.items():
            neighbors = self.essay_index.get_related_essays(slug, n=self.n_neighbors)

            for neighbor_slug in neighbors:
                if neighbor_slug in self.locations and neighbor_slug != slug:
                    # Create bidirectional edge
                    edge = Edge(source=slug, target=neighbor_slug, weight=0.1)
                    location.edges[neighbor_slug] = edge

                    # Add reverse edge if not exists
                    neighbor_loc = self.locations[neighbor_slug]
                    if slug not in neighbor_loc.edges:
                        reverse_edge = Edge(source=neighbor_slug, target=slug, weight=0.1)
                        neighbor_loc.edges[slug] = reverse_edge

        n_edges = len(self.get_all_edges())
        print(f"Created {n_edges} edges (sparse mode, {self.n_neighbors} neighbors each)")

    def _init_full(self):
        """Connect all locations to each other."""
        slugs = list(self.locations.keys())
        for i, slug_a in enumerate(slugs):
            for slug_b in slugs[i+1:]:
                edge_ab = Edge(source=slug_a, target=slug_b, weight=0.1)
                edge_ba = Edge(source=slug_b, target=slug_a, weight=0.1)

                self.locations[slug_a].edges[slug_b] = edge_ab
                self.locations[slug_b].edges[slug_a] = edge_ba

        n_edges = len(self.get_all_edges())
        print(f"Created {n_edges} edges (full mode)")

    def add_location(self, location: Location):
        """Add a new location (e.g., meta-location)."""
        self.locations[location.slug] = location
        self._edge_cache = None

    def remove_location(self, slug: str):
        """Remove a location and its edges."""
        if slug in self.locations:
            del self.locations[slug]

            # Remove edges pointing to this location
            for loc in self.locations.values():
                if slug in loc.edges:
                    del loc.edges[slug]

            self._edge_cache = None

    def get_all_edges(self) -> list[Edge]:
        """Get all unique edges in the graph."""
        if self._edge_cache is not None:
            return self._edge_cache

        seen = set()
        edges = []

        for loc in self.locations.values():
            for target_slug, edge in loc.edges.items():
                # Use sorted tuple as key to avoid duplicates
                key = tuple(sorted([edge.source, edge.target]))
                if key not in seen:
                    edges.append(edge)
                    seen.add(key)

        self._edge_cache = edges
        return edges

    def get_edge(self, slug_a: str, slug_b: str) -> Optional[Edge]:
        """Get edge between two locations if it exists."""
        if slug_a in self.locations:
            return self.locations[slug_a].edges.get(slug_b)
        return None

    def iter_locations(self) -> Iterator[Location]:
        """Iterate over all locations."""
        return iter(self.locations.values())

    def iter_base_locations(self) -> Iterator[Location]:
        """Iterate over base (non-meta) locations only."""
        for loc in self.locations.values():
            if not loc.slug.startswith('meta_'):
                yield loc

    def iter_meta_locations(self) -> Iterator[Location]:
        """Iterate over meta-locations only."""
        for loc in self.locations.values():
            if loc.slug.startswith('meta_'):
                yield loc

    @property
    def n_locations(self) -> int:
        return len(self.locations)

    @property
    def n_base_locations(self) -> int:
        return sum(1 for _ in self.iter_base_locations())

    @property
    def n_meta_locations(self) -> int:
        return sum(1 for _ in self.iter_meta_locations())

    @property
    def n_edges(self) -> int:
        return len(self.get_all_edges())

    @property
    def total_population(self) -> int:
        return sum(len(loc.population) for loc in self.locations.values())

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        edges = self.get_all_edges()
        weights = [e.weight for e in edges]

        deeply_stuck = sum(1 for loc in self.locations.values()
                          if loc.stagnation_state.deeply_stuck)

        return {
            'n_locations': self.n_locations,
            'n_base_locations': self.n_base_locations,
            'n_meta_locations': self.n_meta_locations,
            'n_edges': len(edges),
            'total_population': self.total_population,
            'avg_edge_weight': np.mean(weights) if weights else 0,
            'max_edge_weight': max(weights) if weights else 0,
            'min_edge_weight': min(weights) if weights else 0,
            'deeply_stuck_count': deeply_stuck,
            'deeply_stuck_ratio': deeply_stuck / self.n_locations if self.n_locations > 0 else 0,
        }

    def get_best_global(self) -> tuple[Optional[str], float]:
        """Find the location with the best hypothesis."""
        best_slug = None
        best_fitness = float('-inf')

        for slug, loc in self.locations.items():
            if loc.best_ever_fitness > best_fitness:
                best_fitness = loc.best_ever_fitness
                best_slug = slug

        return best_slug, best_fitness

    def get_strongest_edges(self, n: int = 5) -> list[Edge]:
        """Get the N strongest edges by weight."""
        edges = self.get_all_edges()
        edges.sort(key=lambda e: e.weight, reverse=True)
        return edges[:n]

    def to_dict(self) -> dict:
        """Serialize graph state."""
        return {
            'statistics': self.get_statistics(),
            'locations': {slug: loc.to_dict() for slug, loc in self.locations.items()},
            'strongest_edges': [e.to_dict() for e in self.get_strongest_edges(10)],
        }


def create_graph(
    essay_index: Optional[EssayIndex] = None,
    population_per_location: int = 15,
    n_neighbors: int = 3,
    mode: str = "sparse",
    max_locations: Optional[int] = None,
) -> LocationGraph:
    """
    Convenience function to create and initialize a graph.

    Args:
        essay_index: Optional pre-loaded index
        population_per_location: Population size per location
        n_neighbors: Number of neighbors for sparse init
        mode: "sparse" or "full"
        max_locations: Limit number of locations (for testing)

    Returns:
        Initialized LocationGraph
    """
    graph = LocationGraph(
        essay_index=essay_index,
        population_per_location=population_per_location,
        n_neighbors=n_neighbors,
    )
    graph.initialize(mode=mode, max_locations=max_locations)
    return graph


if __name__ == "__main__":
    # Test graph creation
    graph = create_graph(n_neighbors=3)

    print("\nGraph statistics:")
    for key, value in graph.get_statistics().items():
        print(f"  {key}: {value}")

    print("\nStrongest edges:")
    for edge in graph.get_strongest_edges(5):
        print(f"  {edge}")

    print("\nSample locations:")
    for loc in list(graph.iter_locations())[:5]:
        print(f"  {loc}")
        print(f"    Context: {len(loc.context_region)} chars")
        print(f"    Target chunks: {len(loc.target_chunks)}")
        print(f"    Edges: {list(loc.edges.keys())}")
