"""
Checkpoint system for graph-based cognitive search.

Saves and loads the full state of a search, allowing:
- Resume from any checkpoint
- Change hyperparameters on resume
- Change code and continue with existing populations
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .location_graph import LocationGraph
    from .location import Location
    from .meta_location import MetaLocation

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from cognitive_gen.meta_search import SparseHypothesis


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""
    generation: int
    timestamp: str
    n_locations: int
    n_meta_locations: int
    best_fitness: float
    best_location: str
    total_generations_target: int
    hyperparameters: dict


def serialize_hypothesis(hyp: SparseHypothesis) -> dict:
    """Serialize a SparseHypothesis to dict."""
    return {
        'values': hyp.values,
        'origins': hyp.origins,  # Track dimension lineage
    }


def deserialize_hypothesis(data: dict) -> SparseHypothesis:
    """Deserialize a SparseHypothesis from dict."""
    return SparseHypothesis(
        values=data.get('values', {}),
        origins=data.get('origins', {}),  # Restore lineage if present
    )


def serialize_location(location: 'Location') -> dict:
    """Serialize a Location to dict."""
    from .meta_location import MetaLocation

    data = {
        'slug': location.slug,
        'essay_content': location.essay_content,
        'context_region': location.context_region,
        'target_region': location.target_region,
        'baseline_ppl': location.baseline_ppl,
        'generation': location.generation,
        'population': [serialize_hypothesis(h) for h in location.population],
        'fitness_scores': location.fitness_scores.copy(),
        'best_ever': serialize_hypothesis(location.best_ever) if location.best_ever else None,
        'best_ever_fitness': location.best_ever_fitness,
        'edges': {
            target: {
                'source': edge.source,
                'target': edge.target,
                'weight': edge.weight,
                'successful_crossings': edge.successful_crossings,
                'failed_crossings': edge.failed_crossings,
            }
            for target, edge in location.edges.items()
        },
        'stagnation_state': {
            'diversity_injections': location.stagnation_state.diversity_injections,
            'fitness_at_injection': location.stagnation_state.fitness_at_injection,
            'deeply_stuck': location.stagnation_state.deeply_stuck,
            'generations_since_injection': location.stagnation_state.generations_since_injection,
            # Save the stagnation detector's history
            'history': location.stagnation_state.stagnation.history,
        },
        'is_meta': isinstance(location, MetaLocation),
    }

    # Add meta-location specific fields
    if isinstance(location, MetaLocation):
        data['member_slugs'] = location.member_slugs
        data['creation_generation'] = location.creation_generation

    return data


def deserialize_location(data: dict, essay_index=None) -> 'Location':
    """Deserialize a Location from dict."""
    from .location import Location, LocationStagnationState
    from .meta_location import MetaLocation
    from .edge import Edge
    from cognitive_gen.meta_search import StagnationDetector

    # Reconstruct edges
    edges = {}
    for target, edge_data in data.get('edges', {}).items():
        edge = Edge(
            source=edge_data['source'],
            target=edge_data['target'],
            weight=edge_data['weight'],
            successful_crossings=edge_data['successful_crossings'],
            failed_crossings=edge_data['failed_crossings'],
        )
        edges[target] = edge

    # Reconstruct stagnation state
    stag_data = data.get('stagnation_state', {})
    stagnation_detector = StagnationDetector()
    # Restore the stagnation history if present
    stagnation_detector.history = stag_data.get('history', [])
    stagnation_state = LocationStagnationState(
        stagnation=stagnation_detector,
        diversity_injections=stag_data.get('diversity_injections', 0),
        fitness_at_injection=stag_data.get('fitness_at_injection'),
        deeply_stuck=stag_data.get('deeply_stuck', False),
        generations_since_injection=stag_data.get('generations_since_injection', 0),
    )

    # Reconstruct population
    population = [deserialize_hypothesis(h) for h in data.get('population', [])]

    # Reconstruct best_ever
    best_ever = None
    if data.get('best_ever'):
        best_ever = deserialize_hypothesis(data['best_ever'])

    if data.get('is_meta'):
        # Create a minimal MetaLocation
        # We need to create it differently since we don't have the original locations
        location = Location(
            slug=data['slug'],
            essay_content=data['essay_content'],
            context_region=data['context_region'],
            target_region=data['target_region'],
            population=population,
            fitness_scores=data.get('fitness_scores', []),
            edges=edges,
            stagnation_state=stagnation_state,
            best_ever=best_ever,
            best_ever_fitness=data.get('best_ever_fitness', float('-inf')),
        )
        location.baseline_ppl = data.get('baseline_ppl', 0.0)
        location.generation = data.get('generation', 0)
        # Mark as meta
        location.is_meta = True
        location.member_slugs = data.get('member_slugs', [])
        location.creation_generation = data.get('creation_generation', 0)
    else:
        location = Location(
            slug=data['slug'],
            essay_content=data['essay_content'],
            context_region=data['context_region'],
            target_region=data['target_region'],
            population=population,
            fitness_scores=data.get('fitness_scores', []),
            edges=edges,
            stagnation_state=stagnation_state,
            best_ever=best_ever,
            best_ever_fitness=data.get('best_ever_fitness', float('-inf')),
        )
        location.baseline_ppl = data.get('baseline_ppl', 0.0)
        location.generation = data.get('generation', 0)

    return location


class CheckpointManager:
    """Manage checkpoints for graph search."""

    def __init__(
        self,
        checkpoint_dir: str = "/tmp/graph_search_checkpoints",
        save_interval: int = 5,  # Save every N generations
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

    def should_save(self, generation: int) -> bool:
        """Check if we should save a checkpoint at this generation."""
        return generation > 0 and generation % self.save_interval == 0

    def get_checkpoint_path(self, generation: int) -> Path:
        """Get path for a checkpoint at a given generation."""
        return self.checkpoint_dir / f"checkpoint_gen{generation:04d}.json"

    def save_checkpoint(
        self,
        graph: 'LocationGraph',
        generation: int,
        total_generations: int,
        best_fitness: float,
        best_location: str,
        hyperparameters: dict,
        meta_manager_state: dict,
    ) -> Path:
        """Save a checkpoint of the current state."""

        # Serialize all locations
        locations_data = {}
        for slug, location in graph.locations.items():
            locations_data[slug] = serialize_location(location)

        # Build checkpoint data
        checkpoint = {
            'metadata': {
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'n_locations': graph.n_locations,
                'n_meta_locations': graph.n_meta_locations,
                'best_fitness': best_fitness,
                'best_location': best_location,
                'total_generations_target': total_generations,
                'hyperparameters': hyperparameters,
            },
            'locations': locations_data,
            'meta_manager': meta_manager_state,
        }

        # Save to file
        path = self.get_checkpoint_path(generation)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        # Also save as 'latest'
        latest_path = self.checkpoint_dir / "checkpoint_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        return path

    def load_checkpoint(self, path: Optional[str] = None) -> dict:
        """Load a checkpoint from file.

        Args:
            path: Path to checkpoint file. If None, loads latest.

        Returns:
            Checkpoint data dict with 'metadata', 'locations', 'meta_manager'
        """
        if path is None:
            path = self.checkpoint_dir / "checkpoint_latest.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, 'r') as f:
            checkpoint = json.load(f)

        return checkpoint

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("checkpoint_gen*.json")):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    checkpoints.append({
                        'path': str(path),
                        'generation': data['metadata']['generation'],
                        'timestamp': data['metadata']['timestamp'],
                        'best_fitness': data['metadata']['best_fitness'],
                        'n_locations': data['metadata']['n_locations'],
                    })
            except Exception as e:
                print(f"Warning: Could not read checkpoint {path}: {e}")

        return checkpoints

    def restore_graph(self, checkpoint: dict) -> 'LocationGraph':
        """Restore a LocationGraph from checkpoint data."""
        from .location_graph import LocationGraph

        graph = LocationGraph()

        # Restore all locations
        for slug, loc_data in checkpoint['locations'].items():
            location = deserialize_location(loc_data)
            graph.locations[slug] = location

        return graph

    def get_start_generation(self, checkpoint: dict) -> int:
        """Get the generation to resume from."""
        return checkpoint['metadata']['generation']
