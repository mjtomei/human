"""
Graph-based cognitive hypothesis search.

This module implements a multi-location evolutionary search where:
- Each blog post is a "location" with its own population
- Populations can migrate and cross-breed between locations
- Successful breeding strengthens connections
- Strong connections lead to meta-location creation
"""

from .edge import Edge
from .location import Location, LocationStagnationState
from .location_graph import LocationGraph
from .migration import MigrationSystem
from .meta_location import MetaLocation, MetaLocationManager
from .graph_search import GraphSearch, GraphSearchResult
from .scraper import BlogScraper, EssayData
from .essay_index import EssayIndex, EssayMetadata
from .checkpoint import CheckpointManager, CheckpointMetadata

__all__ = [
    'Edge',
    'Location',
    'LocationStagnationState',
    'LocationGraph',
    'MigrationSystem',
    'MetaLocation',
    'MetaLocationManager',
    'GraphSearch',
    'GraphSearchResult',
    'BlogScraper',
    'EssayData',
    'EssayIndex',
    'EssayMetadata',
    'CheckpointManager',
    'CheckpointMetadata',
]
