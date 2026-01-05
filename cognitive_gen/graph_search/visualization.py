"""
Text-based visualization for graph search progress.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from datetime import datetime

if TYPE_CHECKING:
    from .location_graph import LocationGraph
    from .location import Location
    from .meta_location import MetaLocationManager


LIVE_FILE = Path("/tmp/graph_search_live.txt")


def progress_bar(value: float, width: int = 10, filled: str = "█", empty: str = "░") -> str:
    """Create a text progress bar."""
    n_filled = int(value * width)
    n_empty = width - n_filled
    return filled * n_filled + empty * n_empty


def format_fitness(fitness: float) -> str:
    """Format fitness as percentage improvement."""
    return f"{fitness * 100:.1f}%"


def get_status_symbol(location: 'Location') -> str:
    """Get status symbol for a location."""
    if location.slug.startswith('meta_'):
        return "★"  # Meta-location
    if location.stagnation_state.deeply_stuck:
        return "◆"  # Deeply stuck
    if location.stagnation_state.severity() > 0.5:
        return "▲"  # Stagnating
    return "●"  # Progressing


def get_status_label(location: 'Location') -> str:
    """Get status label for a location."""
    if location.slug.startswith('meta_'):
        return "META"
    if location.stagnation_state.deeply_stuck:
        return "DEEP"
    if location.stagnation_state.severity() > 0.5:
        return "STAG"
    return ""


class GraphVisualizer:
    """
    Text-based visualization of graph search progress.

    Writes to /tmp/graph_search_live.txt for monitoring with:
        watch -n 2 cat /tmp/graph_search_live.txt
    """

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or LIVE_FILE
        self.events: list[str] = []  # Recent events log
        self.max_events = 10

    def add_event(self, event: str):
        """Add an event to the log."""
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def write_startup(
        self,
        graph: 'LocationGraph',
        total_generations: int,
        resuming_from: Optional[int] = None,
    ):
        """Write startup status."""
        if resuming_from is not None:
            # When resuming, show "RESUMED" and current generation
            title = f"  GRAPH-BASED COGNITIVE SEARCH - RESUMED AT GEN {resuming_from}"
            status = f"  Resuming from generation {resuming_from}..."
        else:
            title = "  GRAPH-BASED COGNITIVE SEARCH - STARTING UP"
            status = "  Initializing populations..."

        lines = [
            "=" * 70,
            title,
            "=" * 70,
            "",
            f"  Locations: {graph.n_locations}",
            f"  Edges: {graph.n_edges}",
            f"  Population per location: {graph.population_per_location}",
            f"  Total generations: {total_generations}",
            "",
            status,
            "",
            "=" * 70,
            f"  Watch: watch -n 2 cat {self.output_path}",
            "=" * 70,
        ]

        self.output_path.write_text("\n".join(lines))

    def write(
        self,
        graph: 'LocationGraph',
        generation: int,
        total_generations: int,
        meta_manager: 'MetaLocationManager',
        best_global_slug: Optional[str] = None,
        best_global_fitness: float = 0.0,
    ):
        """Write current status to file."""
        stats = graph.get_statistics()

        # Header
        lines = [
            "=" * 70,
            f"  GRAPH-BASED COGNITIVE SEARCH - GENERATION {generation}/{total_generations}",
            "=" * 70,
            "",
        ]

        # Graph structure
        lines.extend([
            "  GRAPH STRUCTURE",
            "  " + "─" * 20,
            f"  Locations: {stats['n_base_locations']} base + {stats['n_meta_locations']} meta = {stats['n_locations']} total",
            f"  Edges: {stats['n_edges']} (avg weight: {stats['avg_edge_weight']:.3f})",
        ])

        # Threshold info
        meta_status = meta_manager.get_status(graph)
        if meta_status['calibrated']:
            threshold = meta_manager.get_merge_threshold(graph)
            lines.append(
                f"  Merge threshold: {threshold:.3f} "
                f"(base: {meta_status['base_threshold']:.3f}, "
                f"{stats['deeply_stuck_ratio']:.0%} deeply stuck)"
            )
        else:
            lines.append("  Merge threshold: Not calibrated yet")

        lines.append("")

        # Location health table
        lines.extend([
            "  LOCATION HEALTH                                      Compute",
            "  " + "─" * 54 + "  " + "─" * 8,
        ])

        # Sort locations: progressing first, then stagnating, then deeply stuck
        sorted_locs = sorted(
            graph.locations.values(),
            key=lambda l: (
                l.stagnation_state.deeply_stuck,
                l.stagnation_state.severity(),
                -l.best_ever_fitness,
            ),
        )

        # Show up to 15 locations
        for loc in sorted_locs[:15]:
            severity = loc.stagnation_state.severity()
            bar = progress_bar(1 - severity)  # Invert so full = healthy
            fitness = format_fitness(loc.best_ever_fitness)
            symbol = get_status_symbol(loc)
            label = get_status_label(loc)
            compute = f"{int(loc.get_compute_weight() * 100)}%"

            name = loc.slug[:14].ljust(14)
            lines.append(
                f"  {name} [{bar}] {severity:.2f}  best: {fitness:>5}  {symbol} {label:4}  {compute:>4}"
            )

        if len(sorted_locs) > 15:
            lines.append(f"  ... and {len(sorted_locs) - 15} more locations")

        lines.append("")
        lines.append("  ● = progressing  ▲ = stagnating  ◆ = deeply stuck  ★ = meta-location")
        lines.append("")

        # Strongest edges
        lines.extend([
            "  STRONGEST EDGES (candidates for meta-location)",
            "  " + "─" * 46,
        ])

        for edge in graph.get_strongest_edges(5):
            lines.append(
                f"  {edge.source} ←→ {edge.target}".ljust(30) +
                f"weight: {edge.weight:.2f}  ({edge.successful_crossings} successful / {edge.failed_crossings} failed)"
            )

        lines.append("")

        # Best hypothesis
        if best_global_slug and best_global_slug in graph.locations:
            best_loc = graph.locations[best_global_slug]
            lines.extend([
                "  BEST HYPOTHESIS (global)",
                "  " + "─" * 24,
                f"  Location: {best_global_slug}",
                f"  Fitness: {format_fitness(best_global_fitness)} improvement over baseline",
            ])

            if best_loc.best_ever:
                lines.append(f"  Dimensions ({best_loc.best_ever.n_active} active):")
                for dim in list(best_loc.best_ever.active_dimensions)[:5]:
                    value = best_loc.best_ever.values.get(dim, "")
                    if value and len(str(value)) > 40:
                        value = str(value)[:40] + "..."
                    dim_display = dim.replace('_', ' ').title()
                    lines.append(f"    {dim_display}: \"{value}\"")
                if best_loc.best_ever.n_active > 5:
                    lines.append(f"    ... and {best_loc.best_ever.n_active - 5} more")

        lines.append("")

        # Recent events
        if self.events:
            lines.extend([
                "  RECENT EVENTS",
                "  " + "─" * 15,
            ])
            for event in self.events[-5:]:
                lines.append(f"  {event}")
            lines.append("")

        # Footer
        lines.extend([
            "=" * 70,
            f"  Watch: watch -n 2 cat {self.output_path}",
            f"  Last update: {datetime.now().strftime('%H:%M:%S')}",
            "=" * 70,
        ])

        self.output_path.write_text("\n".join(lines))

    def write_final(
        self,
        graph: 'LocationGraph',
        generation: int,
        best_global_slug: Optional[str],
        best_global_fitness: float,
        runtime_seconds: float,
    ):
        """Write final results."""
        stats = graph.get_statistics()

        lines = [
            "=" * 70,
            "  GRAPH-BASED COGNITIVE SEARCH - COMPLETE",
            "=" * 70,
            "",
            f"  Generations completed: {generation}",
            f"  Runtime: {runtime_seconds:.1f} seconds",
            "",
            "  FINAL GRAPH",
            "  " + "─" * 12,
            f"  Base locations: {stats['n_base_locations']}",
            f"  Meta-locations: {stats['n_meta_locations']}",
            f"  Total edges: {stats['n_edges']}",
            "",
            "  BEST RESULT",
            "  " + "─" * 12,
            f"  Location: {best_global_slug}",
            f"  Fitness: {format_fitness(best_global_fitness)}",
            "",
        ]

        if best_global_slug and best_global_slug in graph.locations:
            best_loc = graph.locations[best_global_slug]
            if best_loc.best_ever:
                lines.append("  DIMENSIONS:")
                for dim in best_loc.best_ever.active_dimensions:
                    value = best_loc.best_ever.values.get(dim, "")
                    dim_display = dim.replace('_', ' ').title()
                    lines.append(f"    {dim_display}:")
                    lines.append(f"      \"{value}\"")
                lines.append("")

        lines.extend([
            "=" * 70,
        ])

        self.output_path.write_text("\n".join(lines))
