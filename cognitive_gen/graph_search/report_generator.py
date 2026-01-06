"""
Interactive HTML report generator for cognitive hypothesis search.

Generates reports from checkpoint data with:
- Dimension histograms showing positive/negative/missing breakdown
- Click-through to see individual occurrences
- Presence/absence classification using LLM
- Dimension origin tracking (where/when each value first appeared)
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import os


# Cache directory for LLM classifications
CLASSIFICATION_CACHE_DIR = Path("/tmp/classification_cache")
CHECKPOINT_DIR = Path("/tmp/graph_search_checkpoints")


def infer_origins_from_checkpoints(
    current_checkpoint: dict,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict:
    """
    Infer dimension value origins by scanning past checkpoints.

    For each (location, dimension, value) tuple in the current checkpoint,
    finds the earliest checkpoint where that exact value appeared and which
    location it was in.

    Args:
        current_checkpoint: The current checkpoint data
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Dict mapping (location, dimension) -> {'location': origin_loc, 'generation': origin_gen}
    """
    # Build set of (dim, value) pairs we need to find origins for
    values_to_trace = {}  # (dim, value_hash) -> list of current locations

    locations = current_checkpoint.get('locations', {})
    for slug, loc_data in locations.items():
        best_ever = loc_data.get('best_ever')
        if not best_ever:
            continue

        values = best_ever.get('values', {})
        origins = best_ever.get('origins', {})

        for dim, value in values.items():
            if value is None:
                continue
            # Skip if we already have origin data
            if dim in origins and origins[dim]:
                continue

            # Use value hash to match exact values
            value_hash = hashlib.md5(str(value).encode()).hexdigest()[:16]
            key = (dim, value_hash)
            if key not in values_to_trace:
                values_to_trace[key] = {'value': value, 'current_locs': []}
            values_to_trace[key]['current_locs'].append(slug)

    if not values_to_trace:
        print("All values have origin data, no inference needed")
        return {}

    print(f"Inferring origins for {len(values_to_trace)} unique (dimension, value) pairs...")

    # Origins we've discovered: (dim, value_hash) -> {'location': ..., 'generation': ...}
    discovered_origins = {}

    # Get sorted list of checkpoint files (oldest first)
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_gen*.json"))

    # Scan checkpoints from oldest to newest
    for cp_path in checkpoint_files:
        try:
            with open(cp_path) as f:
                cp_data = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not read {cp_path}: {e}")
            continue

        cp_gen = cp_data.get('metadata', {}).get('generation', 0)
        cp_locations = cp_data.get('locations', {})

        # Check each location's population for our target values
        for loc_slug, loc_data in cp_locations.items():
            # Check best_ever
            best_ever = loc_data.get('best_ever')
            if best_ever:
                values = best_ever.get('values', {})
                for dim, value in values.items():
                    if value is None:
                        continue
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()[:16]
                    key = (dim, value_hash)

                    # If this is a value we're tracing and we haven't found its origin yet
                    if key in values_to_trace and key not in discovered_origins:
                        discovered_origins[key] = {
                            'location': loc_slug,
                            'generation': cp_gen,
                        }

            # Also check population members
            population = loc_data.get('population', [])
            for hyp in population:
                values = hyp.get('values', {})
                for dim, value in values.items():
                    if value is None:
                        continue
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()[:16]
                    key = (dim, value_hash)

                    if key in values_to_trace and key not in discovered_origins:
                        discovered_origins[key] = {
                            'location': loc_slug,
                            'generation': cp_gen,
                        }

        # Early exit if we've found all origins
        if len(discovered_origins) >= len(values_to_trace):
            break

    # Build final mapping: (current_location, dim) -> origin
    result = {}
    for (dim, value_hash), info in values_to_trace.items():
        origin = discovered_origins.get((dim, value_hash))
        if origin:
            for current_loc in info['current_locs']:
                result[(current_loc, dim)] = origin

    found = len([k for k in values_to_trace if k in discovered_origins])
    print(f"  Found origins for {found}/{len(values_to_trace)} values")

    return result


def get_cache_key(checkpoint_path: Path) -> str:
    """Generate a cache key based on checkpoint file content hash."""
    content = checkpoint_path.read_bytes()
    return hashlib.md5(content).hexdigest()[:12]


def load_cached_classifications(cache_key: str) -> Optional[dict]:
    """Load cached classifications if available."""
    cache_file = CLASSIFICATION_CACHE_DIR / f"classifications_{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def save_cached_classifications(cache_key: str, classifications: dict) -> None:
    """Save classifications to cache."""
    CLASSIFICATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CLASSIFICATION_CACHE_DIR / f"classifications_{cache_key}.json"
    with open(cache_file, 'w') as f:
        json.dump(classifications, f)
    print(f"Cached classifications to {cache_file}")


def classify_values_with_llm(values_to_classify: list[tuple[str, str, str]]) -> dict:
    """
    Classify dimension values using a local LLM.

    Args:
        values_to_classify: List of (location, dimension, value) tuples

    Returns:
        Dict mapping (location, dimension) -> classification
    """
    from cognitive_gen.local_models import TransformersServer

    if not values_to_classify:
        return {}

    # Filter out missing values (None, empty)
    items_to_classify = []
    results = {}

    for loc, dim, value in values_to_classify:
        if value is None:
            results[(loc, dim)] = 'missing'
        elif isinstance(value, (list, dict)):
            results[(loc, dim)] = 'positive' if value else 'missing'
        elif not str(value).strip():
            results[(loc, dim)] = 'missing'
        else:
            items_to_classify.append((loc, dim, str(value).strip()))

    if not items_to_classify:
        return results

    print(f"Classifying {len(items_to_classify)} dimension values with LLM...")

    # Load a small, fast model for classification
    server = TransformersServer("mistralai/Mistral-7B-Instruct-v0.3")

    # Create classification prompts
    prompts = []
    for loc, dim, value in items_to_classify:
        prompt = f"""Classify this psychological description's POLARITY and INTENSITY.

Dimension: {dim.replace('_', ' ')}
Description: "{value}"

POLARITY:
- ZERO: No information provided / not addressed / "unknown" with no claim
- NEGATIVE: Claims ABSENCE of a trait or behavior
- POSITIVE: Claims PRESENCE of a trait or behavior

INTENSITY (for NEGATIVE and POSITIVE only):
- LOW: Weak/hedged ("may", "might", "possibly", "seems to slightly", "little evidence")
- MEDIUM: Moderate ("appears to", "likely", "suggests", "tends to", "does not appear")
- HIGH: Strong/definitive ("clearly", "strongly", "definitely", "always", "never", "absolutely")

Examples:
- "Unknown" or "Not enough information" → ZERO
- "There may be little indication of X" → NEGATIVE-LOW
- "The writer does not appear to have X" → NEGATIVE-MEDIUM
- "There is absolutely no evidence of X" → NEGATIVE-HIGH
- "They may sometimes feel X" → POSITIVE-LOW
- "The writer appears to value X" → POSITIVE-MEDIUM
- "They strongly believe in X" → POSITIVE-HIGH

Reply with only one of: ZERO, NEGATIVE-LOW, NEGATIVE-MEDIUM, NEGATIVE-HIGH, POSITIVE-LOW, POSITIVE-MEDIUM, POSITIVE-HIGH"""
        prompts.append(prompt)

    # Process in batches of 32
    batch_size = 32
    all_responses = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    import time
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = i // batch_size + 1
        batch_start = time.time()
        responses = server.generate_batch(batch, max_tokens=10)
        all_responses.extend(responses)
        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        items_done = min(i + batch_size, len(prompts))
        rate = items_done / elapsed if elapsed > 0 else 0
        remaining = (len(prompts) - items_done) / rate if rate > 0 else 0
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} items in {batch_time:.1f}s | "
              f"{items_done}/{len(prompts)} done ({rate:.1f}/s) | ETA: {remaining:.0f}s")

    # Parse responses
    counts = {
        'zero': 0,
        'negative-low': 0, 'negative-medium': 0, 'negative-high': 0,
        'positive-low': 0, 'positive-medium': 0, 'positive-high': 0
    }
    for (loc, dim, value), response in zip(items_to_classify, all_responses):
        response_clean = response.strip().upper()

        # Check for ZERO first
        if 'ZERO' in response_clean:
            results[(loc, dim)] = 'zero'
            counts['zero'] += 1
            continue

        is_negative = 'NEGATIVE' in response_clean and 'POSITIVE' not in response_clean
        polarity = 'negative' if is_negative else 'positive'

        if 'HIGH' in response_clean:
            intensity = 'high'
        elif 'LOW' in response_clean:
            intensity = 'low'
        else:
            intensity = 'medium'

        classification = f"{polarity}-{intensity}"
        results[(loc, dim)] = classification
        counts[classification] += 1

    neg_total = counts['negative-low'] + counts['negative-medium'] + counts['negative-high']
    pos_total = counts['positive-low'] + counts['positive-medium'] + counts['positive-high']
    print(f"  Classification complete: {counts['zero']} zero, "
          f"{neg_total} negative (L:{counts['negative-low']} M:{counts['negative-medium']} H:{counts['negative-high']}), "
          f"{pos_total} positive (L:{counts['positive-low']} M:{counts['positive-medium']} H:{counts['positive-high']})")

    # Clean up model to free GPU memory
    del server
    import torch
    torch.cuda.empty_cache()

    return results


def classify_value_simple(value) -> str:
    """Simple fallback classification without LLM."""
    if value is None:
        return 'missing'
    if isinstance(value, (list, dict)):
        return 'positive' if value else 'missing'
    value_str = str(value).strip()
    if not value_str:
        return 'missing'
    return 'positive'


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint data from JSON file."""
    with open(checkpoint_path) as f:
        return json.load(f)


def generate_report(
    checkpoint_data: dict,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    use_llm: bool = True,
) -> None:
    """Generate interactive HTML report from checkpoint data.

    Args:
        checkpoint_data: Loaded checkpoint data
        output_path: Where to write the HTML report
        checkpoint_path: Original checkpoint file (for cache key)
        use_llm: Whether to use LLM for classification (default True)
    """

    metadata = checkpoint_data.get('metadata', {})
    locations = checkpoint_data.get('locations', {})
    n_locations = len(locations)

    generation = metadata.get('generation', '?')
    total_gens = metadata.get('total_generations_target', '?')
    best_fitness = metadata.get('best_fitness', 0)
    best_location = metadata.get('best_location', 'unknown')
    n_meta = metadata.get('n_meta_locations', 0)

    # Collect all values that need classification
    values_to_classify = []  # (location, dimension, value)
    location_values = {}  # (location, dimension) -> value

    for slug, loc_data in locations.items():
        population = loc_data.get('population', [])
        if not population:
            continue

        best_hyp = population[0]
        values = best_hyp.get('values', {})

        for dim, value in values.items():
            values_to_classify.append((slug, dim, value))
            location_values[(slug, dim)] = value

    # Get classifications (from cache or LLM)
    classifications = {}  # (location, dimension) -> 'positive'/'negative'/'missing'

    if use_llm and checkpoint_path:
        cache_key = get_cache_key(checkpoint_path)
        cached = load_cached_classifications(cache_key)

        if cached:
            print(f"Loaded {len(cached)} classifications from cache")
            # Convert string keys back to tuples
            classifications = {tuple(k.split('|')): v for k, v in cached.items()}
        else:
            classifications = classify_values_with_llm(values_to_classify)
            # Convert tuple keys to strings for JSON serialization
            cache_data = {f"{k[0]}|{k[1]}": v for k, v in classifications.items()}
            save_cached_classifications(cache_key, cache_data)
    else:
        # Fallback to simple classification
        for loc, dim, value in values_to_classify:
            classifications[(loc, dim)] = classify_value_simple(value)

    # Extract origins from checkpoint data or infer from past checkpoints
    origins = {}  # (location, dim) -> {'location': origin_loc, 'generation': origin_gen}

    # First, collect any origins that exist in the checkpoint
    has_any_origins = False
    for slug, loc_data in locations.items():
        best_ever = loc_data.get('best_ever', {})
        if best_ever and best_ever.get('origins'):
            for dim, origin in best_ever.get('origins', {}).items():
                if origin:
                    origins[(slug, dim)] = origin
                    has_any_origins = True

    # If no origins in checkpoint, infer from past checkpoints
    if not has_any_origins and checkpoint_path:
        print("No origin data in checkpoint, inferring from past checkpoints...")
        inferred = infer_origins_from_checkpoints(checkpoint_data, checkpoint_path.parent)
        origins.update(inferred)

    # Build dimension occurrence data with classification and origins
    dim_occurrences = {}  # dim -> list of {location, fitness, value, classification, origin}

    for slug, loc_data in locations.items():
        fitness = loc_data.get('best_ever_fitness', loc_data.get('best_fitness', 0))
        population = loc_data.get('population', [])
        if not population:
            continue

        best_hyp = population[0]
        values = best_hyp.get('values', {})

        for dim, value in values.items():
            if dim not in dim_occurrences:
                dim_occurrences[dim] = []

            classification = classifications.get((slug, dim), 'positive')
            origin = origins.get((slug, dim))

            dim_occurrences[dim].append({
                'location': slug,
                'fitness': fitness,
                'value': value if value is not None else '(null)',
                'classification': classification,
                'origin': origin,  # {'location': ..., 'generation': ...} or None
            })

    # Generate HTML
    html = generate_html(
        generation=generation,
        total_gens=total_gens,
        n_locations=n_locations,
        n_meta=n_meta,
        best_fitness=best_fitness,
        best_location=best_location,
        locations=locations,
        dim_occurrences=dim_occurrences,
        checkpoint_data=checkpoint_data,
    )

    output_path.write_text(html)
    print(f"Report written to {output_path}")


def generate_html(
    generation: int,
    total_gens: int,
    n_locations: int,
    n_meta: int,
    best_fitness: float,
    best_location: str,
    locations: dict,
    dim_occurrences: dict,
    checkpoint_data: dict,
) -> str:
    """Generate the full HTML content."""

    # Deduplicate values: create lookup table and replace values with IDs
    # This reduces file size by ~90% since same values appear across many locations
    value_to_id = {}  # value string -> numeric ID
    id_to_value = []  # index = ID, value = string

    for dim, occurrences in dim_occurrences.items():
        for occ in occurrences:
            value = occ.get('value', '')
            if value not in value_to_id:
                value_to_id[value] = len(id_to_value)
                id_to_value.append(value)
            # Replace value with ID
            occ['v'] = value_to_id[value]
            del occ['value']

    # Convert to JSON
    dim_occ_json = json.dumps(dim_occurrences)
    values_json = json.dumps(id_to_value)

    # Slim down checkpoint data - only include what the JS actually uses
    # Values are already in DIM_OCCURRENCES, so we don't need population here
    slim_checkpoint = {
        'metadata': checkpoint_data.get('metadata', {}),
        'locations': {}
    }
    for slug, loc_data in checkpoint_data.get('locations', {}).items():
        # Slim down stagnation history - only keep what's needed for the graph
        stag = loc_data.get('stagnation_state', {})
        slim_stag = {
            'history': [{'best': h.get('best', 0)} for h in stag.get('history', [])],
            'diversity_injections': stag.get('diversity_injections', 0),
            'deeply_stuck': stag.get('deeply_stuck', False)
        }
        slim_checkpoint['locations'][slug] = {
            'best_ever_fitness': loc_data.get('best_ever_fitness'),
            'stagnation_state': slim_stag,
        }
    checkpoint_json = json.dumps(slim_checkpoint)

    n_edges = checkpoint_data.get('metadata', {}).get('hyperparameters', {})
    # Try to count edges from graph statistics if available
    graph_stats = checkpoint_data.get('graph_statistics', {})
    n_edges = graph_stats.get('n_edges', 58)  # Default from earlier observation

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Hypothesis Search - Gen {generation} Report</title>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --bg-section: #0f3460;
            --text-primary: #e6e6e6;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --accent-light: #ff6b6b;
            --border: #3a3a5c;
            --bg-modal: rgba(0,0,0,0.85);
            --positive: #4ade80;
            --negative: #f87171;
            --missing: #3a3a5c;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{ color: var(--accent); margin-bottom: 10px; font-size: 1.8em; }}
        .meta {{ color: var(--text-secondary); margin-bottom: 30px; font-size: 0.9em; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: var(--bg-card);
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid var(--accent);
        }}
        .stat-value {{ font-size: 1.5em; color: var(--accent-light); font-weight: bold; }}
        .stat-label {{ color: var(--text-secondary); font-size: 0.85em; }}
        .location {{
            background: var(--bg-card);
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .location > summary {{
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-section);
            user-select: none;
        }}
        .location > summary:hover {{ background: #1a4a7c; }}
        .location > summary::marker {{ display: none; }}
        .location > summary::-webkit-details-marker {{ display: none; }}
        .location-name {{ font-weight: bold; font-size: 1.1em; }}
        .location-fitness {{
            background: var(--accent);
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .location-content {{ padding: 15px 20px; }}
        .group {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid var(--border);
        }}
        .group > summary {{
            padding: 10px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}
        .group > summary:hover {{ background: rgba(255,255,255,0.05); }}
        .group > summary::marker {{ display: none; }}
        .group > summary::-webkit-details-marker {{ display: none; }}
        .group-name {{ font-weight: 600; color: var(--accent-light); }}
        .group-count {{ color: var(--text-secondary); font-size: 0.85em; }}
        .group-content {{ padding: 10px 15px; border-top: 1px solid var(--border); }}
        .dimension {{ margin-bottom: 12px; padding-left: 10px; border-left: 3px solid var(--border); }}
        .dimension:last-child {{ margin-bottom: 0; }}
        .dimension.positive-high {{ border-left-color: var(--positive); border-left-width: 4px; }}
        .dimension.positive-medium {{ border-left-color: var(--positive); opacity: 0.85; }}
        .dimension.positive-low {{ border-left-color: var(--positive); opacity: 0.65; border-left-style: dashed; }}
        .dimension.negative-high {{ border-left-color: var(--negative); border-left-width: 4px; }}
        .dimension.negative-medium {{ border-left-color: var(--negative); opacity: 0.85; }}
        .dimension.negative-low {{ border-left-color: var(--negative); opacity: 0.65; border-left-style: dashed; }}
        .dimension.zero {{ border-left-color: var(--text-secondary); opacity: 0.5; border-left-style: dotted; }}
        .dim-name {{ color: var(--accent); font-weight: 500; margin-bottom: 3px; }}
        .dim-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 3px; flex-wrap: wrap; }}
        .dim-origin {{ font-size: 0.75em; padding: 2px 6px; border-radius: 3px; background: rgba(233, 69, 96, 0.15); color: var(--accent); }}
        .dim-origin.native {{ background: rgba(76, 175, 80, 0.15); color: var(--positive); }}
        .dim-value {{ color: var(--text-secondary); font-size: 0.9em; }}
        .histogram {{ background: var(--bg-card); padding: 20px; border-radius: 10px; margin-top: 30px; }}
        .histogram h2 {{ color: var(--accent); margin-bottom: 15px; }}
        .hist-group {{ margin-bottom: 20px; }}
        .hist-group-name {{ color: var(--accent-light); font-weight: 600; margin-bottom: 8px; }}
        .hist-bar-container {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
            cursor: pointer;
            padding: 3px 5px;
            border-radius: 4px;
            transition: background 0.2s;
        }}
        .hist-bar-container:hover {{
            background: rgba(233, 69, 96, 0.2);
        }}
        .hist-label {{
            width: 220px;
            font-size: 0.85em;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .hist-bar {{
            flex: 1;
            height: 18px;
            background: var(--missing);
            border-radius: 3px;
            overflow: hidden;
            margin: 0 10px;
            display: flex;
        }}
        .hist-pos-high {{ height: 100%; background: var(--positive); }}
        .hist-pos-med {{ height: 100%; background: var(--positive); opacity: 0.7; }}
        .hist-pos-low {{ height: 100%; background: var(--positive); opacity: 0.4; }}
        .hist-neg-high {{ height: 100%; background: var(--negative); }}
        .hist-neg-med {{ height: 100%; background: var(--negative); opacity: 0.7; }}
        .hist-neg-low {{ height: 100%; background: var(--negative); opacity: 0.4; }}
        .hist-zero {{ height: 100%; background: var(--text-secondary); opacity: 0.3; }}
        .hist-score {{
            width: 50px;
            text-align: right;
            font-size: 0.85em;
            font-weight: 600;
            color: var(--text-secondary);
        }}
        .hist-score.positive {{ color: var(--positive); }}
        .hist-score.negative {{ color: var(--negative); }}
        .arrow {{ transition: transform 0.2s; color: var(--text-secondary); }}
        details[open] > summary .arrow {{ transform: rotate(90deg); }}
        .controls {{ margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; }}
        button {{
            background: var(--bg-section);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        button:hover {{ background: var(--accent); border-color: var(--accent); }}
        .sort-btn {{
            padding: 4px 12px;
            font-size: 0.8em;
        }}
        .sort-btn.active {{
            background: var(--accent);
            border-color: var(--accent);
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 0.85em;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-box {{
            width: 14px;
            height: 14px;
            border-radius: 2px;
        }}
        .legend-box.positive {{ background: var(--positive); }}
        .legend-box.negative {{ background: var(--negative); }}
        .legend-box.missing {{ background: var(--missing); }}
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-modal);
            z-index: 1000;
            overflow-y: auto;
        }}
        .modal.active {{ display: block; }}
        .modal-content {{
            background: var(--bg-card);
            margin: 40px auto;
            padding: 25px;
            border-radius: 12px;
            max-width: 900px;
            max-height: calc(100vh - 80px);
            overflow-y: auto;
            position: relative;
        }}
        .modal-close {{
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 1.5em;
            cursor: pointer;
            color: var(--text-secondary);
            background: none;
            border: none;
            padding: 5px 10px;
        }}
        .modal-close:hover {{ color: var(--accent); background: none; }}
        .modal-title {{
            color: var(--accent);
            font-size: 1.4em;
            margin-bottom: 5px;
            padding-right: 40px;
        }}
        .modal-subtitle {{
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .occurrence {{
            background: var(--bg-section);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
            border-left: 3px solid var(--accent);
        }}
        .occurrence.positive {{ border-left-color: var(--positive); }}
        .occurrence.negative {{ border-left-color: var(--negative); }}
        .occurrence.zero {{ border-left-color: var(--text-secondary); border-left-style: dotted; }}
        .occurrence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .occurrence-location {{
            font-weight: 600;
            color: var(--accent-light);
        }}
        .essay-link {{
            color: inherit;
            text-decoration: none;
        }}
        .essay-link:hover {{
            text-decoration: underline;
        }}
        .occurrence-badges {{
            display: flex;
            gap: 8px;
        }}
        .occurrence-fitness {{
            background: var(--accent);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }}
        .occurrence-class {{
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            color: white;
        }}
        .occurrence-class.positive-high {{ background: var(--positive); color: #000; }}
        .occurrence-class.positive-medium {{ background: var(--positive); color: #000; opacity: 0.8; }}
        .occurrence-class.positive-low {{ background: var(--positive); color: #000; opacity: 0.6; }}
        .occurrence-class.negative {{ background: var(--negative); color: #000; }}
        .occurrence-origin {{
            background: #6366f1;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 500;
        }}
        .occurrence-origin.native {{
            background: #475569;
            color: #94a3b8;
        }}
        .occurrence-value {{
            color: var(--text-primary);
            font-size: 0.95em;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <h1>Cognitive Hypothesis Search Report</h1>
    <div class="meta">Generation {generation} of {total_gens} | Generated {datetime.now().strftime('%Y-%m-%d')} |
        <a href="https://github.com/mjtomei/human" target="_blank" class="essay-link">GitHub</a> &middot;
        <a href="http://mtomei.com/cluster_analysis.html" class="essay-link">Cluster Analysis</a>
    </div>
    <div class="stats">
        <div class="stat-card"><div class="stat-value">{n_locations}</div><div class="stat-label">Locations</div></div>
        <div class="stat-card"><div class="stat-value">{best_fitness*100:.1f}%</div><div class="stat-label">Best Fitness ({best_location})</div></div>
        <div class="stat-card"><div class="stat-value">{n_edges}</div><div class="stat-label">Edges</div></div>
        <div class="stat-card"><div class="stat-value">{n_meta}</div><div class="stat-label">Meta-Locations</div></div>
    </div>
    <div class="controls">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
        <button onclick="expandLocations()">Show All Locations</button>
        <span style="color: var(--border); margin: 0 4px;">|</span>
        <span style="color: var(--text-secondary); font-size: 0.85em;">Filter:</span>
        <button id="filter-all" class="sort-btn active" onclick="setLocationFilter('all')">All</button>
        <button id="filter-base" class="sort-btn" onclick="setLocationFilter('base')">Base Only</button>
        <button id="filter-meta" class="sort-btn" onclick="setLocationFilter('meta')">Meta Only</button>
    </div>
    <div id="locations"><p style="color: var(--text-secondary);">Loading...</p></div>
    <div class="histogram" id="histogram">
        <h2>Dimension Histogram (Best per Location)</h2>
        <div class="legend" style="flex-wrap: wrap;">
            <div class="legend-item"><div class="legend-box positive"></div><span>+High</span></div>
            <div class="legend-item"><div class="legend-box positive" style="opacity:0.7"></div><span>+Med</span></div>
            <div class="legend-item"><div class="legend-box positive" style="opacity:0.4"></div><span>+Low</span></div>
            <div class="legend-item"><div class="legend-box negative"></div><span>-High</span></div>
            <div class="legend-item"><div class="legend-box negative" style="opacity:0.7"></div><span>-Med</span></div>
            <div class="legend-item"><div class="legend-box negative" style="opacity:0.4"></div><span>-Low</span></div>
            <div class="legend-item"><div class="legend-box" style="background:var(--text-secondary);opacity:0.3"></div><span>Zero</span></div>
            <div class="legend-item"><div class="legend-box missing"></div><span>Missing</span></div>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 10px;">Click any dimension to see all occurrences. Bar width = total locations ({n_locations}).</p>
        <div style="margin-bottom: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <span style="color: var(--text-secondary); font-size: 0.85em;">Sort by:</span>
            <button id="sort-weighted" class="sort-btn active" onclick="sortHistogram('weighted')">Score</button>
            <button id="sort-positive" class="sort-btn" onclick="sortHistogram('positive')">% Positive</button>
            <button id="sort-negative" class="sort-btn" onclick="sortHistogram('negative')">% Negative</button>
            <button id="sort-any" class="sort-btn" onclick="sortHistogram('any')">% Any</button>
            <span style="color: var(--border); margin: 0 4px;">|</span>
            <span style="color: var(--text-secondary); font-size: 0.85em;">Score mode:</span>
            <button id="score-net" class="sort-btn active" onclick="setScoreMode('net')">Net</button>
            <button id="score-pos" class="sort-btn" onclick="setScoreMode('pos')">+Only</button>
            <button id="score-neg" class="sort-btn" onclick="setScoreMode('neg')">-Only</button>
            <span style="color: var(--border); margin: 0 4px;">|</span>
            <button class="sort-btn" onclick="expandHistGroups()">Expand All</button>
            <button class="sort-btn" onclick="collapseHistGroups()">Collapse All</button>
        </div>
        <div id="hist-content">Loading...</div>
    </div>
    <div class="histogram" id="shared-values">
        <h2>Shared Values (Crossover Spread)</h2>
        <p style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 10px;">Dimension values that appear in multiple locations, sorted by occurrence count. Higher counts suggest successful crossover spread.</p>
        <div style="margin-bottom: 15px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <span style="color: var(--text-secondary); font-size: 0.85em;">Min occurrences:</span>
            <button id="min-2" class="sort-btn active" onclick="setMinOccurrences(2)">2+</button>
            <button id="min-3" class="sort-btn" onclick="setMinOccurrences(3)">3+</button>
            <button id="min-5" class="sort-btn" onclick="setMinOccurrences(5)">5+</button>
            <button id="min-10" class="sort-btn" onclick="setMinOccurrences(10)">10+</button>
        </div>
        <div id="shared-content">Loading...</div>
    </div>
    <div id="modal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <div id="modal-body"></div>
        </div>
    </div>
    <script>
    const checkpointData = {checkpoint_json};
    const DIM_OCCURRENCES = {dim_occ_json};
    const VALUES_LOOKUP = {values_json};
    const TOTAL_LOCATIONS = {n_locations};

    // Helper to get value from ID
    function getValue(occ) {{ return VALUES_LOOKUP[occ.v] || ''; }}

    // Groups from cognitive_gen/dimension_pool.py
    const DIMENSION_GROUPS = {{
        'existential': [
            'relationship_to_existence', 'cosmic_stance', 'mortality_awareness',
            'meaning_framework', 'relationship_to_time', 'ontological_position',
            'relationship_to_infinity',
        ],
        'identity': [
            'core_belief', 'self_image', 'shadow_self', 'identity_they_reject',
        ],
        'intellectual': [
            'intellectual_stance', 'relationship_to_uncertainty',
            'how_they_know_things', 'relationship_to_truth', 'what_counts_as_evidence',
        ],
        'moral': [
            'moral_framework', 'what_outrages_them', 'what_they_protect',
        ],
        'embodied': [
            'body_state', 'preverbal_feeling', 'where_tension_lives',
            'somatic_memory', 'felt_sense_of_world', 'breathing_pattern',
            'posture_toward_existence',
        ],
        'relational': [
            'stance_toward_reader', 'who_they_write_for', 'what_they_want_reader_to_feel',
            'audience_relationship', 'relationship_to_authority', 'relationship_to_collective',
            'what_they_offer',
        ],
        'shadow': [
            'what_they_hide', 'what_they_cant_say_directly', 'forbidden_knowledge',
            'secret_shame', 'unacknowledged_desire', 'deepest_fear', 'deepest_desire',
        ],
        'dark': [
            'the_cruelty_they_enjoy', 'who_they_resent', 'the_lie_they_tell_themselves',
            'their_capacity_for_violence', 'what_they_refuse_to_forgive', 'what_they_want_to_destroy',
            'the_envy_they_wont_admit', 'their_contempt', 'the_power_they_crave',
            'their_petty_satisfactions', 'the_mask_they_wear', 'what_they_would_do_if_no_one_knew',
        ],
        'clinical': [
            'narcissistic_supply', 'machiavellian_calculation', 'psychopathic_detachment',
            'magical_thinking', 'paranoid_ideation', 'ideas_of_reference', 'apophenia',
            'perceptual_disturbance', 'dissociative_tendency', 'identity_fragmentation',
            'derealization', 'grandiose_fantasy', 'persecution_complex', 'messianic_tendency',
            'obsessive_fixation', 'compulsive_ritual', 'intrusive_thoughts',
            'abandonment_terror', 'engulfment_fear', 'object_inconstancy',
        ],
        'temporal': [
            'relationship_to_past', 'relationship_to_future', 'relationship_to_present',
            'what_haunts_them', 'what_they_anticipate', 'sense_of_urgency',
        ],
        'aesthetic': [
            'what_they_notice', 'what_they_find_beautiful', 'what_they_find_ugly',
            'how_they_see_details', 'sensory_orientation', 'relationship_to_language',
        ],
        'creative': [
            'active_preoccupation', 'creative_mission', 'life_circumstance',
            'emotional_weather', 'central_tension',
        ],
        'psychodynamic': [
            'the_wound', 'the_compensation',
        ],
        'deadly_sins': [
            'pride', 'greed', 'lust', 'envy', 'gluttony', 'wrath', 'sloth',
        ],
        'fruits_of_spirit': [
            'love_agape', 'joy', 'peace', 'patience', 'kindness',
            'goodness', 'faithfulness', 'gentleness', 'self_control',
        ],
    }};

    function getDimensionGroup(dim) {{
        for (const [group, dims] of Object.entries(DIMENSION_GROUPS)) {{
            if (dims.includes(dim)) return group;
        }}
        return 'other';
    }}

    function formatDimName(name) {{
        return name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
    }}

    // Create a link to the essay on mtomei.com (only for base locations, not meta)
    function essayLink(slug, className = '') {{
        if (slug.startsWith('meta_')) {{
            return '<span class="' + className + '">' + slug + '</span>';
        }}
        return '<a href="http://mtomei.com/' + slug + '.html" target="_blank" class="essay-link ' + className + '">' + slug + '</a>';
    }}

    function showDimensionOccurrences(dimName) {{
        const modal = document.getElementById('modal');
        const modalBody = document.getElementById('modal-body');
        const allOccurrences = DIM_OCCURRENCES[dimName] || [];
        const group = getDimensionGroup(dimName);

        // Filter occurrences based on current filter
        const occurrences = allOccurrences.filter(occ => {{
            if (locationFilter === 'base' && isMetaLocation(occ.location)) return false;
            if (locationFilter === 'meta' && !isMetaLocation(occ.location)) return false;
            return true;
        }});

        // Count filtered locations for missing calculation
        let filteredLocationCount = 0;
        for (const slug of Object.keys(checkpointData.locations)) {{
            if (locationFilter === 'all' ||
                (locationFilter === 'base' && !isMetaLocation(slug)) ||
                (locationFilter === 'meta' && isMetaLocation(slug))) {{
                filteredLocationCount++;
            }}
        }}

        // Count all categories
        const counts = {{ ph: 0, pm: 0, pl: 0, nh: 0, nm: 0, nl: 0, zero: 0 }};
        let totalScore = 0;
        for (const occ of occurrences) {{
            const score = INTENSITY_WEIGHTS[occ.classification] || 0;
            totalScore += score;
            if (occ.classification === 'positive-high') counts.ph++;
            else if (occ.classification === 'positive-medium') counts.pm++;
            else if (occ.classification === 'positive-low') counts.pl++;
            else if (occ.classification === 'negative-high') counts.nh++;
            else if (occ.classification === 'negative-medium') counts.nm++;
            else if (occ.classification === 'negative-low') counts.nl++;
            else if (occ.classification === 'zero') counts.zero++;
        }}
        const missCount = filteredLocationCount - occurrences.length;
        const posTotal = counts.ph + counts.pm + counts.pl;
        const negTotal = counts.nh + counts.nm + counts.nl;

        let html = '<div class="modal-title">' + formatDimName(dimName) + '</div>';
        html += '<div class="modal-subtitle">Group: ' + formatDimName(group) + ' | ';
        html += '<strong>Score: ' + totalScore + '</strong> | ';
        html += '<span style="color: var(--positive)">' + posTotal + ' pos</span> ';
        html += '<span style="color: var(--positive); opacity: 0.7">(H:' + counts.ph + ' M:' + counts.pm + ' L:' + counts.pl + ')</span>, ';
        html += '<span style="color: var(--negative)">' + negTotal + ' neg</span> ';
        html += '<span style="color: var(--negative); opacity: 0.7">(H:' + counts.nh + ' M:' + counts.nm + ' L:' + counts.nl + ')</span>, ';
        html += '<span style="color: var(--text-secondary)">' + counts.zero + ' zero, ' + missCount + ' missing</span></div>';

        // Sort by score (weight) descending, then by fitness
        const sorted = [...occurrences].sort((a, b) => {{
            const scoreA = INTENSITY_WEIGHTS[a.classification] || 0;
            const scoreB = INTENSITY_WEIGHTS[b.classification] || 0;
            if (scoreB !== scoreA) return scoreB - scoreA;
            return b.fitness - a.fitness;
        }});

        for (const occ of sorted) {{
            const fitnessPct = (occ.fitness * 100).toFixed(1);
            const score = INTENSITY_WEIGHTS[occ.classification] || 0;
            const classLabel = occ.classification.replace('-', ' ').replace(/\\b\\w/g, c => c.toUpperCase());
            const baseClass = occ.classification.startsWith('positive') ? 'positive' :
                              occ.classification.startsWith('negative') ? 'negative' : 'zero';
            html += '<div class="occurrence ' + baseClass + '">';
            html += '<div class="occurrence-header">';
            html += essayLink(occ.location, 'occurrence-location');
            html += '<div class="occurrence-badges">';
            // Show origin if different from current location
            if (occ.origin && occ.origin.location !== occ.location) {{
                html += '<span class="occurrence-origin" title="Originated at ' + occ.origin.location + ' gen ' + occ.origin.generation + '">↖ ' + essayLink(occ.origin.location) + ' (g' + occ.origin.generation + ')</span>';
            }} else if (occ.origin) {{
                html += '<span class="occurrence-origin native" title="Native to this location">● native (g' + occ.origin.generation + ')</span>';
            }}
            const scoreLabel = score >= 0 ? '+' + score : score;
            html += '<span class="occurrence-class ' + baseClass + '" title="Score: ' + score + '">' + classLabel + ' (' + scoreLabel + ')</span>';
            html += '<span class="occurrence-fitness">' + fitnessPct + '% fitness</span>';
            html += '</div></div>';
            html += '<div class="occurrence-value">' + getValue(occ) + '</div>';
            html += '</div>';
        }}

        modalBody.innerHTML = html;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }}

    function closeModal() {{
        document.getElementById('modal').classList.remove('active');
        document.body.style.overflow = '';
    }}

    function getOrigin(location, dimName) {{
        // Look up origin for a dimension at a specific location
        const occurrences = DIM_OCCURRENCES[dimName] || [];
        const occ = occurrences.find(o => o.location === location);
        return occ?.origin || null;
    }}

    document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') closeModal();
    }});
    document.getElementById('modal').addEventListener('click', (e) => {{
        if (e.target.id === 'modal') closeModal();
    }});

    function classifyValue(value) {{
        if (value === null || value === undefined) return 'missing';
        if (Array.isArray(value) || typeof value === 'object') {{
            return Object.keys(value).length > 0 ? 'positive' : 'missing';
        }}
        const str = String(value).trim();
        if (!str) return 'missing';

        // Positive override patterns (hedged but positive inferences)
        const posOverridePatterns = [
            /it'?s?\\s+not\\s+explicitly\\s+stated,?\\s+but\\b/i,
            /while\\s+not\\s+explicitly/i,
            /although\\s+not\\s+explicitly/i,
            /though\\s+not\\s+explicitly/i,
            /not\\s+explicitly\\s+(?:stated|described|mentioned),?\\s+but\\b/i,
            /\\bsuggests\\s+(?:a|an?|that|the)\\b/i,
            /\\bmay\\s+\\w*\\s*(?:be|have|reflect|indicate|suggest)\\b/i,
            /\\bcould\\s+\\w*\\s*(?:be|have|reflect|indicate|suggest)\\b/i,
            /\\bmight\\s+\\w*\\s*(?:be|have|reflect|indicate|suggest)\\b/i,
            /\\bseems\\s+to\\b/i,
            /\\bappears\\s+to\\b/i,
            /\\bdisregard\\s+for\\b/i,
            /\\bwhat\\s+(?:triggers|outrages|angers|bothers)\\b/i,
            /\\bthe\\s+(?:disregard|exploitation|abuse|neglect|mistreatment)\\b/i,
            /,\\s*as\\s+they\\s+are\\b/i,
        ];
        for (const pat of posOverridePatterns) {{
            if (pat.test(str)) return 'positive';
        }}

        const negPatterns = [
            /\\bno\\s+(?:indication|evidence|clear|explicit|specific)/i,
            /\\bunclear\\b/i,
            /\\bcannot\\s+(?:determine|be\\s+determined|infer|identify)/i,
            /\\bdifficult\\s+to\\s+(?:determine|identify|assess)/i,
            /\\bunknown\\b/i,
            /\\black\\s+of\\b/i,
            /\\babsence\\s+of\\b/i,
            /\\bdoes\\s+not\\s+(?:provide|exhibit|show|indicate|express|reveal|seem|suggest)/i,
            /\\bthere\\s+is\\s+no\\b/i,
            /\\bnull\\b/i,
            /^n\\/a$/i,
            /^none$/i,
            /\\bit\\s+is\\s+unclear\\b/i,
            /\\bis\\s+not\\s+(?:clear|explicitly|possible)/i,
            /\\blittle\\s+to\\s+no\\b/i,
            /\\bminimal\\b/i,
            /^low$/i,
            /:\\s*low\\b/i,
            /\\bnot\\s+evident\\b/i,
            /\\bnot\\s+applicable\\b/i,
        ];

        for (const pat of negPatterns) {{
            if (pat.test(str)) return 'negative';
        }}
        return 'positive';
    }}

    // Build classification lookup from DIM_OCCURRENCES (pre-computed by LLM)
    const classificationLookup = {{}};
    for (const [dim, occurrences] of Object.entries(DIM_OCCURRENCES)) {{
        for (const occ of occurrences) {{
            classificationLookup[occ.location + '|' + dim] = occ.classification;
        }}
    }}

    function getClassification(location, dim) {{
        return classificationLookup[location + '|' + dim] || 'positive';
    }}

    function renderFitnessGraph(history, width = 300, height = 60) {{
        if (!history || history.length < 2) return '<div style="color: var(--text-secondary); font-size: 0.8em;">No history yet</div>';

        const padding = 5;
        const graphWidth = width - padding * 2;
        const graphHeight = height - padding * 2;

        // Extract best fitness values
        const values = history.map(h => h.best);
        const minVal = Math.min(...values) * 0.95;
        const maxVal = Math.max(...values) * 1.05;
        const range = maxVal - minVal || 0.01;

        // Create SVG path
        const points = values.map((v, i) => {{
            const x = padding + (i / (values.length - 1)) * graphWidth;
            const y = padding + graphHeight - ((v - minVal) / range) * graphHeight;
            return `${{x}},${{y}}`;
        }});

        const pathD = 'M ' + points.join(' L ');

        // Create area fill
        const areaD = pathD + ` L ${{padding + graphWidth}},${{padding + graphHeight}} L ${{padding}},${{padding + graphHeight}} Z`;

        return `<svg width="${{width}}" height="${{height}}" style="background: var(--bg-section); border-radius: 4px;">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:var(--accent);stop-opacity:0.3"/>
                    <stop offset="100%" style="stop-color:var(--accent);stop-opacity:0.05"/>
                </linearGradient>
            </defs>
            <path d="${{areaD}}" fill="url(#grad)"/>
            <path d="${{pathD}}" fill="none" stroke="var(--accent)" stroke-width="2"/>
            <text x="${{padding}}" y="${{height - 2}}" fill="var(--text-secondary)" font-size="9">Gen 1</text>
            <text x="${{width - padding - 30}}" y="${{height - 2}}" fill="var(--text-secondary)" font-size="9">Gen ${{values.length}}</text>
            <text x="${{width - padding}}" y="${{padding + 10}}" fill="var(--positive)" font-size="10" text-anchor="end">${{(maxVal * 100).toFixed(0)}}%</text>
            <text x="${{width - padding}}" y="${{height - padding - 5}}" fill="var(--text-secondary)" font-size="10" text-anchor="end">${{(minVal * 100).toFixed(0)}}%</text>
        </svg>`;
    }}

    let locationFilter = 'all';

    function setLocationFilter(filter) {{
        locationFilter = filter;
        document.querySelectorAll('.controls .sort-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('filter-' + filter).classList.add('active');
        renderLocations();
        renderHistogram();
        renderSharedValues();
    }}

    function isMetaLocation(slug) {{
        return slug.startsWith('meta_');
    }}

    function renderLocations() {{
        const container = document.getElementById('locations');
        const locations = checkpointData.locations;
        const sorted = Object.entries(locations)
            .filter(([slug, loc]) => {{
                if (locationFilter === 'all') return true;
                if (locationFilter === 'base') return !isMetaLocation(slug);
                if (locationFilter === 'meta') return isMetaLocation(slug);
                return true;
            }})
            .sort((a, b) => {{
                return (b[1].best_ever_fitness || 0) - (a[1].best_ever_fitness || 0);
            }});
        let html = '';
        if (sorted.length === 0) {{
            html = '<p style="color: var(--text-secondary);">No locations match the current filter.</p>';
        }}
        for (const [slug, loc] of sorted) {{
            const fitness = ((loc.best_ever_fitness || 0) * 100).toFixed(1);
            // Get values from DIM_OCCURRENCES instead of population (to avoid duplication)
            const groups = {{}};
            for (const [dim, occurrences] of Object.entries(DIM_OCCURRENCES)) {{
                const occ = occurrences.find(o => o.location === slug);
                if (!occ) continue;
                const group = getDimensionGroup(dim);
                if (!groups[group]) groups[group] = [];
                groups[group].push({{name: dim, value: getValue(occ), classification: occ.classification, origin: occ.origin, location: slug}});
            }}

            // Get fitness history from stagnation_state
            const history = loc.stagnation_state?.history || [];
            const stagnation = loc.stagnation_state || {{}};
            const severity = history.length >= 5 ? Math.min(1, (history.length - history.findLastIndex((h, i, arr) => i > 0 && h.best > arr[i-1].best + 0.005) || history.length) / 10) : 0;
            const statusIcon = stagnation.deeply_stuck ? '◆' : severity > 0.5 ? '▲' : '●';
            const statusColor = stagnation.deeply_stuck ? 'var(--negative)' : severity > 0.5 ? '#fbbf24' : 'var(--positive)';

            html += '<details class="location"><summary>' + essayLink(slug, 'location-name') + '<span><span style="color:' + statusColor + '; margin-right: 8px;">' + statusIcon + '</span><span class="location-fitness">' + fitness + '%</span><span class="arrow">&#9654;</span></span></summary><div class="location-content">';

            // Add fitness graph
            html += '<div style="margin-bottom: 15px;">';
            html += '<div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 5px;">Fitness History (best per generation)</div>';
            html += renderFitnessGraph(history);
            html += '<div style="color: var(--text-secondary); font-size: 0.8em; margin-top: 5px;">Diversity injections: ' + (stagnation.diversity_injections || 0) + '</div>';
            html += '</div>';

            for (const [group, dims] of Object.entries(groups).sort()) {{
                html += '<details class="group"><summary><span class="group-name">' + formatDimName(group) + '</span><span><span class="group-count">' + dims.length + ' dimensions</span><span class="arrow">&#9654;</span></span></summary><div class="group-content">';
                for (const dim of dims) {{
                    let originBadge = '';
                    if (dim.origin && dim.origin.location !== dim.location) {{
                        originBadge = '<span class="dim-origin" title="Originated at ' + dim.origin.location + ' gen ' + dim.origin.generation + '">↖ ' + essayLink(dim.origin.location) + ' (g' + dim.origin.generation + ')</span>';
                    }} else if (dim.origin) {{
                        originBadge = '<span class="dim-origin native" title="Native to this location">● native (g' + dim.origin.generation + ')</span>';
                    }}
                    html += '<div class="dimension ' + dim.classification + '"><div class="dim-header"><span class="dim-name">' + formatDimName(dim.name) + '</span>' + originBadge + '</div><div class="dim-value">' + dim.value + '</div></div>';
                }}
                html += '</div></details>';
            }}
            html += '</div></details>';
        }}
        container.innerHTML = html;
    }}

    // Intensity weights for sorting
    // Weights: positive adds, negative subtracts
    const INTENSITY_WEIGHTS = {{
        'positive-high': 4, 'positive-medium': 2, 'positive-low': 1,
        'negative-high': -4, 'negative-medium': -2, 'negative-low': -1,
        'zero': 0, 'missing': 0
    }};

    // Compute histogram data with filter
    function computeHistogramData() {{
        const dimStats = {{}};
        const groupWeightedScores = {{}};
        const groupLocationsPositive = {{}};
        const groupLocationsNegative = {{}};
        const groupLocationsAny = {{}};

        // Count filtered locations
        let filteredLocationCount = 0;
        for (const slug of Object.keys(checkpointData.locations)) {{
            if (locationFilter === 'all' ||
                (locationFilter === 'base' && !isMetaLocation(slug)) ||
                (locationFilter === 'meta' && isMetaLocation(slug))) {{
                filteredLocationCount++;
            }}
        }}

        for (const [dim, occurrences] of Object.entries(DIM_OCCURRENCES)) {{
            let ph = 0, pm = 0, pl = 0, nh = 0, nm = 0, nl = 0, zero = 0;
            let weightedSum = 0, posScore = 0, negScore = 0;
            const group = getDimensionGroup(dim);

            if (!groupLocationsPositive[group]) groupLocationsPositive[group] = new Set();
            if (!groupLocationsNegative[group]) groupLocationsNegative[group] = new Set();
            if (!groupLocationsAny[group]) groupLocationsAny[group] = new Set();
            if (!groupWeightedScores[group]) groupWeightedScores[group] = {{ net: 0, pos: 0, neg: 0 }};

            for (const occ of occurrences) {{
                // Apply filter
                if (locationFilter === 'base' && isMetaLocation(occ.location)) continue;
                if (locationFilter === 'meta' && !isMetaLocation(occ.location)) continue;

                groupLocationsAny[group].add(occ.location);
                const weight = INTENSITY_WEIGHTS[occ.classification] || 0;
                weightedSum += weight;
                groupWeightedScores[group].net += weight;

                if (occ.classification === 'positive-high') {{
                    ph++; posScore += 4;
                    groupLocationsPositive[group].add(occ.location);
                    groupWeightedScores[group].pos += 4;
                }} else if (occ.classification === 'positive-medium') {{
                    pm++; posScore += 2;
                    groupLocationsPositive[group].add(occ.location);
                    groupWeightedScores[group].pos += 2;
                }} else if (occ.classification === 'positive-low') {{
                    pl++; posScore += 1;
                    groupLocationsPositive[group].add(occ.location);
                    groupWeightedScores[group].pos += 1;
                }} else if (occ.classification === 'negative-high') {{
                    nh++; negScore += 4;
                    groupLocationsNegative[group].add(occ.location);
                    groupWeightedScores[group].neg += 4;
                }} else if (occ.classification === 'negative-medium') {{
                    nm++; negScore += 2;
                    groupLocationsNegative[group].add(occ.location);
                    groupWeightedScores[group].neg += 2;
                }} else if (occ.classification === 'negative-low') {{
                    nl++; negScore += 1;
                    groupLocationsNegative[group].add(occ.location);
                    groupWeightedScores[group].neg += 1;
                }} else if (occ.classification === 'zero') {{
                    zero++;
                }}
            }}
            const total = ph + pm + pl + nh + nm + nl + zero;
            const miss = filteredLocationCount - total;
            const positive = ph + pm + pl;
            const negative = nh + nm + nl;
            dimStats[dim] = {{ ph, pm, pl, nh, nm, nl, zero, positive, negative, missing: miss, weighted: weightedSum, posScore, negScore, total: filteredLocationCount }};
        }}

        const byGroup = {{}};
        for (const dim of Object.keys(dimStats)) {{
            const group = getDimensionGroup(dim);
            if (!byGroup[group]) byGroup[group] = [];
            byGroup[group].push(dim);
        }}

        const groupCoverage = {{}};
        for (const group of Object.keys(byGroup)) {{
            const scores = groupWeightedScores[group] || {{ net: 0, pos: 0, neg: 0 }};
            groupCoverage[group] = {{
                positive: ((groupLocationsPositive[group]?.size || 0) / filteredLocationCount) * 100,
                negative: ((groupLocationsNegative[group]?.size || 0) / filteredLocationCount) * 100,
                any: ((groupLocationsAny[group]?.size || 0) / filteredLocationCount) * 100,
                weighted: scores.net,
                weightedPos: scores.pos,
                weightedNeg: scores.neg,
            }};
        }}

        return {{ dimStats, byGroup, groupCoverage, filteredLocationCount }};
    }}

    let currentSortMode = 'weighted';
    let currentScoreMode = 'net';

    function sortHistogram(mode) {{
        currentSortMode = mode;
        document.querySelectorAll('#histogram .sort-btn').forEach(btn => {{
            if (btn.id.startsWith('sort-')) btn.classList.remove('active');
        }});
        document.getElementById('sort-' + mode).classList.add('active');
        renderHistogram();
    }}

    function setScoreMode(mode) {{
        currentScoreMode = mode;
        document.querySelectorAll('#histogram .sort-btn').forEach(btn => {{
            if (btn.id.startsWith('score-')) btn.classList.remove('active');
        }});
        document.getElementById('score-' + mode).classList.add('active');
        renderHistogram();
    }}

    function getScoreForMode(stats, mode) {{
        if (mode === 'pos') return stats.posScore || 0;
        if (mode === 'neg') return stats.negScore || 0;
        return stats.weighted || 0;
    }}

    function getGroupScoreForMode(coverage, mode) {{
        if (mode === 'pos') return coverage.weightedPos || 0;
        if (mode === 'neg') return coverage.weightedNeg || 0;
        return coverage.weighted || 0;
    }}

    function renderHistogram() {{
        const container = document.getElementById('hist-content');
        const {{ dimStats, byGroup, groupCoverage, filteredLocationCount }} = computeHistogramData();

        const sortedGroups = Object.entries(byGroup).sort((a, b) => {{
            if (currentSortMode === 'weighted') {{
                return getGroupScoreForMode(groupCoverage[b[0]], currentScoreMode) - getGroupScoreForMode(groupCoverage[a[0]], currentScoreMode);
            }}
            return (groupCoverage[b[0]][currentSortMode] || 0) - (groupCoverage[a[0]][currentSortMode] || 0);
        }});

        let html = '';
        for (const [group, dims] of sortedGroups) {{
            if (currentSortMode === 'negative') {{
                dims.sort((a, b) => dimStats[b].negative - dimStats[a].negative);
            }} else if (currentSortMode === 'weighted') {{
                dims.sort((a, b) => getScoreForMode(dimStats[b], currentScoreMode) - getScoreForMode(dimStats[a], currentScoreMode));
            }} else {{
                dims.sort((a, b) => dimStats[b].positive - dimStats[a].positive);
            }}

            const coverage = groupCoverage[group];
            const groupScore = getGroupScoreForMode(coverage, currentScoreMode);
            const scoreModeLabel = currentScoreMode === 'pos' ? '+score' : currentScoreMode === 'neg' ? '-score' : 'score';
            const label = currentSortMode === 'weighted' ? scoreModeLabel + ': ' + groupScore.toFixed(0) :
                          currentSortMode === 'positive' ? coverage.positive.toFixed(0) + '% positive' :
                          currentSortMode === 'negative' ? coverage.negative.toFixed(0) + '% negative' :
                          coverage.any.toFixed(0) + '% any';

            html += '<details class="hist-group"><summary style="cursor: pointer; margin-bottom: 8px;"><span class="hist-group-name">' + formatDimName(group) + ' (' + label + ')</span></summary>';

            for (const dim of dims) {{
                const stats = dimStats[dim];
                // Calculate bar widths for all 7 categories (relative to filtered count)
                const phPct = (stats.ph / filteredLocationCount) * 100;
                const pmPct = (stats.pm / filteredLocationCount) * 100;
                const plPct = (stats.pl / filteredLocationCount) * 100;
                const nhPct = (stats.nh / filteredLocationCount) * 100;
                const nmPct = (stats.nm / filteredLocationCount) * 100;
                const nlPct = (stats.nl / filteredLocationCount) * 100;
                const zeroPct = (stats.zero / filteredLocationCount) * 100;

                const dimScore = getScoreForMode(stats, currentScoreMode);
                html += '<div class="hist-bar-container" onclick="showDimensionOccurrences(\\'' + dim + '\\')">';
                html += '<span class="hist-label" title="' + formatDimName(dim) + ' (' + scoreModeLabel + ': ' + dimScore + ')">' + formatDimName(dim) + '</span>';
                html += '<div class="hist-bar">';
                // Positive bars (green, varying intensity)
                html += '<div class="hist-pos-high" style="width: ' + phPct + '%" title="+High: ' + stats.ph + '"></div>';
                html += '<div class="hist-pos-med" style="width: ' + pmPct + '%" title="+Med: ' + stats.pm + '"></div>';
                html += '<div class="hist-pos-low" style="width: ' + plPct + '%" title="+Low: ' + stats.pl + '"></div>';
                // Negative bars (red, varying intensity)
                html += '<div class="hist-neg-high" style="width: ' + nhPct + '%" title="-High: ' + stats.nh + '"></div>';
                html += '<div class="hist-neg-med" style="width: ' + nmPct + '%" title="-Med: ' + stats.nm + '"></div>';
                html += '<div class="hist-neg-low" style="width: ' + nlPct + '%" title="-Low: ' + stats.nl + '"></div>';
                // Zero bar (gray)
                html += '<div class="hist-zero" style="width: ' + zeroPct + '%" title="Zero: ' + stats.zero + '"></div>';
                html += '</div>';
                const displayScore = dimScore;
                const scoreClass = currentScoreMode === 'neg' ? 'negative' :
                                   currentScoreMode === 'pos' ? 'positive' :
                                   (displayScore > 0 ? 'positive' : displayScore < 0 ? 'negative' : '');
                const scorePrefix = displayScore > 0 ? '+' : '';
                html += '<span class="hist-score ' + scoreClass + '">' + scorePrefix + displayScore + '</span>';
                html += '</div>';
            }}
            html += '</details>';
        }}
        container.innerHTML = html;
    }}

    function expandAll() {{ document.querySelectorAll('details').forEach(d => d.open = true); }}
    function collapseAll() {{ document.querySelectorAll('details').forEach(d => d.open = false); }}
    function expandHistGroups() {{ document.querySelectorAll('.hist-group').forEach(d => d.open = true); }}
    function collapseHistGroups() {{ document.querySelectorAll('.hist-group').forEach(d => d.open = false); }}
    function expandLocations() {{
        document.querySelectorAll('.location').forEach(d => d.open = true);
        document.querySelectorAll('.group').forEach(d => d.open = false);
    }}

    // Shared values analysis (with filter support)
    function computeSharedValuesData() {{
        const valueMap = {{}};  // dim -> value -> [{{location, fitness, classification}}]

        for (const [dim, occurrences] of Object.entries(DIM_OCCURRENCES)) {{
            if (!valueMap[dim]) valueMap[dim] = {{}};
            for (const occ of occurrences) {{
                // Apply filter
                if (locationFilter === 'base' && isMetaLocation(occ.location)) continue;
                if (locationFilter === 'meta' && !isMetaLocation(occ.location)) continue;

                const val = String(getValue(occ)).trim();
                if (!val || val === '(null)') continue;
                if (!valueMap[dim][val]) valueMap[dim][val] = [];
                valueMap[dim][val].push({{
                    location: occ.location,
                    fitness: occ.fitness,
                    classification: occ.classification,
                    origin: occ.origin
                }});
            }}
        }}

        // Flatten to list of {{dim, value, count, locations, classification}}
        const shared = [];
        for (const [dim, values] of Object.entries(valueMap)) {{
            for (const [value, locs] of Object.entries(values)) {{
                if (locs.length >= 2) {{
                    shared.push({{
                        dim: dim,
                        value: value,
                        count: locs.length,
                        locations: locs,
                        classification: locs[0].classification,
                        avgFitness: locs.reduce((s, l) => s + l.fitness, 0) / locs.length
                    }});
                }}
            }}
        }}

        // Sort by count descending
        shared.sort((a, b) => b.count - a.count);
        return shared;
    }}

    let minOccurrences = 2;

    function setMinOccurrences(n) {{
        minOccurrences = n;
        document.querySelectorAll('#shared-values .sort-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('min-' + n).classList.add('active');
        renderSharedValues();
    }}

    function renderSharedValues() {{
        const container = document.getElementById('shared-content');
        const sharedData = computeSharedValuesData();
        const filtered = sharedData.filter(s => s.count >= minOccurrences);

        if (filtered.length === 0) {{
            container.innerHTML = '<p style="color: var(--text-secondary);">No values appear in ' + minOccurrences + '+ locations.</p>';
            return;
        }}

        let html = '<div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 10px;">' +
                   filtered.length + ' shared values found</div>';

        for (const item of filtered) {{
            const truncValue = item.value.length > 80 ? item.value.substring(0, 80) + '...' : item.value;
            const clsClass = item.classification.startsWith('positive') ? 'positive' :
                             item.classification.startsWith('negative') ? 'negative' : '';

            // Find the origin (earliest generation)
            let originLoc = null;
            let originGen = Infinity;
            for (const loc of item.locations) {{
                if (loc.origin && loc.origin.generation < originGen) {{
                    originGen = loc.origin.generation;
                    originLoc = loc.origin.location;
                }}
            }}

            html += '<details class="group" style="margin-bottom: 8px;">';
            html += '<summary style="cursor: pointer;">';
            html += '<span style="color: var(--accent-light); font-weight: 600;">' + formatDimName(item.dim) + '</span>';
            html += '<span style="color: var(--text-secondary); margin-left: 10px;">(' + item.count + ' locations)</span>';
            if (originLoc) {{
                html += '<span class="dim-origin" style="margin-left: 8px;" title="First appeared at ' + originLoc + ' gen ' + originGen + '">↖ ' + essayLink(originLoc) + ' (g' + originGen + ')</span>';
            }}
            html += '<span style="color: var(--text-secondary); margin-left: 5px; font-size: 0.85em;">"' + truncValue + '"</span>';
            html += '</summary>';
            html += '<div style="padding: 10px; border-left: 3px solid var(--' + clsClass + '); margin: 5px 0;">';
            html += '<div style="color: var(--text-primary); margin-bottom: 8px; white-space: pre-wrap; line-height: 1.5;">' + item.value + '</div>';

            // Show locations with origin info
            html += '<div style="color: var(--text-secondary); font-size: 0.85em; margin-top: 10px;">';
            html += '<strong>Locations:</strong></div>';
            html += '<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 5px;">';
            for (const loc of item.locations.sort((a, b) => (a.origin?.generation || 999) - (b.origin?.generation || 999))) {{
                const isNative = loc.origin && loc.origin.location === loc.location;
                const originClass = isNative ? 'native' : '';
                const originTitle = loc.origin ? (isNative ? 'Native (g' + loc.origin.generation + ')' : 'From ' + loc.origin.location + ' (g' + loc.origin.generation + ')') : '';
                html += '<span class="dim-origin ' + originClass + '" title="' + originTitle + '">' + essayLink(loc.location);
                if (loc.origin) {{
                    html += ' <span style="opacity: 0.7">(g' + loc.origin.generation + ')</span>';
                }}
                html += '</span>';
            }}
            html += '</div>';

            html += '<div style="color: var(--text-secondary); font-size: 0.85em; margin-top: 8px;">';
            html += '<strong>Avg fitness:</strong> ' + (item.avgFitness * 100).toFixed(1) + '%</div>';
            html += '</div>';
            html += '</details>';
        }}

        container.innerHTML = html;
    }}

    renderLocations();
    renderHistogram();
    renderSharedValues();
    </script>
</body>
</html>'''

    return html


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate interactive HTML report from checkpoint')
    parser.add_argument('checkpoint', type=Path, help='Path to checkpoint JSON file')
    parser.add_argument('-o', '--output', type=Path, default=None, help='Output HTML file path')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM classification (use simple fallback)')

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1

    checkpoint_data = load_checkpoint(args.checkpoint)

    if args.output is None:
        gen = checkpoint_data.get('metadata', {}).get('generation', 'unknown')
        args.output = Path(f'results/interactive_report_gen{gen}.html')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_report(
        checkpoint_data,
        args.output,
        checkpoint_path=args.checkpoint,
        use_llm=not args.no_llm,
    )

    return 0


if __name__ == '__main__':
    exit(main())
