"""
Interactive HTML report generator for cognitive hypothesis search.

Generates reports from checkpoint data with:
- Dimension histograms showing positive/negative/missing breakdown
- Click-through to see individual occurrences
- Presence/absence classification using LLM
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional


# Cache directory for LLM classifications
CLASSIFICATION_CACHE_DIR = Path("/tmp/classification_cache")


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
        prompt = f"""Classify this psychological dimension description as either POSITIVE (substantive content about the trait) or NEGATIVE (trait is absent, minimal, or unknown).

Dimension: {dim.replace('_', ' ')}
Description: "{value}"

NEGATIVE if ANY of these apply:
- Says "Low", "Minimal", "Little", "None", "N/A"
- Says trait is "not present", "not evident", "not apparent", "not applicable"
- Says "unknown", "unclear", "cannot determine", "difficult to assess"
- Says there is "no indication", "no evidence", "no clear"
- Describes absence or lack ("does not show", "does not exhibit", "lacks")
- Only present "to a degree", "beyond what is needed", or similar qualified absence

POSITIVE only if it makes a substantive claim about what IS present (even if hedged with "may", "seems", "suggests").

Reply with only: POSITIVE or NEGATIVE"""
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
    pos_count = 0
    neg_count = 0
    for (loc, dim, value), response in zip(items_to_classify, all_responses):
        response_clean = response.strip().upper()
        if 'NEGATIVE' in response_clean:
            results[(loc, dim)] = 'negative'
            neg_count += 1
        else:
            results[(loc, dim)] = 'positive'
            pos_count += 1

    print(f"  Classification complete: {pos_count} positive, {neg_count} negative")

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

    # Build dimension occurrence data with classification
    dim_occurrences = {}  # dim -> list of {location, fitness, value, classification}

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
            dim_occurrences[dim].append({
                'location': slug,
                'fitness': fitness,
                'value': value if value is not None else '(null)',
                'classification': classification,
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

    # Convert dim_occurrences to JSON for JavaScript
    dim_occ_json = json.dumps(dim_occurrences, indent=2)
    checkpoint_json = json.dumps(checkpoint_data, indent=2)

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
        .dimension {{ margin-bottom: 12px; padding-left: 10px; border-left: 2px solid var(--border); }}
        .dimension:last-child {{ margin-bottom: 0; }}
        .dimension.positive {{ border-left-color: var(--positive); }}
        .dimension.negative {{ border-left-color: var(--negative); }}
        .dim-name {{ color: var(--accent); font-weight: 500; margin-bottom: 3px; }}
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
            position: relative;
        }}
        .hist-positive {{
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            background: var(--positive);
        }}
        .hist-negative {{
            position: absolute;
            right: 0;
            top: 0;
            height: 100%;
            background: var(--negative);
        }}
        .hist-count {{
            width: 80px;
            text-align: right;
            font-size: 0.8em;
            color: var(--text-secondary);
            display: flex;
            gap: 8px;
            justify-content: flex-end;
        }}
        .hist-count .pos {{ color: var(--positive); }}
        .hist-count .neg {{ color: var(--negative); }}
        .hist-count .miss {{ color: var(--text-secondary); }}
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
        .occurrence-class.positive {{ background: var(--positive); color: #000; }}
        .occurrence-class.negative {{ background: var(--negative); color: #000; }}
        .occurrence-value {{
            color: var(--text-primary);
            font-size: 0.95em;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <h1>Cognitive Hypothesis Search Report</h1>
    <div class="meta">Generation {generation} of {total_gens} | Generated {datetime.now().strftime('%Y-%m-%d')}</div>
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
    </div>
    <div id="locations"><p style="color: var(--text-secondary);">Loading...</p></div>
    <div class="histogram" id="histogram">
        <h2>Dimension Histogram (Best per Location)</h2>
        <div class="legend">
            <div class="legend-item"><div class="legend-box positive"></div><span>Positive (presence)</span></div>
            <div class="legend-item"><div class="legend-box missing"></div><span>Missing</span></div>
            <div class="legend-item"><div class="legend-box negative"></div><span>Negative (absence)</span></div>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.9em; margin-bottom: 10px;">Click any dimension to see all occurrences. Bar width = total locations ({n_locations}).</p>
        <div style="margin-bottom: 15px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <span style="color: var(--text-secondary); font-size: 0.85em;">Sort groups by:</span>
            <button id="sort-positive" class="sort-btn active" onclick="sortHistogram('positive')">% Positive</button>
            <button id="sort-negative" class="sort-btn" onclick="sortHistogram('negative')">% Negative</button>
            <button id="sort-any" class="sort-btn" onclick="sortHistogram('any')">% Any</button>
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
    const TOTAL_LOCATIONS = {n_locations};

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

    function showDimensionOccurrences(dimName) {{
        const modal = document.getElementById('modal');
        const modalBody = document.getElementById('modal-body');
        const occurrences = DIM_OCCURRENCES[dimName] || [];
        const group = getDimensionGroup(dimName);

        const posCount = occurrences.filter(o => o.classification === 'positive').length;
        const negCount = occurrences.filter(o => o.classification === 'negative').length;
        const missCount = TOTAL_LOCATIONS - occurrences.length;

        let html = '<div class="modal-title">' + formatDimName(dimName) + '</div>';
        html += '<div class="modal-subtitle">Group: ' + formatDimName(group) + ' | ';
        html += '<span style="color: var(--positive)">' + posCount + ' positive</span>, ';
        html += '<span style="color: var(--negative)">' + negCount + ' negative</span>, ';
        html += '<span style="color: var(--text-secondary)">' + missCount + ' missing</span></div>';

        // Sort by fitness descending
        const sorted = [...occurrences].sort((a, b) => b.fitness - a.fitness);

        for (const occ of sorted) {{
            const fitnessPct = (occ.fitness * 100).toFixed(1);
            const classLabel = occ.classification === 'positive' ? 'Positive' : 'Negative';
            html += '<div class="occurrence ' + occ.classification + '">';
            html += '<div class="occurrence-header">';
            html += '<span class="occurrence-location">' + occ.location + '</span>';
            html += '<div class="occurrence-badges">';
            html += '<span class="occurrence-class ' + occ.classification + '">' + classLabel + '</span>';
            html += '<span class="occurrence-fitness">' + fitnessPct + '% fitness</span>';
            html += '</div></div>';
            html += '<div class="occurrence-value">' + occ.value + '</div>';
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

    function renderLocations() {{
        const container = document.getElementById('locations');
        const locations = checkpointData.locations;
        const sorted = Object.entries(locations).sort((a, b) => {{
            return (b[1].best_ever_fitness || 0) - (a[1].best_ever_fitness || 0);
        }});
        let html = '';
        for (const [slug, loc] of sorted) {{
            const fitness = ((loc.best_ever_fitness || 0) * 100).toFixed(1);
            const best = loc.population[0]?.values || {{}};
            const groups = {{}};
            for (const [dim, value] of Object.entries(best)) {{
                const group = getDimensionGroup(dim);
                if (!groups[group]) groups[group] = [];
                const cls = getClassification(slug, dim);
                groups[group].push({{name: dim, value: value, classification: cls}});
            }}
            html += '<details class="location"><summary><span class="location-name">' + slug + '</span><span><span class="location-fitness">' + fitness + '%</span><span class="arrow">&#9654;</span></span></summary><div class="location-content">';
            for (const [group, dims] of Object.entries(groups).sort()) {{
                html += '<details class="group"><summary><span class="group-name">' + formatDimName(group) + '</span><span><span class="group-count">' + dims.length + ' dimensions</span><span class="arrow">&#9654;</span></span></summary><div class="group-content">';
                for (const dim of dims) {{
                    html += '<div class="dimension ' + dim.classification + '"><div class="dim-name">' + formatDimName(dim.name) + '</div><div class="dim-value">' + dim.value + '</div></div>';
                }}
                html += '</div></details>';
            }}
            html += '</div></details>';
        }}
        container.innerHTML = html;
    }}

    // Precompute histogram data once
    const histogramData = (function() {{
        const dimStats = {{}};
        const groupLocationsPositive = {{}};
        const groupLocationsNegative = {{}};
        const groupLocationsAny = {{}};

        for (const [dim, occurrences] of Object.entries(DIM_OCCURRENCES)) {{
            let pos = 0, neg = 0;
            const group = getDimensionGroup(dim);

            if (!groupLocationsPositive[group]) groupLocationsPositive[group] = new Set();
            if (!groupLocationsNegative[group]) groupLocationsNegative[group] = new Set();
            if (!groupLocationsAny[group]) groupLocationsAny[group] = new Set();

            for (const occ of occurrences) {{
                groupLocationsAny[group].add(occ.location);
                if (occ.classification === 'positive') {{
                    pos++;
                    groupLocationsPositive[group].add(occ.location);
                }} else if (occ.classification === 'negative') {{
                    neg++;
                    groupLocationsNegative[group].add(occ.location);
                }}
            }}
            const miss = TOTAL_LOCATIONS - occurrences.length;
            dimStats[dim] = {{ positive: pos, negative: neg, missing: miss }};
        }}

        const byGroup = {{}};
        for (const dim of Object.keys(dimStats)) {{
            const group = getDimensionGroup(dim);
            if (!byGroup[group]) byGroup[group] = [];
            byGroup[group].push(dim);
        }}

        const groupCoverage = {{}};
        for (const group of Object.keys(byGroup)) {{
            groupCoverage[group] = {{
                positive: ((groupLocationsPositive[group]?.size || 0) / TOTAL_LOCATIONS) * 100,
                negative: ((groupLocationsNegative[group]?.size || 0) / TOTAL_LOCATIONS) * 100,
                any: ((groupLocationsAny[group]?.size || 0) / TOTAL_LOCATIONS) * 100,
            }};
        }}

        return {{ dimStats, byGroup, groupCoverage }};
    }})();

    let currentSortMode = 'positive';

    function sortHistogram(mode) {{
        currentSortMode = mode;
        document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('sort-' + mode).classList.add('active');
        renderHistogram();
    }}

    function renderHistogram() {{
        const container = document.getElementById('hist-content');
        const {{ dimStats, byGroup, groupCoverage }} = histogramData;

        const sortedGroups = Object.entries(byGroup).sort((a, b) => {{
            return (groupCoverage[b[0]][currentSortMode] || 0) - (groupCoverage[a[0]][currentSortMode] || 0);
        }});

        let html = '';
        for (const [group, dims] of sortedGroups) {{
            if (currentSortMode === 'negative') {{
                dims.sort((a, b) => dimStats[b].negative - dimStats[a].negative);
            }} else {{
                dims.sort((a, b) => dimStats[b].positive - dimStats[a].positive);
            }}

            const coverage = groupCoverage[group];
            const label = currentSortMode === 'positive' ? coverage.positive.toFixed(0) + '% positive' :
                          currentSortMode === 'negative' ? coverage.negative.toFixed(0) + '% negative' :
                          coverage.any.toFixed(0) + '% any';

            html += '<details class="hist-group"><summary style="cursor: pointer; margin-bottom: 8px;"><span class="hist-group-name">' + formatDimName(group) + ' (' + label + ')</span></summary>';

            for (const dim of dims) {{
                const stats = dimStats[dim];
                const posPct = (stats.positive / TOTAL_LOCATIONS) * 100;
                const negPct = (stats.negative / TOTAL_LOCATIONS) * 100;

                html += '<div class="hist-bar-container" onclick="showDimensionOccurrences(\\'' + dim + '\\')">';
                html += '<span class="hist-label" title="' + formatDimName(dim) + '">' + formatDimName(dim) + '</span>';
                html += '<div class="hist-bar">';
                html += '<div class="hist-positive" style="width: ' + posPct + '%"></div>';
                html += '<div class="hist-negative" style="width: ' + negPct + '%"></div>';
                html += '</div>';
                html += '<span class="hist-count">';
                html += '<span class="pos">' + stats.positive + '</span>/';
                html += '<span class="neg">' + stats.negative + '</span>/';
                html += '<span class="miss">' + stats.missing + '</span>';
                html += '</span>';
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

    // Shared values analysis
    const sharedValuesData = (function() {{
        const valueMap = {{}};  // dim -> value -> [{{location, fitness, classification}}]

        for (const [dim, occurrences] of Object.entries(DIM_OCCURRENCES)) {{
            if (!valueMap[dim]) valueMap[dim] = {{}};
            for (const occ of occurrences) {{
                const val = String(occ.value).trim();
                if (!val || val === '(null)') continue;
                if (!valueMap[dim][val]) valueMap[dim][val] = [];
                valueMap[dim][val].push({{
                    location: occ.location,
                    fitness: occ.fitness,
                    classification: occ.classification
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
    }})();

    let minOccurrences = 2;

    function setMinOccurrences(n) {{
        minOccurrences = n;
        document.querySelectorAll('#shared-values .sort-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('min-' + n).classList.add('active');
        renderSharedValues();
    }}

    function renderSharedValues() {{
        const container = document.getElementById('shared-content');
        const filtered = sharedValuesData.filter(s => s.count >= minOccurrences);

        if (filtered.length === 0) {{
            container.innerHTML = '<p style="color: var(--text-secondary);">No values appear in ' + minOccurrences + '+ locations.</p>';
            return;
        }}

        let html = '<div style="color: var(--text-secondary); font-size: 0.85em; margin-bottom: 10px;">' +
                   filtered.length + ' shared values found</div>';

        for (const item of filtered) {{
            const truncValue = item.value.length > 100 ? item.value.substring(0, 100) + '...' : item.value;
            const locList = item.locations.map(l => l.location).join(', ');
            const clsClass = item.classification === 'positive' ? 'positive' : 'negative';

            html += '<details class="group" style="margin-bottom: 8px;">';
            html += '<summary style="cursor: pointer;">';
            html += '<span style="color: var(--accent-light); font-weight: 600;">' + formatDimName(item.dim) + '</span>';
            html += '<span style="color: var(--text-secondary); margin-left: 10px;">(' + item.count + ' locations, avg fitness: ' + (item.avgFitness * 100).toFixed(1) + '%)</span>';
            html += '</summary>';
            html += '<div style="padding: 10px; border-left: 3px solid var(--' + clsClass + '); margin: 5px 0;">';
            html += '<div style="color: var(--text-primary); margin-bottom: 8px;">"' + truncValue + '"</div>';
            html += '<div style="color: var(--text-secondary); font-size: 0.85em;">Locations: ' + locList + '</div>';
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
