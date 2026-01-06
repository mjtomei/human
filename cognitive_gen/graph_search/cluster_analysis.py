#!/usr/bin/env python3
"""
Cluster Analysis for Graph-Based Cognitive Search

Analyzes how dimension values cluster and spread between essays,
tracks improvement over generations, and identifies bridge essays.

Usage:
    python cluster_analysis.py /tmp/graph_search_checkpoints/
    python cluster_analysis.py /tmp/graph_search_checkpoints/ --output results/cluster_analysis.html
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import math


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_base_locations(checkpoint: dict) -> dict:
    """Extract only base (non-meta) locations."""
    return {
        name: data for name, data in checkpoint['locations'].items()
        if not name.startswith('meta_')
    }


def get_location_values(loc_data: dict) -> set:
    """Extract (dimension, value) pairs from a location's best hypothesis."""
    best_ever = loc_data.get('best_ever', {})
    values = best_ever.get('values', {}) if isinstance(best_ever, dict) else {}
    result = set()
    for dim, value in values.items():
        if isinstance(value, str):
            # Truncate for comparison
            result.add((dim, value[:100]))
    return result


def compute_shared_values(checkpoint: dict) -> dict:
    """Compute shared value counts between all location pairs."""
    locations = get_base_locations(checkpoint)
    location_values = {name: get_location_values(data) for name, data in locations.items()}

    overlaps = {}
    for loc1 in location_values:
        for loc2 in location_values:
            if loc1 < loc2:
                overlap = len(location_values[loc1] & location_values[loc2])
                overlaps[(loc1, loc2)] = overlap

    return overlaps, location_values


def compute_total_shared(location_values: dict) -> Counter:
    """Compute total shared values for each location."""
    total_shared = Counter()
    for loc, values in location_values.items():
        for (dim, val) in values:
            shared_count = sum(
                1 for other, other_vals in location_values.items()
                if other != loc and (dim, val) in other_vals
            )
            total_shared[loc] += shared_count
    return total_shared


def load_all_checkpoints(checkpoint_dir: str) -> list:
    """Load all checkpoints in order."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    for path in sorted(checkpoint_dir.glob('checkpoint_gen*.json')):
        gen = int(path.stem.split('gen')[1])
        checkpoints.append((gen, load_checkpoint(path)))
    return checkpoints


def track_fitness_over_time(checkpoints: list) -> dict:
    """Track fitness improvement for each base location over generations."""
    fitness_history = defaultdict(list)

    for gen, checkpoint in checkpoints:
        for name, data in get_base_locations(checkpoint).items():
            fitness = data.get('best_ever_fitness', 0) or 0
            baseline = data.get('baseline_ppl', 1)
            fitness_history[name].append({
                'generation': gen,
                'fitness': fitness,
                'baseline_ppl': baseline,
                'improvement_pct': fitness * 100
            })

    return dict(fitness_history)


def find_late_bloomers(fitness_history: dict, threshold_gen: int = 10, min_improvement: float = 0.3) -> list:
    """
    Find essays that started with low improvement but improved significantly later.

    Args:
        fitness_history: Dict of location -> list of {generation, fitness, ...}
        threshold_gen: Generation to use as "early" cutoff
        min_improvement: Minimum improvement delta to be considered a late bloomer
    """
    late_bloomers = []

    for loc, history in fitness_history.items():
        early = [h for h in history if h['generation'] <= threshold_gen]
        late = [h for h in history if h['generation'] > threshold_gen]

        if not early or not late:
            continue

        early_best = max(h['fitness'] for h in early)
        late_best = max(h['fitness'] for h in late)

        improvement = late_best - early_best

        if improvement >= min_improvement:
            late_bloomers.append({
                'location': loc,
                'early_fitness': early_best,
                'late_fitness': late_best,
                'improvement': improvement,
                'early_gen': max(h['generation'] for h in early),
                'late_gen': next(h['generation'] for h in late if h['fitness'] == late_best)
            })

    return sorted(late_bloomers, key=lambda x: -x['improvement'])


def find_bridge_essays(overlaps: dict, target_loc: str, top_n: int = 5) -> list:
    """Find essays that share the most values with a target essay."""
    bridges = []
    for (loc1, loc2), count in overlaps.items():
        if loc1 == target_loc:
            bridges.append((loc2, count))
        elif loc2 == target_loc:
            bridges.append((loc1, count))

    return sorted(bridges, key=lambda x: -x[1])[:top_n]


def generate_html_report(
    checkpoint_dir: str,
    output_path: str,
    final_checkpoint: dict,
    overlaps: dict,
    location_values: dict,
    total_shared: Counter,
    fitness_history: dict,
    late_bloomers: list
):
    """Generate an interactive HTML report with visualizations."""

    locations = list(get_base_locations(final_checkpoint).keys())

    # Prepare data for JavaScript
    nodes_data = []
    for loc in locations:
        loc_data = final_checkpoint['locations'][loc]
        nodes_data.append({
            'id': loc,
            'shared': total_shared[loc],
            'fitness': (loc_data.get('best_ever_fitness', 0) or 0) * 100,
            'baseline_ppl': loc_data.get('baseline_ppl', 1)
        })

    edges_data = []
    for (loc1, loc2), weight in overlaps.items():
        if weight > 0:
            edges_data.append({
                'source': loc1,
                'target': loc2,
                'weight': weight
            })

    # Find bridge essays for late bloomers
    late_bloomer_bridges = {}
    for lb in late_bloomers:
        bridges = find_bridge_essays(overlaps, lb['location'])
        late_bloomer_bridges[lb['location']] = bridges

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Essay Cluster Analysis</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1, h2, h3 {{
            color: #fff;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .section {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        #network {{
            width: 100%;
            height: 600px;
            background: #0f0f23;
            border-radius: 8px;
        }}
        .node {{
            cursor: pointer;
        }}
        .node text {{
            font-size: 10px;
            fill: #fff;
        }}
        .link {{
            stroke: #4a9eff;
            stroke-opacity: 0.3;
        }}
        .tooltip {{
            position: absolute;
            background: #222;
            border: 1px solid #444;
            padding: 10px;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
            z-index: 1000;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #0f3460;
            color: #fff;
        }}
        tr:hover {{
            background: #1f4068;
        }}
        .late-bloomer {{
            background: #2d4a22;
        }}
        .bridge-badge {{
            display: inline-block;
            background: #4a9eff;
            color: #fff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-right: 4px;
        }}
        .controls {{
            margin-bottom: 15px;
        }}
        .controls label {{
            margin-right: 15px;
        }}
        .controls input[type="range"] {{
            width: 200px;
            vertical-align: middle;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        .method-box {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
        }}
        .highlight {{
            color: #4a9eff;
        }}
        #fitness-chart {{
            width: 100%;
            height: 400px;
            background: #0f0f23;
            border-radius: 8px;
        }}
        .filter-btn {{
            background: #0f3460;
            color: #fff;
            border: 1px solid #333;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 8px;
        }}
        .filter-btn:hover {{
            background: #1f4068;
        }}
        .filter-btn.active {{
            background: #4a9eff;
            border-color: #4a9eff;
        }}
        .essay-link {{
            color: #4a9eff;
            text-decoration: none;
        }}
        .essay-link:hover {{
            text-decoration: underline;
        }}
        .bridge-badge .essay-link {{
            color: #fff;
        }}
        .bridge-badge .essay-link:hover {{
            color: #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Essay Cluster Analysis</h1>
        <p style="color: var(--text-secondary); margin-top: -10px; margin-bottom: 20px;">
            <a href="https://github.com/mjtomei/human" target="_blank" class="essay-link">GitHub Repository</a> &middot;
            <a href="http://mtomei.com/graph_search_report_gen49.html" class="essay-link">Dimension Analysis Report</a>
        </p>

        <div class="section">
            <h2>Network Visualization</h2>
            <div class="controls">
                <label>
                    Min edge weight: <span id="weight-display">5</span>
                    <input type="range" id="weight-slider" min="0" max="30" value="5">
                </label>
                <label>
                    Node size by:
                    <select id="size-metric">
                        <option value="shared">Shared values</option>
                        <option value="fitness">Fitness %</option>
                    </select>
                </label>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff6b6b;"></div>
                    <span>Core cluster (high connectivity)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ecdc4;"></div>
                    <span>Bridge essays</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffe66d;"></div>
                    <span>Outliers (low connectivity)</span>
                </div>
            </div>
            <div id="network"></div>
        </div>

        <div class="section">
            <h2>Late Bloomers & Bridge Essays</h2>
            <p>Essays that started with &lt;10% improvement but improved significantly after generation 10:</p>
            <table>
                <thead>
                    <tr>
                        <th>Essay</th>
                        <th>Early Fitness (gen &le;10)</th>
                        <th>Final Fitness</th>
                        <th>Improvement</th>
                        <th>Bridge Essays (shared values)</th>
                    </tr>
                </thead>
                <tbody id="late-bloomers-table">
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Methodology</h2>
            <div class="method-box">
<span class="highlight">1. Value Extraction:</span>
   For each essay (location), extract (dimension, value) pairs from the best hypothesis.
   Values are truncated to 100 chars for comparison.

<span class="highlight">2. Shared Value Computation:</span>
   For each pair of essays (A, B), count how many (dimension, value) pairs they share.
   shared(A, B) = |values(A) ∩ values(B)|

<span class="highlight">3. Total Connectivity:</span>
   For each essay, sum shared values across all other essays.
   connectivity(A) = Σ shared(A, X) for all X ≠ A

<span class="highlight">4. Clustering:</span>
   Essays with high mutual shared values form clusters.
   Core cluster: reason, memes, horus, violence, reproduction, tantra (40-45 shared pairs)
   Outliers: communes, water (0 shared with anyone)

<span class="highlight">5. Late Bloomer Detection:</span>
   Compare fitness at generation &le;10 vs fitness at generation &gt;10.
   Late bloomers improved by &ge;30% after early generations.

<span class="highlight">6. Bridge Essay Identification:</span>
   For late bloomers, find essays with highest shared value count.
   These bridges may have contributed successful dimension values via crossover.
            </div>
        </div>

        <div class="section">
            <h2>Fitness Over Time</h2>
            <div class="controls">
                <label>Show essays:</label>
                <button id="filter-all" class="filter-btn active" onclick="setFitnessFilter('all')">All Essays</button>
                <button id="filter-late" class="filter-btn" onclick="setFitnessFilter('late-bloomers')">Late Bloomers Only</button>
            </div>
            <div id="fitness-chart"></div>
        </div>

        <div class="section">
            <h2>Cluster Membership</h2>
            <table>
                <thead>
                    <tr>
                        <th>Essay</th>
                        <th>Total Shared</th>
                        <th>Final Fitness</th>
                        <th>Baseline PPL</th>
                        <th>Cluster</th>
                    </tr>
                </thead>
                <tbody id="cluster-table">
                </tbody>
            </table>
        </div>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script>
        // Deep copy nodes to avoid mutation issues
        const nodesData = {json.dumps(nodes_data)};
        const edges = {json.dumps(edges_data)};
        const fitnessHistory = {json.dumps(fitness_history)};
        const lateBloomerBridges = {json.dumps(late_bloomer_bridges)};
        const lateBloomerData = {json.dumps(late_bloomers)};

        // Create working copy of nodes with initial positions
        let nodes = nodesData.map((n, i) => ({{
            ...n,
            x: 400 + Math.cos(i * 2 * Math.PI / nodesData.length) * 200,
            y: 300 + Math.sin(i * 2 * Math.PI / nodesData.length) * 200
        }}));

        // Determine cluster membership
        function getCluster(shared) {{
            if (shared >= 400) return {{ name: 'Core', color: '#ff6b6b' }};
            if (shared >= 200) return {{ name: 'Bridge', color: '#4ecdc4' }};
            return {{ name: 'Outlier', color: '#ffe66d' }};
        }}

        // Helper to create essay link
        function essayLink(slug) {{
            if (slug.startsWith('meta_')) {{
                return slug;
            }}
            return `<a href="http://mtomei.com/${{slug}}.html" target="_blank" class="essay-link">${{slug}}</a>`;
        }}

        // Populate cluster table
        const clusterTable = document.getElementById('cluster-table');
        [...nodes].sort((a, b) => b.shared - a.shared).forEach(node => {{
            const cluster = getCluster(node.shared);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${{essayLink(node.id)}}</td>
                <td>${{node.shared}}</td>
                <td>${{node.fitness.toFixed(1)}}%</td>
                <td>${{node.baseline_ppl.toFixed(1)}}</td>
                <td><span style="color: ${{cluster.color}}">${{cluster.name}}</span></td>
            `;
            clusterTable.appendChild(row);
        }});

        // Populate late bloomers table
        const lbTable = document.getElementById('late-bloomers-table');
        lateBloomerData.forEach(lb => {{
            const bridges = lateBloomerBridges[lb.location] || [];
            const bridgeHtml = bridges.map(b =>
                `<span class="bridge-badge">${{essayLink(b[0])}} (${{b[1]}})</span>`
            ).join('');

            const row = document.createElement('tr');
            row.className = 'late-bloomer';
            row.innerHTML = `
                <td>${{essayLink(lb.location)}}</td>
                <td>${{(lb.early_fitness * 100).toFixed(1)}}%</td>
                <td>${{(lb.late_fitness * 100).toFixed(1)}}%</td>
                <td>+${{(lb.improvement * 100).toFixed(1)}}%</td>
                <td>${{bridgeHtml || 'None'}}</td>
            `;
            lbTable.appendChild(row);
        }});

        // Network visualization
        const width = document.getElementById('network').clientWidth || 800;
        const height = 600;
        const padding = 50;

        const svg = d3.select('#network')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const tooltip = d3.select('#tooltip');

        let simulation = null;
        let minWeight = 5;

        function updateNetwork() {{
            // Stop any existing simulation
            if (simulation) {{
                simulation.stop();
            }}

            svg.selectAll('*').remove();

            // Reset node positions to avoid NaN issues
            nodes.forEach((n, i) => {{
                if (isNaN(n.x) || isNaN(n.y) || n.x === undefined || n.y === undefined) {{
                    n.x = width/2 + Math.cos(i * 2 * Math.PI / nodes.length) * 200;
                    n.y = height/2 + Math.sin(i * 2 * Math.PI / nodes.length) * 200;
                }}
                // Clear any fixed positions
                n.fx = null;
                n.fy = null;
            }});

            // Create fresh edge copies to avoid D3 mutation
            const filteredEdges = edges
                .filter(e => e.weight >= minWeight)
                .map(e => ({{ source: e.source, target: e.target, weight: e.weight }}));

            const sizeMetric = document.getElementById('size-metric').value;
            const maxSize = Math.max(...nodes.map(n => sizeMetric === 'fitness' ? n.fitness : (n.shared || 1)));

            // Create simulation with boundary forces
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(filteredEdges).id(d => d.id).distance(80).strength(d => Math.min(d.weight / 30, 1)))
                .force('charge', d3.forceManyBody().strength(-150))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(35))
                .force('x', d3.forceX(width / 2).strength(0.05))
                .force('y', d3.forceY(height / 2).strength(0.05));

            const link = svg.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(filteredEdges)
                .enter().append('line')
                .attr('class', 'link')
                .attr('stroke-width', d => Math.sqrt(d.weight) / 2);

            const node = svg.append('g')
                .attr('class', 'nodes')
                .selectAll('g')
                .data(nodes)
                .enter().append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            // Click handler to open essay
            function openEssay(event, d) {{
                if (!d.id.startsWith('meta_')) {{
                    window.open(`http://mtomei.com/${{d.id}}.html`, '_blank');
                }}
            }}

            node.append('circle')
                .attr('r', d => {{
                    const val = sizeMetric === 'fitness' ? d.fitness : (d.shared || 1);
                    return 8 + (val / maxSize) * 17;
                }})
                .attr('fill', d => getCluster(d.shared).color)
                .attr('stroke', '#fff')
                .attr('stroke-width', 1.5)
                .style('cursor', d => d.id.startsWith('meta_') ? 'default' : 'pointer')
                .on('click', openEssay)
                .on('mouseover', (event, d) => {{
                    tooltip.style('display', 'block')
                        .html(`<strong>${{d.id}}</strong><br>
                               Shared: ${{d.shared}}<br>
                               Fitness: ${{d.fitness.toFixed(1)}}%<br>
                               Baseline PPL: ${{d.baseline_ppl.toFixed(1)}}<br>
                               Cluster: ${{getCluster(d.shared).name}}`);
                }})
                .on('mousemove', (event) => {{
                    tooltip.style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                }})
                .on('mouseout', () => tooltip.style('display', 'none'));

            node.append('text')
                .attr('dx', 12)
                .attr('dy', 4)
                .style('cursor', d => d.id.startsWith('meta_') ? 'default' : 'pointer')
                .on('click', openEssay)
                .text(d => d.id);

            simulation.on('tick', () => {{
                // Constrain nodes to stay within bounds
                nodes.forEach(d => {{
                    d.x = Math.max(padding, Math.min(width - padding, d.x));
                    d.y = Math.max(padding, Math.min(height - padding, d.y));
                }});

                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
            }});
        }}

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = Math.max(padding, Math.min(width - padding, event.x));
            d.fy = Math.max(padding, Math.min(height - padding, event.y));
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        document.getElementById('weight-slider').addEventListener('input', (e) => {{
            minWeight = parseInt(e.target.value);
            document.getElementById('weight-display').textContent = minWeight;
            updateNetwork();
        }});

        document.getElementById('size-metric').addEventListener('change', updateNetwork);

        updateNetwork();

        // Fitness over time chart
        const chartWidth = document.getElementById('fitness-chart').clientWidth || 800;
        const chartHeight = 400;
        const margin = {{ top: 20, right: 150, bottom: 40, left: 60 }};

        const chartSvg = d3.select('#fitness-chart')
            .append('svg')
            .attr('width', chartWidth)
            .attr('height', chartHeight);

        // Debug: log which essays are in fitnessHistory
        console.log('Essays in fitnessHistory:', Object.keys(fitnessHistory));
        console.log('Total essays:', Object.keys(fitnessHistory).length);

        const allGens = new Set();
        Object.values(fitnessHistory).forEach(hist => {{
            hist.forEach(h => allGens.add(h.generation));
        }});
        const generations = Array.from(allGens).sort((a, b) => a - b);

        const xScale = d3.scaleLinear()
            .domain([0, Math.max(...generations)])
            .range([margin.left, chartWidth - margin.right]);

        // Dynamic Y scale based on actual data (with minimum of 0, max of actual max + 5%)
        const allFitness = Object.values(fitnessHistory).flatMap(h => h.map(d => d.improvement_pct));
        const minFitness = Math.min(0, Math.min(...allFitness));
        const maxFitness = Math.max(...allFitness) + 5;

        const yScale = d3.scaleLinear()
            .domain([minFitness, maxFitness])
            .range([chartHeight - margin.bottom, margin.top]);

        // Static elements (axes)
        chartSvg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${{chartHeight - margin.bottom}})`)
            .call(d3.axisBottom(xScale).ticks(10))
            .attr('color', '#888');

        chartSvg.append('g')
            .attr('class', 'y-axis')
            .attr('transform', `translate(${{margin.left}},0)`)
            .call(d3.axisLeft(yScale).ticks(10).tickFormat(d => d + '%'))
            .attr('color', '#888');

        chartSvg.append('text')
            .attr('x', chartWidth / 2)
            .attr('y', chartHeight - 5)
            .attr('fill', '#888')
            .attr('text-anchor', 'middle')
            .text('Generation');

        chartSvg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -chartHeight / 2)
            .attr('y', 15)
            .attr('fill', '#888')
            .attr('text-anchor', 'middle')
            .text('Fitness (% improvement)');

        // Vertical line at gen 10
        chartSvg.append('line')
            .attr('x1', xScale(10))
            .attr('x2', xScale(10))
            .attr('y1', margin.top)
            .attr('y2', chartHeight - margin.bottom)
            .attr('stroke', '#666')
            .attr('stroke-dasharray', '5,5');

        chartSvg.append('text')
            .attr('x', xScale(10) + 5)
            .attr('y', margin.top + 10)
            .attr('fill', '#666')
            .attr('font-size', '10px')
            .text('Gen 10 (early/late threshold)');

        // Container for lines
        const linesGroup = chartSvg.append('g').attr('class', 'lines-group');
        const labelsGroup = chartSvg.append('g').attr('class', 'labels-group');

        const line = d3.line()
            .x(d => xScale(d.generation))
            .y(d => yScale(d.improvement_pct))
            .defined(d => !isNaN(d.improvement_pct))
            .curve(d3.curveMonotoneX);

        const colorScale = d3.scaleOrdinal(d3.schemeTableau10);
        const lateBloomerLocs = new Set(lateBloomerData.map(lb => lb.location));

        let currentFilter = 'all';

        function renderFitnessChart() {{
            linesGroup.selectAll('*').remove();
            labelsGroup.selectAll('*').remove();

            const entries = Object.entries(fitnessHistory);

            entries.forEach(([loc, hist], i) => {{
                if (hist.length === 0) return;

                const isLateBloomer = lateBloomerLocs.has(loc);
                let opacity, strokeWidth, showLabel;

                if (currentFilter === 'late-bloomers') {{
                    if (!isLateBloomer) return;
                    opacity = 1;
                    strokeWidth = 2.5;
                    showLabel = true;
                }} else {{
                    opacity = isLateBloomer ? 1 : 0.3;
                    strokeWidth = isLateBloomer ? 2.5 : 1;
                    showLabel = isLateBloomer;
                }}

                linesGroup.append('path')
                    .datum(hist)
                    .attr('fill', 'none')
                    .attr('stroke', colorScale(loc))
                    .attr('stroke-width', strokeWidth)
                    .attr('stroke-opacity', opacity)
                    .attr('d', line)
                    .attr('class', `line-${{loc}}`)
                    .on('mouseover', function() {{
                        d3.select(this).attr('stroke-width', 4).attr('stroke-opacity', 1);
                        tooltip.style('display', 'block').html(`<strong>${{loc}}</strong>`);
                    }})
                    .on('mousemove', (event) => {{
                        tooltip.style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 10) + 'px');
                    }})
                    .on('mouseout', function() {{
                        d3.select(this).attr('stroke-width', strokeWidth).attr('stroke-opacity', opacity);
                        tooltip.style('display', 'none');
                    }});

                // Label at end
                const lastPoint = hist[hist.length - 1];
                if (showLabel && lastPoint) {{
                    labelsGroup.append('text')
                        .attr('x', xScale(lastPoint.generation) + 5)
                        .attr('y', yScale(lastPoint.improvement_pct))
                        .attr('fill', colorScale(loc))
                        .attr('font-size', '11px')
                        .attr('dominant-baseline', 'middle')
                        .text(loc);
                }}
            }});
        }}

        // Initialize chart
        renderFitnessChart();

        // Fitness filter handler
        function setFitnessFilter(filter) {{
            currentFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(filter === 'all' ? 'filter-all' : 'filter-late').classList.add('active');
            renderFitnessChart();
        }}
        // Make it globally accessible
        window.setFitnessFilter = setFitnessFilter;
    </script>
    <footer style="margin-top: 40px; padding: 20px; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 0.85em;">
        <p><strong>Note:</strong> The "water" essay has an artificially low baseline perplexity (1.13) due to a truncation bug where essays exceeding 2048 tokens have their context length computed without truncation. This causes perplexity to be measured on context tokens rather than target tokens. The bug affects only "water" (the longest essay at ~3700 tokens). Other essays' results are unaffected.</p>
    </footer>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    return html


def main():
    parser = argparse.ArgumentParser(description='Analyze essay clusters from graph search')
    parser.add_argument('checkpoint_dir', help='Directory containing checkpoint files')
    parser.add_argument('-o', '--output', default='results/cluster_analysis.html',
                        help='Output HTML file path')
    parser.add_argument('--print-summary', action='store_true',
                        help='Print text summary to stdout')
    args = parser.parse_args()

    print(f"Loading checkpoints from {args.checkpoint_dir}...")
    checkpoints = load_all_checkpoints(args.checkpoint_dir)
    print(f"  Loaded {len(checkpoints)} checkpoints")

    # Use final checkpoint for clustering
    final_gen, final_checkpoint = checkpoints[-1]
    print(f"  Final checkpoint: generation {final_gen}")

    # Compute shared values
    print("Computing shared values...")
    overlaps, location_values = compute_shared_values(final_checkpoint)
    total_shared = compute_total_shared(location_values)

    # Track fitness over time
    print("Tracking fitness over time...")
    fitness_history = track_fitness_over_time(checkpoints)

    # Find late bloomers
    print("Finding late bloomers...")
    late_bloomers = find_late_bloomers(fitness_history, threshold_gen=10, min_improvement=0.10)

    if args.print_summary:
        print("\n" + "=" * 80)
        print("CLUSTER ANALYSIS SUMMARY")
        print("=" * 80)

        print("\nTop connected essays (core cluster):")
        for loc, count in total_shared.most_common(10):
            print(f"  {loc}: {count} shared values")

        print("\nOutlier essays (few shared values):")
        for loc, count in total_shared.most_common()[-5:]:
            print(f"  {loc}: {count} shared values")

        print("\nLate bloomers (improved >10% after gen 10):")
        for lb in late_bloomers:
            bridges = find_bridge_essays(overlaps, lb['location'])
            bridge_str = ', '.join(f"{b[0]}({b[1]})" for b in bridges[:3])
            print(f"  {lb['location']}: {lb['early_fitness']*100:.1f}% -> {lb['late_fitness']*100:.1f}% (+{lb['improvement']*100:.1f}%)")
            print(f"    Bridges: {bridge_str}")

    # Generate HTML report
    print(f"\nGenerating HTML report at {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    generate_html_report(
        args.checkpoint_dir,
        args.output,
        final_checkpoint,
        overlaps,
        location_values,
        total_shared,
        fitness_history,
        late_bloomers
    )

    print("Done!")
    return late_bloomers


if __name__ == '__main__':
    main()
