"""Experiment runner for comparing baseline vs cognitive-conditioned generation."""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from scipy import stats

from .context import CognitiveContext
from .generator import CognitiveGenerator, GenerationResult
from .detector import AIDetector


@dataclass
class SampleResult:
    """Result for a single generated sample."""

    sample_id: int
    text_type: str
    mode: str  # "baseline" or "cognitive"
    surface_text: str
    inner_monologue: Optional[str]
    ai_probability: float
    human_probability: float
    detection_label: str
    word_count: int
    context_summary: Optional[str] = None


@dataclass
class ExperimentResults:
    """Aggregated results from an experiment run."""

    timestamp: str
    samples_per_condition: int
    text_types: list[str]
    total_samples: int

    # Per-condition statistics
    baseline_mean_ai_prob: float
    baseline_std_ai_prob: float
    cognitive_mean_ai_prob: float
    cognitive_std_ai_prob: float

    # Statistical test
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d

    # Per text-type breakdown
    by_text_type: dict

    # All individual samples
    samples: list[SampleResult]


def run_experiment(
    samples_per_condition: int = 5,
    text_types: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    temperature: float = 1.0,
    verbose: bool = True,
) -> ExperimentResults:
    """
    Run the cognitive context experiment.

    Generates text samples with and without cognitive conditioning,
    runs them through an AI detector, and performs statistical analysis.

    Args:
        samples_per_condition: Number of samples per text type per condition
        text_types: List of text types to test (default: all four)
        output_dir: Directory to save results (default: ./results)
        temperature: Sampling temperature for generation
        verbose: Whether to print progress

    Returns:
        ExperimentResults with all data and analysis
    """
    if text_types is None:
        text_types = ["personal_essay", "creative_fiction", "email", "message"]

    if output_dir is None:
        output_dir = "./results"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("COGNITIVE CONTEXT TEXT GENERATION EXPERIMENT")
        print("=" * 60)
        print(f"Samples per condition: {samples_per_condition}")
        print(f"Text types: {text_types}")
        print(f"Total samples: {samples_per_condition * len(text_types) * 2}")
        print()

    # Initialize components
    if verbose:
        print("Initializing generator...")
    generator = CognitiveGenerator()

    if verbose:
        print("Initializing detector...")
    detector = AIDetector()

    all_samples: list[SampleResult] = []
    sample_id = 0

    # Generate samples for each text type
    for text_type in text_types:
        if verbose:
            print(f"\n--- {text_type.upper()} ---")

        # Baseline samples
        if verbose:
            print(f"Generating {samples_per_condition} baseline samples...")

        for i in range(samples_per_condition):
            result = generator.generate_baseline(text_type, temperature=temperature)
            detection = detector.detect(result.surface_text)

            sample = SampleResult(
                sample_id=sample_id,
                text_type=text_type,
                mode="baseline",
                surface_text=result.surface_text,
                inner_monologue=None,
                ai_probability=detection.ai_probability,
                human_probability=detection.human_probability,
                detection_label=detection.label,
                word_count=len(result.surface_text.split()),
            )
            all_samples.append(sample)
            sample_id += 1

            if verbose:
                print(
                    f"  Baseline {i + 1}: AI prob = {detection.ai_probability:.3f}"
                )

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        # Cognitive samples
        if verbose:
            print(f"Generating {samples_per_condition} cognitive samples...")

        for i in range(samples_per_condition):
            result = generator.generate_with_context(
                text_type, temperature=temperature, include_inner_monologue=True
            )
            detection = detector.detect(result.surface_text)

            # Summarize context for logging
            context_summary = None
            if result.context:
                context_summary = f"Goals: {result.context.explicit_goals[:2]}, State: {result.context.emotional_state}"

            sample = SampleResult(
                sample_id=sample_id,
                text_type=text_type,
                mode="cognitive",
                surface_text=result.surface_text,
                inner_monologue=result.inner_monologue,
                ai_probability=detection.ai_probability,
                human_probability=detection.human_probability,
                detection_label=detection.label,
                word_count=len(result.surface_text.split()),
                context_summary=context_summary,
            )
            all_samples.append(sample)
            sample_id += 1

            if verbose:
                print(
                    f"  Cognitive {i + 1}: AI prob = {detection.ai_probability:.3f}"
                )

            # Small delay to avoid rate limiting
            time.sleep(0.5)

    # Analyze results
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

    baseline_probs = [s.ai_probability for s in all_samples if s.mode == "baseline"]
    cognitive_probs = [s.ai_probability for s in all_samples if s.mode == "cognitive"]

    baseline_mean = sum(baseline_probs) / len(baseline_probs)
    baseline_std = (
        sum((p - baseline_mean) ** 2 for p in baseline_probs) / len(baseline_probs)
    ) ** 0.5

    cognitive_mean = sum(cognitive_probs) / len(cognitive_probs)
    cognitive_std = (
        sum((p - cognitive_mean) ** 2 for p in cognitive_probs) / len(cognitive_probs)
    ) ** 0.5

    # T-test
    t_stat, p_value = stats.ttest_ind(baseline_probs, cognitive_probs)

    # Cohen's d effect size
    pooled_std = (
        (baseline_std**2 + cognitive_std**2) / 2
    ) ** 0.5
    effect_size = (baseline_mean - cognitive_mean) / pooled_std if pooled_std > 0 else 0

    if verbose:
        print(f"\nBaseline AI probability:  {baseline_mean:.3f} (+/- {baseline_std:.3f})")
        print(f"Cognitive AI probability: {cognitive_mean:.3f} (+/- {cognitive_std:.3f})")
        print(f"\nDifference: {baseline_mean - cognitive_mean:.3f}")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Effect size (Cohen's d): {effect_size:.3f}")

        if p_value < 0.05:
            if cognitive_mean < baseline_mean:
                print("\n>>> SIGNIFICANT: Cognitive conditioning REDUCES AI detection")
            else:
                print("\n>>> SIGNIFICANT: Cognitive conditioning INCREASES AI detection")
        else:
            print("\n>>> NOT SIGNIFICANT: No reliable difference detected")

    # Per text-type breakdown
    by_text_type = {}
    for text_type in text_types:
        type_baseline = [
            s.ai_probability
            for s in all_samples
            if s.mode == "baseline" and s.text_type == text_type
        ]
        type_cognitive = [
            s.ai_probability
            for s in all_samples
            if s.mode == "cognitive" and s.text_type == text_type
        ]

        by_text_type[text_type] = {
            "baseline_mean": sum(type_baseline) / len(type_baseline),
            "cognitive_mean": sum(type_cognitive) / len(type_cognitive),
            "difference": sum(type_baseline) / len(type_baseline)
            - sum(type_cognitive) / len(type_cognitive),
        }

        if verbose:
            print(f"\n{text_type}:")
            print(f"  Baseline: {by_text_type[text_type]['baseline_mean']:.3f}")
            print(f"  Cognitive: {by_text_type[text_type]['cognitive_mean']:.3f}")
            print(f"  Diff: {by_text_type[text_type]['difference']:.3f}")

    # Create results object
    results = ExperimentResults(
        timestamp=datetime.now().isoformat(),
        samples_per_condition=samples_per_condition,
        text_types=text_types,
        total_samples=len(all_samples),
        baseline_mean_ai_prob=baseline_mean,
        baseline_std_ai_prob=baseline_std,
        cognitive_mean_ai_prob=cognitive_mean,
        cognitive_std_ai_prob=cognitive_std,
        t_statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        by_text_type=by_text_type,
        samples=[],  # Will add serializable version
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"experiment_{timestamp}.json"

    # Convert to serializable format
    results_dict = {
        "timestamp": results.timestamp,
        "samples_per_condition": results.samples_per_condition,
        "text_types": results.text_types,
        "total_samples": results.total_samples,
        "baseline_mean_ai_prob": results.baseline_mean_ai_prob,
        "baseline_std_ai_prob": results.baseline_std_ai_prob,
        "cognitive_mean_ai_prob": results.cognitive_mean_ai_prob,
        "cognitive_std_ai_prob": results.cognitive_std_ai_prob,
        "t_statistic": results.t_statistic,
        "p_value": results.p_value,
        "effect_size": results.effect_size,
        "by_text_type": results.by_text_type,
        "samples": [asdict(s) for s in all_samples],
    }

    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {results_file}")

    # Store samples in results
    results.samples = all_samples

    return results


def print_sample_comparison(results: ExperimentResults, text_type: str = "personal_essay"):
    """Print example samples side by side for qualitative comparison."""
    baseline_samples = [
        s for s in results.samples if s.mode == "baseline" and s.text_type == text_type
    ]
    cognitive_samples = [
        s for s in results.samples if s.mode == "cognitive" and s.text_type == text_type
    ]

    if baseline_samples and cognitive_samples:
        print(f"\n{'=' * 60}")
        print(f"SAMPLE COMPARISON: {text_type}")
        print("=" * 60)

        print("\n--- BASELINE SAMPLE ---")
        print(f"AI Probability: {baseline_samples[0].ai_probability:.3f}")
        print(f"\n{baseline_samples[0].surface_text}")

        print("\n--- COGNITIVE SAMPLE ---")
        print(f"AI Probability: {cognitive_samples[0].ai_probability:.3f}")
        if cognitive_samples[0].context_summary:
            print(f"Context: {cognitive_samples[0].context_summary}")
        print(f"\n{cognitive_samples[0].surface_text}")

        if cognitive_samples[0].inner_monologue:
            print("\n--- INNER MONOLOGUE (not shown in output) ---")
            print(cognitive_samples[0].inner_monologue[:500] + "...")
