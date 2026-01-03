#!/usr/bin/env python3
"""
Run shadow dimension experiments:
1. V1.1: Fixed 28 dimensions (20 original + 8 dark/shadow)
2. Meta with grace period: Sparse dimensions with minimum refinement time

Both include the new shadow/dark dimension pool.
"""

import json
import sys
import random
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '.')

from cognitive_gen.meta_search import (
    MetaSearch, MetaScorer, MetaHypothesisGenerator,
    SparseHypothesis, StagnationDetector,
    DIMENSION_GROUPS, DIMENSION_TO_GROUP, ALL_DIMENSIONS,
)
from cognitive_gen.dimension_pool import (
    V1_1_DIMENSIONS, V1_1_LINKAGE_GROUPS, V1_1_DIMENSION_TO_GROUP,
    DIMENSION_POOL,
)
from cognitive_gen.reverse_inference import PUBLIC_FIGURES


class MetaHypothesisGeneratorV1_1(MetaHypothesisGenerator):
    """Generator that uses V1.1 dimensions (including shadow/dark)."""

    def generate_v1_seeded(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
    ) -> list[SparseHypothesis]:
        """Generate hypotheses using V1.1 dimensions (28 total including shadow)."""
        import json

        samples_text = "\n\n---\n\n".join(writing_samples)

        prompts = []
        for i in range(n_hypotheses):
            prompt = f"""Analyze these writing samples and generate a hypothesis about the writer's cognitive/psychological state.

Writing samples:
{samples_text}

Generate cognitive state hypothesis #{i+1}. Provide values for 8-15 of these dimensions (leave others blank).
Be HONEST about the dark aspects - we're looking for psychological truth, not flattery.

EMBODIED:
- body_state: Physical sensation
- preverbal_feeling: Feeling before words

INTELLECTUAL:
- core_belief: Fundamental truth they hold
- intellectual_stance: How they approach ideas
- what_they_notice: What draws their attention

MORAL:
- moral_framework: How they judge right/wrong
- what_outrages_them: What they can't tolerate
- what_they_protect: What they feel responsible for

RELATIONAL:
- stance_toward_reader: How they position vis-a-vis audience
- who_they_write_for: Real or imagined audience
- what_they_want_reader_to_feel: Intended effect

TEMPORAL:
- relationship_to_past: How history weighs on them
- relationship_to_future: Hope, dread, indifference
- sense_of_urgency: Time pressure

AESTHETIC:
- what_they_find_beautiful: Aesthetic values
- what_they_find_ugly: Aesthetic aversions
- relationship_to_language: Tool, art, weapon, window

HIDDEN:
- what_they_cant_say_directly: The unspeakable
- the_wound: The injury that shapes the voice
- the_compensation: How they cope with the wound

DARK/SHADOW (be honest here):
- the_cruelty_they_enjoy: What harm or dominance they secretly enjoy
- who_they_resent: People or types they harbor resentment toward
- the_lie_they_tell_themselves: Core self-deception protecting their self-image
- their_capacity_for_violence: What violence (physical, emotional, social) they're capable of
- what_they_refuse_to_forgive: Grudges, unforgiven wounds
- their_contempt: Who or what they look down on
- the_power_they_crave: Hidden desires for control or dominance
- what_they_would_do_if_no_one_knew: Actions they'd take without consequence

Return as JSON object. Be specific, grounded in the text, and HONEST about darker aspects.
Try a DIFFERENT angle than other hypotheses - focus on different aspects."""
            prompts.append(prompt)

        print(f"  Generating {n_hypotheses} V1.1 hypotheses in batch...")
        if self.use_local:
            responses = self.local_llm.generate_batch(prompts, max_new_tokens=1200)
        else:
            responses = [self._generate(p, 1200) for p in prompts]

        results = []
        for text in responses:
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    h = json.loads(text[start:end])
                    filtered = {d: h.get(d) for d in V1_1_DIMENSIONS if h.get(d)}
                    if filtered:
                        results.append(SparseHypothesis(values=filtered))
            except Exception as e:
                pass

        print(f"  Successfully parsed {len(results)} hypotheses")
        return results

GENERATIONS = 25
POPULATION_SIZE = 40  # Larger population for better exploration


class MetaSearchWithGrace(MetaSearch):
    """
    Meta search with grace period for dimensions.

    Dimensions must exist for `grace_period` generations before they can be removed.
    This ensures dimensions are properly refined before being judged.
    """

    def __init__(
        self,
        *args,
        grace_period: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.grace_period = grace_period
        # Track dimension ages: {hypothesis_id: {dim: age}}
        self.dimension_ages = defaultdict(lambda: defaultdict(int))
        self.hypothesis_counter = 0

    def _get_hypothesis_id(self, hypothesis: SparseHypothesis) -> int:
        """Get or assign an ID to a hypothesis for tracking."""
        if not hasattr(hypothesis, '_grace_id'):
            hypothesis._grace_id = self.hypothesis_counter
            self.hypothesis_counter += 1
        return hypothesis._grace_id

    def _inherit_ages(self, child: SparseHypothesis, parent: SparseHypothesis):
        """Inherit dimension ages from parent to child."""
        child_id = self._get_hypothesis_id(child)
        parent_id = self._get_hypothesis_id(parent)

        for dim in child.active_dimensions:
            if dim in self.dimension_ages[parent_id]:
                # Inherit age from parent
                self.dimension_ages[child_id][dim] = self.dimension_ages[parent_id][dim]
            else:
                # New dimension starts at age 0
                self.dimension_ages[child_id][dim] = 0

    def _increment_ages(self, population: list[SparseHypothesis]):
        """Increment ages for all dimensions in all hypotheses."""
        for hyp in population:
            hyp_id = self._get_hypothesis_id(hyp)
            for dim in hyp.active_dimensions:
                self.dimension_ages[hyp_id][dim] += 1

    def mutate(
        self,
        hypothesis: SparseHypothesis,
        writing_samples: list[str],
        stagnation_severity: float,
    ) -> SparseHypothesis:
        """Apply mutations with grace period protection for dimension removal."""

        result = hypothesis.copy()
        hyp_id = self._get_hypothesis_id(hypothesis)
        result_id = self._get_hypothesis_id(result)

        # Copy ages to new hypothesis
        self.dimension_ages[result_id] = self.dimension_ages[hyp_id].copy()

        # Adaptive mutation rate
        mutation_rate = self.base_mutation_rate + (0.5 - self.base_mutation_rate) * stagnation_severity

        # Dimension addition (only if not in v1 mode)
        if not self.use_v1_mode and random.random() < self.add_dim_rate and result.n_active < self.max_dims:
            inactive = [d for d in self.allowed_dims if d not in result.active_dimensions]
            if inactive:
                new_dim = random.choice(inactive)
                value = self.generator.generate_dimension_value(new_dim, writing_samples)
                if value:
                    result.values[new_dim] = value
                    self.dimension_ages[result_id][new_dim] = 0  # New dimension starts at age 0

        # Dimension removal WITH GRACE PERIOD
        if not self.use_v1_mode and random.random() < self.remove_dim_rate and result.n_active > self.min_dims:
            # Only remove dimensions that have existed for grace_period generations
            removable = [
                dim for dim in result.active_dimensions
                if self.dimension_ages[result_id].get(dim, 0) >= self.grace_period
            ]
            if removable:
                dim_to_remove = random.choice(removable)
                result.values[dim_to_remove] = None
                del self.dimension_ages[result_id][dim_to_remove]

        # Value mutation (same as parent class)
        if random.random() < mutation_rate and result.active_dimensions:
            if stagnation_severity > 0.7:
                weights = {'vary': 0.1, 'intensify': 0.1, 'soften': 0.1,
                          'opposite': 0.4, 'linked': 0.2, 'new': 0.1}
            elif stagnation_severity > 0.3:
                weights = {'vary': 0.3, 'intensify': 0.15, 'soften': 0.15,
                          'opposite': 0.2, 'linked': 0.15, 'new': 0.05}
            else:
                weights = {'vary': 0.4, 'intensify': 0.2, 'soften': 0.2,
                          'opposite': 0.1, 'linked': 0.1, 'new': 0.0}

            mutation_type = random.choices(
                list(weights.keys()),
                weights=list(weights.values())
            )[0]

            if mutation_type == 'linked':
                active_groups = defaultdict(list)
                for dim in result.active_dimensions:
                    group = self.dim_to_group.get(dim, "other")
                    active_groups[group].append(dim)

                if active_groups:
                    group_name = random.choice(list(active_groups.keys()))
                    group_dims = active_groups[group_name]
                    result = self.generator.mutator.mutate_linked_group(
                        result, group_dims, writing_samples
                    )
            elif mutation_type == 'new':
                dim = random.choice(result.active_dimensions)
                new_value = self.generator.generate_dimension_value(dim, writing_samples)
                if new_value:
                    result.values[dim] = new_value
            else:
                dim = random.choice(result.active_dimensions)
                old_value = result.values[dim]
                try:
                    mutator = self.generator.mutator
                    new_value = getattr(mutator, mutation_type)(dim, old_value)
                    result.values[dim] = new_value
                except Exception as e:
                    print(f"Mutation error: {e}")

        return result

    def search(self, writing_samples: list[str], generations: int = 25, verbose: bool = True):
        """Override search to track dimension ages across generations."""
        # Call parent search but intercept to track ages
        # For simplicity, we'll just call parent and let the mutate override handle it
        return super().search(writing_samples, generations, verbose)


def run_v1_1_experiment(samples: list[str]) -> dict:
    """Run V1.1 experiment with shadow dimensions."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: V1.1 MODE (28 fixed dimensions including shadow)")
    print("="*70 + "\n")

    # Use v1 format scorer for consistency
    scorer = MetaScorer(model_name='mistralai/Mistral-7B-v0.3', use_v1_format=True)
    generator = MetaHypothesisGeneratorV1_1(use_local=True)  # Use V1.1 generator

    # Create search with V1.1 dimensions
    search = MetaSearch(
        scorer=scorer,
        generator=generator,
        population_size=POPULATION_SIZE,
        use_v1_mode=True,  # Fixed dimensions mode
    )

    # Override to use V1.1 dimensions and groups
    search.linkage_groups = list(V1_1_LINKAGE_GROUPS.values())
    search.dim_to_group = V1_1_DIMENSION_TO_GROUP
    search.allowed_dims = set(V1_1_DIMENSIONS)

    result = search.search(
        writing_samples=samples,
        generations=GENERATIONS,
    )

    return {
        'experiment': 'v1.1',
        'baseline': result.baseline_perplexity,
        'best': result.best_perplexity,
        'improvement': result.improvement,
        'n_dimensions': len(V1_1_DIMENSIONS),
        'best_hypothesis': result.best_hypothesis.to_dict() if hasattr(result.best_hypothesis, 'to_dict') else dict(result.best_hypothesis.values),
    }


def run_meta_grace_experiment(samples: list[str]) -> dict:
    """Run meta search with grace period and shadow dimensions."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: META SEARCH WITH GRACE PERIOD (sparse + shadow)")
    print("="*70 + "\n")

    # Use default format scorer
    scorer = MetaScorer(model_name='mistralai/Mistral-7B-v0.3', use_v1_format=False)
    generator = MetaHypothesisGenerator(use_local=True)

    # Create search with grace period
    search = MetaSearchWithGrace(
        scorer=scorer,
        generator=generator,
        population_size=POPULATION_SIZE,
        use_v1_mode=False,
        grace_period=5,  # Dimensions must exist for 5 generations before removal
        min_dims=5,
        max_dims=35,
    )

    result = search.search(
        writing_samples=samples,
        generations=GENERATIONS,
    )

    return {
        'experiment': 'meta_grace',
        'baseline': result.baseline_perplexity,
        'best': result.best_perplexity,
        'improvement': result.improvement,
        'grace_period': 5,
        'n_active_dims': result.best_hypothesis.n_active,
        'best_hypothesis': result.best_hypothesis.to_dict() if hasattr(result.best_hypothesis, 'to_dict') else dict(result.best_hypothesis.values),
    }


def main():
    """Run both shadow experiments."""

    # Get mtomei samples
    samples = PUBLIC_FIGURES['mtomei']['samples']
    print(f"Mat Tomei samples: {len(samples)}")
    print(f"Generations: {GENERATIONS}, Population: {POPULATION_SIZE}")

    results = {}

    # Run V1.1 experiment
    results['v1_1'] = run_v1_1_experiment(samples)
    print("\n" + "="*70)
    print("V1.1 RESULTS")
    print(f"Baseline: {results['v1_1']['baseline']:.3f}")
    print(f"Best: {results['v1_1']['best']:.3f}")
    print(f"Improvement: {results['v1_1']['improvement']*100:.1f}%")
    print(f"Dimensions: {results['v1_1']['n_dimensions']}")
    print("="*70)

    # Run meta grace experiment
    results['meta_grace'] = run_meta_grace_experiment(samples)
    print("\n" + "="*70)
    print("META GRACE RESULTS")
    print(f"Baseline: {results['meta_grace']['baseline']:.3f}")
    print(f"Best: {results['meta_grace']['best']:.3f}")
    print(f"Improvement: {results['meta_grace']['improvement']*100:.1f}%")
    print(f"Active dimensions: {results['meta_grace']['n_active_dims']}")
    print("="*70)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SHADOW DIMENSION EXPERIMENTS")
    print("="*70)
    print(f"V1.1 (28 fixed):    Baseline {results['v1_1']['baseline']:.3f} -> Best {results['v1_1']['best']:.3f} ({results['v1_1']['improvement']*100:.1f}%)")
    print(f"Meta Grace (sparse): Baseline {results['meta_grace']['baseline']:.3f} -> Best {results['meta_grace']['best']:.3f} ({results['meta_grace']['improvement']*100:.1f}%)")
    print("="*70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"results/shadow_experiments_{timestamp}.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {outfile}")

    return results


if __name__ == "__main__":
    main()
