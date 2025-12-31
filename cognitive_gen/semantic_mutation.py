"""
Semantic mutation operators using Claude.

Instead of random value replacement, generate meaningful variations
of dimension values using language understanding.
"""

import json
import random
from typing import Optional
import anthropic

from cognitive_gen.cognitive_state import CognitiveState, get_state_class


class SemanticMutator:
    """Generate semantic mutations of cognitive state dimensions."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def vary(self, dimension: str, value: str) -> str:
        """Generate a slight variation of a value."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""For the cognitive dimension "{dimension}", generate a slight variation of:
"{value}"

Return ONLY the new value, nothing else. Keep similar meaning but use different words or angle."""
            }]
        )
        return response.content[0].text.strip().strip('"')

    def intensify(self, dimension: str, value: str) -> str:
        """Make a value more extreme."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""For the cognitive dimension "{dimension}", make this more extreme/intense:
"{value}"

Return ONLY the intensified version, nothing else."""
            }]
        )
        return response.content[0].text.strip().strip('"')

    def soften(self, dimension: str, value: str) -> str:
        """Make a value more subtle."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""For the cognitive dimension "{dimension}", make this more subtle/muted:
"{value}"

Return ONLY the softened version, nothing else."""
            }]
        )
        return response.content[0].text.strip().strip('"')

    def opposite(self, dimension: str, value: str) -> str:
        """Generate the psychological opposite."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""For the cognitive dimension "{dimension}", generate the psychological opposite of:
"{value}"

Return ONLY the opposite value, nothing else."""
            }]
        )
        return response.content[0].text.strip().strip('"')

    def blend(self, dimension: str, value_a: str, value_b: str) -> str:
        """Combine two values into a synthesis."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""For the cognitive dimension "{dimension}", blend these two psychological states into a coherent synthesis:
A: "{value_a}"
B: "{value_b}"

Return ONLY the blended value, nothing else."""
            }]
        )
        return response.content[0].text.strip().strip('"')

    def mutate_linked_group(
        self,
        hypothesis: dict,
        group: list[str],
        writing_samples: list[str],
    ) -> dict:
        """Mutate linked dimensions together to maintain coherence."""
        current_values = {d: hypothesis.get(d) for d in group if hypothesis.get(d)}

        if not current_values:
            return hypothesis

        samples_text = "\n---\n".join(writing_samples[:2])  # Use first 2 samples

        prompt = f"""Given these linked cognitive dimensions for a writer:

{chr(10).join(f'{d}: {v}' for d, v in current_values.items())}

And samples of their writing:
{samples_text}

Generate a coherent variation that keeps the psychological structure but shifts the specifics.
These dimensions are linked - they should remain internally consistent.

Return as JSON object with the same keys. Return ONLY the JSON, nothing else."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Parse JSON
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                new_values = json.loads(text[start:end])
            else:
                return hypothesis
        except json.JSONDecodeError:
            return hypothesis

        mutated = hypothesis.copy()
        for d, v in new_values.items():
            if d in group:
                mutated[d] = v

        return mutated


class HypothesisGenerator:
    """Generate cognitive hypotheses from writing samples."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", state_version: str = "v1"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.state_class = get_state_class(state_version)
        self.state_version = state_version

    def generate_initial_hypotheses(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
    ) -> list:
        """Generate initial hypotheses by analyzing the writing."""

        # Generate in batches to avoid overwhelming the model
        batch_size = min(5, n_hypotheses)  # Max 5 at a time
        all_hypotheses = []

        samples_text = "\n\n---\n\n".join(writing_samples)
        dimensions = self.state_class.get_dimensions()

        # Only show a subset of dimensions in prompt to reduce complexity
        key_dimensions = dimensions[:15] if len(dimensions) > 15 else dimensions
        dim_list = "\n".join(f"- {d}" for d in key_dimensions)

        remaining = n_hypotheses
        while remaining > 0:
            batch = min(batch_size, remaining)

            prompt = f"""Analyze these writing samples and generate {batch} different hypotheses about the writer's cognitive/psychological state.

Writing samples:
{samples_text}

For each hypothesis, provide values for some of these dimensions (use 5-10 dimensions per hypothesis, leave others blank):
{dim_list}

Generate {batch} DISTINCT cognitive state hypotheses. Each should emphasize different aspects.
Be specific and grounded in the actual text.

Return as a JSON array of objects. Return ONLY valid JSON, nothing else.
Example format: [{{"body_state": "...", "core_belief": "..."}}, {{"body_state": "...", "core_belief": "..."}}]"""

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.content[0].text

                # Parse JSON
                start = text.find('[')
                end = text.rfind(']') + 1
                if start == -1 or end == 0:
                    print("Warning: Could not parse JSON from response")
                    remaining -= batch
                    continue

                hypotheses_data = json.loads(text[start:end])

                for h in hypotheses_data:
                    state = self.state_class.from_dict(h)
                    all_hypotheses.append(state)

            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error: {e}")
            except Exception as e:
                print(f"Warning: API error: {e}")

            remaining -= batch

        return all_hypotheses

    def generate_dimension_value(
        self,
        dimension: str,
        writing_samples: list[str],
    ) -> str:
        """Generate a single dimension value."""

        samples_text = "\n---\n".join(writing_samples[:2])

        prompt = f"""Based on these writing samples, generate a value for the cognitive dimension "{dimension}":

{samples_text}

Return ONLY the dimension value, nothing else. Be specific and grounded in the text."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip().strip('"')

    def generate_dimension_candidates(
        self,
        dimension: str,
        writing_samples: list[str],
        n_candidates: int = 5,
    ) -> list[str]:
        """Generate multiple candidate values for a dimension."""

        samples_text = "\n---\n".join(writing_samples[:2])

        prompt = f"""Based on these writing samples, generate {n_candidates} different possible values for the cognitive dimension "{dimension}":

{samples_text}

Return as a JSON array of strings. Return ONLY the JSON array, nothing else.
Each value should be a distinct interpretation of this dimension for this writer."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text

        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        return []
