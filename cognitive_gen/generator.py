"""Hierarchical text generator using cognitive context conditioning."""

import anthropic
from dataclasses import dataclass
from typing import Optional

from .context import CognitiveContext, generate_random_context


@dataclass
class GenerationResult:
    """Result of a text generation."""

    surface_text: str
    inner_monologue: Optional[str] = None
    context: Optional[CognitiveContext] = None
    mode: str = "baseline"


# Writing prompts for different text types
WRITING_PROMPTS = {
    "personal_essay": (
        "Write a short personal essay (150-250 words) about a meaningful moment "
        "from your past. It could be a realization, a turning point, an ordinary "
        "moment that stuck with you, or something you're still processing. "
        "Write in first person."
    ),
    "creative_fiction": (
        "Write the opening of a short story (150-250 words). Draw the reader in "
        "with a specific scene, character, or moment. Don't try to tell a complete "
        "story - just create an opening that makes someone want to keep reading."
    ),
    "email": (
        "Write an email (100-200 words) to a friend you haven't spoken to in a "
        "few years. You're reaching out to reconnect. Don't be overly formal or "
        "overly casual - find the natural tone for reaching out after time has passed."
    ),
    "message": (
        "Write a text message (50-100 words) apologizing to a friend for missing "
        "an event that was important to them. You had a legitimate reason but still "
        "feel bad about it. Keep it natural - this is a text, not a formal letter."
    ),
}


class CognitiveGenerator:
    """
    Generator that produces text with and without cognitive context conditioning.

    The hypothesis is that text generated with rich cognitive context will feel
    more human because it has "subtext" - underlying thoughts and motivations
    that subtly influence the surface text.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def generate_baseline(
        self,
        text_type: str,
        temperature: float = 1.0,
    ) -> GenerationResult:
        """
        Generate text without cognitive context - standard LLM generation.

        Args:
            text_type: One of "personal_essay", "creative_fiction", "email", "message"
            temperature: Sampling temperature

        Returns:
            GenerationResult with the generated text
        """
        prompt = WRITING_PROMPTS.get(text_type, WRITING_PROMPTS["personal_essay"])

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = response.content[0].text if response.content else ""

        return GenerationResult(
            surface_text=text,
            inner_monologue=None,
            context=None,
            mode="baseline",
        )

    def generate_with_context(
        self,
        text_type: str,
        context: Optional[CognitiveContext] = None,
        temperature: float = 1.0,
        include_inner_monologue: bool = True,
    ) -> GenerationResult:
        """
        Generate text with cognitive context conditioning.

        This is the experimental condition: we inject a rich cognitive context
        and optionally generate an inner monologue before the surface text.

        Args:
            text_type: One of "personal_essay", "creative_fiction", "email", "message"
            context: Cognitive context to use (generated randomly if not provided)
            temperature: Sampling temperature
            include_inner_monologue: Whether to generate inner monologue first

        Returns:
            GenerationResult with the generated text and context used
        """
        if context is None:
            context = generate_random_context(text_type)

        prompt = WRITING_PROMPTS.get(text_type, WRITING_PROMPTS["personal_essay"])
        context_prompt = context.to_prompt()

        inner_monologue = None

        if include_inner_monologue:
            # Stage 1: Generate inner monologue
            inner_monologue = self._generate_inner_monologue(
                context_prompt, prompt, temperature
            )

            # Stage 2: Generate surface text conditioned on context + inner monologue
            surface_text = self._generate_surface_with_monologue(
                context_prompt, prompt, inner_monologue, temperature
            )
        else:
            # Generate directly with context but no inner monologue
            surface_text = self._generate_surface_direct(
                context_prompt, prompt, temperature
            )

        return GenerationResult(
            surface_text=surface_text,
            inner_monologue=inner_monologue,
            context=context,
            mode="cognitive",
        )

    def _generate_inner_monologue(
        self,
        context_prompt: str,
        writing_prompt: str,
        temperature: float,
    ) -> str:
        """Generate the inner monologue / stream of consciousness."""
        system = f"""You are simulating the inner mental experience of a person about to write something.

{context_prompt}

Given this cognitive context, generate a stream of consciousness (100-150 words) showing what's going through this person's mind as they prepare to write. Include:
- Fleeting thoughts and associations
- Hesitations and second-guessing
- Emotional undercurrents
- Half-formed intentions
- Things they're trying not to think about

This should feel like real internal experience, not a neat summary. Include sentence fragments, contradictions, and tangents."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate the inner monologue for someone about to: {writing_prompt}",
                }
            ],
        )

        return response.content[0].text if response.content else ""

    def _generate_surface_with_monologue(
        self,
        context_prompt: str,
        writing_prompt: str,
        inner_monologue: str,
        temperature: float,
    ) -> str:
        """Generate surface text conditioned on context and inner monologue."""
        system = f"""You are writing AS this person. Your job is to produce the actual text they would write.

{context_prompt}

INNER EXPERIENCE (what's going through their mind - do not include this in output):
{inner_monologue}

Now write the actual text this person would produce. The cognitive context and inner monologue should subtly influence your word choice, rhythm, what you include and leave out - but should NOT be explicitly mentioned. The output should read as natural human writing, not as a description of someone writing.

Write ONLY the requested text, nothing else. No meta-commentary, no explanations."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": writing_prompt,
                }
            ],
        )

        return response.content[0].text if response.content else ""

    def _generate_surface_direct(
        self,
        context_prompt: str,
        writing_prompt: str,
        temperature: float,
    ) -> str:
        """Generate surface text conditioned only on context (no inner monologue)."""
        system = f"""You are writing AS this person.

{context_prompt}

Write the actual text this person would produce. The cognitive context should subtly influence your word choice, rhythm, what you include and leave out - but should NOT be explicitly mentioned. The output should read as natural human writing.

Write ONLY the requested text, nothing else. No meta-commentary, no explanations."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": writing_prompt,
                }
            ],
        )

        return response.content[0].text if response.content else ""
