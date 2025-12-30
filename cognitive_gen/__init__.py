"""Cognitive Context Text Generation - Testing the subtext hypothesis."""

from .context import CognitiveContext, generate_random_context
from .generator import CognitiveGenerator
from .detector import AIDetector
from .experiment import run_experiment

__all__ = [
    "CognitiveContext",
    "generate_random_context",
    "CognitiveGenerator",
    "AIDetector",
    "run_experiment",
]
