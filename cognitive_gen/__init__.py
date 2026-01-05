"""Cognitive Context Text Generation - Testing the subtext hypothesis."""

from .context import CognitiveContext, generate_random_context

# Optional imports that require additional dependencies
try:
    from .generator import CognitiveGenerator
except ImportError:
    CognitiveGenerator = None

try:
    from .detector import AIDetector
except ImportError:
    AIDetector = None

try:
    from .experiment import run_experiment
except ImportError:
    run_experiment = None

__all__ = [
    "CognitiveContext",
    "generate_random_context",
    "CognitiveGenerator",
    "AIDetector",
    "run_experiment",
]
