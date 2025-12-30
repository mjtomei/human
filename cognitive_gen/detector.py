"""AI text detection using open-source models."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of AI detection on a text sample."""

    text: str
    ai_probability: float
    human_probability: float
    label: str  # "AI" or "Human" based on higher probability


class AIDetector:
    """
    Wrapper around HuggingFace AI detection models.

    Uses roberta-base-openai-detector by default, which was trained to
    distinguish GPT-2 outputs from human text. While not perfectly calibrated
    for modern models like Claude, it provides a baseline signal.
    """

    def __init__(
        self,
        model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta",
        device: Optional[str] = None,
    ):
        """
        Initialize the detector.

        Args:
            model_name: HuggingFace model to use for detection
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading detector model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Detector loaded on {self.device}")

    def detect(self, text: str) -> DetectionResult:
        """
        Detect whether text is AI-generated.

        Args:
            text: The text to analyze

        Returns:
            DetectionResult with probabilities and label
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Model outputs: index 0 = "Fake" (AI), index 1 = "Real" (Human)
        # But let's verify by checking the config
        ai_prob = probs[0][0].item()
        human_prob = probs[0][1].item()

        label = "AI" if ai_prob > human_prob else "Human"

        return DetectionResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            ai_probability=ai_prob,
            human_probability=human_prob,
            label=label,
        )

    def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
        """
        Detect AI-generation for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of DetectionResults
        """
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results

    def get_ai_score(self, text: str) -> float:
        """
        Get just the AI probability score for a text.

        Args:
            text: The text to analyze

        Returns:
            Probability (0-1) that the text is AI-generated
        """
        return self.detect(text).ai_probability


class MultiDetector:
    """
    Runs multiple detection models and aggregates results.

    This helps account for the fact that different detectors have
    different strengths and biases.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize with multiple detector models."""
        self.detectors = {}

        # Primary detector
        self.detectors["roberta-openai"] = AIDetector(
            model_name="openai-community/roberta-base-openai-detector",
            device=device,
        )

        # Could add more detectors here:
        # self.detectors["roberta-chatgpt"] = AIDetector(
        #     model_name="Hello-SimpleAI/chatgpt-detector-roberta",
        #     device=device,
        # )

    def detect(self, text: str) -> dict[str, DetectionResult]:
        """Run all detectors on the text."""
        return {
            name: detector.detect(text) for name, detector in self.detectors.items()
        }

    def get_average_ai_score(self, text: str) -> float:
        """Get average AI probability across all detectors."""
        scores = [detector.get_ai_score(text) for detector in self.detectors.values()]
        return sum(scores) / len(scores)
