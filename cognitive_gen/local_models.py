"""
Local model serving with batching support.

Supports multiple backends:
- transformers (default, simple but slow)
- vllm (fast, requires vllm package)
- ollama (easy setup, requires ollama server)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """Configuration for a local model."""
    name: str
    model_id: str
    backend: str = "transformers"  # "transformers", "vllm", "ollama"
    max_tokens: int = 1000
    temperature: float = 0.7


# Common open models for hypothesis generation
AVAILABLE_MODELS = {
    # Mistral family
    "mistral-7b-instruct": ModelConfig(
        name="Mistral-7B-Instruct",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "mixtral-8x7b-instruct": ModelConfig(
        name="Mixtral-8x7B-Instruct",
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
    # Llama family
    "llama3-8b-instruct": ModelConfig(
        name="Llama-3-8B-Instruct",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "llama3-70b-instruct": ModelConfig(
        name="Llama-3-70B-Instruct",
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    # Qwen family (good for reasoning)
    "qwen2-7b-instruct": ModelConfig(
        name="Qwen2-7B-Instruct",
        model_id="Qwen/Qwen2-7B-Instruct",
    ),
    "qwen2-72b-instruct": ModelConfig(
        name="Qwen2-72B-Instruct",
        model_id="Qwen/Qwen2-72B-Instruct",
    ),
}


class ModelServer(ABC):
    """Abstract base class for model servers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate a single response."""
        pass

    @abstractmethod
    def generate_batch(self, prompts: list[str], max_tokens: int = 500) -> list[str]:
        """Generate responses for multiple prompts (batched for efficiency)."""
        pass


class TransformersServer(ModelServer):
    """Serve models using HuggingFace transformers."""

    def __init__(self, model_id: str, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.device = next(self.model.parameters()).device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # For batch generation

        print(f"Model loaded on {self.device}")

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for instruct model."""
        try:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            return f"[INST] {prompt} [/INST]"

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate single response."""
        text = self._format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def generate_batch(self, prompts: list[str], max_tokens: int = 500) -> list[str]:
        """Generate responses for multiple prompts."""
        if not prompts:
            return []

        # Format all prompts
        texts = [self._format_prompt(p) for p in prompts]

        # Tokenize with padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode each response
        responses = []
        for i, output in enumerate(outputs):
            # Find where the input ends
            input_len = inputs['input_ids'][i].shape[0]
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses


class VLLMServer(ModelServer):
    """Serve models using vLLM for high throughput."""

    def __init__(self, model_id: str, tensor_parallel_size: int = 1):
        try:
            from vllm import LLM, SamplingParams
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        print(f"Loading {model_id} with vLLM...")
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype="float16",
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("vLLM model loaded")

    def _format_prompt(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            return f"[INST] {prompt} [/INST]"

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        results = self.generate_batch([prompt], max_tokens)
        return results[0] if results else ""

    def generate_batch(self, prompts: list[str], max_tokens: int = 500) -> list[str]:
        if not prompts:
            return []

        texts = [self._format_prompt(p) for p in prompts]
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        outputs = self.llm.generate(texts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


class OllamaServer(ModelServer):
    """Serve models using Ollama."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests not installed. Run: pip install requests")

        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

        # Check if model is available
        try:
            response = requests.get(f"{base_url}/api/tags")
            models = [m["name"] for m in response.json().get("models", [])]
            if model_name not in models and f"{model_name}:latest" not in models:
                print(f"Warning: {model_name} not found. Available: {models}")
                print(f"Pull with: ollama pull {model_name}")
        except:
            print(f"Warning: Could not connect to Ollama at {base_url}")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.requests.post(
            self.generate_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                }
            }
        )
        return response.json().get("response", "").strip()

    def generate_batch(self, prompts: list[str], max_tokens: int = 500) -> list[str]:
        # Ollama doesn't support true batching, so we do sequential
        # Could use async for parallelism
        return [self.generate(p, max_tokens) for p in prompts]


def get_server(
    model_key: str = "mistral-7b-instruct",
    backend: str = "transformers",
    **kwargs
) -> ModelServer:
    """Get a model server instance."""

    if model_key in AVAILABLE_MODELS:
        config = AVAILABLE_MODELS[model_key]
        model_id = config.model_id
    else:
        # Assume it's a direct model ID
        model_id = model_key

    if backend == "transformers":
        return TransformersServer(model_id, **kwargs)
    elif backend == "vllm":
        return VLLMServer(model_id, **kwargs)
    elif backend == "ollama":
        return OllamaServer(model_key, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# HYPOTHESIS GENERATOR WITH BATCHING
# =============================================================================

class BatchedHypothesisGenerator:
    """Generate hypotheses using a local model server with batching."""

    def __init__(self, server: ModelServer):
        self.server = server

    def generate_hypotheses(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
        dims_per_hypothesis: int = 8,
    ) -> list[dict]:
        """Generate multiple hypotheses in batched calls."""
        from cognitive_gen.dimension_pool import V1_DIMENSIONS

        samples_text = "\n\n---\n\n".join(writing_samples)

        # Create prompts for batch generation
        prompts = []
        for i in range(n_hypotheses):
            prompt = f"""Analyze these writing samples and generate a hypothesis about the writer's cognitive/psychological state.

Writing samples:
{samples_text}

Generate a cognitive state hypothesis. Provide values for 5-10 of these dimensions (leave others blank):

- body_state: Physical sensation
- preverbal_feeling: Feeling before words
- core_belief: Fundamental truth they hold
- intellectual_stance: How they approach ideas
- what_they_notice: What draws their attention
- moral_framework: How they judge right/wrong
- what_outrages_them: What they can't tolerate
- what_they_protect: What they feel responsible for
- stance_toward_reader: How they position vis-a-vis audience
- who_they_write_for: Real or imagined audience
- what_they_want_reader_to_feel: Intended effect
- relationship_to_past: How history weighs on them
- relationship_to_future: Hope, dread, indifference
- sense_of_urgency: Time pressure
- what_they_find_beautiful: Aesthetic values
- what_they_find_ugly: Aesthetic aversions
- relationship_to_language: Tool, art, weapon, window
- what_they_cant_say_directly: The unspeakable
- the_wound: The injury that shapes the voice
- the_compensation: How they cope with the wound

Return as a JSON object. Be specific and grounded in the text.
Hypothesis #{i+1} - try a DIFFERENT angle than other hypotheses might take."""
            prompts.append(prompt)

        # Generate all at once (batched)
        print(f"  Generating {n_hypotheses} hypotheses in batch...")
        responses = self.server.generate_batch(prompts, max_tokens=1000)

        # Parse responses
        hypotheses = []
        for response in responses:
            try:
                # Find JSON in response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                    # Filter to v1 dimensions
                    filtered = {d: data.get(d) for d in V1_DIMENSIONS if data.get(d)}
                    if filtered:
                        hypotheses.append(filtered)
            except:
                pass

        print(f"  Successfully parsed {len(hypotheses)} hypotheses")
        return hypotheses

    def mutate_value(self, dimension: str, value: str, mutation_type: str) -> str:
        """Mutate a single dimension value."""
        if mutation_type == "vary":
            prompt = f'For "{dimension}", generate a slight variation of: "{value}"\nReturn ONLY the new value.'
        elif mutation_type == "intensify":
            prompt = f'For "{dimension}", make this more extreme: "{value}"\nReturn ONLY the intensified value.'
        elif mutation_type == "soften":
            prompt = f'For "{dimension}", make this more subtle: "{value}"\nReturn ONLY the softened value.'
        elif mutation_type == "opposite":
            prompt = f'For "{dimension}", generate the psychological opposite of: "{value}"\nReturn ONLY the opposite value.'
        else:
            return value

        response = self.server.generate(prompt, max_tokens=100)
        return response.strip().strip('"') or value

    def mutate_batch(
        self,
        mutations: list[tuple[str, str, str]]  # [(dim, value, type), ...]
    ) -> list[str]:
        """Mutate multiple values in batch."""
        prompts = []
        for dim, value, mutation_type in mutations:
            if mutation_type == "vary":
                prompt = f'For "{dim}", generate a slight variation of: "{value}"\nReturn ONLY the new value.'
            elif mutation_type == "intensify":
                prompt = f'For "{dim}", make this more extreme: "{value}"\nReturn ONLY the intensified value.'
            elif mutation_type == "soften":
                prompt = f'For "{dim}", make this more subtle: "{value}"\nReturn ONLY the softened value.'
            elif mutation_type == "opposite":
                prompt = f'For "{dim}", generate the psychological opposite of: "{value}"\nReturn ONLY the opposite value.'
            else:
                prompt = f'Return: "{value}"'
            prompts.append(prompt)

        responses = self.server.generate_batch(prompts, max_tokens=100)
        return [r.strip().strip('"') for r in responses]
