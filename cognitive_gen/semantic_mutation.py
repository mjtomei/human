"""
Semantic mutation operators using Claude or local LLM.

Instead of random value replacement, generate meaningful variations
of dimension values using language understanding.
"""

import json
import random
from typing import Optional

import anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_gen.cognitive_state import CognitiveState, get_state_class


# =============================================================================
# LOCAL LLM FOR V1 (singleton, shared with meta_search if both loaded)
# =============================================================================

class LocalLLMV1:
    """Local LLM for v1 hypothesis generation and mutation."""

    _instance = None

    def __new__(cls, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._loaded = False
            cls._instance._model = None
            cls._instance._tokenizer = None
        return cls._instance

    def _ensure_loaded(self):
        if not self._loaded:
            print(f"Loading generation model: {self._model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"
            self._loaded = True
            print(f"Generation model loaded on {next(self._model.parameters()).device}")

    def _format_prompt(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            return f"[INST] {prompt} [/INST]"

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        self._ensure_loaded()
        text = self._format_prompt(prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 500) -> list[str]:
        if not prompts:
            return []

        self._ensure_loaded()
        texts = [self._format_prompt(p) for p in prompts]

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        responses = []
        for i, output in enumerate(outputs):
            input_len = (inputs['attention_mask'][i] == 1).sum().item()
            response = self._tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses


class SemanticMutator:
    """Generate semantic mutations of cognitive state dimensions."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", use_local: bool = False):
        self.use_local = use_local
        if use_local:
            self.local_llm = LocalLLMV1()
        else:
            self.client = anthropic.Anthropic()
            self.model = model

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        if self.use_local:
            return self.local_llm.generate(prompt, max_new_tokens=max_tokens)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

    def vary(self, dimension: str, value: str) -> str:
        """Generate a slight variation of a value."""
        prompt = f"""For the cognitive dimension "{dimension}", generate a slight variation of:
"{value}"

Return ONLY the new value, nothing else. Keep similar meaning but use different words or angle."""
        return self._generate(prompt, 100).strip('"')

    def intensify(self, dimension: str, value: str) -> str:
        """Make a value more extreme."""
        prompt = f"""For the cognitive dimension "{dimension}", make this more extreme/intense:
"{value}"

Return ONLY the intensified version, nothing else."""
        return self._generate(prompt, 100).strip('"')

    def soften(self, dimension: str, value: str) -> str:
        """Make a value more subtle."""
        prompt = f"""For the cognitive dimension "{dimension}", make this more subtle/muted:
"{value}"

Return ONLY the softened version, nothing else."""
        return self._generate(prompt, 100).strip('"')

    def opposite(self, dimension: str, value: str) -> str:
        """Generate the psychological opposite."""
        prompt = f"""For the cognitive dimension "{dimension}", generate the psychological opposite of:
"{value}"

Return ONLY the opposite value, nothing else."""
        return self._generate(prompt, 100).strip('"')

    def blend(self, dimension: str, value_a: str, value_b: str) -> str:
        """Combine two values into a synthesis."""
        prompt = f"""For the cognitive dimension "{dimension}", blend these two psychological states into a coherent synthesis:
A: "{value_a}"
B: "{value_b}"

Return ONLY the blended value, nothing else."""
        return self._generate(prompt, 100).strip('"')

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

        samples_text = "\n---\n".join(writing_samples[:2])

        prompt = f"""Given these linked cognitive dimensions for a writer:

{chr(10).join(f'{d}: {v}' for d, v in current_values.items())}

And samples of their writing:
{samples_text}

Generate a coherent variation that keeps the psychological structure but shifts the specifics.
These dimensions are linked - they should remain internally consistent.

Return as JSON object with the same keys. Return ONLY the JSON, nothing else."""

        try:
            text = self._generate(prompt, 300)
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                new_values = json.loads(text[start:end])
                mutated = hypothesis.copy()
                for d, v in new_values.items():
                    if d in group:
                        mutated[d] = v
                return mutated
        except:
            pass

        return hypothesis


class HypothesisGenerator:
    """Generate cognitive hypotheses from writing samples."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", state_version: str = "v1", use_local: bool = False):
        self.use_local = use_local
        if use_local:
            self.local_llm = LocalLLMV1()
        else:
            self.client = anthropic.Anthropic()
            self.model = model
        self.state_class = get_state_class(state_version)
        self.state_version = state_version

    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        if self.use_local:
            return self.local_llm.generate(prompt, max_new_tokens=max_tokens)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

    def generate_initial_hypotheses(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 10,
    ) -> list:
        """Generate initial hypotheses with BATCHED generation for local LLM."""

        samples_text = "\n\n---\n\n".join(writing_samples)
        dimensions = self.state_class.get_dimensions()
        key_dimensions = dimensions[:15] if len(dimensions) > 15 else dimensions
        dim_list = "\n".join(f"- {d}" for d in key_dimensions)

        if self.use_local:
            # Batched generation for local LLM
            prompts = []
            for i in range(n_hypotheses):
                prompt = f"""Analyze these writing samples and generate a hypothesis about the writer's cognitive/psychological state.

Writing samples:
{samples_text}

Generate hypothesis #{i+1}. Provide values for 5-10 of these dimensions (leave others blank):
{dim_list}

Return as a JSON object. Be specific and grounded in the text.
Try a DIFFERENT angle than other hypotheses."""
                prompts.append(prompt)

            print(f"  Generating {n_hypotheses} hypotheses in batch...")
            responses = self.local_llm.generate_batch(prompts, max_new_tokens=1000)

            all_hypotheses = []
            for text in responses:
                try:
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start >= 0 and end > start:
                        h = json.loads(text[start:end])
                        state = self.state_class.from_dict(h)
                        all_hypotheses.append(state)
                except:
                    pass

            print(f"  Successfully parsed {len(all_hypotheses)} hypotheses")
            return all_hypotheses

        else:
            # Original API-based generation
            batch_size = min(5, n_hypotheses)
            all_hypotheses = []
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
                    text = self._generate(prompt, 4000)
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    if start >= 0 and end > start:
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

        return self._generate(prompt, 100).strip('"')

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

        text = self._generate(prompt, 500)

        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        return []
