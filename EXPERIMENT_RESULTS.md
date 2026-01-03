# Experiment 5 Results: Reverse Inference

## Overview

Testing approaches to infer cognitive states from writing samples using perplexity scoring with genetic search.

- **Scoring model**: mistralai/Mistral-7B-v0.3 (base model)
- **Generation model**: mistralai/Mistral-7B-Instruct-v0.3 (local)
- **Population**: 30 hypotheses
- **Generations**: 25

## Methods Compared

1. **Meta V1Mode**: Uses meta-search with fixed 20 v1-style cognitive dimensions and linkage groups
2. **Meta Complete**: Uses meta-search with dynamic dimension discovery (10-16 dimensions)

## Final Results

| Writer | Meta V1Mode | Meta Complete |
|--------|-------------|---------------|
| **Kafka** | 80.9% (±1.9%) | 81.3% (±1.0%) |
| **Nietzsche** | 61.7% (±1.4%) | 62.9% (±1.9%) |
| **Hemingway** | 55.0% (±1.3%) | 48.1% (±4.2%) |

## Key Findings

1. **Both methods achieve comparable high improvements on Kafka and Nietzsche**
   - Kafka: ~81% improvement with both methods
   - Nietzsche: ~62% improvement with both methods

2. **Meta V1Mode outperforms on Hemingway** (55.0% vs 48.1%)
   - The fixed 20 dimensions may work better for certain writers
   - Meta Complete had higher variance on Hemingway (±4.2%)

3. **Dynamic dimensions in Meta Complete**
   - Uses 10-16 dimensions (vs fixed 20 in V1Mode)
   - Discovers novel dimensions like `deepest_desire`, `secret_shame`, `somatic_memory`, `cosmic_stance`

4. **Consistent high improvements**
   - All experiments show 48-81% perplexity improvement over baseline
   - Results are reproducible across runs (low standard deviations)

## Raw Results

### Meta V1Mode

#### Kafka
```json
[
  {"run": 1, "improvement": 83.48, "baseline_ppl": 15.92, "best_ppl": 2.63},
  {"run": 2, "improvement": 79.06, "baseline_ppl": 15.92, "best_ppl": 3.33},
  {"run": 3, "improvement": 80.25, "baseline_ppl": 15.92, "best_ppl": 3.14}
]
```

#### Nietzsche
```json
[
  {"run": 1, "improvement": 62.27, "baseline_ppl": 8.26, "best_ppl": 3.12},
  {"run": 2, "improvement": 63.14, "baseline_ppl": 8.26, "best_ppl": 3.04},
  {"run": 3, "improvement": 59.83, "baseline_ppl": 8.26, "best_ppl": 3.32}
]
```

#### Hemingway
```json
[
  {"run": 1, "improvement": 55.54, "baseline_ppl": 7.04, "best_ppl": 3.13},
  {"run": 2, "improvement": 53.22, "baseline_ppl": 7.04, "best_ppl": 3.29},
  {"run": 3, "improvement": 56.36, "baseline_ppl": 7.04, "best_ppl": 3.07}
]
```

### Meta Complete

#### Kafka
```json
[
  {"run": 1, "improvement": 81.43, "baseline_ppl": 20.68, "best_ppl": 3.84, "dims": 10},
  {"run": 2, "improvement": 82.45, "baseline_ppl": 20.68, "best_ppl": 3.63, "dims": 10},
  {"run": 3, "improvement": 79.94, "baseline_ppl": 20.68, "best_ppl": 4.15, "dims": 14}
]
```

#### Nietzsche
```json
[
  {"run": 1, "improvement": 62.27, "baseline_ppl": 11.11, "best_ppl": 4.19, "dims": 10},
  {"run": 2, "improvement": 60.99, "baseline_ppl": 11.11, "best_ppl": 4.34, "dims": 16},
  {"run": 3, "improvement": 65.44, "baseline_ppl": 11.11, "best_ppl": 3.84, "dims": 11}
]
```

#### Hemingway
```json
[
  {"run": 1, "improvement": 44.61, "baseline_ppl": 6.72, "best_ppl": 3.72, "dims": 12},
  {"run": 2, "improvement": 45.63, "baseline_ppl": 6.72, "best_ppl": 3.65, "dims": 11},
  {"run": 3, "improvement": 53.95, "baseline_ppl": 6.72, "best_ppl": 3.10, "dims": 11}
]
```

## V1 Dimensions (Fixed 20)

The Meta V1Mode uses these fixed cognitive dimensions:
- `body_state`, `preverbal_feeling`, `core_belief`, `intellectual_stance`
- `what_they_notice`, `moral_framework`, `what_outrages_them`, `what_they_protect`
- `stance_toward_reader`, `who_they_write_for`, `what_they_want_reader_to_feel`
- `relationship_to_past`, `relationship_to_future`, `sense_of_urgency`
- `what_they_find_beautiful`, `what_they_find_ugly`, `what_they_cant_say_directly`
- `the_wound`, `the_compensation`, `relationship_to_language`

## Meta Complete Discovered Dimensions

Top dimensions discovered across runs (by frequency in top 25%):
- `deepest_desire`, `deepest_fear`, `somatic_memory`
- `cosmic_stance`, `relationship_to_authority`, `shadow_self`
- `felt_sense_of_world`, `ontological_position`, `active_preoccupation`
- `relationship_to_truth`, `secret_shame`, `forbidden_knowledge`
- `stance_toward_reader`, `mortality_awareness`, `relationship_to_infinity`

## Notes

- Baseline perplexity varies between V1Mode and Complete due to different context formatting
- Meta V1Mode uses v1-style formatting with writing samples wrapper
- Meta Complete uses sparse hypothesis format without wrapper
- All experiments use local models only (no API costs)

## Bug Fixes Applied

### Target Text Leakage (Critical)
Previously, the `writing_samples` (which included the target text) was passed to hypothesis generators, allowing them to potentially quote or paraphrase the target directly. Fixed by:
- Storing `self._context_samples = context_samples` (excludes target)
- Passing `context_samples` instead of `writing_samples` to all generation methods

### Perplexity Measurement
- Fixed newline format before target: `\n` -> `\n\n`
- Fixed context length calculation to include writing samples wrapper in v1 mode
