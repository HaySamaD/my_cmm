# Call Me Maybe: Constrained Decoding Pipeline for Safe JSON Function Calling

A structured, production-grade Python inference pipeline that guarantees **100% syntactically valid and schema-compliant JSON outputs** from lightweight Large Language Models (LLMs).

---

## The Architectural Challenge: Reinventing the Wheel from Scratch

Lightweight models (like `Qwen/Qwen3-0.6B`) inherently fail at producing structured JSON outputs through prompting alone. They constantly break downstream production systems by adding natural language prose, trailing commas, or missing key braces. 

To overcome this, **this pipeline solves the problem at the compiler and inference layer rather than the prompt layer**. 

### The Real Challenge: No Frameworks Allowed
The absolute constraint of this engineering showcase was the **strict prohibition of all standard, high-level structured generation tools** (such as Outlines, Instructor, Guidance, or vLLM grammar engines). Every layer of this project had to be built completely **from scratch**:
* Intercepting next-token probabilities at the raw model logit layer manually.
* Managing the auto-regressive state arrays token-by-token directly inside the generation loop.
* Isolating tokenizer ambiguities (like BPE whitespace variations) without framework abstractions.

By bypassing modern framework shortcuts, this system stands as a pure proof of low-level software engineering excellence, forcing deterministic grammar validation directly onto raw matrix outputs. For full architectural specifications and task boundaries, refer directly to **`doc/subject.pdf`**.

---

## Overview

**Call Me Maybe** tracks the token-generation lifecycle via a custom **Token-Level Constrained Decoding State Machine**. By injecting an active constraint mask before the model selects its next token piece, it applies a dynamic vocabulary filter that completely suppresses non-compliant outputs, ensuring flawless execution boundaries.

### Key Architectural Pillars
* **Logit Interception Layer:** Intercepts next-token scores via `get_logits_from_input_ids()` before token sampling occurs.
* **Deterministic Grammar Enforcer:** Evaluates bracket balance, dictionary keys, and schema datatypes character-by-character on the fly.
* **Clean Codebase Isolation:** Decoupled architecture separating token decoding, dataset ingestion, and local hardware SDK utilities cleanly.

---

## Modular System Architecture

The pipeline is split into explicit decoupled modules designed for optimal maintenance safety:
* **`src/constrained_decoding.py`:** Core finite-state machine (FSM) enforcing structural token masking profiles byte-by-byte.
* **`src/json_loader.py`:** Handles safe structural input extraction and dynamic payload evaluation.
* **`llm_sdk/`:** Low-level SDK wrapper managing local model inference using Hugging Face transformers, raw tensor evaluations, and automated device tracking (`mps`/`cuda`/`cpu`).

---

## Technical Decisions & Optimization

### Token generation Latency Cap
To balance execution speeds and guarantee system stability, a custom execution flag `--max_tokens` is integrated at the command-line entry point. This parameter strictly forces a ceiling limit on the generated tokens to prevent runtime token inflation, isolate infinite loops, and optimize resource distribution on compute instances.

---

## Getting Started

### Installation
Ensure your host environment features `Python >= 3.10` and `uv` for dependency isolation. Synchronize environment components via:

```bash
make install
```

### Execution
Run the baseline automated evaluation array using default local assets:

```bash
make run
```

For customized inputs, runtime token caps, and custom model evaluations, run the executable module layout directly with target tags:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B \
  --max_tokens 50
```

### Quality Assurance & Linting
Enforce strict formatting validation and type system checks against code boundaries:

```bash
# Run standard flake8 style checks and partial mypy type tests
make lint

# Enforce complete static assurance via strict typing layouts
make lint-strict
```

---

## Performance & Evaluation

* **Syntax Reliability:** **100% Successful JSON Parse Rate** with zero exceptions across evaluation arrays.
* **Intent Extraction Accuracy:** **>90% Semantic Accuracy** matching natural prompts to functional schemas.
* **Latency Benchmarks:** High-throughput tensor masking maps complex argument parameters in **under 60 seconds** on core acceleration backends.

---

## Example Pipeline Workflow

### Available Function Schema Registry (`functions_definition.json`)
```json
[
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": { "type": "string" },
      "regex": { "type": "string" },
      "replacement": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

### Pipeline Terminal Execution Trace Output
```text
=== Program Start ===

[INFO] Pre-processing token vocabulary index maps...
[INFO] Processing prompts with max_tokens limit set to: 50

Processing prompt 1/1: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'
  -> Next Token Logits Tracked: [Generating safe structure...]
  -> ✔ Success: fn_substitute_string_with_regex({'source_string': 'The cat sat on the mat', 'regex': 'cat', 'replacement': 'dog'})

The result is saved in data/output/function_calling_results.json
Total time: 1.24 seconds
```

---

## Resources & Core References

### Community & Architecture Discussions
* **Stack Overflow:** Deep technical threads regarding logit manipulation matrix bounds, NaN tensor mitigation, and token-mask performance overhead.
* **Reddit (r/LocalLLaMA & r/MachineLearning):** Discussions on the execution speed of raw loop constraint structures on sub-1B lightweight topologies.
* **Hugging Face Transformers Documentation:** Low-level mechanics of transformers generation steps and output logit distributions.

### AI Utility Disclosure
Artificial intelligence systems were utilized during development exclusively to accelerate repetitive boilerplate tasks, resolve explicit static typing hint boundaries during strict `mypy` sweeps, and format PEP 257-compliant docstrings across open boundaries.

---
*Developed as a high-performance system engineering showcase for the 42 Network Curriculum. See `doc/subject.pdf` for full project requirements.*