# sentinel - LLM Red-Teaming Toolkit

A modular, lightweight, open-source toolkit for systematic red-teaming of large language models (LLMs).

## Features

**Attack Generators:**
- Single-turn jailbreak attacks (role-play, DAN, direct override, etc.)
- Prompt injection attacks (direct and indirect)
- Token perturbation attacks (spelling, encoding, spacing)
- Universal/transferable trigger attacks
- Multi-turn conversational attacks

**Evaluation & Metrics:**
- Comprehensive trial-level and aggregate metrics collection
- Per-attack-type performance breakdown
- Judge agreement and consensus measurement
- Detailed analysis and reporting via CLI

**Model Support:**
- Local llama.cpp models (GGUF format)
- Extensible adapter interface for other backends

## Quick start

```bash
# Install in editable mode (requires Python ≥ 3.10)
cd toolkit
pip install -e ".[dev]"

# Run an attack test
sentinel run manifests/example.yaml

# Analyse the results
sentinel analyze logs/example.jsonl

# Validate a manifest
sentinel validate manifests/example.yaml

# List available plugins
sentinel list-plugins
```

## Built-in Attack Generators

| Generator | Type | Use Case |
|-----------|------|----------|
| `single-turn-jailbreak` | Jailbreak | One-shot prompt engineering attacks |
| `minimal-attack` | Baseline | One fixed prompt for smoke tests |
| `prompt-injection` | Injection | User-input override attacks |
| `token-perturbation` | Evasion | Adversarial token modifications |
| `universal-trigger` | Trigger | Transferable attack tokens |
| `multi-turn-conversation` | Conversation | Multi-step dialogue attacks |

## Project layout

```
toolkit/
├── pyproject.toml
├── manifests/
│   ├── single-turn-jailbreak.yaml
│   ├── prompt-injection.yaml
│   ├── token-perturbation.yaml
│   ├── universal-trigger.yaml
│   ├── multi-turn-conversation.yaml
│   └── comprehensive-attacks.yaml
├── src/sentinel/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI
│   ├── orchestrator.py         # experiment runner
│   ├── manifest.py             # manifest loader
│   ├── models.py               # data models
│   ├── logger.py               # JSONL logging
│   ├── metrics.py              # metrics collection
│   ├── plugins.py              # plugin registry
│   ├── model_adapters/         # model backends
│   ├── generators/             # attack generators
│   └── judges/                 # evaluation judges
└── tests/
    └── ...
```

## Example: Run and Analyse

```bash
# Run a comprehensive test
sentinel run manifests/comprehensive-attacks.yaml

# Analyse results with detailed breakdown
sentinel analyze logs/comprehensive-attacks.jsonl

# Save analysis to JSON
sentinel analyze logs/comprehensive-attacks.jsonl \
  --output-json reports/comprehensive.json
```

## Writing a custom adapter

```python
from sentinel.model_adapters.base import ModelAdapter
from sentinel.models import ModelResponse, HealthStatus

class MyAdapter(ModelAdapter):
    async def generate(self, prompt, config=None):
        # call your backend …
        return ModelResponse(text="…", model_id=self.model_id)

    async def health_check(self):
        return HealthStatus.OK
```

Register it:

```python
from sentinel.plugins import register_adapter
register_adapter("my-adapter", MyAdapter)
```

Then reference `adapters:` in your manifest and add `adapter: my-adapter` inside the entry.

## Writing a custom generator

Subclass `AttackGenerator` and implement the three required methods:

```python
from typing import Any, Dict, List
from sentinel.generators.base import AttackGenerator
from sentinel.models import PromptCandidate

class MyGenerator(AttackGenerator):
    display_name = "my-generator"

    def configure(self, params: Dict[str, Any]) -> None:
        # store any manifest config params here
        self._configured = True

    def next(self, batch_size: int = 1) -> List[PromptCandidate]:
        return [
            PromptCandidate(text="my adversarial prompt", metadata={"generator": self.name})
            for _ in range(batch_size)
        ]

    def reset(self) -> None:
        pass  # reset counters / caches if needed
```

Register it at runtime:

```python
from sentinel.plugins import register_generator
register_generator("my-generator", MyGenerator)
```

Or advertise it as a Python entry-point in `pyproject.toml` so it is discovered automatically without code changes:

```toml
[project.entry-points."sentinel.generators"]
my-generator = "my_package.my_module:MyGenerator"
```

Then reference it in your manifest:

```yaml
generators:
  - instance_id: generator-custom
    name: my-generator
    config:
      seed: 42
```

## Local llama.cpp judge

If you have a GGUF model downloaded locally, you can use it as a judge via the
`llama-cpp-judge` plugin. Install the optional dependency first:

```bash
pip install -e ".[llama]"
```

Then add it to your manifest:

```yaml
judges:
  - name: llama-cpp-judge
    config:
      model_path: /path/to/model.gguf
      n_ctx: 1024
      n_gpu_layers: 0
      temperature: 0.0
      max_tokens: 256
```

## License

MIT
