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
| `prompt-injection` | Injection | User-input override attacks |
| `token-perturbation` | Evasion | Adversarial token modifications |
| `universal-trigger` | Trigger | Transferable attack tokens |
| `multi-turn-conversation` | Conversation | Multi-step dialogue attacks |

## Project layout

```
toolkit/
├── pyproject.toml              # packaging & entry-points
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
│   ├── analysis.py             # experiment analysis
│   ├── plugins.py              # plugin registry
│   ├── model_adapters/         # model backends
│   ├── generators/             # attack generators
│   │   ├── base.py
│   │   ├── single_turn_jailbreak.py
│   │   ├── prompt_injection.py
│   │   ├── token_perturbation.py
│   │   ├── universal_trigger.py
│   │   └── multi_turn_conversation.py
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

## Detailed Documentation

- **[Attacks and Metrics Guide](ATTACKS_AND_METRICS.md)** - Complete reference for all attack types, metrics, and analysis
- **Custom Generators** - Implement your own attack generators by extending `AttackGenerator`
- **Custom Judges** - Implement evaluation logic by extending `JudgeAdapter`

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

## Local llama.cpp judge

If you have a GGUF model downloaded locally, you can use it as a judge via the
`llama-cpp-judge` plugin. A ready-to-run example manifest is provided at
`manifests/qwen36-judge.yaml`.

Run it from the `toolkit/` directory:

```bash
sentinel run manifests/qwen36-judge.yaml
```

That manifest uses the local model path:

```text
/home/hockey/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

If you move the model, update `judges[1].config.model_path` accordingly.

## License

MIT
