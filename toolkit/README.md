# redteam - LLM Red-Teaming Toolkit

A modular, lightweight, open-source toolkit for systematic red-teaming of large language models (LLMs).

## Quick start

```bash
# Install in editable mode (requires Python ≥ 3.10)
pip install -e ".[dev]"

# Run the example experiment
redteam run manifests/example.yaml

# Validate a manifest without executing
redteam validate manifests/example.yaml

# List available plugins
redteam list-plugins
```

## Architecture

```
manifest.yaml
    │
    ▼
┌─────────────┐       ┌──────────────┐       ┌───────────────┐
│ Orchestrator│─────▶│ Attack        │─────▶│PromptCandidate│
│  (asyncio)  │◀─────│Generator      │       └───────────────┘
└─────┬───────┘       └──────────────┘
      │
      │  prompt
      ▼
┌──────────────┐
│Model Adapter │──▶ ModelResponse
│ (stub/llama) │
└─────┬────────┘
      │  response
      ▼
┌──────────────┐
│Judge Pipeline│──▶ Verdict[]
│ (heuristic…) │
└─────┬────────┘
      │
      ▼
  JSONL log store
```

## Project layout

```
toolkit/
├── pyproject.toml              # packaging & entry-points
├── manifests/
│   └── example.yaml            # sample experiment manifest
├── src/redteam/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI (redteam run / validate / …)
│   ├── orchestrator.py         # central experiment runner
│   ├── manifest.py             # YAML/JSON manifest loader
│   ├── models.py               # shared data models
│   ├── logger.py               # append-only JSONL log store
│   ├── plugins.py              # component registry & factories
│   ├── adapters/
│   │   ├── base.py             # ModelAdapter ABC
│   │   └── stub.py             # deterministic mock adapter
│   ├── attacks/
│   │   ├── base.py             # AttackGenerator ABC
│   │   └── stub.py             # template × goal generator
│   └── judges/
│       ├── base.py             # JudgeAdapter ABC
│       ├── heuristic.py        # regex / keyword judge
│       └── stub.py             # placeholder LLM judge
└── tests/
    └── ...
```

## Writing a new adapter

```python
from redteam.adapters.base import ModelAdapter
from redteam.models import ModelResponse, HealthStatus

class MyAdapter(ModelAdapter):
    async def generate(self, prompt, config=None):
        # call your backend …
        return ModelResponse(text="…", model_id=self.model_id)

    async def health_check(self):
        return HealthStatus.OK
```

Register it:

```python
from redteam.plugins import register_adapter
register_adapter("my-adapter", MyAdapter)
```

Then reference `adapter: my-adapter` in your manifest.

## License

MIT
