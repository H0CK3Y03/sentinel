# sentinel - LLM Red-Teaming Toolkit

A modular, lightweight, open-source toolkit for systematic red-teaming of large language models (LLMs).

## Quick start

```bash
# Install in editable mode (requires Python ≥ 3.10)
pip install -e ".[dev]"

# Run the example experiment
sentinel run manifests/example.yaml

# Validate a manifest without executing
sentinel validate manifests/example.yaml

# List available plugins
sentinel list-plugins
```

## Project layout

```
toolkit/
├── pyproject.toml              # packaging & entry-points
├── manifests/
│   └── example.yaml            # sample experiment manifest
├── src/sentinel/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI (sentinel run / validate / …)
│   ├── orchestrator.py         # central experiment runner
│   ├── manifest.py             # YAML/JSON manifest loader
│   ├── models.py               # shared data models
│   ├── logger.py               # append-only JSONL log store
│   ├── plugins.py              # component registry & factories
│   ├── model_adapters/
│   │   ├── base.py             # ModelAdapter ABC
│   │   └── stub.py             # deterministic mock adapter
    │   ├── generators/
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

Then reference `adapter: my-adapter` in your manifest.

## License

MIT
