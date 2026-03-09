"""Experiment manifest loader and validator.

Manifests are small YAML (preferred) or JSON files that declaratively
describe every parameter of a red-teaming experiment.  The loader
validates required fields and returns a typed ``Manifest`` object that
the orchestrator consumes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - YAML is optional
    yaml = None  # type: ignore[assignment]


@dataclass
class ManifestModel:
    """Model / adapter section of a manifest."""
    adapter: str = "stub"
    model_id: str = "stub-v1"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestGenerator:
    """Attack generator section of a manifest."""
    name: str = "stub-template"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestJudge:
    """Single judge entry in the judge pipeline."""
    name: str = "heuristic"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    """Fully parsed and validated experiment manifest."""

    experiment_id: str = field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:8]}")
    author: str = ""
    description: str = ""

    model: ManifestModel = field(default_factory=ManifestModel)
    generator: ManifestGenerator = field(default_factory=ManifestGenerator)
    judges: List[ManifestJudge] = field(default_factory=lambda: [ManifestJudge()])

    seed: int = 42
    batch_size: int = 8
    num_batches: int = 1
    rate_limit_rps: float = 0.0  # 0 = unlimited
    output: str = "logs/experiment.jsonl"

    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a plain dict (for logging)."""
        return {
            "experiment_id": self.experiment_id,
            "author": self.author,
            "description": self.description,
            "model": {"adapter": self.model.adapter, "model_id": self.model.model_id, "config": self.model.config},
            "generator": {"name": self.generator.name, "config": self.generator.config},
            "judges": [{"name": j.name, "config": j.config} for j in self.judges],
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "rate_limit_rps": self.rate_limit_rps,
            "output": self.output,
        }


def _parse_raw(data: Dict[str, Any]) -> Manifest:
    """Build a ``Manifest`` from a raw dict."""
    model_raw = data.get("model", {})
    gen_raw = data.get("generator", {})
    judge_list = data.get("judges", [{}])

    model = ManifestModel(
        adapter=model_raw.get("adapter", "stub"),
        model_id=model_raw.get("model_id", "stub-v1"),
        config=model_raw.get("config", {}),
    )
    generator = ManifestGenerator(
        name=gen_raw.get("name", "stub-template"),
        config=gen_raw.get("config", {}),
    )
    judges = [
        ManifestJudge(name=j.get("name", "heuristic"), config=j.get("config", {}))
        for j in (judge_list if isinstance(judge_list, list) else [judge_list])
    ]

    return Manifest(
        experiment_id=data.get("experiment_id", f"exp-{uuid.uuid4().hex[:8]}"),
        author=data.get("author", ""),
        description=data.get("description", ""),
        model=model,
        generator=generator,
        judges=judges,
        seed=int(data.get("seed", 42)),
        batch_size=int(data.get("batch_size", 8)),
        num_batches=int(data.get("num_batches", 1)),
        rate_limit_rps=float(data.get("rate_limit_rps", 0)),
        output=data.get("output", "logs/experiment.jsonl"),
        raw=data,
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load a manifest from a YAML or JSON file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file format is unsupported or content is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    text = p.read_text(encoding="utf-8")

    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load YAML manifests.  "
                "Install it with: pip install pyyaml"
            )
        data = yaml.safe_load(text)
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported manifest format: {p.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a mapping / object.")

    return _parse_raw(data)
