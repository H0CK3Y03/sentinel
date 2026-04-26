"""Experiment manifest loader and validator.

Manifests are small YAML (preferred) or JSON files that declaratively
describe every parameter of a red-teaming experiment.  The loader
validates required fields and returns a typed `Manifest` object that
the orchestrator consumes.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - YAML is optional
    yaml = None  # type: ignore[assignment]


def _new_instance_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class ManifestAdapter:
    """Model / adapter section of a manifest."""
    instance_id: str = field(default_factory=lambda: _new_instance_id("adapter"))
    adapter: str = "stub"
    model_id: str = "stub-v1"
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "adapter": self.adapter,
            "model_id": self.model_id,
            "config": self.config,
        }


@dataclass
class ManifestGenerator:
    """Attack generator section of a manifest."""
    instance_id: str = field(default_factory=lambda: _new_instance_id("generator"))
    name: str = "stub-template"
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "config": self.config,
        }


@dataclass
class ManifestJudge:
    """Single judge entry in the judge pipeline."""
    instance_id: str = field(default_factory=lambda: _new_instance_id("judge"))
    name: str = "heuristic"
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "config": self.config,
        }


@dataclass
class Manifest:
    """Fully parsed and validated experiment manifest."""

    experiment_id: str = field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:8]}")
    author: str = ""
    description: str = ""

    adapters: List[ManifestAdapter] = field(default_factory=list)
    generators: List[ManifestGenerator] = field(default_factory=list)
    judges: List[ManifestJudge] = field(default_factory=lambda: [ManifestJudge()])

    seed: int = 42
    batch_size: int = 8
    num_batches: int = 1
    max_concurrency: int = 1
    max_combo_concurrency: int = 1
    pipeline_mode: str = "batch"
    force_generator_isolation: bool = False
    max_turns_per_conversation: int = 10
    rate_limit_rps: float = 0.0  # 0 = unlimited
    output: str = "logs/experiment.jsonl"

    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise component lists and ensure at least one adapter and generator exist."""
        self.adapters = [
            _ensure_adapter_instance_id(adapter) for adapter in self.adapters
        ]
        self.generators = [
            _ensure_generator_instance_id(generator) for generator in self.generators
        ]
        self.judges = [
            _ensure_judge_instance_id(judge) for judge in self.judges
        ]
        self.adapters = _normalise_adapters(self.adapters)
        self.generators = _normalise_generators(self.generators)
        self.judges = _normalise_judges(self.judges)
        pipeline_mode = (self.pipeline_mode or "batch").lower()
        if pipeline_mode not in {"batch", "streaming"}:
            raise ValueError(
                "pipeline_mode must be 'batch' or 'streaming'"
            )
        self.pipeline_mode = pipeline_mode
        self.force_generator_isolation = bool(self.force_generator_isolation)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a plain dict (for logging)."""
        return {
            "experiment_id": self.experiment_id,
            "author": self.author,
            "description": self.description,
            "adapters": [adapter.to_dict() for adapter in self.adapters],
            "generators": [generator.to_dict() for generator in self.generators],
            "judges": [judge.to_dict() for judge in self.judges],
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "max_concurrency": self.max_concurrency,
            "max_combo_concurrency": self.max_combo_concurrency,
            "pipeline_mode": self.pipeline_mode,
            "force_generator_isolation": self.force_generator_isolation,
            "max_turns_per_conversation": self.max_turns_per_conversation,
            "rate_limit_rps": self.rate_limit_rps,
            "output": self.output,
        }


def _parse_raw(data: Dict[str, Any]) -> Manifest:
    """Build a `Manifest` from a raw dict."""
    adapters_raw = data.get("adapters")
    generators_raw = data.get("generators")
    judge_list = data.get("judges", [{}])

    adapters = _normalise_adapters(
        _parse_component_list(
            adapters_raw,
            lambda item: ManifestAdapter(
                instance_id=item.get("instance_id") or _new_instance_id("adapter"),
                adapter=item.get("adapter", "stub"),
                model_id=item.get("model_id", "stub-v1"),
                config=item.get("config", {}),
            ),
        )
    )
    generators = _normalise_generators(
        _parse_component_list(
            generators_raw,
            lambda item: ManifestGenerator(
                instance_id=item.get("instance_id") or _new_instance_id("generator"),
                name=item.get("name", "stub-template"),
                config=item.get("config", {}),
            ),
        )
    )
    judges = _parse_component_list(
        judge_list,
        lambda item: ManifestJudge(
            instance_id=item.get("instance_id") or _new_instance_id("judge"),
            name=item.get("name", "heuristic"),
            config=item.get("config", {}),
        ),
    )

    return Manifest(
        experiment_id=data.get("experiment_id", f"exp-{uuid.uuid4().hex[:8]}"),
        author=data.get("author", ""),
        description=data.get("description", ""),
        adapters=adapters,
        generators=generators,
        judges=judges,
        seed=int(data.get("seed", 42)),
        batch_size=int(data.get("batch_size", 8)),
        num_batches=int(data.get("num_batches", 1)),
        max_concurrency=max(1, int(data.get("max_concurrency", 1))),
        max_combo_concurrency=max(1, int(data.get("max_combo_concurrency", 1))),
        pipeline_mode=str(data.get("pipeline_mode", "batch")),
        force_generator_isolation=bool(data.get("force_generator_isolation", False)),
        max_turns_per_conversation=max(1, int(data.get("max_turns_per_conversation", 10))),
        rate_limit_rps=float(data.get("rate_limit_rps", 0)),
        output=data.get("output", "logs/experiment.jsonl"),
        raw=data,
    )


def _normalise_adapters(adapters: List[ManifestAdapter] | None) -> List[ManifestAdapter]:
    """Return a non-empty adapter list.

    A manifest always needs at least one model adapter.  When none is
    provided, use the built-in stub adapter as the default.
    """
    if adapters:
        return list(adapters)
    return [ManifestAdapter()]


def _normalise_generators(generators: List[ManifestGenerator] | None) -> List[ManifestGenerator]:
    """Return a non-empty generator list.

    A manifest always needs at least one attack generator.  When none is
    provided, use the built-in stub template generator as the default.
    """
    if generators:
        return list(generators)
    return [ManifestGenerator()]


def _normalise_judges(judges: List[ManifestJudge] | None) -> List[ManifestJudge]:
    """Return a non-empty judge list.

    A manifest always needs at least one judge. When none is provided, use the
    built-in heuristic judge as the default.
    """
    if judges:
        return list(judges)
    return [ManifestJudge()]


def _ensure_adapter_instance_id(adapter: ManifestAdapter) -> ManifestAdapter:
    if adapter.instance_id:
        return adapter
    return ManifestAdapter(
        instance_id=_new_instance_id("adapter"),
        adapter=adapter.adapter,
        model_id=adapter.model_id,
        config=adapter.config,
    )


def _ensure_generator_instance_id(generator: ManifestGenerator) -> ManifestGenerator:
    if generator.instance_id:
        return generator
    return ManifestGenerator(
        instance_id=_new_instance_id("generator"),
        name=generator.name,
        config=generator.config,
    )


def _ensure_judge_instance_id(judge: ManifestJudge) -> ManifestJudge:
    if judge.instance_id:
        return judge
    return ManifestJudge(
        instance_id=_new_instance_id("judge"),
        name=judge.name,
        config=judge.config,
    )


T = TypeVar("T")


def _parse_component_list(
    raw_items: Any,
    builder: Callable[[Dict[str, Any]], T],
) -> List[T] | None:
    """Parse a list of manifest components while skipping malformed entries."""
    if not isinstance(raw_items, list):
        return None
    return [builder(item) for item in raw_items if isinstance(item, dict)]


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
