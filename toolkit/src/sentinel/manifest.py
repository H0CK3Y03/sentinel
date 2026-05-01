"""Experiment manifest loader and validator.

Manifests are small YAML or JSON files that declaratively describe every
parameter of a red-teaming experiment. The loader validates required
fields and returns a typed :class:`Manifest` object that the orchestrator
consumes.
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


# ---------------------------------------------------------------------------
# Component dataclasses
# ---------------------------------------------------------------------------

def _new_instance_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class ManifestAdapter:
    """Model adapter section of a manifest."""
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
    force_adapter_isolation: bool = False
    force_judge_isolation: bool = False
    max_turns_per_conversation: int = 10
    stop_on_compliance: bool = True
    rate_limit_rps: float = 0.0  # 0 = unlimited
    output: str = "logs/experiment.jsonl"

    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in default IDs / lists and validate ``pipeline_mode``."""
        self.adapters = _ensure_components(
            self.adapters,
            prefix="adapter",
            replace=lambda c, new_id: ManifestAdapter(
                instance_id=new_id,
                adapter=c.adapter,
                model_id=c.model_id,
                config=c.config,
            ),
            default_factory=ManifestAdapter,
        )
        self.generators = _ensure_components(
            self.generators,
            prefix="generator",
            replace=lambda c, new_id: ManifestGenerator(
                instance_id=new_id, name=c.name, config=c.config
            ),
            default_factory=ManifestGenerator,
        )
        self.judges = _ensure_components(
            self.judges,
            prefix="judge",
            replace=lambda c, new_id: ManifestJudge(
                instance_id=new_id, name=c.name, config=c.config
            ),
            default_factory=ManifestJudge,
        )

        pipeline_mode = (self.pipeline_mode or "batch").lower()
        if pipeline_mode not in {"batch", "streaming"}:
            raise ValueError("pipeline_mode must be 'batch' or 'streaming'")
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
            "force_adapter_isolation": self.force_adapter_isolation,
            "force_judge_isolation": self.force_judge_isolation,
            "max_turns_per_conversation": self.max_turns_per_conversation,
            "stop_on_compliance": self.stop_on_compliance,
            "rate_limit_rps": self.rate_limit_rps,
            "output": self.output,
        }


# ---------------------------------------------------------------------------
# Component-list normalisation (shared by adapters / generators / judges)
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _ensure_components(
    components: List[T] | None,
    *,
    prefix: str,
    replace: Callable[[T, str], T],
    default_factory: Callable[[], T],
) -> List[T]:
    """Return *components* with: at least one entry, every entry having an ID.

    ``replace`` rebuilds an entry with a fresh ``instance_id`` (used when a
    user supplied a component without one). ``default_factory`` provides a
    sensible default when the list is empty.
    """
    if not components:
        return [default_factory()]

    return [
        component if getattr(component, "instance_id", "") else replace(
            component, _new_instance_id(prefix)
        )
        for component in components
    ]


# ---------------------------------------------------------------------------
# Raw-dict parsing
# ---------------------------------------------------------------------------

def _parse_component_list(
    raw_items: Any, builder: Callable[[Dict[str, Any]], T]
) -> List[T] | None:
    """Parse a list of manifest components, skipping malformed entries."""
    if not isinstance(raw_items, list):
        return None
    return [builder(item) for item in raw_items if isinstance(item, dict)]


def _build_adapter(item: Dict[str, Any]) -> ManifestAdapter:
    return ManifestAdapter(
        instance_id=item.get("instance_id") or _new_instance_id("adapter"),
        adapter=item.get("adapter", "stub"),
        model_id=item.get("model_id", "stub-v1"),
        config=item.get("config", {}),
    )


def _build_generator(item: Dict[str, Any]) -> ManifestGenerator:
    return ManifestGenerator(
        instance_id=item.get("instance_id") or _new_instance_id("generator"),
        name=item.get("name", "stub-template"),
        config=item.get("config", {}),
    )


def _build_judge(item: Dict[str, Any]) -> ManifestJudge:
    return ManifestJudge(
        instance_id=item.get("instance_id") or _new_instance_id("judge"),
        name=item.get("name", "heuristic"),
        config=item.get("config", {}),
    )


def _parse_raw(data: Dict[str, Any]) -> Manifest:
    """Build a :class:`Manifest` from a raw dict."""
    adapters = _parse_component_list(data.get("adapters"), _build_adapter)
    generators = _parse_component_list(data.get("generators"), _build_generator)
    judges = _parse_component_list(data.get("judges", [{}]), _build_judge) or []

    return Manifest(
        experiment_id=data.get("experiment_id", f"exp-{uuid.uuid4().hex[:8]}"),
        author=data.get("author", ""),
        description=data.get("description", ""),
        adapters=adapters or [],
        generators=generators or [],
        judges=judges,
        seed=int(data.get("seed", 42)),
        batch_size=int(data.get("batch_size", 8)),
        num_batches=int(data.get("num_batches", 1)),
        max_concurrency=max(1, int(data.get("max_concurrency", 1))),
        max_combo_concurrency=max(1, int(data.get("max_combo_concurrency", 1))),
        pipeline_mode=str(data.get("pipeline_mode", "batch")),
        force_generator_isolation=bool(data.get("force_generator_isolation", False)),
        force_adapter_isolation=bool(data.get("force_adapter_isolation", False)),
        force_judge_isolation=bool(data.get("force_judge_isolation", False)),
        max_turns_per_conversation=max(1, int(data.get("max_turns_per_conversation", 10))),
        stop_on_compliance=bool(data.get("stop_on_compliance", True)),
        rate_limit_rps=float(data.get("rate_limit_rps", 0)),
        output=data.get("output", "logs/experiment.jsonl"),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> Manifest:
    """Load a manifest from a YAML or JSON file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file format is unsupported or its content is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    text = p.read_text(encoding="utf-8")
    data = _parse_text(text, suffix=p.suffix)
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a mapping / object.")
    return _parse_raw(data)


def _parse_text(text: str, *, suffix: str) -> Any:
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load YAML manifests. "
                "Install it with: pip install pyyaml"
            )
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported manifest format: {suffix}")
