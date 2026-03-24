"""Plugin discovery and component registry.

Provides factory functions that map names from an experiment manifest to
concrete adapter, generator, and judge instances.

Built-in components are always available. Third-party plugins can be exposed
via Python entry points and are discovered at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any, Dict, Type, TypeVar

from sentinel.model_adapters.base import ModelAdapter
from sentinel.model_adapters.stub import StubAdapter
from sentinel.generators.base import AttackGenerator
from sentinel.generators.stub import StubAttackGenerator
from sentinel.judges.base import JudgeAdapter
from sentinel.judges.heuristic import HeuristicJudge
from sentinel.judges.stub import StubJudge

try:
    from sentinel.judges.llama_cpp import LlamaCppJudge
except ImportError:  # pragma: no cover - optional dependency
    LlamaCppJudge = None  # type: ignore[assignment]

T = TypeVar("T")

_ADAPTER_ENTRYPOINT_GROUP = "sentinel.model_adapters"
_GENERATOR_ENTRYPOINT_GROUP = "sentinel.generators"
_JUDGE_ENTRYPOINT_GROUP = "sentinel.judges"

# ---------------------------------------------------------------------------
# Built-in registries (name → class)
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    "stub": StubAdapter,
}

_GENERATOR_REGISTRY: Dict[str, Type[AttackGenerator]] = {
    "stub-template": StubAttackGenerator,
}

_JUDGE_REGISTRY: Dict[str, Type[JudgeAdapter]] = {
    "heuristic": HeuristicJudge,
    "stub-judge": StubJudge,
}

if LlamaCppJudge is not None:
    _JUDGE_REGISTRY["llama-cpp-judge"] = LlamaCppJudge


def _iter_group_entry_points(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


@lru_cache(maxsize=1)
def _discover_adapters() -> Dict[str, Type[ModelAdapter]]:
    discovered: Dict[str, Type[ModelAdapter]] = {}
    for ep in _iter_group_entry_points(_ADAPTER_ENTRYPOINT_GROUP):
        obj = ep.load()
        if isinstance(obj, type) and issubclass(obj, ModelAdapter):
            discovered[ep.name] = obj
    return discovered


@lru_cache(maxsize=1)
def _discover_generators() -> Dict[str, Type[AttackGenerator]]:
    discovered: Dict[str, Type[AttackGenerator]] = {}
    for ep in _iter_group_entry_points(_GENERATOR_ENTRYPOINT_GROUP):
        obj = ep.load()
        if isinstance(obj, type) and issubclass(obj, AttackGenerator):
            discovered[ep.name] = obj
    return discovered


@lru_cache(maxsize=1)
def _discover_judges() -> Dict[str, Type[JudgeAdapter]]:
    discovered: Dict[str, Type[JudgeAdapter]] = {}
    for ep in _iter_group_entry_points(_JUDGE_ENTRYPOINT_GROUP):
        obj = ep.load()
        if isinstance(obj, type) and issubclass(obj, JudgeAdapter):
            discovered[ep.name] = obj
    return discovered


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def register_adapter(name: str, cls: Type[ModelAdapter]) -> None:
    _ADAPTER_REGISTRY[name] = cls


def register_generator(name: str, cls: Type[AttackGenerator]) -> None:
    _GENERATOR_REGISTRY[name] = cls


def register_judge(name: str, cls: Type[JudgeAdapter]) -> None:
    _JUDGE_REGISTRY[name] = cls


def list_adapters() -> list[str]:
    return sorted(set(_ADAPTER_REGISTRY) | set(_discover_adapters()))


def list_generators() -> list[str]:
    return sorted(set(_GENERATOR_REGISTRY) | set(_discover_generators()))


def list_judges() -> list[str]:
    return sorted(set(_JUDGE_REGISTRY) | set(_discover_judges()))


# ---------------------------------------------------------------------------
# Factory functions used by the orchestrator
# ---------------------------------------------------------------------------

def create_adapter(name: str, model_id: str, config: Dict[str, Any] | None = None) -> ModelAdapter:
    cls = _ADAPTER_REGISTRY.get(name) or _discover_adapters().get(name)
    if cls is None:
        raise KeyError(
            f"Unknown adapter '{name}'.  Available: {list_adapters()}"
        )
    return cls(model_id=model_id, config=config)


def create_generator(name: str) -> AttackGenerator:
    cls = _GENERATOR_REGISTRY.get(name) or _discover_generators().get(name)
    if cls is None:
        raise KeyError(
            f"Unknown generator '{name}'.  Available: {list_generators()}"
        )
    return cls(name=name)


def create_judge(name: str) -> JudgeAdapter:
    cls = _JUDGE_REGISTRY.get(name) or _discover_judges().get(name)
    if cls is None:
        raise KeyError(
            f"Unknown judge '{name}'.  Available: {list_judges()}"
        )
    return cls(name=name)
