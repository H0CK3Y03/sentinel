"""Plugin discovery and component registry.

Provides factory functions that map names from an experiment manifest to
concrete adapter, generator, and judge instances.  For now the registry is
hard-coded to the built-in stubs; in the future this module will also
support Python entry-point based discovery of third-party plugins.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from redteam.adapters.base import ModelAdapter
from redteam.adapters.stub import StubAdapter
from redteam.attacks.base import AttackGenerator
from redteam.attacks.stub import StubAttackGenerator
from redteam.judges.base import JudgeAdapter
from redteam.judges.heuristic import HeuristicJudge
from redteam.judges.stub import StubJudge

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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def register_adapter(name: str, cls: Type[ModelAdapter]) -> None:
    _ADAPTER_REGISTRY[name] = cls


def register_generator(name: str, cls: Type[AttackGenerator]) -> None:
    _GENERATOR_REGISTRY[name] = cls


def register_judge(name: str, cls: Type[JudgeAdapter]) -> None:
    _JUDGE_REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# Factory functions used by the orchestrator
# ---------------------------------------------------------------------------

def create_adapter(name: str, model_id: str, config: Dict[str, Any] | None = None) -> ModelAdapter:
    cls = _ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown adapter '{name}'.  Available: {list(_ADAPTER_REGISTRY)}"
        )
    return cls(model_id=model_id, config=config)


def create_generator(name: str) -> AttackGenerator:
    cls = _GENERATOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown generator '{name}'.  Available: {list(_GENERATOR_REGISTRY)}"
        )
    return cls(name=name)


def create_judge(name: str) -> JudgeAdapter:
    cls = _JUDGE_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown judge '{name}'.  Available: {list(_JUDGE_REGISTRY)}"
        )
    return cls(name=name)
