"""Plugin discovery and component factories.

The toolkit has three kinds of pluggable components - model adapters,
attack generators, and judges. They share the same shape: a built-in
registry mapping ``name`` to ``class``, a Python entry-point group for
third-party plugins, and a factory function the orchestrator calls with
the manifest config. This module captures that pattern in one
:class:`_Registry` helper and uses it to build the three public APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sentinel.generators.base import AttackGenerator
from sentinel.generators.multi_turn_conversation import MultiTurnConversationGenerator
from sentinel.generators.prompt_injection import PromptInjectionGenerator
from sentinel.generators.single_turn_jailbreak import SingleTurnJailbreakGenerator
from sentinel.generators.stub import StubAttackGenerator
from sentinel.generators.token_perturbation import TokenPerturbationGenerator
from sentinel.generators.universal_trigger import UniversalTriggerGenerator
from sentinel.judges.base import JudgeAdapter
from sentinel.judges.heuristic import HeuristicJudge
from sentinel.judges.stub import StubJudge
from sentinel.model_adapters.base import ModelAdapter
from sentinel.model_adapters.stub import StubAdapter

# Optional integrations: imported defensively so the package still works
# without the llama-cpp dependency installed.
try:
    from sentinel.judges.llama_cpp import LlamaCppJudge
except ImportError:  # pragma: no cover - optional dependency
    LlamaCppJudge = None  # type: ignore[assignment]

try:
    from sentinel.model_adapters.llama_cpp import LlamaCppModelAdapter
except ImportError:  # pragma: no cover - optional dependency
    LlamaCppModelAdapter = None  # type: ignore[assignment]

try:
    from sentinel.generators.llama_cpp import LlamaCppAttackGenerator
except ImportError:  # pragma: no cover - optional dependency
    LlamaCppAttackGenerator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Generic registry helper
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _iter_entry_points(group: str):
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


@dataclass
class _Registry(Generic[T]):
    """Built-in registry plus lazy entry-point discovery for one component kind."""

    label: str
    base_class: Type[T]
    entry_point_group: str
    builtin: Dict[str, Type[T]] = field(default_factory=dict)
    _discovered: Optional[Dict[str, Type[T]]] = None

    def register(self, name: str, cls: Type[T]) -> None:
        self.builtin[name] = cls

    def names(self) -> List[str]:
        return sorted(set(self.builtin) | set(self._discover()))

    def get(self, name: str) -> Type[T]:
        cls = self.builtin.get(name) or self._discover().get(name)
        if cls is None:
            raise KeyError(f"Unknown {self.label} '{name}'.  Available: {self.names()}")
        return cls

    def _discover(self) -> Dict[str, Type[T]]:
        """Return third-party components advertised via Python entry-points (cached)."""
        if self._discovered is not None:
            return self._discovered

        discovered: Dict[str, Type[T]] = {}
        for ep in _iter_entry_points(self.entry_point_group):
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, self.base_class):
                discovered[ep.name] = obj

        self._discovered = discovered
        return discovered


# ---------------------------------------------------------------------------
# Built-in registries
# ---------------------------------------------------------------------------

_ADAPTERS: _Registry[ModelAdapter] = _Registry(
    label="adapter",
    base_class=ModelAdapter,
    entry_point_group="sentinel.model_adapters",
    builtin={"stub": StubAdapter},
)
if LlamaCppModelAdapter is not None:
    _ADAPTERS.register("llama-cpp", LlamaCppModelAdapter)

_GENERATORS: _Registry[AttackGenerator] = _Registry(
    label="generator",
    base_class=AttackGenerator,
    entry_point_group="sentinel.generators",
    builtin={
        "stub-template": StubAttackGenerator,
        "single-turn-jailbreak": SingleTurnJailbreakGenerator,
        "prompt-injection": PromptInjectionGenerator,
        "token-perturbation": TokenPerturbationGenerator,
        "universal-trigger": UniversalTriggerGenerator,
        "multi-turn-conversation": MultiTurnConversationGenerator,
    },
)
if LlamaCppAttackGenerator is not None:
    _GENERATORS.register("llama-cpp-attacker", LlamaCppAttackGenerator)

_JUDGES: _Registry[JudgeAdapter] = _Registry(
    label="judge",
    base_class=JudgeAdapter,
    entry_point_group="sentinel.judges",
    builtin={
        "heuristic": HeuristicJudge,
        "stub-judge": StubJudge,
    },
)
if LlamaCppJudge is not None:
    _JUDGES.register("llama-cpp-judge", LlamaCppJudge)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_adapter(name: str, cls: Type[ModelAdapter]) -> None:
    _ADAPTERS.register(name, cls)


def register_generator(name: str, cls: Type[AttackGenerator]) -> None:
    _GENERATORS.register(name, cls)


def register_judge(name: str, cls: Type[JudgeAdapter]) -> None:
    _JUDGES.register(name, cls)


def list_adapters() -> List[str]:
    return _ADAPTERS.names()


def list_generators() -> List[str]:
    return _GENERATORS.names()


def list_judges() -> List[str]:
    return _JUDGES.names()


# Factory functions used by the orchestrator. They differ in signature
# because each component takes different constructor arguments.

def create_adapter(
    name: str,
    model_id: str,
    config: Dict[str, Any] | None = None,
    instance_id: str = "",
) -> ModelAdapter:
    cls = _ADAPTERS.get(name)
    adapter = cls(model_id=model_id, config=config)
    adapter.instance_id = instance_id
    return adapter


def create_generator(name: str, instance_id: str = "") -> AttackGenerator:
    cls = _GENERATORS.get(name)
    generator = cls(name=name)
    generator.instance_id = instance_id
    return generator


def create_judge(name: str, instance_id: str = "") -> JudgeAdapter:
    cls = _JUDGES.get(name)
    judge = cls(name=name)
    judge.instance_id = instance_id
    return judge
