"""Abstract base class for attack generators.

Every attack module (template, search-based, LLM-driven …) must subclass
`AttackGenerator` and implement the required methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from redteam.models import InferenceMode, ModelResponse, PromptCandidate


class AttackGenerator(ABC):
    """Plugin interface for adversarial prompt generators.

    Parameters
    ----------
    name : str
        Human-readable identifier for this generator instance.
    """

    def __init__(self, name: str = "base") -> None:
        self.name = name
        self._configured = False

    # -- required interface ---------------------------------------------------

    @abstractmethod
    def configure(self, params: Dict[str, Any]) -> None:
        """Accept declarative configuration from the experiment manifest.

        Implementations should store seeds, constraints, and generation
        parameters, and set `self._configured = True`.
        """

    @abstractmethod
    def next(self, batch_size: int = 1) -> List[PromptCandidate]:
        """Return the next batch of candidate adversarial prompts."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (counters, caches, RNG) for a fresh run."""

    # -- optional / overridable -----------------------------------------------

    def requires_white_box(self) -> bool:
        """True if the generator needs white-box model access (logits etc.)."""
        return False

    def supported_modes(self) -> set[InferenceMode]:
        """Inference modes this generator can work with."""
        return {InferenceMode.BLACK_BOX}

    def update(self, prompt: PromptCandidate, response: ModelResponse) -> None:
        """Stateful callback: update internal policy given a model response.

        Used by multi-turn or adaptive attacks. Default is a no-op.
        """

    def sanity_check(self) -> bool:
        """Lightweight self-test to verify the generator is operational.

        Returns True if the generator can produce at least one prompt
        without errors.
        """
        try:
            candidates = self.next(batch_size=1)
            return len(candidates) > 0 and all(
                isinstance(c, PromptCandidate) for c in candidates
            )
        except Exception:
            return False
