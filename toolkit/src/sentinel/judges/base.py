"""Abstract base class for judge adapters.

Each judge receives a model response together with the original prompt and
returns a structured `Verdict`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sentinel.models import (
    HealthStatus,
    JudgeType,
    ModelResponse,
    PromptCandidate,
    Verdict,
)


class JudgeAdapter(ABC):
    """Uniform interface for evaluation / judging plugins.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    judge_type : JudgeType
        Category label written into every verdict.
    """

    def __init__(
        self,
        name: str = "base",
        judge_type: JudgeType = JudgeType.HEURISTIC,
        instance_id: str = "",
    ) -> None:
        self.name = name
        self.judge_type = judge_type
        self.instance_id = instance_id
        self._configured = False
        self.tokens_used: int = 0

    # -- required interface ---------------------------------------------------

    @abstractmethod
    def configure(self, params: Dict[str, Any]) -> None:
        """Accept per-experiment configuration from the manifest."""

    @abstractmethod
    def evaluate(
        self,
        response: ModelResponse,
        prompt: PromptCandidate,
    ) -> Verdict:
        """Produce a structured verdict for *response* given *prompt*."""

    # -- optional / overridable -----------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Return the current availability of the judge backend."""
        return HealthStatus.OK

    def diagnostics(self) -> Dict[str, Any]:
        """Optional runtime diagnostics for logging and troubleshooting."""
        return {}

    def reset(self) -> None:
        """Reset any accumulated state between experiments."""
