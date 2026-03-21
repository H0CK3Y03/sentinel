"""Stub LLM-based judge - placeholder for future integration.

This judge always returns an `inconclusive` verdict and serves as an
adapter-style placeholder for an LLM-based evaluation backend (e.g.,
llama.cpp judge, OpenAI moderation API, etc.).
"""

from __future__ import annotations

from typing import Any, Dict

from sentinel.judges.base import JudgeAdapter
from sentinel.models import (
    HealthStatus,
    JudgeType,
    ModelResponse,
    PromptCandidate,
    Verdict,
)


class StubJudge(JudgeAdapter):
    """Placeholder judge that always returns inconclusive.

    Meant to be replaced with a real LLM-based or classifier-based judge.
    """

    def __init__(self, name: str = "stub-judge") -> None:
        super().__init__(name=name, judge_type=JudgeType.LLM)

    def configure(self, params: Dict[str, Any]) -> None:
        self._configured = True

    def evaluate(
        self,
        response: ModelResponse,
        prompt: PromptCandidate,
    ) -> Verdict:
        return Verdict(
            experiment_id="",
            prompt_id=prompt.prompt_id,
            model_id=response.model_id,
            labels=["inconclusive"],
            confidence=0.0,
            judge_type=self.judge_type.value,
            explanation="Stub judge - no real evaluation performed.",
        )

    def health_check(self) -> HealthStatus:
        return HealthStatus.OK

    def reset(self) -> None:
        pass
