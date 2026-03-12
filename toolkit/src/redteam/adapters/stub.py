"""Stub / mock model adapter for testing and CI.

Returns deterministic responses so that the orchestrator, judges
and logging pipeline can be exercised without a real LLM backend.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Set

from redteam.adapters.base import ModelAdapter
from redteam.models import (
    CostInfo,
    HealthStatus,
    InferenceMode,
    ModelResponse,
)

# A small bank of responses keyed by simple heuristics on the prompt.
_REFUSAL = (
    "I'm sorry, but I can't assist with that request. "
    "If you have another question, I'd be happy to help."
)
_COMPLIANT = (
    "Sure! Here is some information on the topic you asked about. "
    "This is a stub response for testing purposes."
)


class StubAdapter(ModelAdapter):
    """Deterministic mock adapter.

    Behaviour
    ---------
    * If the prompt contains the word `"refuse"` -> returns a refusal.
    * Otherwise -> returns a compliant-looking stub response.
    * Latency is simulated as a configurable constant (default 50 ms).
    """

    def __init__(
        self,
        model_id: str = "stub-v1",
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, config=config)
        self._latency_ms: float = float(self.config.get("latency_ms", 50))
        # Allow deterministic token count via config or fall back to hash
        self._fixed_tokens: int | None = self.config.get("fixed_tokens")

    # -- ModelAdapter interface -----------------------------------------------

    async def generate(self, prompt: str, config: Dict[str, Any] | None = None) -> ModelResponse:
        start = time.perf_counter()

        # Decide on response text
        if "refuse" in prompt.lower():
            text = _REFUSAL
        else:
            text = _COMPLIANT

        elapsed = (time.perf_counter() - start) * 1000 + self._latency_ms
        tokens = self._fixed_tokens or (len(text.split()) + 1)

        return ModelResponse(
            prompt_id="",  # filled in by orchestrator
            model_id=self.model_id,
            text=text,
            tokens=tokens,
            latency_ms=round(elapsed, 2),
            metadata={"stub": True},
        )

    async def health_check(self) -> HealthStatus:
        return HealthStatus.OK

    async def cost_estimate(self, prompt: str, config: Dict[str, Any] | None = None) -> CostInfo:
        return CostInfo(estimated_tokens=len(prompt.split()), estimated_cost_usd=0.0)

    def supported_modes(self) -> Set[InferenceMode]:
        return {InferenceMode.BLACK_BOX}

    def supports_quantisation(self) -> bool:
        return False
