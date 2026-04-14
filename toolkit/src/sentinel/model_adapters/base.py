"""Abstract base class for model adapters.

Every concrete adapter (local llama.cpp, API, mock, custom) must subclass
`ModelAdapter` and implement the three required methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Set

from sentinel.models import (
    CostInfo,
    HealthStatus,
    InferenceMode,
    ModelResponse,
)


class ModelAdapter(ABC):
    """Uniform interface for heterogeneous LLM backends.

    Parameters
    ----------
    model_id : str
        Human-readable identifier for the backend / model combination.
    config : dict
        Backend-specific options forwarded from the experiment manifest.
    """

    def __init__(
        self,
        model_id: str,
        config: Dict[str, Any] | None = None,
        instance_id: str = "",
    ) -> None:
        self.model_id = model_id
        self.config: Dict[str, Any] = config or {}
        self.instance_id = instance_id

    # -- required interface ---------------------------------------------------

    @abstractmethod
    async def generate(self, prompt: str, config: Dict[str, Any] | None = None) -> ModelResponse:
        """Send *prompt* to the backend and return a normalised response."""

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Return the current availability of the backend."""

    # -- optional / overridable -----------------------------------------------

    async def cost_estimate(self, prompt: str, config: Dict[str, Any] | None = None) -> CostInfo:
        """Return an *estimate* of cost / token usage for *prompt*."""
        return CostInfo()

    def supported_modes(self) -> Set[InferenceMode]:
        """Declare which inference modes the adapter supports."""
        return {InferenceMode.BLACK_BOX}

    def supports_quantisation(self) -> bool:
        """Whether the adapter can serve quantised models."""
        return False

    # -- lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """Clean-up hook called at the end of an experiment."""
