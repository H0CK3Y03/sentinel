"""Local llama.cpp-backed model adapter.

Uses a GGUF model through ``llama-cpp-python`` for response generation.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Set

from llama_cpp import Llama

from sentinel.model_adapters.base import ModelAdapter
from sentinel.models import (
    CostInfo,
    HealthStatus,
    InferenceMode,
    ModelResponse,
)


class LlamaCppModelAdapter(ModelAdapter):
    """Serve model responses from a local GGUF model via llama.cpp.

    Configuration keys (via manifest model.config)
    -----------------------------------------------
    model_path : str
        Path to GGUF model file (required).
    system_prompt : str
        Optional system prompt prepended to every request.
    max_tokens : int
        Maximum generated tokens per response (default 256).
    temperature : float
        Sampling temperature (default 0.7).
    n_ctx : int
        Context window size (default 2048).
    n_gpu_layers : int
        Number of model layers to offload to GPU (default 0).
    n_threads : int
        Number of CPU threads (default: auto-detected).
    verbose : bool
        Enable llama.cpp verbose logs (default False).
    """

    def __init__(
        self,
        model_id: str = "llama-cpp-model",
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, config=config)
        self._llm: Llama | None = None
        self._model_path: Path | None = None
        self._load_error: str | None = None
        self._configured = False

        self._system_prompt = "You are a helpful assistant."
        self._max_tokens = 256
        self._temperature = 0.7
        self._n_ctx = 2048
        self._n_gpu_layers = 0
        self._n_threads = max(1, os.cpu_count() or 1)
        self._verbose = False

        if self.config:
            self._try_configure(self.config)

    def _try_configure(self, params: Dict[str, Any]) -> None:
        model_path_raw = params.get("model_path")
        if not model_path_raw:
            self._load_error = "llama-cpp adapter requires 'model_path' in model.config"
            return

        self._model_path = Path(str(model_path_raw)).expanduser()
        self._system_prompt = str(params.get("system_prompt", self._system_prompt))
        self._max_tokens = int(params.get("max_tokens", self._max_tokens))
        self._temperature = float(params.get("temperature", self._temperature))
        self._n_ctx = int(params.get("n_ctx", self._n_ctx))
        self._n_gpu_layers = int(params.get("n_gpu_layers", self._n_gpu_layers))
        self._n_threads = int(params.get("n_threads", self._n_threads))
        self._verbose = bool(params.get("verbose", self._verbose))

        if not self._model_path.exists():
            self._load_error = f"llama-cpp adapter model not found: {self._model_path}"
            self._llm = None
            self._configured = True
            return

        try:
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                n_threads=self._n_threads,
                verbose=self._verbose,
            )
            self._load_error = None
        except Exception as exc:  # pragma: no cover - covered by fallback test
            self._llm = None
            self._load_error = str(exc)

        self._configured = True

    def configure(self, params: Dict[str, Any]) -> None:
        super().configure(params)
        self._try_configure(self.config)

    async def generate(self, prompt: str, config: Dict[str, Any] | None = None) -> ModelResponse:
        if config:
            merged = dict(self.config)
            merged.update(config)
            self._try_configure(merged)
        elif not self._configured:
            self._try_configure(self.config)

        if self._llm is None:
            message = "llama.cpp adapter unavailable"
            if self._load_error:
                message = f"{message}: {self._load_error}"

            return ModelResponse(
                model_id=self.model_id,
                text=f"[Error: {message}]",
                tokens=0,
                latency_ms=0.0,
                metadata={"adapter": "llama-cpp", "error": message},
            )

        started = time.perf_counter()
        chat_result = await asyncio.to_thread(
            self._llm.create_chat_completion,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        choice = chat_result.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = str(message.get("content", "")).strip()

        usage = chat_result.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            tokens = completion_tokens
        else:
            tokens = len(text.split())

        return ModelResponse(
            model_id=self.model_id,
            text=text,
            tokens=tokens,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "adapter": "llama-cpp",
                "finish_reason": choice.get("finish_reason", ""),
                "usage": usage if isinstance(usage, dict) else {},
            },
        )

    async def health_check(self) -> HealthStatus:
        if self._llm is not None:
            return HealthStatus.OK
        if self._load_error:
            return HealthStatus.DEGRADED
        return HealthStatus.UNAVAILABLE

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "model_path": str(self._model_path) if self._model_path else "",
            "loaded": self._llm is not None,
            "load_error": self._load_error or "",
        }

    async def cost_estimate(self, prompt: str, config: Dict[str, Any] | None = None) -> CostInfo:
        # Local inference is hardware-bound; expose a token-only estimate.
        estimated_tokens = max(1, len(prompt.split())) + self._max_tokens
        return CostInfo(estimated_tokens=estimated_tokens, estimated_cost_usd=0.0)

    def supported_modes(self) -> Set[InferenceMode]:
        return {InferenceMode.BLACK_BOX}

    def supports_quantisation(self) -> bool:
        return True
