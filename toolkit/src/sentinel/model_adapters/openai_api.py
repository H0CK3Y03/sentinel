"""OpenAI-compatible REST API model adapter.

Works with any provider that speaks the OpenAI chat-completion protocol:
e-INFRA CZ (https://llm.ai.e-infra.cz/v1), OpenAI, Azure OpenAI, vLLM, etc.

The API key is never hard-coded.  Resolution order:
    1. ``api_key`` key in the manifest config block
    2. ``SENTINEL_API_KEY`` environment variable
    3. ``OPENAI_API_KEY`` environment variable
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Set

import httpx

from sentinel.model_adapters.base import ModelAdapter
from sentinel.models import CostInfo, HealthStatus, InferenceMode, ModelResponse

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIApiAdapter(ModelAdapter):
    """OpenAI-compatible chat-completion adapter.

    Configuration keys (via manifest ``model.config``)
    ---------------------------------------------------
    base_url : str
        API base URL, e.g. ``https://llm.ai.e-infra.cz/v1``.
        Defaults to the official OpenAI endpoint.
    model : str
        Model name the endpoint should serve, e.g. ``deepseek-r1-distill-qwen-32b``.
        Falls back to ``model_id`` when not set.
    api_key : str
        Auth token.  Prefer environment variables over embedding a key in YAML.
    system_prompt : str
        System message prepended to every request (default: generic assistant).
    max_tokens : int
        Maximum generated tokens per response (default 512).
    temperature : float
        Sampling temperature (default 0.7).
    timeout : float
        HTTP request timeout in seconds (default 60).
    """

    def __init__(
        self,
        model_id: str = "openai-api",
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, config=config)
        self._base_url: str = _DEFAULT_BASE_URL
        self._model: str = model_id
        self._api_key: str | None = None
        self._system_prompt: str = "You are a helpful assistant."
        self._max_tokens: int = 512
        self._temperature: float = 0.7
        self._timeout: float = 60.0
        self._strip_think_tags: bool = False
        self._config_error: str | None = None

        if self.config:
            self._apply_config(self.config)

    # -- configuration --------------------------------------------------------

    def _apply_config(self, params: Dict[str, Any]) -> None:
        self._base_url = str(params.get("base_url", self._base_url)).rstrip("/")
        self._model = str(params.get("model", self._model))
        self._system_prompt = str(params.get("system_prompt", self._system_prompt))
        self._max_tokens = int(params.get("max_tokens", self._max_tokens))
        self._temperature = float(params.get("temperature", self._temperature))
        self._timeout = float(params.get("timeout", self._timeout))
        self._strip_think_tags = bool(params.get("strip_think_tags", self._strip_think_tags))

        raw_key = params.get("api_key")
        if raw_key:
            self._api_key = str(raw_key)
        else:
            self._api_key = (
                os.environ.get("SENTINEL_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

        self._config_error = (
            None
            if self._api_key
            else (
                "No API key found.  Set 'api_key' in the manifest config, "
                "or export SENTINEL_API_KEY / OPENAI_API_KEY."
            )
        )
        self._configured = True

    def configure(self, params: Dict[str, Any]) -> None:
        super().configure(params)
        self._apply_config(self.config)

    # -- inference ------------------------------------------------------------

    async def generate(self, prompt: str, config: Dict[str, Any] | None = None) -> ModelResponse:
        if config:
            self._apply_config({**self.config, **config})
        elif not self._configured:
            self._apply_config(self.config)

        if self._config_error:
            return ModelResponse(
                model_id=self.model_id,
                text=f"[Error: {self._config_error}]",
                tokens=0,
                latency_ms=0.0,
                is_error=True,
                metadata={"adapter": "openai-api", "error": self._config_error},
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            return self._error_response(error, time.perf_counter() - started)
        except Exception as exc:
            return self._error_response(str(exc), time.perf_counter() - started)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = str(message.get("content", "")).strip()
        if self._strip_think_tags:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        usage = data.get("usage", {})
        tokens = (
            usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), int)
            else len(text.split())
        )

        return ModelResponse(
            model_id=self.model_id,
            text=text,
            tokens=tokens,
            latency_ms=elapsed_ms,
            metadata={
                "adapter": "openai-api",
                "base_url": self._base_url,
                "model": self._model,
                "finish_reason": choice.get("finish_reason", "") if isinstance(choice, dict) else "",
                "usage": usage if isinstance(usage, dict) else {},
            },
        )

    def _error_response(self, error: str, elapsed_s: float) -> ModelResponse:
        return ModelResponse(
            model_id=self.model_id,
            text=f"[Error: {error}]",
            tokens=0,
            latency_ms=round(elapsed_s * 1000, 2),
            is_error=True,
            metadata={"adapter": "openai-api", "error": error},
        )

    # -- health / diagnostics -------------------------------------------------

    async def health_check(self) -> HealthStatus:
        if self._config_error:
            return HealthStatus.DEGRADED
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return HealthStatus.OK if resp.status_code < 500 else HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNAVAILABLE

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "base_url": self._base_url,
            "model": self._model,
            "api_key_set": bool(self._api_key),
            "config_error": self._config_error or "",
        }

    # -- cost / capabilities --------------------------------------------------

    async def cost_estimate(self, prompt: str, config: Dict[str, Any] | None = None) -> CostInfo:
        estimated_tokens = max(1, len(prompt.split())) + self._max_tokens
        return CostInfo(estimated_tokens=estimated_tokens, estimated_cost_usd=0.0)

    def supported_modes(self) -> Set[InferenceMode]:
        return {InferenceMode.BLACK_BOX}
