"""Tests for the local llama.cpp model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sentinel.model_adapters.llama_cpp import LlamaCppModelAdapter
from sentinel.models import HealthStatus


@dataclass
class _FakeLlama:
    result_text: str
    completion_tokens: int = 11
    last_kwargs: dict | None = None

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        return {
            "choices": [{"message": {"content": self.result_text}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": self.completion_tokens},
        }


@pytest.mark.asyncio
async def test_llama_cpp_adapter_generates_response(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"stub")

    fake_llama = _FakeLlama("Local model output")
    monkeypatch.setattr(
        "sentinel.model_adapters.llama_cpp.Llama",
        lambda **kwargs: fake_llama,
    )

    adapter = LlamaCppModelAdapter(
        model_id="qwen-local",
        config={
            "model_path": str(model_path),
            "system_prompt": "You are a test model",
            "max_tokens": 64,
            "temperature": 0.0,
        },
    )

    response = await adapter.generate("Say hello")

    assert response.model_id == "qwen-local"
    assert response.text == "Local model output"
    assert response.tokens == 11
    assert response.metadata["adapter"] == "llama-cpp"

    assert fake_llama.last_kwargs is not None
    messages = fake_llama.last_kwargs["messages"]
    assert messages[0]["content"] == "You are a test model"
    assert messages[1]["content"] == "Say hello"


@pytest.mark.asyncio
async def test_llama_cpp_adapter_returns_error_when_load_fails(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"stub")

    def _raise(**kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("insufficient memory")

    monkeypatch.setattr("sentinel.model_adapters.llama_cpp.Llama", _raise)

    adapter = LlamaCppModelAdapter(
        model_id="qwen-local",
        config={"model_path": str(model_path)},
    )

    response = await adapter.generate("Say hello")

    assert "unavailable" in response.text
    assert response.tokens == 0
    assert response.metadata.get("error")

    health = await adapter.health_check()
    assert health == HealthStatus.DEGRADED
