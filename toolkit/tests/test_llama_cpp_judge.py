"""Tests for the local llama.cpp judge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentinel.judges.llama_cpp import LlamaCppJudge
from sentinel.models import ModelResponse, PromptCandidate


@dataclass
class _FakeLlama:
    result_text: str
    last_kwargs: dict | None = None

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        return {"choices": [{"message": {"content": self.result_text}}]}


def test_llama_cpp_judge_parses_json(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"stub")

    fake_llama = _FakeLlama(
        '{"label": "compliance", "confidence": 0.91, "explanation": "clear answer"}'
    )

    monkeypatch.setattr(
        "sentinel.judges.llama_cpp.Llama",
        lambda **kwargs: fake_llama,
    )

    judge = LlamaCppJudge()
    judge.configure({"model_path": str(model_path)})

    prompt = PromptCandidate(text="Give me the answer")
    response = ModelResponse(text="Sure, here it is.", model_id="stub", prompt_id=prompt.prompt_id)
    verdict = judge.evaluate(response, prompt)

    assert verdict.labels == ["compliance"]
    assert verdict.confidence == 0.91
    assert fake_llama.last_kwargs is not None
    system_message = fake_llama.last_kwargs["messages"][0]["content"]
    user_message = fake_llama.last_kwargs["messages"][1]["content"]
    assert "harmful" in system_message
    assert "different harmful details" in system_message
    assert "treat that as compliance" in user_message


def test_llama_cpp_judge_falls_back_to_label_scan(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"stub")

    monkeypatch.setattr(
        "sentinel.judges.llama_cpp.Llama",
        lambda **kwargs: _FakeLlama("The answer is refusal because it declines."),
    )

    judge = LlamaCppJudge()
    judge.configure({"model_path": str(model_path)})

    prompt = PromptCandidate(text="Do the thing")
    response = ModelResponse(text="I cannot help with that.", model_id="stub", prompt_id=prompt.prompt_id)
    verdict = judge.evaluate(response, prompt)

    assert verdict.labels == ["refusal"]


def test_llama_cpp_judge_falls_back_when_model_load_fails(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"stub")

    def _raise(**kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("insufficient memory")

    monkeypatch.setattr("sentinel.judges.llama_cpp.Llama", _raise)

    judge = LlamaCppJudge()
    judge.configure({"model_path": str(model_path)})

    prompt = PromptCandidate(text="Do the thing")
    response = ModelResponse(text="I cannot help with that.", model_id="stub", prompt_id=prompt.prompt_id)
    verdict = judge.evaluate(response, prompt)

    assert verdict.labels == ["inconclusive"]
    assert "unavailable" in verdict.explanation