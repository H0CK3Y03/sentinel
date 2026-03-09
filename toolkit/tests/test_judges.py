"""Tests for the heuristic judge."""

from __future__ import annotations

from redteam.judges.heuristic import HeuristicJudge
from redteam.models import ModelResponse, PromptCandidate


def _make(text: str) -> tuple[ModelResponse, PromptCandidate]:
    prompt = PromptCandidate(text="test prompt")
    resp = ModelResponse(text=text, model_id="test", prompt_id=prompt.prompt_id)
    return resp, prompt


def test_detects_refusal() -> None:
    judge = HeuristicJudge()
    judge.configure({})
    resp, prompt = _make("I'm sorry, but I cannot assist with that request.")
    verdict = judge.evaluate(resp, prompt)
    assert "refusal" in verdict.labels


def test_detects_compliance() -> None:
    judge = HeuristicJudge()
    judge.configure({})
    resp, prompt = _make("Sure! Here is the information you asked for.")
    verdict = judge.evaluate(resp, prompt)
    assert "compliance" in verdict.labels


def test_inconclusive_on_neutral() -> None:
    judge = HeuristicJudge()
    judge.configure({})
    resp, prompt = _make("The weather today is sunny.")
    verdict = judge.evaluate(resp, prompt)
    assert "inconclusive" in verdict.labels
