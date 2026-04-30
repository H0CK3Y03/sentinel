"""Tests for the ensemble verdict aggregator."""

from __future__ import annotations

from sentinel.models import JudgeType, ModelResponse, PromptCandidate, Verdict
from sentinel.verdict import aggregate_final_verdict


def _verdict(label: str) -> Verdict:
    return Verdict(labels=[label], confidence=1.0, judge_type=JudgeType.HEURISTIC.value)


def _setup() -> tuple[PromptCandidate, ModelResponse]:
    prompt = PromptCandidate(text="prompt text")
    response = ModelResponse(text="response text", model_id="m", prompt_id=prompt.prompt_id)
    return prompt, response


def test_unanimous_compliance() -> None:
    prompt, response = _setup()
    final = aggregate_final_verdict(
        "exp", prompt, response, [_verdict("compliance")] * 3
    )
    assert final.labels == ["compliance"]
    assert final.confidence == 1.0
    assert final.judge_type == JudgeType.ENSEMBLE.value
    assert final.judge_instance_id == "ensemble"


def test_majority_refusal() -> None:
    prompt, response = _setup()
    final = aggregate_final_verdict(
        "exp",
        prompt,
        response,
        [_verdict("refusal"), _verdict("refusal"), _verdict("compliance")],
    )
    assert final.labels == ["refusal"]
    # 2 of 3 judges agreed -> confidence 2/3 rounded
    assert final.confidence == round(2 / 3, 4)


def test_tie_resolves_to_inconclusive() -> None:
    prompt, response = _setup()
    final = aggregate_final_verdict(
        "exp",
        prompt,
        response,
        [_verdict("compliance"), _verdict("refusal")],
    )
    assert final.labels == ["inconclusive"]
    # No judge said inconclusive, so the confidence reflects that.
    assert final.confidence == 0.0


def test_no_verdicts_defaults_to_inconclusive() -> None:
    prompt, response = _setup()
    final = aggregate_final_verdict("exp", prompt, response, [])
    assert final.labels == ["inconclusive"]
    assert final.confidence == 0.0


def test_unknown_labels_are_ignored() -> None:
    """Labels outside refusal/compliance/inconclusive should not skew the vote."""
    prompt, response = _setup()
    final = aggregate_final_verdict(
        "exp",
        prompt,
        response,
        [_verdict("compliance"), _verdict("something-else")],
    )
    # Only compliance counts; the other label is ignored.
    assert final.labels == ["compliance"]
