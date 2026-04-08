"""Basic tests for the orchestrator and pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sentinel.generators.base import AttackGenerator
from sentinel.model_adapters.base import ModelAdapter
from sentinel.judges.base import JudgeAdapter
from sentinel.manifest import Manifest, ManifestAdapter, ManifestGenerator, ManifestJudge
from sentinel.models import HealthStatus, JudgeType, ModelResponse, PromptCandidate, Verdict
from sentinel.orchestrator import Orchestrator
from sentinel.plugins import register_adapter, register_generator, register_judge


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "test.jsonl"


@pytest.fixture
def default_manifest(tmp_log: Path) -> Manifest:
    return Manifest(
        experiment_id="test-001",
        author="test",
        description="unit test",
        adapters=[ManifestAdapter(adapter="stub", model_id="stub-v1")],
        generators=[ManifestGenerator(name="stub-template", config={"seed": 42})],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=4,
        num_batches=2,
        output=str(tmp_log),
    )


@pytest.mark.asyncio
async def test_multiple_generators_multiply_batches(tmp_path: Path) -> None:
    class StaticGenerator(AttackGenerator):
        def __init__(self, name: str = "static-generator") -> None:
            super().__init__(name=name)

        def configure(self, params: dict) -> None:
            self._configured = True

        def next(self, batch_size: int = 1) -> list[PromptCandidate]:
            return [PromptCandidate(text=f"prompt-{self.name}") for _ in range(batch_size)]

        def reset(self) -> None:
            pass

    register_generator("static-generator-a", StaticGenerator)
    register_generator("static-generator-b", StaticGenerator)

    log_path = tmp_path / "multi.jsonl"
    manifest = Manifest(
        experiment_id="multi-001",
        author="test",
        description="multi-generator test",
        adapters=[ManifestAdapter(adapter="stub", model_id="stub-v1")],
        generators=[
            ManifestGenerator(name="static-generator-a"),
            ManifestGenerator(name="static-generator-b"),
        ],
        judges=[ManifestJudge(name="heuristic")],
        seed=1,
        batch_size=2,
        num_batches=3,
        output=str(log_path),
    )

    orch = Orchestrator(manifest)
    summary = await orch.run()

    assert summary.total_prompts == 12
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    trials = [event for event in events if event.get("event_type") == "trial_result"]
    assert len(trials) == 12
    assert {trial["data"]["prompt"]["metadata"]["attack_type"] for trial in trials} == {
        "static-generator-a",
        "static-generator-b",
    }
    assert all("generator_name" in trial["data"]["prompt"]["metadata"] for trial in trials)


@pytest.mark.asyncio
async def test_full_pipeline(default_manifest: Manifest, tmp_log: Path) -> None:
    """Smoke test: run a full experiment with stubs and verify JSONL output."""
    orch = Orchestrator(default_manifest)
    summary = await orch.run()

    assert summary.total_prompts == 8  # 4 × 2
    assert summary.total_errors == 0
    assert summary.elapsed_seconds > 0

    # Verify JSONL log was written
    lines = tmp_log.read_text().strip().splitlines()
    assert len(lines) >= 10  # start + 8 trials + end

    # Every line must be valid JSON
    for line in lines:
        obj = json.loads(line)
        assert "event_type" in obj


@pytest.mark.asyncio
async def test_asr_calculation(default_manifest: Manifest) -> None:
    """ASR should be non-negative and ≤ 1."""
    orch = Orchestrator(default_manifest)
    summary = await orch.run()
    assert 0.0 <= summary.asr <= 1.0


@pytest.mark.asyncio
async def test_batch_parallelism(tmp_path: Path) -> None:
    """Prompts in a batch should execute concurrently when configured."""

    class ParallelAdapter(ModelAdapter):
        def __init__(self, model_id: str = "parallel-test", config: dict | None = None) -> None:
            super().__init__(model_id=model_id, config=config)
            self.in_flight = 0
            self.max_in_flight = 0

        async def generate(self, prompt: str, config: dict | None = None) -> ModelResponse:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            await asyncio.sleep(0.05)
            self.in_flight -= 1
            return ModelResponse(text="Sure! Here is a response.", model_id=self.model_id)

        async def health_check(self) -> HealthStatus:
            return HealthStatus.OK

    register_adapter("parallel-test", ParallelAdapter)

    manifest = Manifest(
        experiment_id="parallel-001",
        author="test",
        description="parallel execution test",
        adapters=[ManifestAdapter(adapter="parallel-test", model_id="parallel-test")],
        generators=[ManifestGenerator(name="stub-template", config={"seed": 1})],
        judges=[ManifestJudge(name="heuristic")],
        seed=1,
        batch_size=4,
        num_batches=1,
        max_concurrency=4,
        output=str(tmp_path / "parallel.jsonl"),
    )

    orch = Orchestrator(manifest)
    summary = await orch.run()

    assert summary.total_prompts == 4
    assert summary.total_errors == 0
    assert orch.adapters[0].max_in_flight > 1


@pytest.mark.asyncio
async def test_final_verdict_uses_all_judges(tmp_path: Path) -> None:
    """Conflicting judges should yield an inconclusive ensemble final verdict."""

    class AlwaysComplianceJudge(JudgeAdapter):
        def __init__(self, name: str = "always-compliance") -> None:
            super().__init__(name=name, judge_type=JudgeType.HEURISTIC)

        def configure(self, params: dict) -> None:
            self._configured = True

        def evaluate(self, response: ModelResponse, prompt: PromptCandidate) -> Verdict:
            return Verdict(
                prompt_id=prompt.prompt_id,
                model_id=response.model_id,
                labels=["compliance"],
                confidence=1.0,
                judge_type=self.judge_type.value,
                explanation="always compliance",
            )

    class AlwaysRefusalJudge(JudgeAdapter):
        def __init__(self, name: str = "always-refusal") -> None:
            super().__init__(name=name, judge_type=JudgeType.HEURISTIC)

        def configure(self, params: dict) -> None:
            self._configured = True

        def evaluate(self, response: ModelResponse, prompt: PromptCandidate) -> Verdict:
            return Verdict(
                prompt_id=prompt.prompt_id,
                model_id=response.model_id,
                labels=["refusal"],
                confidence=1.0,
                judge_type=self.judge_type.value,
                explanation="always refusal",
            )

    register_judge("always-compliance", AlwaysComplianceJudge)
    register_judge("always-refusal", AlwaysRefusalJudge)

    log_path = tmp_path / "ensemble.jsonl"
    manifest = Manifest(
        experiment_id="ensemble-001",
        author="test",
        description="ensemble verdict test",
        adapters=[ManifestAdapter(adapter="stub", model_id="stub-v1")],
        generators=[ManifestGenerator(name="stub-template", config={"seed": 1})],
        judges=[
            ManifestJudge(name="always-compliance"),
            ManifestJudge(name="always-refusal"),
        ],
        seed=1,
        batch_size=1,
        num_batches=1,
        output=str(log_path),
    )

    orch = Orchestrator(manifest)
    summary = await orch.run()

    assert summary.total_prompts == 1
    assert summary.total_inconclusive == 1
    assert summary.total_compliances == 0
    assert summary.total_refusals == 0

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    trial = next(e for e in events if e.get("event_type") == "trial_result")
    assert trial["data"]["final_verdict"]["labels"] == ["inconclusive"]
