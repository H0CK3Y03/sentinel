"""Basic tests for the orchestrator and pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sentinel.model_adapters.base import ModelAdapter
from sentinel.manifest import Manifest, ManifestModel, ManifestGenerator, ManifestJudge
from sentinel.models import HealthStatus, ModelResponse
from sentinel.orchestrator import Orchestrator
from sentinel.plugins import register_adapter


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "test.jsonl"


@pytest.fixture
def default_manifest(tmp_log: Path) -> Manifest:
    return Manifest(
        experiment_id="test-001",
        author="test",
        description="unit test",
        model=ManifestModel(adapter="stub", model_id="stub-v1"),
        generator=ManifestGenerator(name="stub-template", config={"seed": 42}),
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=4,
        num_batches=2,
        output=str(tmp_log),
    )


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
        model=ManifestModel(adapter="parallel-test", model_id="parallel-test"),
        generator=ManifestGenerator(name="stub-template", config={"seed": 1}),
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
    assert orch.adapter.max_in_flight > 1
