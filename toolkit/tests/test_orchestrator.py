"""Basic tests for the orchestrator and pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from redteam.manifest import Manifest, ManifestModel, ManifestGenerator, ManifestJudge
from redteam.orchestrator import Orchestrator


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
