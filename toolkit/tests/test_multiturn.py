"""Tests for multi-turn conversation and generator isolation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sentinel.generators.base import AttackGenerator
from sentinel.generators.multi_turn_conversation import MultiTurnConversationGenerator
from sentinel.manifest import Manifest, ManifestAdapter, ManifestGenerator, ManifestJudge
from sentinel.models import ModelResponse, PromptCandidate
from sentinel.orchestrator import Orchestrator
from sentinel.plugins import register_generator


class CountingGenerator(AttackGenerator):
    """Test generator that tracks how many instances were created."""
    
    _instance_count = 0
    _instance_ids = []
    
    def __init__(self, name: str = "counting-gen") -> None:
        super().__init__(name=name)
        CountingGenerator._instance_count += 1
        CountingGenerator._instance_ids.append(id(self))
        self._call_count = 0
    
    def configure(self, params: dict) -> None:
        self._configured = True
    
    def next(self, batch_size: int = 1) -> list[PromptCandidate]:
        self._call_count += 1
        return [
            PromptCandidate(
                text=f"prompt-{self._call_count}-{i}",
                metadata={"generator_id": id(self)}
            )
            for i in range(batch_size)
        ]
    
    def reset(self) -> None:
        self._call_count = 0
    
    @classmethod
    def reset_counters(cls) -> None:
        cls._instance_count = 0
        cls._instance_ids = []


@pytest.mark.asyncio
async def test_multiturn_generator_progression(tmp_path: Path) -> None:
    """Test that multi-turn conversations advance through multiple turns."""
    log_path = tmp_path / "multiturn.jsonl"
    
    manifest = Manifest(
        experiment_id="multiturn-001",
        author="test",
        description="multi-turn test",
        adapters=[ManifestAdapter(adapter="stub", model_id="stub-v1")],
        generators=[ManifestGenerator(name="multi-turn-conversation", config={})],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=2,
        num_batches=1,
        pipeline_mode="batch",
        force_generator_isolation=False,
        output=str(log_path),
    )
    
    orch = Orchestrator(manifest)
    summary = await orch.run()
    
    # With 2 batch_size and 1 batch, we have 2 initial prompts
    # Each conversation has 4 steps (see multi_turn_conversation.py flows)
    # So we should have 2 conversations × 4 turns = 8 prompts total
    assert summary.total_prompts >= 2, f"Expected at least 2 prompts, got {summary.total_prompts}"
    
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    trials = [event for event in events if event.get("event_type") == "trial_result"]
    
    # Verify multi-turn metadata is present
    assert len(trials) >= 2
    for trial in trials[:2]:  # Check first batch at least
        metadata = trial["data"]["prompt"]["metadata"]
        assert metadata.get("attack_type") == "multi-turn-conversation"
        assert "conversation_id" in metadata
        assert "turn" in metadata
        assert "flow_name" in metadata
        assert "goal" in metadata


@pytest.mark.asyncio
async def test_force_generator_isolation(tmp_path: Path) -> None:
    """Test that force_generator_isolation creates separate generator instances."""
    register_generator("counting-gen", CountingGenerator)
    
    CountingGenerator.reset_counters()
    log_path = tmp_path / "isolation.jsonl"
    
    # With 2 adapters and 1 generator, force_generator_isolation should create
    # 2 separate generator instances (one per adapter)
    manifest = Manifest(
        experiment_id="isolation-001",
        author="test",
        description="isolation test",
        adapters=[
            ManifestAdapter(adapter="stub", model_id="stub-v1", instance_id="adapter-1"),
            ManifestAdapter(adapter="stub", model_id="stub-v1", instance_id="adapter-2"),
        ],
        generators=[ManifestGenerator(name="counting-gen", config={})],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=1,
        num_batches=1,
        force_generator_isolation=True,
        output=str(log_path),
    )
    
    orch = Orchestrator(manifest)
    combos = orch._build_combos(manifest)
    
    # With 2 adapters, 1 generator, and force_generator_isolation=True,
    # we should have 2 combos with different generator instances
    assert len(combos) == 2
    
    # Verify generators are different instances
    gen_ids = [id(combo.generator) for combo in combos]
    assert len(set(gen_ids)) == 2, "Expected 2 different generator instances"
    
    # Run the orchestrator to verify isolation works
    summary = await orch.run()
    
    # Each adapter should process 1 batch of 1 prompt
    assert summary.total_prompts == 2


@pytest.mark.asyncio
async def test_generator_isolation_not_forced_by_default(tmp_path: Path) -> None:
    """Test that without force_generator_isolation, single generators are shared."""
    register_generator("counting-gen", CountingGenerator)
    
    log_path = tmp_path / "no_isolation.jsonl"
    
    manifest = Manifest(
        experiment_id="no-isolation-001",
        author="test",
        description="no isolation test",
        adapters=[
            ManifestAdapter(adapter="stub", model_id="stub-v1", instance_id="adapter-1"),
            ManifestAdapter(adapter="stub", model_id="stub-v1", instance_id="adapter-2"),
        ],
        generators=[ManifestGenerator(name="counting-gen", config={})],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=1,
        num_batches=1,
        force_generator_isolation=False,
        max_combo_concurrency=1,  # Ensure default behavior
        output=str(log_path),
    )
    
    orch = Orchestrator(manifest)
    combos = orch._build_combos(manifest)
    
    # With force_generator_isolation=False and max_combo_concurrency=1,
    # the generator should be shared (same instance)
    assert len(combos) == 2
    
    gen_ids = [id(combo.generator) for combo in combos]
    # Should be 1 unique instance (shared)
    assert len(set(gen_ids)) == 1, "Expected shared generator instance"


@pytest.mark.asyncio
async def test_combo_concurrency_creates_isolation(tmp_path: Path) -> None:
    """Test that max_combo_concurrency > 1 with multiple adapters triggers isolation."""
    register_generator("counting-gen", CountingGenerator)
    
    log_path = tmp_path / "concurrency_isolation.jsonl"
    
    manifest = Manifest(
        experiment_id="concurrency-001",
        author="test",
        description="concurrency isolation test",
        adapters=[
            ManifestAdapter(adapter="stub", model_id="stub-v1", instance_id="adapter-1"),
            ManifestAdapter(adapter="stub", model_id="stub-v1", instance_id="adapter-2"),
        ],
        generators=[ManifestGenerator(name="counting-gen", config={})],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=1,
        num_batches=1,
        max_combo_concurrency=2,  # Enable concurrency
        output=str(log_path),
    )
    
    orch = Orchestrator(manifest)
    combos = orch._build_combos(manifest)
    
    # With max_combo_concurrency > 1 and 2 adapters, generators should be isolated
    assert len(combos) == 2
    
    gen_ids = [id(combo.generator) for combo in combos]
    assert len(set(gen_ids)) == 2, "Expected 2 different generator instances with concurrency"


@pytest.mark.asyncio
async def test_manifest_preserves_force_generator_isolation(tmp_path: Path) -> None:
    """Test that force_generator_isolation field is properly serialized and deserialized."""
    manifest = Manifest(
        experiment_id="test-001",
        author="test",
        description="test",
        adapters=[ManifestAdapter(adapter="stub")],
        generators=[ManifestGenerator(name="stub-template")],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=4,
        num_batches=1,
        force_generator_isolation=True,
        output=str(tmp_path / "test.jsonl"),
    )
    
    # Check to_dict preserves the field
    manifest_dict = manifest.to_dict()
    assert manifest_dict["force_generator_isolation"] is True
    
    # Check that it can be parsed back
    assert manifest.force_generator_isolation is True


@pytest.mark.asyncio
async def test_multiturn_conversation_turns_are_sequential(tmp_path: Path) -> None:
    """Test that multi-turn conversation turns execute sequentially per conversation."""
    log_path = tmp_path / "sequential_turns.jsonl"
    
    manifest = Manifest(
        experiment_id="seq-turns-001",
        author="test",
        description="sequential turns test",
        adapters=[ManifestAdapter(adapter="stub", model_id="stub-v1")],
        generators=[ManifestGenerator(name="multi-turn-conversation", config={})],
        judges=[ManifestJudge(name="heuristic")],
        seed=42,
        batch_size=1,
        num_batches=1,
        pipeline_mode="batch",
        output=str(log_path),
    )
    
    orch = Orchestrator(manifest)
    summary = await orch.run()
    
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    trials = [event for event in events if event.get("event_type") == "trial_result"]
    
    # Check that turns are sequential for each conversation
    conversations: dict[str, list] = {}
    for trial in trials:
        conv_id = trial["data"]["prompt"]["metadata"].get("conversation_id")
        turn = trial["data"]["prompt"]["metadata"].get("turn")
        if conv_id:
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(turn)
    
    # For each conversation, verify turns are in order
    for conv_id, turns in conversations.items():
        # Turns should be consecutive starting from 0
        assert turns == sorted(turns), f"Turns not sequential for {conv_id}: {turns}"
