"""Tests for manifest loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.manifest import load_manifest, Manifest


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        "experiment_id: yaml-test\n"
        "model:\n  adapter: stub\n  model_id: stub-v1\n"
        "generators:\n"
        "  - name: stub-template\n"
        "  - name: single-turn-jailbreak\n"
        "judges:\n  - name: heuristic\n"
        "batch_size: 4\n"
        "num_batches: 1\n"
        "output: /tmp/test.jsonl\n"
    )
    m = load_manifest(p)
    assert isinstance(m, Manifest)
    assert m.experiment_id == "yaml-test"
    assert m.model.adapter == "stub"
    assert len(m.generators) == 2
    assert m.generator.name == "stub-template"


def test_load_json(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "experiment_id": "json-test",
        "model": {"adapter": "stub"},
        "batch_size": 2,
    }))
    m = load_manifest(p)
    assert m.experiment_id == "json-test"


def test_load_legacy_single_generator(tmp_path: Path) -> None:
    p = tmp_path / "legacy.yaml"
    p.write_text(
        "experiment_id: legacy-test\n"
        "generator:\n  name: stub-template\n"
        "judges:\n  - name: heuristic\n"
    )
    m = load_manifest(p)
    assert len(m.generators) == 1
    assert m.generator.name == "stub-template"


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest("/nonexistent/manifest.yaml")
