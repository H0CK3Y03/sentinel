"""Component construction and lifecycle for the orchestrator.

A single experiment runs the cross product of every adapter and every
generator (a "combo"), so this module centralises:

* the dataclasses that bundle a combo with its judges,
* the rules for when components must be cloned per combo (isolation),
* a deduplication pass so configure / health-check is run once per concrete
  instance rather than once per combo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from sentinel.generators.base import AttackGenerator
from sentinel.judges.base import JudgeAdapter
from sentinel.manifest import Manifest, ManifestAdapter, ManifestGenerator, ManifestJudge
from sentinel.model_adapters.base import ModelAdapter
from sentinel.plugins import create_adapter, create_generator, create_judge


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Combo:
    """Adapter + generator pair with the judges that evaluate its responses."""

    adapter: ModelAdapter
    adapter_cfg: ManifestAdapter
    generator: AttackGenerator
    generator_cfg: ManifestGenerator
    judges: List[JudgeAdapter] = field(default_factory=list)
    judge_cfgs: List[ManifestJudge] = field(default_factory=list)


@dataclass
class ComponentSet:
    """Unique component instances + manifest configs collected from the combos.

    Each ``*_cfgs`` dict is keyed by ``id(component)`` so that, when isolation
    duplicates instances, every concrete instance can still find its own
    manifest entry.
    """

    adapters: List[ModelAdapter]
    adapter_cfgs: Dict[int, ManifestAdapter]
    generators: List[AttackGenerator]
    generator_cfgs: Dict[int, ManifestGenerator]
    judges: List[JudgeAdapter]
    judge_cfgs: Dict[int, ManifestJudge]


# ---------------------------------------------------------------------------
# Combo construction
# ---------------------------------------------------------------------------

def build_combos(
    manifest: Manifest,
    shared_adapters: List[ModelAdapter],
    shared_generators: List[AttackGenerator],
    shared_judges: List[JudgeAdapter],
) -> List[Combo]:
    """Build adapter×generator combos with optional per-combo isolation.

    When isolation is requested (or implied by combo concurrency with
    multiple adapters), fresh component instances are created per combo so
    that stateful generators / judges / adapters do not share mutable state
    across concurrent runs. Otherwise, the shared instances are reused.
    """
    isolate_generators = (
        manifest.force_generator_isolation
        or (manifest.max_combo_concurrency > 1 and len(manifest.adapters) > 1)
    )
    isolate_adapters = manifest.force_adapter_isolation
    isolate_judges = manifest.force_judge_isolation

    combos: List[Combo] = []
    for shared_adapter, adapter_cfg in zip(shared_adapters, manifest.adapters):
        for shared_generator, generator_cfg in zip(shared_generators, manifest.generators):
            adapter = _adapter_for_combo(shared_adapter, adapter_cfg, isolate_adapters)
            generator = _generator_for_combo(shared_generator, generator_cfg, isolate_generators)
            judges = _judges_for_combo(shared_judges, manifest.judges, isolate_judges)
            combos.append(
                Combo(
                    adapter=adapter,
                    adapter_cfg=adapter_cfg,
                    generator=generator,
                    generator_cfg=generator_cfg,
                    judges=judges,
                    judge_cfgs=list(manifest.judges),
                )
            )
    return combos


def _adapter_for_combo(
    shared: ModelAdapter, cfg: ManifestAdapter, isolate: bool
) -> ModelAdapter:
    if not isolate:
        return shared
    return create_adapter(
        name=cfg.adapter,
        model_id=cfg.model_id,
        config=cfg.config,
        instance_id=cfg.instance_id,
    )


def _generator_for_combo(
    shared: AttackGenerator, cfg: ManifestGenerator, isolate: bool
) -> AttackGenerator:
    if not isolate:
        return shared
    return create_generator(cfg.name, instance_id=cfg.instance_id)


def _judges_for_combo(
    shared_judges: List[JudgeAdapter],
    manifest_judges: List[ManifestJudge],
    isolate: bool,
) -> List[JudgeAdapter]:
    if not isolate:
        return list(shared_judges)
    return [create_judge(j.name, instance_id=j.instance_id) for j in manifest_judges]


# ---------------------------------------------------------------------------
# Component deduplication
# ---------------------------------------------------------------------------

def collect_components(combos: List[Combo]) -> ComponentSet:
    """Deduplicate the components across all combos for configure / health-check."""
    return ComponentSet(
        adapters=_unique_by_id([c.adapter for c in combos]),
        adapter_cfgs=_config_map_by_id(
            [c.adapter for c in combos], [c.adapter_cfg for c in combos]
        ),
        generators=_unique_by_id([c.generator for c in combos]),
        generator_cfgs=_config_map_by_id(
            [c.generator for c in combos], [c.generator_cfg for c in combos]
        ),
        judges=_unique_by_id([j for c in combos for j in c.judges]),
        judge_cfgs=_config_map_by_id(
            [j for c in combos for j in c.judges],
            [jc for c in combos for jc in c.judge_cfgs],
        ),
    )


def _unique_by_id(items: List[Any]) -> List[Any]:
    """Return *items* with duplicates (by ``id()``) removed, order preserved."""
    seen: set[int] = set()
    unique: List[Any] = []
    for item in items:
        if id(item) in seen:
            continue
        seen.add(id(item))
        unique.append(item)
    return unique


def _config_map_by_id(
    components: List[Any], configs: List[Any]
) -> Dict[int, Any]:
    """Map ``id(component) -> manifest config``, taking the first sighting only."""
    config_map: Dict[int, Any] = {}
    for component, cfg in zip(components, configs):
        config_map.setdefault(id(component), cfg)
    return config_map
