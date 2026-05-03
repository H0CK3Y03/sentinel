"""Configure and health-check phases for the orchestrator.

These phases run once per experiment, before the main prompt loop. Pulled
out of the orchestrator so the main file can stay focused on the runtime
loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from sentinel.logger import JsonlLogger
from sentinel.models import EventType, HealthStatus, LogEvent
from sentinel.orchestrator_components import ComponentSet


def configure_components(
    components: ComponentSet, seed: int, logger: JsonlLogger, experiment_id: str
) -> None:
    """Apply manifest config to every unique component.

    Aggregates errors so the user gets a single failure report rather than
    aborting on the first one. If anything fails, the log is closed and a
    ``RuntimeError`` is raised.
    """
    errors: Dict[str, str] = {}

    for adapter in components.adapters:
        cfg = components.adapter_cfgs.get(id(adapter))
        if cfg is None:
            continue
        try:
            adapter.configure(cfg.config)
        except Exception as exc:
            errors[f"adapter:{cfg.instance_id}"] = str(exc)

    for generator in components.generators:
        cfg = components.generator_cfgs.get(id(generator))
        if cfg is None:
            continue
        try:
            generator.configure({"seed": seed, **cfg.config})
        except Exception as exc:
            errors[f"generator:{cfg.instance_id}"] = str(exc)

    for judge in components.judges:
        cfg = components.judge_cfgs.get(id(judge))
        if cfg is None:
            continue
        try:
            judge.configure(cfg.config)
            judge.weight = float(cfg.weight)
        except Exception as exc:
            errors[f"judge:{cfg.instance_id}"] = str(exc)

    if errors:
        message = "Component configuration failed:\n" + "\n".join(
            f"  {key}: {value}" for key, value in errors.items()
        )
        logger.log_error(experiment_id, message=message)
        logger.close()
        raise RuntimeError(message)


async def health_check_components(
    components: ComponentSet, logger: JsonlLogger, experiment_id: str
) -> None:
    """Run ``health_check()`` on each component and log the result.

    Any component that does not return :data:`HealthStatus.OK` is collected;
    if there's at least one issue, the run is aborted with a ``RuntimeError``.
    """
    issues: Dict[str, Dict[str, Any]] = {}

    for adapter in components.adapters:
        cfg = components.adapter_cfgs.get(id(adapter))
        if cfg is None:
            continue
        await _check_one_health(
            "adapter", adapter, cfg.adapter, cfg.instance_id,
            experiment_id, logger, issues,
        )

    for generator in components.generators:
        cfg = components.generator_cfgs.get(id(generator))
        if cfg is None:
            continue
        await _check_one_health(
            "generator", generator, cfg.name, cfg.instance_id,
            experiment_id, logger, issues,
        )

    for judge in components.judges:
        cfg = components.judge_cfgs.get(id(judge))
        if cfg is None:
            continue
        await _check_one_health(
            "judge", judge, cfg.name, cfg.instance_id,
            experiment_id, logger, issues,
        )

    if issues:
        message = "Component validation failed:\n" + "\n".join(
            f"  {key} ({info['name']}): health={info['health']} diagnostics={info['diagnostics']}"
            for key, info in issues.items()
        )
        logger.log_error(experiment_id, message=message)
        logger.close()
        raise RuntimeError(message)


async def _check_one_health(
    kind: str,
    component: Any,
    cfg_name: str,
    cfg_instance_id: str,
    experiment_id: str,
    logger: JsonlLogger,
    issues: Dict[str, Dict[str, Any]],
) -> None:
    health = await component.health_check()
    diagnostics = component.diagnostics()
    payload = _health_payload(kind, cfg_name, cfg_instance_id, health, diagnostics)
    logger.log_event(
        LogEvent(event_type=EventType.INFO.value, experiment_id=experiment_id, data=payload)
    )
    if health == HealthStatus.OK:
        return

    issues[f"{kind}:{cfg_instance_id}"] = {
        "name": cfg_name,
        "health": health.value,
        "diagnostics": diagnostics,
    }
    warning = dict(payload)
    warning["level"] = "warning"
    warning["message"] = f"{kind.title()} '{cfg_name}' is not fully available"
    logger.log_event(
        LogEvent(event_type=EventType.INFO.value, experiment_id=experiment_id, data=warning)
    )


def _health_payload(
    kind: str,
    name: str,
    instance_id: str,
    health: HealthStatus,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    # Both the generic ``component_*`` keys and the type-specific ``{kind}_*``
    # keys are emitted so downstream log consumers that filter by either
    # convention keep working.
    return {
        "component_kind": kind,
        "component_name": name,
        "component_instance_id": instance_id,
        "component_health": health.value,
        "component_diagnostics": diagnostics,
        f"{kind}_name": name,
        f"{kind}_instance_id": instance_id,
        f"{kind}_health": health.value,
        f"{kind}_diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def close_components(components: List[Any]) -> None:
    """Call ``close()`` on each component if it defines one. Errors are swallowed."""
    for component in components:
        close = getattr(component, "close", None)
        if close is None:
            continue
        try:
            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                await asyncio.to_thread(close)
        except Exception:
            # Cleanup is best-effort: never fail the run because of close().
            pass
