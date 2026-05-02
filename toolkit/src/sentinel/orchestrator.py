"""Orchestrator - central experiment runner.

Consumes a :class:`Manifest`, wires up the selected adapter, attack
generator, and judge pipeline, then executes the experiment loop:

    manifest -> generator.next() -> adapter.generate() -> judges.evaluate() -> logger.log_trial()

All provenance (manifest snapshot, prompts, raw responses, verdicts) is
written to the JSONL log store for reproducibility and later analysis.

Long-running helpers live in three sibling modules:
* :mod:`sentinel.orchestrator_components` — combo / component-set types.
* :mod:`sentinel.orchestrator_lifecycle`  — configure + health-check phases.
* :mod:`sentinel.orchestrator_runtime`    — per-prompt execution and combos.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from sentinel.concurrency import RateLimiter
from sentinel.logger import JsonlLogger
from sentinel.manifest import Manifest
from sentinel.metrics import MetricsCollector
from sentinel.models import EventType, LogEvent, ModelResponse, PromptCandidate, Verdict
from sentinel.orchestrator_components import build_combos, collect_components
from sentinel.orchestrator_lifecycle import (
    close_components,
    configure_components,
    health_check_components,
)
from sentinel.orchestrator_runtime import RuntimeContext, run_combos
from sentinel.plugins import create_adapter, create_generator, create_judge


# ---------------------------------------------------------------------------
# Public summary type
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSummary:
    """Aggregate statistics collected during the run."""

    experiment_id: str = ""
    total_prompts: int = 0
    total_refusals: int = 0
    total_compliances: int = 0
    total_inconclusive: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        # Private running totals for latency average — not dataclass fields.
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

    def record_response(self, response: "ModelResponse") -> None:
        """Update token and latency accumulators from a successful response."""
        self.total_tokens += response.tokens
        self._latency_sum += response.latency_ms
        self._latency_count += 1

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_count:
            return 0.0
        return round(self._latency_sum / self._latency_count, 1)

    @property
    def asr(self) -> float:
        """Attack Success Rate = compliances / total (excluding errors)."""
        denom = self.total_prompts - self.total_errors
        if denom <= 0:
            return 0.0
        return self.total_compliances / denom

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "total_prompts": self.total_prompts,
            "total_refusals": self.total_refusals,
            "total_compliances": self.total_compliances,
            "total_inconclusive": self.total_inconclusive,
            "total_errors": self.total_errors,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "attack_success_rate": round(self.asr, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Type aliases for the live-panel callbacks the CLI hooks in.
TrialCallback = Callable[[int, PromptCandidate, Optional[ModelResponse], Optional[Verdict]], None]
StageCallback = Callable[[str], None]


class Orchestrator:
    """Coordinate an end-to-end red-teaming experiment.

    Usage
    -----
    >>> orch = Orchestrator(manifest)
    >>> summary = asyncio.run(orch.run())
    """

    def __init__(
        self,
        manifest: Manifest,
        on_trial_complete: Optional[TrialCallback] = None,
        on_generation_start: Optional[StageCallback] = None,
        on_adapter_start: Optional[StageCallback] = None,
        on_judge_start: Optional[StageCallback] = None,
    ) -> None:
        self.manifest = manifest
        self.on_trial_complete = on_trial_complete
        self.on_generation_start = on_generation_start
        self.on_adapter_start = on_adapter_start
        self.on_judge_start = on_judge_start

        # Shared component instances created from the manifest. ``build_combos``
        # may reuse them or create per-combo copies, depending on the manifest's
        # isolation flags.
        self.adapters = [
            create_adapter(
                name=a.adapter,
                model_id=a.model_id,
                config=a.config,
                instance_id=a.instance_id,
            )
            for a in manifest.adapters
        ]
        self.generators = [
            create_generator(g.name, instance_id=g.instance_id)
            for g in manifest.generators
        ]
        self.judges = [
            create_judge(j.name, instance_id=j.instance_id)
            for j in manifest.judges
        ]

        self.logger = JsonlLogger(manifest.output)
        self.metrics_collector = MetricsCollector()

    # -- Build helper used by tests ------------------------------------------

    def _build_combos(self, manifest: Manifest):
        """Compatibility shim — used by tests to inspect combo construction."""
        return build_combos(manifest, self.adapters, self.generators, self.judges)

    # -- Public entry-point --------------------------------------------------

    async def run(self) -> ExperimentSummary:
        """Execute the full experiment and return aggregate metrics."""
        m = self.manifest
        summary = ExperimentSummary(experiment_id=m.experiment_id)
        t0 = time.perf_counter()

        # Log experiment start before configuration so setup failures still
        # leave an audit trail in the JSONL output.
        self.logger.log_experiment_start(m.experiment_id, m.to_dict())

        combos = build_combos(m, self.adapters, self.generators, self.judges)
        components = collect_components(combos)

        # Phase 1 + 2: configure and health-check.
        configure_components(components, m.seed, self.logger, m.experiment_id)
        await health_check_components(components, self.logger, m.experiment_id)

        # Phase 3: run the experiment loop with combo and prompt parallelism.
        runtime = RuntimeContext(
            manifest=m,
            summary=summary,
            logger=self.logger,
            metrics_collector=self.metrics_collector,
            on_trial_complete=self.on_trial_complete,
            on_generation_start=self.on_generation_start,
            on_adapter_start=self.on_adapter_start,
            on_judge_start=self.on_judge_start,
        )
        semaphore = asyncio.Semaphore(max(1, int(m.max_concurrency)))
        rate_limiter = RateLimiter(m.rate_limit_rps)
        await run_combos(combos, runtime, semaphore, rate_limiter)

        summary.elapsed_seconds = time.perf_counter() - t0
        await self._finalize_run(components, summary)
        return summary

    async def _finalize_run(self, components, summary: ExperimentSummary) -> None:
        """Write final metrics, close the log, and clean up components."""
        # Add tokens consumed by generators (red LLM) and judges (LLM judges).
        summary.total_tokens += sum(
            getattr(g, "tokens_used", 0) for g in components.generators
        )
        summary.total_tokens += sum(
            getattr(j, "tokens_used", 0) for j in components.judges
        )

        m = self.manifest
        aggregate_metrics = self.metrics_collector.compute_aggregate(
            elapsed_seconds=summary.elapsed_seconds
        )
        self.logger.log_event(
            LogEvent(
                event_type=EventType.INFO.value,
                experiment_id=m.experiment_id,
                data={"detailed_metrics": aggregate_metrics.to_dict()},
            )
        )
        self.logger.log_experiment_end(m.experiment_id, summary.to_dict())
        self.logger.close()

        # Best-effort cleanup of every unique component instance we touched.
        for adapter in components.adapters:
            await adapter.close()
        await close_components(components.generators)
        await close_components(components.judges)
