"""Orchestrator - central experiment runner.

Consumes a `Manifest`, wires up the selected adapter, attack generator,
and judge pipeline, then executes the experiment loop:

    manifest -> generator.next() -> adapter.generate() -> judges.evaluate() -> logger.log_trial()

All provenance (manifest snapshot, prompts, raw responses, verdicts) is
written to the JSONL log store for reproducibility and later analysis.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from sentinel.model_adapters.base import ModelAdapter
from sentinel.generators.base import AttackGenerator
from sentinel.judges.base import JudgeAdapter
from sentinel.logger import JsonlLogger
from sentinel.manifest import Manifest
from sentinel.metrics import MetricsCollector
from sentinel.models import (
    EventType,
    HealthStatus,
    LogEvent,
    ModelResponse,
    PromptCandidate,
    Verdict,
)
from sentinel.plugins import create_adapter, create_generator, create_judge


# ---------------------------------------------------------------------------
# Experiment summary
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
    elapsed_seconds: float = 0.0

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
            "attack_success_rate": round(self.asr, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class _TrialOutcome:
    """Internal result bundle for a single prompt execution."""

    prompt: PromptCandidate
    response: ModelResponse | None = None
    verdicts: List[Verdict] = field(default_factory=list)
    error: Exception | None = None


class _AsyncRateLimiter:
    """Serialise request starts to respect a maximum prompts-per-second budget."""

    def __init__(self, rate_limit_rps: float) -> None:
        self._interval = 0.0 if rate_limit_rps <= 0 else 1.0 / rate_limit_rps
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return

        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            if self._next_allowed_at > now:
                await asyncio.sleep(self._next_allowed_at - now)
                now = loop.time()

            self._next_allowed_at = max(self._next_allowed_at, now) + self._interval


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Coordinate an end-to-end red-teaming experiment.

    Usage
    -----
    >>> orch = Orchestrator(manifest)
    >>> summary = asyncio.run(orch.run())
    """

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest

        # -- wire up components from the manifest ----------------------------
        self.adapters: List[ModelAdapter] = [
            create_adapter(
                name=adapter.adapter,
                model_id=adapter.model_id,
                config=adapter.config,
                instance_id=adapter.instance_id,
            )
            for adapter in manifest.adapters
        ]
        self.generators: List[AttackGenerator] = [
            create_generator(generator.name, instance_id=generator.instance_id) for generator in manifest.generators
        ]
        self.judges: List[JudgeAdapter] = [
            create_judge(judge.name, instance_id=judge.instance_id) for judge in manifest.judges
        ]
        self.logger = JsonlLogger(manifest.output)
        self.metrics_collector = MetricsCollector()

    @staticmethod
    def _aggregate_final_verdict(
        experiment_id: str,
        prompt: PromptCandidate,
        response: ModelResponse,
        verdicts: List[Verdict],
    ) -> Verdict:
        """Build a single final verdict from all judge outputs.

        Strategy: majority vote across `refusal`, `compliance`, and
        `inconclusive`. Ties resolve to `inconclusive`.
        """
        vote_counts = {"refusal": 0, "compliance": 0, "inconclusive": 0}
        for verdict in verdicts:
            labels = set(verdict.labels)
            for label in vote_counts:
                if label in labels:
                    vote_counts[label] += 1

        if not verdicts:
            winning_label = "inconclusive"
            winning_votes = 0
        else:
            max_votes = max(vote_counts.values())
            winners = [label for label, votes in vote_counts.items() if votes == max_votes]
            if len(winners) == 1:
                winning_label = winners[0]
                winning_votes = max_votes
            else:
                winning_label = "inconclusive"
                winning_votes = vote_counts["inconclusive"]

        confidence = 0.0 if not verdicts else winning_votes / len(verdicts)
        explanation = (
            f"Final verdict by majority vote: {winning_label} "
            f"(refusal={vote_counts['refusal']}, "
            f"compliance={vote_counts['compliance']}, "
            f"inconclusive={vote_counts['inconclusive']})."
        )

        return Verdict(
            experiment_id=experiment_id,
            prompt_id=prompt.prompt_id,
            model_id=response.model_id,
            judge_instance_id="ensemble",
            labels=[winning_label],
            confidence=round(confidence, 4),
            judge_type="ensemble",
            explanation=explanation,
        )

    async def _execute_prompt(
        self,
        adapter: ModelAdapter,
        prompt: PromptCandidate,
        experiment_id: str,
        rate_limiter: _AsyncRateLimiter,
    ) -> _TrialOutcome:
        await rate_limiter.acquire()

        try:
            response = await adapter.generate(prompt.text)
            response.prompt_id = prompt.prompt_id
            response.adapter_instance_id = getattr(adapter, "instance_id", "")

            verdicts: List[Verdict] = []
            for judge in self.judges:
                verdict = judge.evaluate(response, prompt)
                verdict.experiment_id = experiment_id
                verdict.judge_instance_id = getattr(judge, "instance_id", "")
                verdicts.append(verdict)

            return _TrialOutcome(prompt=prompt, response=response, verdicts=verdicts)
        except Exception as exc:
            return _TrialOutcome(prompt=prompt, error=exc)

    # -- public entry-point --------------------------------------------------

    async def run(self) -> ExperimentSummary:
        """Execute the full experiment and return aggregate metrics."""
        m = self.manifest
        summary = ExperimentSummary(experiment_id=m.experiment_id)
        t0 = time.perf_counter()

        # Log experiment start as early as possible so failures during setup
        # still leave an audit trail in the JSONL output.
        self.logger.log_experiment_start(m.experiment_id, m.to_dict())

        # Phase 1: Configure all components and collect errors
        config_errors: Dict[str, str] = {}
        
        for generator, generator_cfg in zip(self.generators, m.generators):
            try:
                generator.configure({"seed": m.seed, **generator_cfg.config})
            except Exception as exc:
                config_errors[f"generator:{generator_cfg.instance_id}"] = str(exc)
        
        for judge, jcfg in zip(self.judges, m.judges):
            try:
                judge.configure(jcfg.config)
            except Exception as exc:
                config_errors[f"judge:{jcfg.instance_id}"] = str(exc)
        
        # If any component failed to configure, fail fast with all errors
        if config_errors:
            error_msg = "Component configuration failed:\n" + "\n".join(
                f"  {k}: {v}" for k, v in config_errors.items()
            )
            self.logger.log_error(
                m.experiment_id,
                message=error_msg,
            )
            self.logger.close()
            raise RuntimeError(error_msg)

        
        # Phase 2: Health check all components and collect issues
        health_issues: Dict[str, tuple[str, Dict[str, Any]]] = {}
        
        for adapter, adapter_cfg in zip(self.adapters, m.adapters):
            health = await adapter.health_check()
            self.logger.log_event(
                LogEvent(
                    event_type=EventType.INFO.value,
                    experiment_id=m.experiment_id,
                    data={
                        "adapter_name": adapter_cfg.adapter,
                        "adapter_health": health.value,
                    },
                )
            )

        for judge in self.judges:
            judge_health = judge.health_check()
            diagnostics = judge.diagnostics()
            self.logger.log_event(
                LogEvent(
                    event_type=EventType.INFO.value,
                    experiment_id=m.experiment_id,
                    data={
                        "judge_name": judge.name,
                        "judge_health": judge_health.value,
                        "judge_diagnostics": diagnostics,
                    },
                )
            )
            if judge_health != HealthStatus.OK:
                health_issues[f"judge:{judge.instance_id}"] = (judge.name, diagnostics)
                self.logger.log_event(
                    LogEvent(
                        event_type=EventType.INFO.value,
                        experiment_id=m.experiment_id,
                        data={
                            "level": "warning",
                            "message": f"Judge '{judge.name}' is not fully available",
                            "judge_name": judge.name,
                            "judge_health": judge_health.value,
                            "judge_diagnostics": diagnostics,
                        },
                    )
                )
        
        # If any component has health issues, fail fast before wasting resources
        if health_issues:
            error_msg = "Component validation failed:\n" + "\n".join(
                f"  {k} ({name}): {diag}" for k, (name, diag) in health_issues.items()
            )
            self.logger.log_error(
                m.experiment_id,
                message=error_msg,
            )
            self.logger.close()
            raise RuntimeError(error_msg)

        # Main loop: batches of prompts
        semaphore = asyncio.Semaphore(max(1, int(m.max_concurrency)))
        rate_limiter = _AsyncRateLimiter(m.rate_limit_rps)
        for adapter, adapter_cfg in zip(self.adapters, m.adapters):
            for generator, generator_cfg in zip(self.generators, m.generators):
                for _batch_idx in range(m.num_batches):
                    prompts = generator.next(batch_size=m.batch_size)

                    async def run_limited(prompt: PromptCandidate) -> _TrialOutcome:
                        async with semaphore:
                            return await self._execute_prompt(
                                adapter,
                                prompt,
                                m.experiment_id,
                                rate_limiter,
                            )

                    outcomes = await asyncio.gather(
                        *(asyncio.create_task(run_limited(prompt)) for prompt in prompts)
                    )

                    for outcome in outcomes:
                        summary.total_prompts += 1
                        if outcome.error is not None or outcome.response is None:
                            summary.total_errors += 1
                            self.logger.log_error(
                                m.experiment_id,
                                message=str(outcome.error) if outcome.error else "Unknown execution error",
                                details={"prompt_id": outcome.prompt.prompt_id},
                            )
                            continue

                        final_verdict = self._aggregate_final_verdict(
                            experiment_id=m.experiment_id,
                            prompt=outcome.prompt,
                            response=outcome.response,
                            verdicts=outcome.verdicts,
                        )

                        if "refusal" in final_verdict.labels:
                            summary.total_refusals += 1
                        elif "compliance" in final_verdict.labels:
                            summary.total_compliances += 1
                        else:
                            summary.total_inconclusive += 1

                        metadata = dict(outcome.prompt.metadata)
                        attack_type = metadata.get("attack_type") or generator_cfg.name
                        metadata.setdefault("attack_type", attack_type)
                        metadata.setdefault("generator_name", generator_cfg.name)
                        metadata.setdefault("generator_instance_id", generator_cfg.instance_id)
                        metadata.setdefault("adapter_name", adapter_cfg.adapter)
                        metadata.setdefault("adapter_instance_id", adapter_cfg.instance_id)
                        enriched_prompt = PromptCandidate(
                            prompt_id=outcome.prompt.prompt_id,
                            text=outcome.prompt.text,
                            metadata=metadata,
                        )

                        # Record detailed metrics
                        self.metrics_collector.record_trial(
                            prompt=enriched_prompt,
                            response=outcome.response,
                            verdicts=outcome.verdicts,
                            final_verdict=final_verdict,
                            attack_type=attack_type,
                        )

                        # Allow adaptive generators to observe the response
                        generator.update(outcome.prompt, outcome.response)

                        # Persist in the original prompt order for reproducibility
                        self.logger.log_trial(
                            m.experiment_id,
                            enriched_prompt,
                            outcome.response,
                            outcome.verdicts,
                            final_verdict=final_verdict,
                        )

        summary.elapsed_seconds = time.perf_counter() - t0

        # Compute detailed aggregate metrics
        aggregate_metrics = self.metrics_collector.compute_aggregate(
            elapsed_seconds=summary.elapsed_seconds
        )

        # Log detailed metrics as final INFO event
        self.logger.log_event(
            LogEvent(
                event_type=EventType.INFO.value,
                experiment_id=m.experiment_id,
                data={
                    "detailed_metrics": aggregate_metrics.to_dict(),
                },
            )
        )

        # Log experiment end
        self.logger.log_experiment_end(m.experiment_id, summary.to_dict())
        self.logger.close()
        for adapter in self.adapters:
            await adapter.close()

        return summary
