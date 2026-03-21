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
from sentinel.models import (
    EventType,
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
        self.adapter: ModelAdapter = create_adapter(
            name=manifest.model.adapter,
            model_id=manifest.model.model_id,
            config=manifest.model.config,
        )
        self.generator: AttackGenerator = create_generator(manifest.generator.name)
        self.judges: List[JudgeAdapter] = [
            create_judge(j.name) for j in manifest.judges
        ]
        self.logger = JsonlLogger(manifest.output)

    async def _execute_prompt(
        self,
        prompt: PromptCandidate,
        experiment_id: str,
        rate_limiter: _AsyncRateLimiter,
    ) -> _TrialOutcome:
        await rate_limiter.acquire()

        try:
            response = await self.adapter.generate(prompt.text)
            response.prompt_id = prompt.prompt_id

            verdicts: List[Verdict] = []
            for judge in self.judges:
                verdict = judge.evaluate(response, prompt)
                verdict.experiment_id = experiment_id
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

        # Configure components
        self.generator.configure({"seed": m.seed, **m.generator.config})
        for judge, jcfg in zip(self.judges, m.judges):
            judge.configure(jcfg.config)

        # Log experiment start (manifest snapshot)
        self.logger.log_experiment_start(m.experiment_id, m.to_dict())

        # Health check
        health = await self.adapter.health_check()
        self.logger.log_event(
            LogEvent(
                event_type=EventType.INFO.value,
                experiment_id=m.experiment_id,
                data={"adapter_health": health.value},
            )
        )

        # Main loop: batches of prompts
        semaphore = asyncio.Semaphore(max(1, int(m.max_concurrency)))
        rate_limiter = _AsyncRateLimiter(m.rate_limit_rps)
        for batch_idx in range(m.num_batches):
            prompts = self.generator.next(batch_size=m.batch_size)
            async def run_limited(prompt: PromptCandidate) -> _TrialOutcome:
                async with semaphore:
                    return await self._execute_prompt(prompt, m.experiment_id, rate_limiter)

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

                # Classify result from first judge for summary stats
                primary = outcome.verdicts[0] if outcome.verdicts else None
                if primary:
                    if "refusal" in primary.labels:
                        summary.total_refusals += 1
                    elif "compliance" in primary.labels:
                        summary.total_compliances += 1
                    else:
                        summary.total_inconclusive += 1

                # Allow adaptive generators to observe the response
                self.generator.update(outcome.prompt, outcome.response)

                # Persist in the original prompt order for reproducibility
                self.logger.log_trial(
                    m.experiment_id,
                    outcome.prompt,
                    outcome.response,
                    outcome.verdicts,
                )

        summary.elapsed_seconds = time.perf_counter() - t0

        # Log experiment end
        self.logger.log_experiment_end(m.experiment_id, summary.to_dict())
        self.logger.close()
        await self.adapter.close()

        return summary
