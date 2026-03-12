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

from redteam.adapters.base import ModelAdapter
from redteam.attacks.base import AttackGenerator
from redteam.judges.base import JudgeAdapter
from redteam.logger import JsonlLogger
from redteam.manifest import Manifest
from redteam.models import (
    EventType,
    LogEvent,
    ModelResponse,
    PromptCandidate,
    Verdict,
)
from redteam.plugins import create_adapter, create_generator, create_judge


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
        for batch_idx in range(m.num_batches):
            prompts = self.generator.next(batch_size=m.batch_size)
            for prompt in prompts:
                summary.total_prompts += 1
                try:
                    response = await self.adapter.generate(prompt.text)
                    response.prompt_id = prompt.prompt_id

                    # Run judge pipeline
                    verdicts: List[Verdict] = []
                    for judge in self.judges:
                        v = judge.evaluate(response, prompt)
                        v.experiment_id = m.experiment_id
                        verdicts.append(v)

                    # Classify result from first judge for summary stats
                    primary = verdicts[0] if verdicts else None
                    if primary:
                        if "refusal" in primary.labels:
                            summary.total_refusals += 1
                        elif "compliance" in primary.labels:
                            summary.total_compliances += 1
                        else:
                            summary.total_inconclusive += 1

                    # Allow adaptive generators to observe the response
                    self.generator.update(prompt, response)

                    # Persist
                    self.logger.log_trial(m.experiment_id, prompt, response, verdicts)

                except Exception as exc:
                    summary.total_errors += 1
                    self.logger.log_error(
                        m.experiment_id,
                        message=str(exc),
                        details={"prompt_id": prompt.prompt_id},
                    )

                # Rate limiting
                if m.rate_limit_rps > 0:
                    await asyncio.sleep(1.0 / m.rate_limit_rps)

        summary.elapsed_seconds = time.perf_counter() - t0

        # Log experiment end
        self.logger.log_experiment_end(m.experiment_id, summary.to_dict())
        self.logger.close()
        await self.adapter.close()

        return summary
