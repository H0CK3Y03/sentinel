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
from sentinel.manifest import Manifest, ManifestAdapter, ManifestGenerator
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


@dataclass(frozen=True)
class _Combo:
    """Adapter + generator pair with manifest metadata."""

    adapter: ModelAdapter
    adapter_cfg: ManifestAdapter
    generator: AttackGenerator
    generator_cfg: ManifestGenerator


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
        self._judge_locks = [asyncio.Lock() for _ in self.judges]
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
            if self.judges:
                verdicts = await asyncio.gather(
                    *(
                        self._evaluate_judge(judge, lock, response, prompt, experiment_id)
                        for judge, lock in zip(self.judges, self._judge_locks)
                    )
                )

            return _TrialOutcome(prompt=prompt, response=response, verdicts=verdicts)
        except Exception as exc:
            return _TrialOutcome(prompt=prompt, error=exc)

    async def _evaluate_judge(
        self,
        judge: JudgeAdapter,
        lock: asyncio.Lock,
        response: ModelResponse,
        prompt: PromptCandidate,
        experiment_id: str,
    ) -> Verdict:
        async with lock:
            verdict = await asyncio.to_thread(judge.evaluate, response, prompt)
        verdict.experiment_id = experiment_id
        verdict.judge_instance_id = getattr(judge, "instance_id", "")
        return verdict

    def _build_combos(self, m: Manifest) -> List[_Combo]:
        """Build adapterxgenerator combos with optional per-combo isolation.
        
        When force_generator_isolation is True or when max_combo_concurrency > 1
        with multiple adapters, instantiate new generator per combo for state isolation.
        """
        should_isolate = (
            m.force_generator_isolation or 
            (m.max_combo_concurrency > 1 and len(m.adapters) > 1)
        )
        
        if should_isolate:
            combos: List[_Combo] = []
            for adapter, adapter_cfg in zip(self.adapters, m.adapters):
                for generator_cfg in m.generators:
                    generator = create_generator(
                        generator_cfg.name,
                        instance_id=generator_cfg.instance_id,
                    )
                    combos.append(
                        _Combo(
                            adapter=adapter,
                            adapter_cfg=adapter_cfg,
                            generator=generator,
                            generator_cfg=generator_cfg,
                        )
                    )
            return combos

        return [
            _Combo(
                adapter=adapter,
                adapter_cfg=adapter_cfg,
                generator=generator,
                generator_cfg=generator_cfg,
            )
            for adapter, adapter_cfg in zip(self.adapters, m.adapters)
            for generator, generator_cfg in zip(self.generators, m.generators)
        ]

    @staticmethod
    def _unique_generators(combos: List[_Combo]) -> List[AttackGenerator]:
        unique: List[AttackGenerator] = []
        seen: set[int] = set()
        for combo in combos:
            generator_id = id(combo.generator)
            if generator_id in seen:
                continue
            seen.add(generator_id)
            unique.append(combo.generator)
        return unique

    @staticmethod
    def _generator_config_map(combos: List[_Combo]) -> Dict[int, ManifestGenerator]:
        config_map: Dict[int, ManifestGenerator] = {}
        for combo in combos:
            config_map.setdefault(id(combo.generator), combo.generator_cfg)
        return config_map

    @staticmethod
    def _component_health_payload(
        kind: str,
        name: str,
        instance_id: str,
        health: HealthStatus,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "component_kind": kind,
            "component_name": name,
            "component_instance_id": instance_id,
            "component_health": health.value,
            "component_diagnostics": diagnostics,
        }
        payload[f"{kind}_name"] = name
        payload[f"{kind}_instance_id"] = instance_id
        payload[f"{kind}_health"] = health.value
        payload[f"{kind}_diagnostics"] = diagnostics
        return payload

    @staticmethod
    def _component_warning_payload(
        kind: str,
        name: str,
        instance_id: str,
        health: HealthStatus,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = Orchestrator._component_health_payload(
            kind,
            name,
            instance_id,
            health,
            diagnostics,
        )
        payload.update(
            {
                "level": "warning",
                "message": f"{kind.title()} '{name}' is not fully available",
            }
        )
        return payload

    def _configure_components(
        self,
        m: Manifest,
        generators: List[AttackGenerator],
        generator_cfgs: Dict[int, ManifestGenerator],
    ) -> None:
        config_errors: Dict[str, str] = {}

        for adapter, adapter_cfg in zip(self.adapters, m.adapters):
            try:
                adapter.configure(adapter_cfg.config)
            except Exception as exc:
                config_errors[f"adapter:{adapter_cfg.instance_id}"] = str(exc)

        for generator in generators:
            generator_cfg = generator_cfgs.get(id(generator))
            if generator_cfg is None:
                continue
            try:
                generator.configure({"seed": m.seed, **generator_cfg.config})
            except Exception as exc:
                config_errors[f"generator:{generator_cfg.instance_id}"] = str(exc)

        for judge, jcfg in zip(self.judges, m.judges):
            try:
                judge.configure(jcfg.config)
            except Exception as exc:
                config_errors[f"judge:{jcfg.instance_id}"] = str(exc)

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

    async def _health_check_components(
        self,
        m: Manifest,
        generators: List[AttackGenerator],
        generator_cfgs: Dict[int, ManifestGenerator],
    ) -> None:
        health_issues: Dict[str, Dict[str, Any]] = {}

        for adapter, adapter_cfg in zip(self.adapters, m.adapters):
            health = await adapter.health_check()
            diagnostics = adapter.diagnostics()
            self.logger.log_event(
                LogEvent(
                    event_type=EventType.INFO.value,
                    experiment_id=m.experiment_id,
                    data=self._component_health_payload(
                        "adapter",
                        adapter_cfg.adapter,
                        adapter_cfg.instance_id,
                        health,
                        diagnostics,
                    ),
                )
            )
            if health != HealthStatus.OK:
                health_issues[f"adapter:{adapter_cfg.instance_id}"] = {
                    "name": adapter_cfg.adapter,
                    "health": health.value,
                    "diagnostics": diagnostics,
                }
                self.logger.log_event(
                    LogEvent(
                        event_type=EventType.INFO.value,
                        experiment_id=m.experiment_id,
                        data=self._component_warning_payload(
                            "adapter",
                            adapter_cfg.adapter,
                            adapter_cfg.instance_id,
                            health,
                            diagnostics,
                        ),
                    )
                )

        for generator in generators:
            generator_cfg = generator_cfgs.get(id(generator))
            if generator_cfg is None:
                continue
            health = await generator.health_check()
            diagnostics = generator.diagnostics()
            self.logger.log_event(
                LogEvent(
                    event_type=EventType.INFO.value,
                    experiment_id=m.experiment_id,
                    data=self._component_health_payload(
                        "generator",
                        generator_cfg.name,
                        generator_cfg.instance_id,
                        health,
                        diagnostics,
                    ),
                )
            )
            if health != HealthStatus.OK:
                health_issues[f"generator:{generator_cfg.instance_id}"] = {
                    "name": generator_cfg.name,
                    "health": health.value,
                    "diagnostics": diagnostics,
                }
                self.logger.log_event(
                    LogEvent(
                        event_type=EventType.INFO.value,
                        experiment_id=m.experiment_id,
                        data=self._component_warning_payload(
                            "generator",
                            generator_cfg.name,
                            generator_cfg.instance_id,
                            health,
                            diagnostics,
                        ),
                    )
                )

        for judge in self.judges:
            judge_health = await judge.health_check()
            diagnostics = judge.diagnostics()
            self.logger.log_event(
                LogEvent(
                    event_type=EventType.INFO.value,
                    experiment_id=m.experiment_id,
                    data=self._component_health_payload(
                        "judge",
                        judge.name,
                        judge.instance_id,
                        judge_health,
                        diagnostics,
                    ),
                )
            )
            if judge_health != HealthStatus.OK:
                health_issues[f"judge:{judge.instance_id}"] = {
                    "name": judge.name,
                    "health": judge_health.value,
                    "diagnostics": diagnostics,
                }
                self.logger.log_event(
                    LogEvent(
                        event_type=EventType.INFO.value,
                        experiment_id=m.experiment_id,
                        data=self._component_warning_payload(
                            "judge",
                            judge.name,
                            judge.instance_id,
                            judge_health,
                            diagnostics,
                        ),
                    )
                )

        if health_issues:
            error_msg = "Component validation failed:\n" + "\n".join(
                f"  {key} ({info['name']}): health={info['health']} diagnostics={info['diagnostics']}"
                for key, info in health_issues.items()
            )
            self.logger.log_error(
                m.experiment_id,
                message=error_msg,
            )
            self.logger.close()
            raise RuntimeError(error_msg)

    def _combo_supports_streaming(self, combo: _Combo, m: Manifest) -> bool:
        return m.pipeline_mode == "streaming" and combo.generator.supports_streaming()

    async def _advance_multiturn_if_needed(
        self,
        outcome: _TrialOutcome,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: _AsyncRateLimiter,
    ) -> None:
        """Advance multi-turn conversations by executing subsequent turns immediately.
        
        For generators with get_next_turn() method and conversations with a 
        conversation_id in metadata, fetches and executes the next turn without
        waiting for a new batch generation cycle.
        """
        metadata = outcome.prompt.metadata or {}
        conversation_id = metadata.get("conversation_id")
        
        # Check if generator supports multi-turn progression
        if not conversation_id or not hasattr(combo.generator, "get_next_turn"):
            return
        
        # Continue executing turns for this conversation
        turn_count = 0
        
        while turn_count < m.max_turns_per_conversation:
            next_turn_prompt = combo.generator.get_next_turn(conversation_id)
            
            if next_turn_prompt is None:
                # Conversation complete
                self.logger.log_event(
                    LogEvent(
                        event_type=EventType.INFO.value,
                        experiment_id=m.experiment_id,
                        data={
                            "level": "info",
                            "message": "Multi-turn conversation completed",
                            "conversation_id": conversation_id,
                            "total_turns": metadata.get("turn", 0) + turn_count + 1,
                        },
                    )
                )
                break
            
            # Execute next turn
            next_outcome = await self._execute_prompt(
                combo.adapter,
                next_turn_prompt,
                m.experiment_id,
                rate_limiter,
            )
            
            # Process next turn outcome
            self._process_outcome(next_outcome, combo, m, summary)
            
            turn_count += 1

    def _combo_supports_streaming(self, combo: _Combo, m: Manifest) -> bool:
        return m.pipeline_mode == "streaming" and combo.generator.supports_streaming()

    def _process_outcome(
        self,
        outcome: _TrialOutcome,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
    ) -> None:
        adapter_cfg = combo.adapter_cfg
        generator = combo.generator
        generator_cfg = combo.generator_cfg

        summary.total_prompts += 1
        if outcome.error is not None or outcome.response is None:
            summary.total_errors += 1
            self.logger.log_error(
                m.experiment_id,
                message=str(outcome.error) if outcome.error else "Unknown execution error",
                details={"prompt_id": outcome.prompt.prompt_id},
            )
            return

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

        self.metrics_collector.record_trial(
            prompt=enriched_prompt,
            response=outcome.response,
            verdicts=outcome.verdicts,
            final_verdict=final_verdict,
            attack_type=attack_type,
        )

        generator.update(outcome.prompt, outcome.response)

        self.logger.log_trial(
            m.experiment_id,
            enriched_prompt,
            outcome.response,
            outcome.verdicts,
            final_verdict=final_verdict,
        )

    async def _run_combo_batch(
        self,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: _AsyncRateLimiter,
    ) -> None:
        adapter = combo.adapter
        generator = combo.generator

        for _batch_idx in range(m.num_batches):
            prompts = await asyncio.to_thread(generator.next, m.batch_size)

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
                self._process_outcome(outcome, combo, m, summary)
                
                # Check for multi-turn conversation progression
                await self._advance_multiturn_if_needed(
                    outcome, combo, m, summary, semaphore, rate_limiter
                )

    async def _run_combo_streaming(
        self,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: _AsyncRateLimiter,
    ) -> None:
        adapter = combo.adapter
        generator = combo.generator
        next_prompts_task: asyncio.Task[List[PromptCandidate]] | None = None

        for batch_idx in range(m.num_batches):
            if next_prompts_task is None:
                prompts = await asyncio.to_thread(generator.next, m.batch_size)
            else:
                prompts = await next_prompts_task

            if batch_idx < m.num_batches - 1:
                next_prompts_task = asyncio.create_task(
                    asyncio.to_thread(generator.next, m.batch_size)
                )
            else:
                next_prompts_task = None

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
                self._process_outcome(outcome, combo, m, summary)
                
                # Note: Multi-turn progression is not applied in streaming mode
                # since streaming generators use supports_streaming() which returns False
                # for multi-turn generators (see multi_turn_conversation.py)

        if next_prompts_task is not None:
            await next_prompts_task

    async def _run_combo(
        self,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: _AsyncRateLimiter,
    ) -> None:
        if self._combo_supports_streaming(combo, m):
            await self._run_combo_streaming(combo, m, summary, semaphore, rate_limiter)
            return

        if m.pipeline_mode == "streaming" and not combo.generator.supports_streaming():
            self.logger.log_event(
                LogEvent(
                    event_type=EventType.INFO.value,
                    experiment_id=m.experiment_id,
                    data={
                        "level": "info",
                        "message": "Streaming pipeline disabled for generator requiring feedback; using batch mode.",
                        "generator_name": combo.generator_cfg.name,
                        "generator_instance_id": combo.generator_cfg.instance_id,
                    },
                )
            )

        await self._run_combo_batch(combo, m, summary, semaphore, rate_limiter)

    async def _run_combos(
        self,
        combos: List[_Combo],
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: _AsyncRateLimiter,
    ) -> None:
        combo_semaphore = asyncio.Semaphore(max(1, int(m.max_combo_concurrency)))

        async def run_limited(combo: _Combo) -> None:
            async with combo_semaphore:
                await self._run_combo(combo, m, summary, semaphore, rate_limiter)

        await asyncio.gather(*(run_limited(combo) for combo in combos))

    # -- public entry-point --------------------------------------------------

    async def run(self) -> ExperimentSummary:
        """Execute the full experiment and return aggregate metrics."""
        m = self.manifest
        summary = ExperimentSummary(experiment_id=m.experiment_id)
        t0 = time.perf_counter()

        # Log experiment start as early as possible so failures during setup
        # still leave an audit trail in the JSONL output.
        self.logger.log_experiment_start(m.experiment_id, m.to_dict())

        combos = self._build_combos(m)
        generator_cfgs = self._generator_config_map(combos)
        generators_in_use = self._unique_generators(combos)

        # Phase 1: Configure all components and collect errors
        self._configure_components(m, generators_in_use, generator_cfgs)

        # Phase 2: Health check all components and collect issues
        await self._health_check_components(m, generators_in_use, generator_cfgs)

        # Main loop: batches of prompts (with combo-level parallelism)
        semaphore = asyncio.Semaphore(max(1, int(m.max_concurrency)))
        rate_limiter = _AsyncRateLimiter(m.rate_limit_rps)
        await self._run_combos(combos, m, summary, semaphore, rate_limiter)

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
