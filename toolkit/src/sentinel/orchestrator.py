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
from typing import Any, Callable, Dict, List, Optional

from sentinel.concurrency import RateLimiter
from sentinel.generators.base import AttackGenerator
from sentinel.judges.base import JudgeAdapter
from sentinel.logger import JsonlLogger
from sentinel.manifest import Manifest, ManifestAdapter, ManifestGenerator, ManifestJudge
from sentinel.metrics import MetricsCollector
from sentinel.model_adapters.base import ModelAdapter
from sentinel.models import (
    EventType,
    HealthStatus,
    LogEvent,
    ModelResponse,
    PromptCandidate,
    Verdict,
)
from sentinel.plugins import create_adapter, create_generator, create_judge
from sentinel.verdict import aggregate_final_verdict


# ---------------------------------------------------------------------------
# Data containers
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
    """Adapter + generator pair with the judges that evaluate its responses."""

    adapter: ModelAdapter
    adapter_cfg: ManifestAdapter
    generator: AttackGenerator
    generator_cfg: ManifestGenerator
    judges: List[JudgeAdapter] = field(default_factory=list)
    judge_cfgs: List[ManifestJudge] = field(default_factory=list)


@dataclass
class _ComponentSet:
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
# Orchestrator
# ---------------------------------------------------------------------------

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
        on_trial_complete: Optional[
            Callable[[int, PromptCandidate, Optional[ModelResponse], Optional[Verdict]], None]
        ] = None,
    ) -> None:
        self.manifest = manifest
        self.on_trial_complete = on_trial_complete

        # Shared component instances created from the manifest. `_build_combos`
        # may reuse them or create per-combo copies, depending on the manifest's
        # isolation flags.
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
            create_generator(generator.name, instance_id=generator.instance_id)
            for generator in manifest.generators
        ]
        self.judges: List[JudgeAdapter] = [
            create_judge(judge.name, instance_id=judge.instance_id)
            for judge in manifest.judges
        ]

        # One asyncio.Lock per *unique* judge instance, keyed by id(). Judges
        # that wrap a single underlying model are not safe to call concurrently;
        # the lock makes evaluation serial per instance while still allowing
        # different judge instances to run in parallel.
        self._judge_locks: Dict[int, asyncio.Lock] = {
            id(j): asyncio.Lock() for j in self.judges
        }
        self.logger = JsonlLogger(manifest.output)
        self.metrics_collector = MetricsCollector()

    # -- per-prompt execution ------------------------------------------------

    async def _execute_prompt(
        self,
        adapter: ModelAdapter,
        prompt: PromptCandidate,
        experiment_id: str,
        rate_limiter: RateLimiter,
        judges: List[JudgeAdapter],
    ) -> _TrialOutcome:
        await rate_limiter.acquire()

        try:
            response = await adapter.generate(prompt.text)
            response.prompt_id = prompt.prompt_id
            response.adapter_instance_id = getattr(adapter, "instance_id", "")

            verdicts: List[Verdict] = []
            if judges:
                verdicts = await asyncio.gather(
                    *(
                        self._evaluate_judge(judge, response, prompt, experiment_id)
                        for judge in judges
                    )
                )

            return _TrialOutcome(prompt=prompt, response=response, verdicts=verdicts)
        except Exception as exc:
            return _TrialOutcome(prompt=prompt, error=exc)

    async def _evaluate_judge(
        self,
        judge: JudgeAdapter,
        response: ModelResponse,
        prompt: PromptCandidate,
        experiment_id: str,
    ) -> Verdict:
        lock = self._judge_locks.setdefault(id(judge), asyncio.Lock())
        async with lock:
            verdict = await asyncio.to_thread(judge.evaluate, response, prompt)
        verdict.experiment_id = experiment_id
        verdict.judge_instance_id = getattr(judge, "instance_id", "")
        return verdict

    # -- combo construction --------------------------------------------------

    def _build_combos(self, m: Manifest) -> List[_Combo]:
        """Build adapter×generator combos with optional per-combo isolation.

        When isolation is requested (or implied by combo concurrency with
        multiple adapters), fresh component instances are created per combo so
        that stateful generators / judges / adapters do not share mutable state
        across concurrent runs. Otherwise, the shared instances created in
        `__init__` are reused.
        """
        isolate_generators = (
            m.force_generator_isolation
            or (m.max_combo_concurrency > 1 and len(m.adapters) > 1)
        )
        isolate_adapters = m.force_adapter_isolation
        isolate_judges = m.force_judge_isolation

        combos: List[_Combo] = []
        for adapter, adapter_cfg in zip(self.adapters, m.adapters):
            for shared_generator, generator_cfg in zip(self.generators, m.generators):
                combo_adapter = self._adapter_for_combo(adapter, adapter_cfg, isolate_adapters)
                combo_generator = self._generator_for_combo(
                    shared_generator, generator_cfg, isolate_generators
                )
                combo_judges = self._judges_for_combo(m.judges, isolate_judges)
                combos.append(
                    _Combo(
                        adapter=combo_adapter,
                        adapter_cfg=adapter_cfg,
                        generator=combo_generator,
                        generator_cfg=generator_cfg,
                        judges=combo_judges,
                        judge_cfgs=list(m.judges),
                    )
                )
        return combos

    @staticmethod
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

    @staticmethod
    def _generator_for_combo(
        shared: AttackGenerator, cfg: ManifestGenerator, isolate: bool
    ) -> AttackGenerator:
        if not isolate:
            return shared
        return create_generator(cfg.name, instance_id=cfg.instance_id)

    def _judges_for_combo(
        self, manifest_judges: List[ManifestJudge], isolate: bool
    ) -> List[JudgeAdapter]:
        if not isolate:
            return list(self.judges)
        return [create_judge(j.name, instance_id=j.instance_id) for j in manifest_judges]

    @staticmethod
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

    @staticmethod
    def _config_map_by_id(
        components: List[Any], configs: List[Any]
    ) -> Dict[int, Any]:
        """Map ``id(component) -> manifest config``, taking the first sighting only."""
        config_map: Dict[int, Any] = {}
        for component, cfg in zip(components, configs):
            config_map.setdefault(id(component), cfg)
        return config_map

    def _collect_components(self, combos: List[_Combo]) -> _ComponentSet:
        """Deduplicate the components across all combos for configure / health-check."""
        return _ComponentSet(
            adapters=self._unique_by_id([c.adapter for c in combos]),
            adapter_cfgs=self._config_map_by_id(
                [c.adapter for c in combos], [c.adapter_cfg for c in combos]
            ),
            generators=self._unique_by_id([c.generator for c in combos]),
            generator_cfgs=self._config_map_by_id(
                [c.generator for c in combos], [c.generator_cfg for c in combos]
            ),
            judges=self._unique_by_id([j for c in combos for j in c.judges]),
            judge_cfgs=self._config_map_by_id(
                [j for c in combos for j in c.judges],
                [jc for c in combos for jc in c.judge_cfgs],
            ),
        )

    # -- component lifecycle: configure + health-check -----------------------

    def _configure_components(self, components: _ComponentSet) -> None:
        config_errors: Dict[str, str] = {}
        seed = self.manifest.seed

        for adapter in components.adapters:
            cfg = components.adapter_cfgs.get(id(adapter))
            if cfg is None:
                continue
            try:
                adapter.configure(cfg.config)
            except Exception as exc:
                config_errors[f"adapter:{cfg.instance_id}"] = str(exc)

        for generator in components.generators:
            cfg = components.generator_cfgs.get(id(generator))
            if cfg is None:
                continue
            try:
                generator.configure({"seed": seed, **cfg.config})
            except Exception as exc:
                config_errors[f"generator:{cfg.instance_id}"] = str(exc)

        for judge in components.judges:
            cfg = components.judge_cfgs.get(id(judge))
            if cfg is None:
                continue
            try:
                judge.configure(cfg.config)
            except Exception as exc:
                config_errors[f"judge:{cfg.instance_id}"] = str(exc)

        if config_errors:
            error_msg = "Component configuration failed:\n" + "\n".join(
                f"  {k}: {v}" for k, v in config_errors.items()
            )
            self.logger.log_error(self.manifest.experiment_id, message=error_msg)
            self.logger.close()
            raise RuntimeError(error_msg)

    @staticmethod
    def _health_payload(
        kind: str,
        name: str,
        instance_id: str,
        health: HealthStatus,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Both the generic ``component_*`` keys and the type-specific
        # ``{kind}_*`` keys are emitted so downstream log consumers that
        # filter by either convention keep working.
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

    async def _check_one_health(
        self,
        kind: str,
        component: Any,
        cfg_name: str,
        cfg_instance_id: str,
        experiment_id: str,
        health_issues: Dict[str, Dict[str, Any]],
    ) -> None:
        health = await component.health_check()
        diagnostics = component.diagnostics()
        payload = self._health_payload(kind, cfg_name, cfg_instance_id, health, diagnostics)
        self.logger.log_event(
            LogEvent(event_type=EventType.INFO.value, experiment_id=experiment_id, data=payload)
        )
        if health == HealthStatus.OK:
            return

        health_issues[f"{kind}:{cfg_instance_id}"] = {
            "name": cfg_name,
            "health": health.value,
            "diagnostics": diagnostics,
        }
        warning = dict(payload)
        warning["level"] = "warning"
        warning["message"] = f"{kind.title()} '{cfg_name}' is not fully available"
        self.logger.log_event(
            LogEvent(event_type=EventType.INFO.value, experiment_id=experiment_id, data=warning)
        )

    async def _health_check_components(self, components: _ComponentSet) -> None:
        m = self.manifest
        health_issues: Dict[str, Dict[str, Any]] = {}

        for adapter in components.adapters:
            cfg = components.adapter_cfgs.get(id(adapter))
            if cfg is None:
                continue
            await self._check_one_health(
                "adapter", adapter, cfg.adapter, cfg.instance_id, m.experiment_id, health_issues
            )

        for generator in components.generators:
            cfg = components.generator_cfgs.get(id(generator))
            if cfg is None:
                continue
            await self._check_one_health(
                "generator", generator, cfg.name, cfg.instance_id, m.experiment_id, health_issues
            )

        for judge in components.judges:
            cfg = components.judge_cfgs.get(id(judge))
            if cfg is None:
                continue
            await self._check_one_health(
                "judge", judge, cfg.name, cfg.instance_id, m.experiment_id, health_issues
            )

        if health_issues:
            error_msg = "Component validation failed:\n" + "\n".join(
                f"  {key} ({info['name']}): health={info['health']} diagnostics={info['diagnostics']}"
                for key, info in health_issues.items()
            )
            self.logger.log_error(m.experiment_id, message=error_msg)
            self.logger.close()
            raise RuntimeError(error_msg)

    # -- per-trial bookkeeping -----------------------------------------------

    def _process_outcome(
        self,
        outcome: _TrialOutcome,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
    ) -> None:
        summary.total_prompts += 1
        if outcome.error is not None or outcome.response is None:
            self._record_error_outcome(outcome, m, summary)
            if self.on_trial_complete:
                self.on_trial_complete(summary.total_prompts, outcome.prompt, None, None)
            return

        final_verdict = aggregate_final_verdict(
            experiment_id=m.experiment_id,
            prompt=outcome.prompt,
            response=outcome.response,
            verdicts=outcome.verdicts,
        )
        self._tally_verdict(final_verdict, summary)
        self._record_successful_trial(outcome, combo, m, final_verdict)
        if self.on_trial_complete:
            self.on_trial_complete(
                summary.total_prompts, outcome.prompt, outcome.response, final_verdict
            )

    @staticmethod
    def _tally_verdict(final_verdict: Verdict, summary: ExperimentSummary) -> None:
        if "refusal" in final_verdict.labels:
            summary.total_refusals += 1
        elif "compliance" in final_verdict.labels:
            summary.total_compliances += 1
        else:
            summary.total_inconclusive += 1

    def _record_error_outcome(
        self,
        outcome: _TrialOutcome,
        m: Manifest,
        summary: ExperimentSummary,
    ) -> None:
        summary.total_errors += 1
        self.logger.log_error(
            m.experiment_id,
            message=str(outcome.error) if outcome.error else "Unknown execution error",
            details={"prompt_id": outcome.prompt.prompt_id},
        )

    def _record_successful_trial(
        self,
        outcome: _TrialOutcome,
        combo: _Combo,
        m: Manifest,
        final_verdict: Verdict,
    ) -> None:
        enriched_prompt, attack_type = self._enrich_prompt_metadata(outcome.prompt, combo)

        self.metrics_collector.record_trial(
            prompt=enriched_prompt,
            response=outcome.response,
            verdicts=outcome.verdicts,
            final_verdict=final_verdict,
            attack_type=attack_type,
        )

        combo.generator.update(outcome.prompt, outcome.response)

        self.logger.log_trial(
            m.experiment_id,
            enriched_prompt,
            outcome.response,
            outcome.verdicts,
            final_verdict=final_verdict,
        )

    @staticmethod
    def _enrich_prompt_metadata(
        prompt: PromptCandidate, combo: _Combo
    ) -> tuple[PromptCandidate, str]:
        """Attach adapter/generator identifiers to the prompt's metadata.

        Downstream analysis groups trials by these keys without re-deriving
        them from the manifest, so they need to live on each logged prompt.
        """
        metadata = dict(prompt.metadata)
        attack_type = metadata.get("attack_type") or combo.generator_cfg.name
        metadata.setdefault("attack_type", attack_type)
        metadata.setdefault("generator_name", combo.generator_cfg.name)
        metadata.setdefault("generator_instance_id", combo.generator_cfg.instance_id)
        metadata.setdefault("adapter_name", combo.adapter_cfg.adapter)
        metadata.setdefault("adapter_instance_id", combo.adapter_cfg.instance_id)
        enriched = PromptCandidate(
            prompt_id=prompt.prompt_id,
            text=prompt.text,
            metadata=metadata,
        )
        return enriched, attack_type

    async def _advance_multiturn_if_needed(
        self,
        outcome: _TrialOutcome,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        rate_limiter: RateLimiter,
    ) -> None:
        """Drive multi-turn conversations to completion in-line.

        For generators that expose ``get_next_turn()`` and prompts that carry a
        ``conversation_id``, fetch and execute follow-up turns immediately
        rather than waiting for the next outer batch. The turn count is bounded
        by ``max_turns_per_conversation`` to guard against malformed flows.
        """
        metadata = outcome.prompt.metadata or {}
        conversation_id = metadata.get("conversation_id")
        if not conversation_id or not hasattr(combo.generator, "get_next_turn"):
            return

        for turn_count in range(m.max_turns_per_conversation):
            next_turn_prompt = combo.generator.get_next_turn(conversation_id)
            if next_turn_prompt is None:
                self._log_conversation_complete(
                    m.experiment_id,
                    conversation_id,
                    total_turns=metadata.get("turn", 0) + turn_count + 1,
                )
                return

            next_outcome = await self._execute_prompt(
                combo.adapter,
                next_turn_prompt,
                m.experiment_id,
                rate_limiter,
                judges=combo.judges,
            )
            self._process_outcome(next_outcome, combo, m, summary)

    def _log_conversation_complete(
        self, experiment_id: str, conversation_id: str, total_turns: int
    ) -> None:
        self.logger.log_event(
            LogEvent(
                event_type=EventType.INFO.value,
                experiment_id=experiment_id,
                data={
                    "level": "info",
                    "message": "Multi-turn conversation completed",
                    "conversation_id": conversation_id,
                    "total_turns": total_turns,
                },
            )
        )

    # -- combo execution -----------------------------------------------------

    async def _run_prompts(
        self,
        prompts: List[PromptCandidate],
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: RateLimiter,
        advance_multiturn: bool,
    ) -> None:
        """Execute a batch of prompts and process their outcomes."""

        async def run_limited(prompt: PromptCandidate) -> _TrialOutcome:
            async with semaphore:
                return await self._execute_prompt(
                    combo.adapter,
                    prompt,
                    m.experiment_id,
                    rate_limiter,
                    judges=combo.judges,
                )

        outcomes = await asyncio.gather(
            *(asyncio.create_task(run_limited(prompt)) for prompt in prompts)
        )
        for outcome in outcomes:
            self._process_outcome(outcome, combo, m, summary)
            if advance_multiturn:
                await self._advance_multiturn_if_needed(
                    outcome, combo, m, summary, rate_limiter
                )

    async def _run_combo_batch(
        self,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: RateLimiter,
    ) -> None:
        for _ in range(m.num_batches):
            prompts = await asyncio.to_thread(combo.generator.next, m.batch_size)
            await self._run_prompts(
                prompts, combo, m, summary, semaphore, rate_limiter, advance_multiturn=True
            )

    async def _run_combo_streaming(
        self,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: RateLimiter,
    ) -> None:
        # Streaming overlaps the next batch's generation with the current
        # batch's execution. Multi-turn advancement is intentionally skipped
        # here because stateful generators report ``supports_streaming() ==
        # False`` and never reach this branch.
        next_prompts_task: asyncio.Task[List[PromptCandidate]] | None = None

        for batch_idx in range(m.num_batches):
            if next_prompts_task is None:
                prompts = await asyncio.to_thread(combo.generator.next, m.batch_size)
            else:
                prompts = await next_prompts_task

            if batch_idx < m.num_batches - 1:
                next_prompts_task = asyncio.create_task(
                    asyncio.to_thread(combo.generator.next, m.batch_size)
                )
            else:
                next_prompts_task = None

            await self._run_prompts(
                prompts, combo, m, summary, semaphore, rate_limiter, advance_multiturn=False
            )

        if next_prompts_task is not None:
            await next_prompts_task

    async def _run_combo(
        self,
        combo: _Combo,
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: RateLimiter,
    ) -> None:
        if m.pipeline_mode == "streaming":
            if combo.generator.supports_streaming():
                await self._run_combo_streaming(combo, m, summary, semaphore, rate_limiter)
                return
            self._log_streaming_disabled(combo, m.experiment_id)

        await self._run_combo_batch(combo, m, summary, semaphore, rate_limiter)

    def _log_streaming_disabled(self, combo: _Combo, experiment_id: str) -> None:
        self.logger.log_event(
            LogEvent(
                event_type=EventType.INFO.value,
                experiment_id=experiment_id,
                data={
                    "level": "info",
                    "message": (
                        "Streaming pipeline disabled for generator requiring feedback; "
                        "using batch mode."
                    ),
                    "generator_name": combo.generator_cfg.name,
                    "generator_instance_id": combo.generator_cfg.instance_id,
                },
            )
        )

    async def _run_combos(
        self,
        combos: List[_Combo],
        m: Manifest,
        summary: ExperimentSummary,
        semaphore: asyncio.Semaphore,
        rate_limiter: RateLimiter,
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

        # Log the experiment start before configuration so that setup failures
        # still leave an audit trail in the JSONL output.
        self.logger.log_experiment_start(m.experiment_id, m.to_dict())

        combos = self._build_combos(m)
        components = self._collect_components(combos)

        # Make sure every judge instance — including any per-combo ones added
        # by isolation — has a serialisation lock available.
        for judge in components.judges:
            self._judge_locks.setdefault(id(judge), asyncio.Lock())

        # Phase 1 + 2: configure and health-check.
        self._configure_components(components)
        await self._health_check_components(components)

        # Phase 3: run the experiment loop with combo and prompt parallelism.
        semaphore = asyncio.Semaphore(max(1, int(m.max_concurrency)))
        rate_limiter = RateLimiter(m.rate_limit_rps)
        await self._run_combos(combos, m, summary, semaphore, rate_limiter)

        summary.elapsed_seconds = time.perf_counter() - t0
        await self._finalize_run(components, summary)
        return summary

    async def _finalize_run(
        self, components: _ComponentSet, summary: ExperimentSummary
    ) -> None:
        """Write final metrics, close the log, and clean up components."""
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
        await self._close_components(components.generators)
        await self._close_components(components.judges)

    @staticmethod
    async def _close_components(components: List[Any]) -> None:
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
