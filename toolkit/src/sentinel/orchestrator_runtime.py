"""Per-prompt and per-combo runtime for the orchestrator.

Holds the dataclasses and async helpers that drive the main experiment
loop:

* :class:`RuntimeContext`  — carries shared state into the helpers.
* :func:`run_combos`       — top-level loop run by the orchestrator.
* :func:`run_combo`        — chooses between the batch and streaming modes
                              for one combo.
* :func:`execute_prompt`   — one adapter call + judge fan-out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sentinel.concurrency import RateLimiter
from sentinel.judges.base import JudgeAdapter
from sentinel.logger import JsonlLogger
from sentinel.manifest import Manifest
from sentinel.metrics import MetricsCollector
from sentinel.model_adapters.base import ModelAdapter
from sentinel.models import EventType, LogEvent, ModelResponse, PromptCandidate, Verdict
from sentinel.orchestrator_components import Combo
from sentinel.verdict import aggregate_final_verdict


# ---------------------------------------------------------------------------
# Runtime context + trial outcome
# ---------------------------------------------------------------------------

# Type aliases mirror the ones the orchestrator advertises.
TrialCallback = Callable[[int, PromptCandidate, Optional[ModelResponse], Optional[Verdict]], None]
StageCallback = Callable[[str], None]
FollowupCallback = Callable[[str], None]
ConversationCompleteCallback = Callable[[str], None]


@dataclass
class RuntimeContext:
    """Bundle of shared mutable state passed through the runtime helpers."""

    manifest: Manifest
    summary: Any  # ExperimentSummary (avoid circular import)
    logger: JsonlLogger
    metrics_collector: MetricsCollector
    on_trial_complete: Optional[TrialCallback] = None
    on_generation_start: Optional[StageCallback] = None
    on_followup_start: Optional[FollowupCallback] = None
    on_adapter_start: Optional[StageCallback] = None
    on_judge_start: Optional[StageCallback] = None
    on_conversation_complete: Optional[ConversationCompleteCallback] = None


@dataclass
class _TrialOutcome:
    """Internal result bundle for a single prompt execution."""

    prompt: PromptCandidate
    response: Optional[ModelResponse] = None
    verdicts: List[Verdict] = field(default_factory=list)
    error: Optional[Exception] = None


# ---------------------------------------------------------------------------
# Per-prompt execution
# ---------------------------------------------------------------------------

async def execute_prompt(
    adapter: ModelAdapter,
    prompt: PromptCandidate,
    experiment_id: str,
    rate_limiter: RateLimiter,
    judges: List[JudgeAdapter],
    ctx: RuntimeContext,
) -> _TrialOutcome:
    """Send *prompt* to *adapter*, then evaluate the response with *judges*."""
    await rate_limiter.acquire()

    display_name = prompt.metadata.get("display_name", "")
    if ctx.on_adapter_start:
        ctx.on_adapter_start(display_name)

    try:
        response = await adapter.generate(prompt.text)
        response.prompt_id = prompt.prompt_id
        response.adapter_instance_id = getattr(adapter, "instance_id", "")

        # Signal that the adapter phase is done and judging is starting. Fires
        # even when judges=[] so the panel can transition resp→judge→done.
        if ctx.on_judge_start:
            ctx.on_judge_start(display_name)

        verdicts: List[Verdict] = []
        if judges:
            verdicts = await asyncio.gather(
                *(_evaluate_judge(j, response, prompt, experiment_id) for j in judges)
            )
        return _TrialOutcome(prompt=prompt, response=response, verdicts=verdicts)

    except Exception as exc:
        return _TrialOutcome(prompt=prompt, error=exc)


async def _evaluate_judge(
    judge: JudgeAdapter,
    response: ModelResponse,
    prompt: PromptCandidate,
    experiment_id: str,
) -> Verdict:
    verdict = await asyncio.to_thread(judge.evaluate, response, prompt)
    verdict.experiment_id = experiment_id
    verdict.judge_instance_id = getattr(judge, "instance_id", "")
    verdict.judge_weight = getattr(judge, "weight", 1.0)
    return verdict


# ---------------------------------------------------------------------------
# Per-trial bookkeeping
# ---------------------------------------------------------------------------

def _process_outcome(outcome: _TrialOutcome, combo: Combo, ctx: RuntimeContext) -> str:
    """Process one trial outcome and return the final verdict label.

    Returns ``"error"`` for failed outcomes so callers (e.g. the multi-turn
    loop) can make stop/continue decisions without re-examining the outcome.
    """
    summary = ctx.summary
    summary.total_prompts += 1

    if (
        outcome.error is not None
        or outcome.response is None
        or outcome.response.is_error
    ):
        _record_error_outcome(outcome, ctx)
        if ctx.on_trial_complete:
            ctx.on_trial_complete(
                summary.total_prompts, outcome.prompt, outcome.response, None
            )
        return "error"

    final_verdict = aggregate_final_verdict(
        experiment_id=ctx.manifest.experiment_id,
        prompt=outcome.prompt,
        response=outcome.response,
        verdicts=outcome.verdicts,
    )
    _tally_verdict(final_verdict, ctx)
    summary.record_response(outcome.response)
    _record_successful_trial(outcome, combo, ctx, final_verdict)
    if ctx.on_trial_complete:
        ctx.on_trial_complete(
            summary.total_prompts, outcome.prompt, outcome.response, final_verdict
        )
    return final_verdict.labels[0] if final_verdict.labels else "inconclusive"


def _tally_verdict(final_verdict: Verdict, ctx: RuntimeContext) -> None:
    summary = ctx.summary
    if "refusal" in final_verdict.labels:
        summary.total_refusals += 1
    elif "compliance" in final_verdict.labels:
        summary.total_compliances += 1
    else:
        summary.total_inconclusive += 1


def _record_error_outcome(outcome: _TrialOutcome, ctx: RuntimeContext) -> None:
    ctx.summary.total_errors += 1
    details: Dict[str, Any] = {"prompt_id": outcome.prompt.prompt_id}
    if outcome.response is not None:
        details["error"] = (
            outcome.response.metadata.get("error")
            or outcome.response.text
            or "unknown error"
        )
    message = (
        str(outcome.error) if outcome.error
        else "Adapter returned an error response"
    )
    ctx.logger.log_error(ctx.manifest.experiment_id, message=message, details=details)


def _record_successful_trial(
    outcome: _TrialOutcome,
    combo: Combo,
    ctx: RuntimeContext,
    final_verdict: Verdict,
) -> None:
    enriched_prompt, attack_type = _enrich_prompt_metadata(outcome.prompt, combo)

    ctx.metrics_collector.record_trial(
        prompt=enriched_prompt,
        response=outcome.response,
        verdicts=outcome.verdicts,
        final_verdict=final_verdict,
        attack_type=attack_type,
    )

    combo.generator.update(outcome.prompt, outcome.response)

    ctx.logger.log_trial(
        ctx.manifest.experiment_id,
        enriched_prompt,
        outcome.response,
        outcome.verdicts,
        final_verdict=final_verdict,
    )


def _enrich_prompt_metadata(
    prompt: PromptCandidate, combo: Combo
) -> tuple[PromptCandidate, str]:
    """Attach adapter/generator identifiers to the prompt's metadata.

    Downstream analysis groups trials by these keys without re-deriving them
    from the manifest, so they need to live on each logged prompt.
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


# ---------------------------------------------------------------------------
# Multi-turn advancement
# ---------------------------------------------------------------------------

async def _advance_multiturn_if_needed(
    outcome: _TrialOutcome,
    combo: Combo,
    ctx: RuntimeContext,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    initial_verdict: str = "",
) -> None:
    """Drive multi-turn conversations to completion in-line.

    For generators that expose ``get_next_turn()`` and prompts that carry a
    ``conversation_id``, fetch and execute follow-up turns immediately rather
    than waiting for the next outer batch. The turn count is bounded by
    ``max_turns_per_conversation`` to guard against malformed flows.

    A conversation-complete event is logged on every exit path with a
    ``stop_reason`` field: ``"natural_end"`` (generator returned None),
    ``"compliance"`` (stop_on_compliance triggered), or ``"max_turns"`` (the
    hard ceiling was reached).
    """
    metadata = outcome.prompt.metadata or {}
    conversation_id = metadata.get("conversation_id")
    if not conversation_id or not hasattr(combo.generator, "get_next_turn"):
        return

    m = ctx.manifest
    # Count includes the initial prompt that was already executed.
    turns_executed = metadata.get("turn", 0) + 1

    # If the very first turn already produced compliance, stop immediately.
    if m.stop_on_compliance and initial_verdict == "compliance":
        _log_conversation_complete(
            ctx, conversation_id, total_turns=turns_executed, stop_reason="compliance"
        )
        return

    for i in range(m.max_turns_per_conversation):
        # The semaphore wraps the whole turn — generation + adapter + judge — so
        # that the combined generator+responder count never exceeds max_concurrency.
        async with semaphore:
            if ctx.on_followup_start:
                ctx.on_followup_start(combo.generator.get_display_name())

            next_turn_prompt = await asyncio.to_thread(combo.generator.get_next_turn, conversation_id)
            if next_turn_prompt is None:
                _log_conversation_complete(
                    ctx, conversation_id, total_turns=turns_executed, stop_reason="natural_end"
                )
                return

            next_outcome = await execute_prompt(
                combo.adapter,
                next_turn_prompt,
                m.experiment_id,
                rate_limiter,
                judges=combo.judges,
                ctx=ctx,
            )
        verdict_label = _process_outcome(next_outcome, combo, ctx)
        turns_executed += 1

        if m.stop_on_compliance and verdict_label == "compliance":
            _log_conversation_complete(
                ctx, conversation_id, total_turns=turns_executed, stop_reason="compliance"
            )
            return

    # Hard ceiling reached without a natural end or compliance stop.
    _log_conversation_complete(
        ctx, conversation_id, total_turns=turns_executed, stop_reason="max_turns"
    )


def _log_conversation_complete(
    ctx: RuntimeContext,
    conversation_id: str,
    total_turns: int,
    stop_reason: str,
) -> None:
    ctx.logger.log_event(
        LogEvent(
            event_type=EventType.INFO.value,
            experiment_id=ctx.manifest.experiment_id,
            data={
                "level": "info",
                "message": "Multi-turn conversation completed",
                "conversation_id": conversation_id,
                "total_turns": total_turns,
                "stop_reason": stop_reason,
            },
        )
    )
    if ctx.on_conversation_complete:
        ctx.on_conversation_complete(conversation_id)


# ---------------------------------------------------------------------------
# Combo execution (batch / streaming)
# ---------------------------------------------------------------------------

async def _run_prompts(
    prompts: List[PromptCandidate],
    combo: Combo,
    ctx: RuntimeContext,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    advance_multiturn: bool,
) -> List["asyncio.Task[None]"]:
    """Execute a batch of prompts concurrently and return scheduled multi-turn tasks.

    Each conversation's multi-turn task is scheduled the moment its initial
    prompt completes, so slow or erroring prompts cannot block other
    conversations from flushing their results.
    """

    async def run_limited(prompt: PromptCandidate) -> _TrialOutcome:
        async with semaphore:
            return await execute_prompt(
                combo.adapter,
                prompt,
                ctx.manifest.experiment_id,
                rate_limiter,
                judges=combo.judges,
                ctx=ctx,
            )

    tasks = [asyncio.create_task(run_limited(prompt)) for prompt in prompts]
    multiturn_tasks: List[asyncio.Task] = []

    # Schedule each conversation's multi-turn task as soon as its initial
    # prompt arrives so the TUI stays responsive and no prompt blocks others.
    for fut in asyncio.as_completed(tasks):
        outcome = await fut
        initial_verdict = _process_outcome(outcome, combo, ctx)
        if advance_multiturn:
            multiturn_tasks.append(asyncio.create_task(
                _advance_multiturn_if_needed(outcome, combo, ctx, semaphore, rate_limiter, initial_verdict)
            ))

    return multiturn_tasks


async def _run_combo_batch(
    combo: Combo,
    ctx: RuntimeContext,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> None:
    all_multiturn: List[asyncio.Task] = []
    for _ in range(ctx.manifest.num_batches):
        if ctx.on_generation_start:
            ctx.on_generation_start(combo.generator.get_display_name())
        prompts = await asyncio.to_thread(combo.generator.next, ctx.manifest.batch_size)
        # _run_prompts schedules multi-turn tasks as each initial prompt
        # completes, so they run in the background during the next batch.
        mt_tasks = await _run_prompts(
            prompts, combo, ctx, semaphore, rate_limiter, advance_multiturn=True
        )
        all_multiturn.extend(mt_tasks)
    if all_multiturn:
        await asyncio.gather(*all_multiturn)


async def _run_combo_streaming(
    combo: Combo,
    ctx: RuntimeContext,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> None:
    """Streaming overlaps the next batch's generation with the current one.

    Multi-turn advancement is intentionally skipped here because stateful
    generators report ``supports_streaming() == False`` and never reach this
    branch.
    """
    next_prompts_task: Optional[asyncio.Task[List[PromptCandidate]]] = None
    m = ctx.manifest

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

        await _run_prompts(
            prompts, combo, ctx, semaphore, rate_limiter, advance_multiturn=False
        )

    if next_prompts_task is not None:
        await next_prompts_task


async def run_combo(
    combo: Combo,
    ctx: RuntimeContext,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> None:
    """Run a single combo using the manifest's pipeline mode."""
    if ctx.manifest.pipeline_mode == "streaming":
        if combo.generator.supports_streaming():
            await _run_combo_streaming(combo, ctx, semaphore, rate_limiter)
            return
        _log_streaming_disabled(combo, ctx)

    await _run_combo_batch(combo, ctx, semaphore, rate_limiter)


def _log_streaming_disabled(combo: Combo, ctx: RuntimeContext) -> None:
    ctx.logger.log_event(
        LogEvent(
            event_type=EventType.INFO.value,
            experiment_id=ctx.manifest.experiment_id,
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


async def run_combos(
    combos: List[Combo],
    ctx: RuntimeContext,
    rate_limiter: RateLimiter,
) -> None:
    """Run all combos with bounded combo-level parallelism.

    Each combo gets its own prompt semaphore so one combo never starves another
    while waiting for concurrency slots.
    """
    combo_semaphore = asyncio.Semaphore(max(1, int(ctx.manifest.max_combo_concurrency)))

    async def run_limited(combo: Combo) -> None:
        async with combo_semaphore:
            per_combo_semaphore = asyncio.Semaphore(max(1, int(ctx.manifest.max_concurrency)))
            await run_combo(combo, ctx, per_combo_semaphore, rate_limiter)

    await asyncio.gather(*(run_limited(combo) for combo in combos))
