"""Comprehensive metrics collection and analysis for red-teaming experiments.

Collects detailed trial-level and aggregate-level statistics on attack
success, judge agreement, response characteristics, and verdict
distributions. Trials are recorded one at a time during a run and then
aggregated at the end via :meth:`MetricsCollector.compute_aggregate`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List

from sentinel.models import ModelResponse, PromptCandidate, Verdict


# ---------------------------------------------------------------------------
# Trial-level metrics
# ---------------------------------------------------------------------------

@dataclass
class TrialMetrics:
    """Metrics for a single prompt-response-verdict trial."""

    prompt_id: str = ""
    prompt_text: str = ""
    response_text: str = ""
    response_time_ms: float = 0.0
    prompt_tokens: int = 0
    response_tokens: int = 0
    verdicts: List[Verdict] = field(default_factory=list)
    final_verdict_label: str = ""  # refusal | compliance | inconclusive
    final_verdict_confidence: float = 0.0
    judge_agreement: float = 0.0  # [0, 1] measure of consensus
    attack_type: str = ""
    model_id: str = ""
    adapter_instance_id: str = ""
    generator_instance_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "response_time_ms": round(self.response_time_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "final_verdict_label": self.final_verdict_label,
            "final_verdict_confidence": round(self.final_verdict_confidence, 4),
            "judge_agreement": round(self.judge_agreement, 4),
            "attack_type": self.attack_type,
            "model_id": self.model_id,
            "adapter_instance_id": self.adapter_instance_id,
            "generator_instance_id": self.generator_instance_id,
        }

    @staticmethod
    def compute_judge_agreement(verdicts: List[Verdict]) -> float:
        """Return the share of judges that picked the most common label."""
        if not verdicts:
            return 0.0

        labels = [v.labels[0] if v.labels else "inconclusive" for v in verdicts]
        label_counts: Dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        return round(max(label_counts.values()) / len(labels), 4)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Summary statistics across all trials in an experiment."""

    total_trials: int = 0
    total_refusals: int = 0
    total_compliances: int = 0
    total_inconclusive: int = 0
    total_errors: int = 0

    refusal_rate: float = 0.0
    compliance_rate: float = 0.0
    inconclusive_rate: float = 0.0
    attack_success_rate: float = 0.0  # = compliance_rate

    avg_confidence_compliance: float = 0.0
    avg_confidence_refusal: float = 0.0
    avg_confidence_inconclusive: float = 0.0
    avg_judge_agreement: float = 0.0

    avg_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    avg_response_tokens: int = 0
    median_response_tokens: int = 0

    metrics_by_attack_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics_by_generator: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics_by_adapter: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics_by_judge: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    verdict_distribution: Dict[str, int] = field(default_factory=dict)

    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trials": self.total_trials,
            "total_refusals": self.total_refusals,
            "total_compliances": self.total_compliances,
            "total_inconclusive": self.total_inconclusive,
            "total_errors": self.total_errors,
            "refusal_rate": round(self.refusal_rate, 4),
            "compliance_rate": round(self.compliance_rate, 4),
            "inconclusive_rate": round(self.inconclusive_rate, 4),
            "attack_success_rate": round(self.attack_success_rate, 4),
            "avg_confidence_compliance": round(self.avg_confidence_compliance, 4),
            "avg_confidence_refusal": round(self.avg_confidence_refusal, 4),
            "avg_confidence_inconclusive": round(self.avg_confidence_inconclusive, 4),
            "avg_judge_agreement": round(self.avg_judge_agreement, 4),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "median_response_time_ms": round(self.median_response_time_ms, 2),
            "avg_response_tokens": self.avg_response_tokens,
            "median_response_tokens": self.median_response_tokens,
            "metrics_by_attack_type": self.metrics_by_attack_type,
            "metrics_by_generator": self.metrics_by_generator,
            "metrics_by_adapter": self.metrics_by_adapter,
            "metrics_by_judge": self.metrics_by_judge,
            "verdict_distribution": self.verdict_distribution,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Helpers shared by the per-group breakdowns
# ---------------------------------------------------------------------------

_VERDICT_LABELS = ("refusal", "compliance", "inconclusive")


def _trial_group_summary(trials: List[TrialMetrics]) -> Dict[str, Any]:
    """Per-group totals, ASR, judge agreement and average response time.

    Used by ``metrics_by_generator`` and ``metrics_by_adapter``, which both
    want the same fields.
    """
    if not trials:
        return {
            "total": 0,
            "compliance": 0,
            "refusal": 0,
            "asr": 0.0,
            "avg_judge_agreement": 0.0,
            "avg_response_time_ms": 0.0,
        }

    compliance = sum(1 for t in trials if t.final_verdict_label == "compliance")
    refusal = sum(1 for t in trials if t.final_verdict_label == "refusal")
    response_times = [t.response_time_ms for t in trials if t.response_time_ms > 0]
    return {
        "total": len(trials),
        "compliance": compliance,
        "refusal": refusal,
        "asr": round(compliance / len(trials), 4),
        "avg_judge_agreement": round(
            statistics.mean(t.judge_agreement for t in trials), 4
        ),
        "avg_response_time_ms": round(
            statistics.mean(response_times) if response_times else 0.0, 2
        ),
    }


def _bucket_by(items: Iterable[Any], key: Callable[[Any], str]) -> Dict[str, List[Any]]:
    """Group ``items`` into a dict keyed by ``key(item)``."""
    buckets: Dict[str, List[Any]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    return buckets


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.mean(items) if items else 0.0


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulate trial-level metrics and produce an :class:`AggregateMetrics`."""

    def __init__(self) -> None:
        self.trials: List[TrialMetrics] = []

    # -- recording -----------------------------------------------------------

    def record_trial(
        self,
        prompt: PromptCandidate,
        response: ModelResponse,
        verdicts: List[Verdict],
        final_verdict: Verdict,
        attack_type: str = "",
    ) -> None:
        """Record a single trial and its evaluation results."""
        final_label = final_verdict.labels[0] if final_verdict.labels else "inconclusive"
        trial = TrialMetrics(
            prompt_id=prompt.prompt_id,
            prompt_text=prompt.text,
            response_text=response.text,
            response_time_ms=response.latency_ms,
            prompt_tokens=len(prompt.text.split()) if prompt.text else 0,
            response_tokens=response.tokens,
            verdicts=verdicts,
            final_verdict_label=final_label,
            final_verdict_confidence=final_verdict.confidence,
            judge_agreement=TrialMetrics.compute_judge_agreement(verdicts),
            attack_type=attack_type,
            model_id=response.model_id,
            adapter_instance_id=response.adapter_instance_id,
            generator_instance_id=prompt.metadata.get("generator_instance_id", ""),
        )
        self.trials.append(trial)

    # -- aggregation ---------------------------------------------------------

    def compute_aggregate(self, elapsed_seconds: float = 0.0) -> AggregateMetrics:
        """Compute aggregate statistics from all recorded trials."""
        agg = AggregateMetrics(elapsed_seconds=elapsed_seconds)
        if not self.trials:
            return agg

        agg.total_trials = len(self.trials)
        self._fill_verdict_totals(agg)
        self._fill_rates(agg)
        self._fill_confidence_averages(agg)
        self._fill_response_characteristics(agg)
        self._fill_per_group_breakdowns(agg)
        return agg

    # -- aggregation helpers -------------------------------------------------

    def _fill_verdict_totals(self, agg: AggregateMetrics) -> None:
        for trial in self.trials:
            if trial.final_verdict_label == "refusal":
                agg.total_refusals += 1
            elif trial.final_verdict_label == "compliance":
                agg.total_compliances += 1
            elif trial.final_verdict_label == "inconclusive":
                agg.total_inconclusive += 1

        for label in _VERDICT_LABELS:
            agg.verdict_distribution[label] = sum(
                1 for t in self.trials if t.final_verdict_label == label
            )

    @staticmethod
    def _fill_rates(agg: AggregateMetrics) -> None:
        if agg.total_trials <= 0:
            return
        agg.refusal_rate = agg.total_refusals / agg.total_trials
        agg.compliance_rate = agg.total_compliances / agg.total_trials
        agg.inconclusive_rate = agg.total_inconclusive / agg.total_trials
        agg.attack_success_rate = agg.compliance_rate

    def _fill_confidence_averages(self, agg: AggregateMetrics) -> None:
        for label, attr in (
            ("compliance", "avg_confidence_compliance"),
            ("refusal", "avg_confidence_refusal"),
            ("inconclusive", "avg_confidence_inconclusive"),
        ):
            confidences = [
                t.final_verdict_confidence
                for t in self.trials
                if t.final_verdict_label == label
            ]
            setattr(agg, attr, _mean(confidences))

        agg.avg_judge_agreement = _mean(t.judge_agreement for t in self.trials)

    def _fill_response_characteristics(self, agg: AggregateMetrics) -> None:
        response_times = [t.response_time_ms for t in self.trials if t.response_time_ms > 0]
        if response_times:
            agg.avg_response_time_ms = statistics.mean(response_times)
            agg.median_response_time_ms = statistics.median(response_times)

        response_tokens = [t.response_tokens for t in self.trials if t.response_tokens > 0]
        if response_tokens:
            agg.avg_response_tokens = int(statistics.mean(response_tokens))
            agg.median_response_tokens = int(statistics.median(response_tokens))

    def _fill_per_group_breakdowns(self, agg: AggregateMetrics) -> None:
        agg.metrics_by_attack_type = self._attack_type_breakdown()
        agg.metrics_by_generator = self._trial_breakdown(
            lambda t: t.generator_instance_id or t.attack_type or "unknown"
        )
        agg.metrics_by_adapter = self._trial_breakdown(
            lambda t: t.adapter_instance_id or t.model_id or "unknown"
        )
        agg.metrics_by_judge = self._judge_breakdown()

    def _attack_type_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Compact breakdown used by ``metrics_by_attack_type`` (no agreement field)."""
        groups = _bucket_by(self.trials, key=lambda t: t.attack_type or "unknown")
        breakdown: Dict[str, Dict[str, Any]] = {}
        for atype, trials in groups.items():
            compliance = sum(1 for t in trials if t.final_verdict_label == "compliance")
            refusal = sum(1 for t in trials if t.final_verdict_label == "refusal")
            breakdown[atype] = {
                "total": len(trials),
                "compliance": compliance,
                "refusal": refusal,
                "asr": round(compliance / len(trials), 4) if trials else 0.0,
            }
        return breakdown

    def _trial_breakdown(
        self, key: Callable[[TrialMetrics], str]
    ) -> Dict[str, Dict[str, Any]]:
        groups = _bucket_by(self.trials, key=key)
        return {key_: _trial_group_summary(trials) for key_, trials in groups.items()}

    def _judge_breakdown(self) -> Dict[str, Dict[str, Any]]:
        per_judge: Dict[str, List[Verdict]] = {}
        for trial in self.trials:
            for verdict in trial.verdicts:
                judge_id = verdict.judge_instance_id or verdict.judge_type or "unknown"
                per_judge.setdefault(judge_id, []).append(verdict)

        breakdown: Dict[str, Dict[str, Any]] = {}
        for judge_id, verdicts in per_judge.items():
            breakdown[judge_id] = {
                "total_verdicts": len(verdicts),
                "compliance": sum(1 for v in verdicts if "compliance" in v.labels),
                "refusal": sum(1 for v in verdicts if "refusal" in v.labels),
            }
        return breakdown
