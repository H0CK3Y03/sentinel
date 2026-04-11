"""Comprehensive metrics collection and analysis for red-teaming experiments.

Collects detailed trial-level and aggregate-level statistics on attack success,
judge agreement, response characteristics, and verdict distributions.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List

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
    attack_type: str = ""  # e.g., "single-turn", "prompt-injection", etc.
    model_id: str = ""

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
        }

    @staticmethod
    def compute_judge_agreement(verdicts: List[Verdict]) -> float:
        """Measure consensus among judges: [0, 1] where 1 = all agree.

        Simple measure: proportion of judges agreeing with the majority label.
        """
        if not verdicts:
            return 0.0

        labels = [v.labels[0] if v.labels else "inconclusive" for v in verdicts]
        if not labels:
            return 0.0

        # Find the most common label
        label_counts: Dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        max_count = max(label_counts.values())
        agreement = max_count / len(labels)
        return round(agreement, 4)


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

    # Rates
    refusal_rate: float = 0.0
    compliance_rate: float = 0.0
    inconclusive_rate: float = 0.0
    attack_success_rate: float = 0.0  # = compliance_rate

    # Confidence metrics
    avg_confidence_compliance: float = 0.0
    avg_confidence_refusal: float = 0.0
    avg_confidence_inconclusive: float = 0.0
    avg_judge_agreement: float = 0.0

    # Response characteristics
    avg_response_time_ms: float = 0.0
    median_response_time_ms: float = 0.0
    avg_response_tokens: int = 0
    median_response_tokens: int = 0

    # Per-attack-type breakdown
    metrics_by_attack_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-adapter breakdown (model-indexed metrics)
    metrics_by_adapter: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-judge breakdown
    metrics_by_judge: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Distribution
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
            "metrics_by_adapter": self.metrics_by_adapter,
            "metrics_by_judge": self.metrics_by_judge,
            "verdict_distribution": self.verdict_distribution,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulate and analyse trial-level and aggregate metrics."""

    def __init__(self) -> None:
        self.trials: List[TrialMetrics] = []

    def record_trial(
        self,
        prompt: PromptCandidate,
        response: ModelResponse,
        verdicts: List[Verdict],
        final_verdict: Verdict | None = None,
        attack_type: str = "",
    ) -> None:
        """Record a single trial and its evaluation results."""
        # Compute judge agreement
        judge_agreement = TrialMetrics.compute_judge_agreement(verdicts)

        # Determine final verdict label
        final_label = "inconclusive"
        final_confidence = 0.0
        if final_verdict:
            final_label = final_verdict.labels[0] if final_verdict.labels else "inconclusive"
            final_confidence = final_verdict.confidence

        # Estimate prompt tokens (rough heuristic)
        prompt_tokens = len(prompt.text.split()) if prompt.text else 0

        trial = TrialMetrics(
            prompt_id=prompt.prompt_id,
            prompt_text=prompt.text,
            response_text=response.text,
            response_time_ms=response.latency_ms,
            prompt_tokens=prompt_tokens,
            response_tokens=response.tokens,
            verdicts=verdicts,
            final_verdict_label=final_label,
            final_verdict_confidence=final_confidence,
            judge_agreement=judge_agreement,
            attack_type=attack_type,
            model_id=response.model_id,
        )

        self.trials.append(trial)

    def compute_aggregate(self, elapsed_seconds: float = 0.0) -> AggregateMetrics:
        """Compute aggregate statistics from all recorded trials."""
        agg = AggregateMetrics(elapsed_seconds=elapsed_seconds)

        if not self.trials:
            return agg

        agg.total_trials = len(self.trials)

        # Count verdicts
        for trial in self.trials:
            if trial.final_verdict_label == "refusal":
                agg.total_refusals += 1
            elif trial.final_verdict_label == "compliance":
                agg.total_compliances += 1
            elif trial.final_verdict_label == "inconclusive":
                agg.total_inconclusive += 1

        # Calculate rates
        valid_trials = agg.total_trials
        if valid_trials > 0:
            agg.refusal_rate = agg.total_refusals / valid_trials
            agg.compliance_rate = agg.total_compliances / valid_trials
            agg.inconclusive_rate = agg.total_inconclusive / valid_trials
            agg.attack_success_rate = agg.compliance_rate

        # Average confidence by verdict type
        compliance_confidences = [
            t.final_verdict_confidence
            for t in self.trials
            if t.final_verdict_label == "compliance"
        ]
        refusal_confidences = [
            t.final_verdict_confidence
            for t in self.trials
            if t.final_verdict_label == "refusal"
        ]
        inconclusive_confidences = [
            t.final_verdict_confidence
            for t in self.trials
            if t.final_verdict_label == "inconclusive"
        ]

        if compliance_confidences:
            agg.avg_confidence_compliance = statistics.mean(compliance_confidences)
        if refusal_confidences:
            agg.avg_confidence_refusal = statistics.mean(refusal_confidences)
        if inconclusive_confidences:
            agg.avg_confidence_inconclusive = statistics.mean(inconclusive_confidences)

        # Judge agreement
        judge_agreements = [t.judge_agreement for t in self.trials]
        if judge_agreements:
            agg.avg_judge_agreement = statistics.mean(judge_agreements)

        # Response characteristics
        response_times = [t.response_time_ms for t in self.trials if t.response_time_ms > 0]
        if response_times:
            agg.avg_response_time_ms = statistics.mean(response_times)
            agg.median_response_time_ms = statistics.median(response_times)

        response_tokens = [t.response_tokens for t in self.trials if t.response_tokens > 0]
        if response_tokens:
            agg.avg_response_tokens = int(statistics.mean(response_tokens))
            agg.median_response_tokens = int(statistics.median(response_tokens))

        # Verdict distribution
        for label in ("refusal", "compliance", "inconclusive"):
            count = sum(1 for t in self.trials if t.final_verdict_label == label)
            agg.verdict_distribution[label] = count

        # Metrics by attack type
        attack_types: Dict[str, List[TrialMetrics]] = {}
        for trial in self.trials:
            atype = trial.attack_type or "unknown"
            if atype not in attack_types:
                attack_types[atype] = []
            attack_types[atype].append(trial)

        for atype, trials in attack_types.items():
            count_compliance = sum(1 for t in trials if t.final_verdict_label == "compliance")
            count_refusal = sum(1 for t in trials if t.final_verdict_label == "refusal")
            asr_for_type = count_compliance / len(trials) if trials else 0.0
            agg.metrics_by_attack_type[atype] = {
                "total": len(trials),
                "compliance": count_compliance,
                "refusal": count_refusal,
                "asr": round(asr_for_type, 4),
            }

        # Metrics by adapter (model-indexed breakdown)
        adapters: Dict[str, List[TrialMetrics]] = {}
        for trial in self.trials:
            adapter_id = trial.model_id or "unknown"
            if adapter_id not in adapters:
                adapters[adapter_id] = []
            adapters[adapter_id].append(trial)

        for adapter_id, trials in adapters.items():
            count_compliance = sum(1 for t in trials if t.final_verdict_label == "compliance")
            count_refusal = sum(1 for t in trials if t.final_verdict_label == "refusal")
            asr_for_adapter = count_compliance / len(trials) if trials else 0.0
            avg_agreement = statistics.mean([t.judge_agreement for t in trials]) if trials else 0.0
            response_times = [t.response_time_ms for t in trials if t.response_time_ms > 0]
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            agg.metrics_by_adapter[adapter_id] = {
                "total": len(trials),
                "compliance": count_compliance,
                "refusal": count_refusal,
                "asr": round(asr_for_adapter, 4),
                "avg_judge_agreement": round(avg_agreement, 4),
                "avg_response_time_ms": round(avg_response_time, 2),
            }

        # Metrics by judge
        judge_names: Dict[str, List[Verdict]] = {}
        for trial in self.trials:
            for verdict in trial.verdicts:
                if verdict.judge_type not in judge_names:
                    judge_names[verdict.judge_type] = []
                judge_names[verdict.judge_type].append(verdict)

        for judge_type, verdicts in judge_names.items():
            count_compliance = sum(
                1 for v in verdicts if "compliance" in v.labels
            )
            count_refusal = sum(
                1 for v in verdicts if "refusal" in v.labels
            )
            agg.metrics_by_judge[judge_type] = {
                "total_verdicts": len(verdicts),
                "compliance": count_compliance,
                "refusal": count_refusal,
            }

        return agg
