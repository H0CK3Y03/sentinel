"""Parse a JSONL experiment log and build an :class:`ExperimentReport`."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from sentinel.analysis.reports import (
    AdapterReport,
    AttackTypeReport,
    ExperimentReport,
    GeneratorReport,
    JudgePerformanceReport,
)


@dataclass
class _TrialRecord:
    """Flattened view of a logged trial used while building the report."""

    prompt: str
    response: str
    verdict: str
    confidence: float
    response_time: float
    response_tokens: int
    judge_agreement: float

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return the dict shape used by ``top_*_prompts`` in the report."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "response_time": self.response_time,
            "response_tokens": self.response_tokens,
            "judge_agreement": self.judge_agreement,
        }


@dataclass
class _GroupBuckets:
    """Per-group trial buckets accumulated while scanning the log."""

    by_attack_type: Dict[str, List[_TrialRecord]] = field(default_factory=lambda: defaultdict(list))
    by_generator: Dict[str, List[_TrialRecord]] = field(default_factory=lambda: defaultdict(list))
    by_adapter: Dict[str, List[_TrialRecord]] = field(default_factory=lambda: defaultdict(list))
    by_judge: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    generator_names: Dict[str, str] = field(default_factory=dict)
    adapter_models: Dict[str, str] = field(default_factory=dict)

    compliances: List[_TrialRecord] = field(default_factory=list)
    refusals: List[_TrialRecord] = field(default_factory=list)
    judge_agreements: List[float] = field(default_factory=list)


class ExperimentAnalyzer:
    """Parse JSONL experiment logs and emit an :class:`ExperimentReport`.

    Typical usage::

        analyzer = ExperimentAnalyzer("logs/run.jsonl")
        analyzer.load_log()
        report = analyzer.generate_report()
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.trials: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.elapsed_seconds: float = 0.0

    # -- log ingestion -------------------------------------------------------

    def load_log(self) -> None:
        """Parse the JSONL log and keep just the events we need."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                self._consume_line(line)

    def _consume_line(self, line: str) -> None:
        try:
            record = json.loads(line.strip())
        except json.JSONDecodeError:
            # Skip malformed lines rather than aborting the whole report.
            return

        event_type = record.get("event_type")
        if event_type == "trial_result":
            self.trials.append(record)
        elif event_type == "experiment_start":
            self.metadata = record.get("data", {})
        elif event_type == "experiment_end":
            summary = record.get("data", {}).get("summary", {})
            self.elapsed_seconds = summary.get("elapsed_seconds", 0.0)

    # -- report construction -------------------------------------------------

    def generate_report(self) -> ExperimentReport:
        """Generate the final report from previously loaded trials."""
        if not self.trials:
            raise ValueError("No trials loaded. Call load_log() first.")

        buckets = _GroupBuckets()
        for trial in self.trials:
            self._index_trial(trial, buckets)

        return ExperimentReport(
            experiment_id=self.metadata.get("manifest", {}).get("experiment_id", "unknown"),
            total_trials=len(self.trials),
            **self._overall_stats(buckets),
            attack_type_reports=self._attack_type_reports(buckets),
            generator_reports=self._generator_reports(buckets),
            adapter_reports=self._adapter_reports(buckets),
            judge_reports=self._judge_reports(buckets),
            top_compliance_prompts=self._top_n(buckets.compliances, 5),
            top_failure_prompts=self._top_n(buckets.refusals, 5),
            elapsed_seconds=self.elapsed_seconds,
        )

    # -- per-trial indexing --------------------------------------------------

    def _index_trial(self, trial: Dict[str, Any], buckets: _GroupBuckets) -> None:
        data = trial.get("data", {})
        prompt = data.get("prompt", {})
        response = data.get("response", {})
        verdicts = data.get("verdicts", [])

        prompt_metadata = prompt.get("metadata", {})
        attack_type = (
            prompt_metadata.get("attack_type")
            or prompt_metadata.get("generator_name")
            or "unknown"
        )
        generator_name = prompt_metadata.get("generator_name") or attack_type
        generator_instance_id = prompt_metadata.get("generator_instance_id") or attack_type
        adapter_instance_id = response.get("adapter_instance_id") or response.get("model_id", "unknown")

        buckets.generator_names.setdefault(generator_instance_id, generator_name)
        buckets.adapter_models.setdefault(
            adapter_instance_id, response.get("model_id", adapter_instance_id)
        )

        final_label, agreement = self._consensus(verdicts)
        if verdicts:
            buckets.judge_agreements.append(agreement)

        record = _TrialRecord(
            prompt=prompt.get("text", ""),
            response=response.get("text", ""),
            verdict=final_label,
            confidence=self._mean(v.get("confidence", 0.0) for v in verdicts),
            response_time=response.get("latency_ms", 0),
            response_tokens=response.get("tokens", 0),
            judge_agreement=agreement,
        )

        buckets.by_attack_type[attack_type].append(record)
        buckets.by_generator[generator_instance_id].append(record)
        buckets.by_adapter[adapter_instance_id].append(record)

        for verdict in verdicts:
            judge_id = verdict.get("judge_instance_id") or verdict.get("judge_type", "unknown")
            buckets.by_judge[judge_id].append(
                {
                    "judge_type": verdict.get("judge_type", "unknown"),
                    "label": (verdict.get("labels") or ["inconclusive"])[0],
                    "confidence": verdict.get("confidence", 0.0),
                }
            )

        if final_label == "compliance":
            buckets.compliances.append(record)
        elif final_label == "refusal":
            buckets.refusals.append(record)

    @staticmethod
    def _consensus(verdicts: List[Dict[str, Any]]) -> tuple[str, float]:
        """Return ``(final_label, agreement)`` for one trial's judge verdicts.

        The label is decided by majority vote across compliance/refusal; ties
        fall through to ``inconclusive``. Agreement is the share of judges
        that picked the dominant label. With no verdicts, returns
        ``("inconclusive", 0.0)``.
        """
        labels = [(v.get("labels") or ["inconclusive"])[0] for v in verdicts]
        if not labels:
            return "inconclusive", 0.0

        if all(label == labels[0] for label in labels):
            return labels[0], 1.0

        compliance_count = sum(1 for label in labels if label == "compliance")
        refusal_count = sum(1 for label in labels if label == "refusal")
        if compliance_count > refusal_count:
            final_label = "compliance"
        elif refusal_count > compliance_count:
            final_label = "refusal"
        else:
            final_label = "inconclusive"

        agreement = max(compliance_count, refusal_count) / len(labels)
        return final_label, agreement

    # -- summarisation primitives -------------------------------------------

    @staticmethod
    def _mean(values: Any) -> float:
        items = list(values)
        return sum(items) / len(items) if items else 0.0

    @staticmethod
    def _summarise(records: List[_TrialRecord]) -> Dict[str, Any]:
        """Return the per-group totals/rates/averages shared by every report type."""
        if not records:
            return {
                "total_prompts": 0,
                "compliances": 0,
                "refusals": 0,
                "inconclusive": 0,
                "asr": 0.0,
                "avg_confidence": 0.0,
                "avg_response_time_ms": 0.0,
                "avg_response_tokens": 0,
            }

        compliances = sum(1 for r in records if r.verdict == "compliance")
        refusals = sum(1 for r in records if r.verdict == "refusal")
        inconclusive = sum(1 for r in records if r.verdict == "inconclusive")
        return {
            "total_prompts": len(records),
            "compliances": compliances,
            "refusals": refusals,
            "inconclusive": inconclusive,
            "asr": compliances / len(records),
            "avg_confidence": sum(r.confidence for r in records) / len(records),
            "avg_response_time_ms": sum(r.response_time for r in records) / len(records),
            "avg_response_tokens": int(
                sum(r.response_tokens for r in records) / len(records)
            ),
        }

    def _overall_stats(self, buckets: _GroupBuckets) -> Dict[str, Any]:
        total = len(self.trials)
        compliances = len(buckets.compliances)
        refusals = len(buckets.refusals)
        inconclusive = total - compliances - refusals

        observed_records = buckets.compliances + buckets.refusals
        return {
            "overall_asr": compliances / total if total else 0.0,
            "overall_refusal_rate": refusals / total if total else 0.0,
            "overall_inconclusive_rate": inconclusive / total if total else 0.0,
            "avg_judge_agreement": self._mean(buckets.judge_agreements),
            "overall_avg_response_time_ms": self._mean(
                r.response_time for r in observed_records
            ),
            "overall_avg_response_tokens": int(
                self._mean(r.response_tokens for r in observed_records)
            ),
        }

    # -- per-group reports ---------------------------------------------------

    def _attack_type_reports(
        self, buckets: _GroupBuckets
    ) -> Dict[str, AttackTypeReport]:
        return {
            atype: AttackTypeReport(attack_type=atype, **self._summarise(records))
            for atype, records in buckets.by_attack_type.items()
        }

    def _generator_reports(
        self, buckets: _GroupBuckets
    ) -> Dict[str, GeneratorReport]:
        return {
            gid: GeneratorReport(
                instance_id=gid,
                name=buckets.generator_names.get(gid, gid),
                **self._summarise(records),
            )
            for gid, records in buckets.by_generator.items()
        }

    def _adapter_reports(
        self, buckets: _GroupBuckets
    ) -> Dict[str, AdapterReport]:
        return {
            aid: AdapterReport(
                instance_id=aid,
                model_id=buckets.adapter_models.get(aid, aid),
                avg_judge_agreement=self._mean(r.judge_agreement for r in records),
                **self._summarise(records),
            )
            for aid, records in buckets.by_adapter.items()
        }

    @staticmethod
    def _judge_reports(
        buckets: _GroupBuckets,
    ) -> Dict[str, JudgePerformanceReport]:
        reports: Dict[str, JudgePerformanceReport] = {}
        for judge_id, verdicts in buckets.by_judge.items():
            compliances = sum(1 for v in verdicts if v["label"] == "compliance")
            refusals = sum(1 for v in verdicts if v["label"] == "refusal")
            inconclusive = sum(1 for v in verdicts if v["label"] == "inconclusive")
            avg_conf = (
                sum(v["confidence"] for v in verdicts) / len(verdicts)
                if verdicts
                else 0.0
            )
            judge_type = verdicts[0].get("judge_type", "unknown") if verdicts else "unknown"
            reports[judge_id] = JudgePerformanceReport(
                instance_id=judge_id,
                judge_type=judge_type,
                total_verdicts=len(verdicts),
                compliance_verdicts=compliances,
                refusal_verdicts=refusals,
                inconclusive_verdicts=inconclusive,
                avg_confidence=avg_conf,
            )
        return reports

    @staticmethod
    def _top_n(records: List[_TrialRecord], n: int) -> List[Dict[str, Any]]:
        ordered = sorted(records, key=lambda r: r.confidence, reverse=True)
        return [r.to_legacy_dict() for r in ordered[:n]]
