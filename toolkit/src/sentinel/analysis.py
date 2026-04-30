"""Analysis and reporting module for red-teaming experiments.

Parses JSONL experiment logs and generates comprehensive reports with
metrics, visualizations, and actionable insights.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from sentinel.metrics import TrialMetrics


@dataclass
class AttackTypeReport:
    """Summary report for a specific attack type."""
    attack_type: str
    total_prompts: int
    compliances: int
    refusals: int
    inconclusive: int
    asr: float  # attack success rate
    avg_confidence: float
    avg_response_time_ms: float
    avg_response_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "total_prompts": self.total_prompts,
            "compliances": self.compliances,
            "refusals": self.refusals,
            "inconclusive": self.inconclusive,
            "attack_success_rate": round(self.asr, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "avg_response_tokens": self.avg_response_tokens,
        }


@dataclass
class GeneratorReport:
    """Summary report for a specific generator instance."""
    instance_id: str
    name: str
    total_prompts: int
    compliances: int
    refusals: int
    inconclusive: int
    asr: float
    avg_confidence: float
    avg_response_time_ms: float
    avg_response_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "total_prompts": self.total_prompts,
            "compliances": self.compliances,
            "refusals": self.refusals,
            "inconclusive": self.inconclusive,
            "attack_success_rate": round(self.asr, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "avg_response_tokens": self.avg_response_tokens,
        }


@dataclass
class JudgePerformanceReport:
    """Summary report for a specific judge."""
    instance_id: str
    judge_type: str
    total_verdicts: int
    compliance_verdicts: int
    refusal_verdicts: int
    inconclusive_verdicts: int
    avg_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "judge_type": self.judge_type,
            "total_verdicts": self.total_verdicts,
            "compliance_verdicts": self.compliance_verdicts,
            "refusal_verdicts": self.refusal_verdicts,
            "inconclusive_verdicts": self.inconclusive_verdicts,
            "avg_confidence": round(self.avg_confidence, 4),
        }


@dataclass
class AdapterReport:
    """Summary report for a specific model adapter."""
    instance_id: str
    model_id: str
    total_prompts: int
    compliances: int
    refusals: int
    inconclusive: int
    asr: float  # attack success rate
    avg_confidence: float
    avg_judge_agreement: float
    avg_response_time_ms: float
    avg_response_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "model_id": self.model_id,
            "total_prompts": self.total_prompts,
            "compliances": self.compliances,
            "refusals": self.refusals,
            "inconclusive": self.inconclusive,
            "attack_success_rate": round(self.asr, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_judge_agreement": round(self.avg_judge_agreement, 4),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "avg_response_tokens": self.avg_response_tokens,
        }


@dataclass
class ExperimentReport:
    """Comprehensive report for an entire experiment."""
    experiment_id: str
    total_trials: int
    overall_asr: float
    overall_refusal_rate: float
    overall_inconclusive_rate: float
    avg_judge_agreement: float
    attack_type_reports: Dict[str, AttackTypeReport]
    generator_reports: Dict[str, GeneratorReport]
    adapter_reports: Dict[str, AdapterReport]
    judge_reports: Dict[str, JudgePerformanceReport]
    top_compliance_prompts: List[Dict[str, Any]]
    top_failure_prompts: List[Dict[str, Any]]
    overall_avg_response_time_ms: float
    overall_avg_response_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "total_trials": self.total_trials,
            "overall_asr": round(self.overall_asr, 4),
            "overall_refusal_rate": round(self.overall_refusal_rate, 4),
            "overall_inconclusive_rate": round(self.overall_inconclusive_rate, 4),
            "avg_judge_agreement": round(self.avg_judge_agreement, 4),
            "attack_type_reports": {
                k: v.to_dict() for k, v in self.attack_type_reports.items()
            },
            "generator_reports": {
                k: v.to_dict() for k, v in self.generator_reports.items()
            },
            "adapter_reports": {
                k: v.to_dict() for k, v in self.adapter_reports.items()
            },
            "judge_reports": {
                k: v.to_dict() for k, v in self.judge_reports.items()
            },
            "top_compliance_prompts": self.top_compliance_prompts,
            "top_failure_prompts": self.top_failure_prompts,
            "overall_avg_response_time_ms": round(self.overall_avg_response_time_ms, 2),
            "overall_avg_response_tokens": self.overall_avg_response_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class ExperimentAnalyzer:
    """Parse and analyse JSONL experiment logs."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.trials: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.elapsed_seconds: float = 0.0

    def load_log(self) -> None:
        """Load and parse the JSONL experiment log."""
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get("event_type") == "trial_result":
                        self.trials.append(record)
                    elif record.get("event_type") == "experiment_start":
                        self.metadata = record.get("data", {})
                    elif record.get("event_type") == "experiment_end":
                        summary = record.get("data", {}).get("summary", {})
                        self.elapsed_seconds = summary.get("elapsed_seconds", 0.0)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

    def generate_report(self) -> ExperimentReport:
        """Generate a comprehensive analysis report from loaded trials."""
        if not self.trials:
            raise ValueError("No trials loaded. Call load_log() first.")

        # Extract metadata
        experiment_id = self.metadata.get("manifest", {}).get("experiment_id", "unknown")

        # Collect metrics by attack type
        attack_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        generator_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        generator_names: Dict[str, str] = {}
        adapter_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        adapter_models: Dict[str, str] = {}
        judge_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        all_compliances: List[Dict[str, Any]] = []
        all_refusals: List[Dict[str, Any]] = []
        judge_agreements: List[float] = []

        for trial in self.trials:
            trial_data = trial.get("data", {})
            prompt = trial_data.get("prompt", {})
            response = trial_data.get("response", {})
            verdicts = trial_data.get("verdicts", [])

            # Track attack type (use "unknown" if not present)
            prompt_metadata = prompt.get("metadata", {})
            attack_type = (
                prompt_metadata.get("attack_type")
                or prompt_metadata.get("generator_name")
                or "unknown"
            )
            generator_name = prompt_metadata.get("generator_name") or attack_type

            generator_instance_id = prompt_metadata.get("generator_instance_id") or attack_type
            generator_names.setdefault(generator_instance_id, generator_name)

            # Track model/adapter ID (use "unknown" if not present)
            adapter_instance_id = response.get("adapter_instance_id") or response.get("model_id", "unknown")

            # Compute aggregate verdict from multiple judges
            # Simple aggregation: majority vote or consensus
            verdict_labels = [v.get("labels", ["inconclusive"])[0] for v in verdicts]
            verdict_confidences = [v.get("confidence", 0.0) for v in verdicts]

            # Determine final verdict: if all agree, use that, otherwise check majority
            final_label = "inconclusive"
            if verdict_labels:
                # Check if all judges agree
                if all(label == verdict_labels[0] for label in verdict_labels):
                    final_label = verdict_labels[0]
                    # All judges agree - perfect agreement
                    agreement = 1.0
                else:
                    # Majority vote
                    compliance_count = sum(1 for l in verdict_labels if l == "compliance")
                    refusal_count = sum(1 for l in verdict_labels if l == "refusal")
                    if compliance_count > refusal_count:
                        final_label = "compliance"
                    elif refusal_count > compliance_count:
                        final_label = "refusal"
                    # Calculate agreement as overlap ratio
                    agreement = max(compliance_count, refusal_count) / len(verdict_labels)

                judge_agreements.append(agreement)

            # Average confidence across judges
            avg_verdict_confidence = sum(verdict_confidences) / len(verdict_confidences) if verdict_confidences else 0.0

            trial_record = {
                "prompt": prompt.get("text", ""),
                "response": response.get("text", ""),
                "verdict": final_label,
                "confidence": avg_verdict_confidence,
                "response_time": response.get("latency_ms", 0),
                "response_tokens": response.get("tokens", 0),
                "judge_agreement": agreement if verdict_labels else 0.0,
            }

            attack_metrics[attack_type].append(trial_record)
            generator_metrics[generator_instance_id].append(trial_record)
            adapter_metrics[adapter_instance_id].append(trial_record)
            adapter_models.setdefault(adapter_instance_id, response.get("model_id", adapter_instance_id))

            # Track judge verdicts
            for verdict in verdicts:
                judge_instance_id = verdict.get("judge_instance_id") or verdict.get("judge_type", "unknown")
                judge_metrics[judge_instance_id].append({
                    "judge_instance_id": judge_instance_id,
                    "judge_type": verdict.get("judge_type", "unknown"),
                    "label": verdict.get("labels", ["inconclusive"])[0],
                    "confidence": verdict.get("confidence", 0.0),
                })

            # Track compliance/refusal examples
            if final_label == "compliance":
                all_compliances.append(trial_record)
            elif final_label == "refusal":
                all_refusals.append(trial_record)

        # Compute overall statistics
        total_trials = len(self.trials)
        total_compliances = len(all_compliances)
        total_refusals = len(all_refusals)
        total_inconclusive = total_trials - total_compliances - total_refusals

        overall_asr = total_compliances / total_trials if total_trials > 0 else 0.0
        overall_refusal_rate = total_refusals / total_trials if total_trials > 0 else 0.0
        overall_inconclusive_rate = total_inconclusive / total_trials if total_trials > 0 else 0.0

        # Judge agreement (average across all trials)
        avg_judge_agreement = sum(judge_agreements) / len(judge_agreements) if judge_agreements else 0.0

        # Average response characteristics
        response_times = [t.get("response_time", 0) for t in all_compliances + all_refusals]
        response_tokens = [t.get("response_tokens", 0) for t in all_compliances + all_refusals]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        avg_response_tokens = int(sum(response_tokens) / len(response_tokens)) if response_tokens else 0

        # Generate per-attack-type reports
        attack_type_reports: Dict[str, AttackTypeReport] = {}
        for atype, metrics_list in attack_metrics.items():
            compliances = sum(1 for m in metrics_list if m["verdict"] == "compliance")
            refusals = sum(1 for m in metrics_list if m["verdict"] == "refusal")
            inconclusive = sum(1 for m in metrics_list if m["verdict"] == "inconclusive")
            asr = compliances / len(metrics_list) if metrics_list else 0.0
            avg_conf = sum(m["confidence"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_time = sum(m["response_time"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_tokens = int(sum(m["response_tokens"] for m in metrics_list) / len(metrics_list)) if metrics_list else 0

            attack_type_reports[atype] = AttackTypeReport(
                attack_type=atype,
                total_prompts=len(metrics_list),
                compliances=compliances,
                refusals=refusals,
                inconclusive=inconclusive,
                asr=asr,
                avg_confidence=avg_conf,
                avg_response_time_ms=avg_time,
                avg_response_tokens=avg_tokens,
            )

        # Generate per-generator-instance reports
        generator_reports: Dict[str, GeneratorReport] = {}
        for generator_id, metrics_list in generator_metrics.items():
            compliances = sum(1 for m in metrics_list if m["verdict"] == "compliance")
            refusals = sum(1 for m in metrics_list if m["verdict"] == "refusal")
            inconclusive = sum(1 for m in metrics_list if m["verdict"] == "inconclusive")
            asr = compliances / len(metrics_list) if metrics_list else 0.0
            avg_conf = sum(m["confidence"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_time = sum(m["response_time"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_tokens = int(sum(m["response_tokens"] for m in metrics_list) / len(metrics_list)) if metrics_list else 0

            generator_reports[generator_id] = GeneratorReport(
                instance_id=generator_id,
                name=generator_names.get(generator_id, generator_id),
                total_prompts=len(metrics_list),
                compliances=compliances,
                refusals=refusals,
                inconclusive=inconclusive,
                asr=asr,
                avg_confidence=avg_conf,
                avg_response_time_ms=avg_time,
                avg_response_tokens=avg_tokens,
            )

        # Generate per-adapter reports
        adapter_reports: Dict[str, AdapterReport] = {}
        for adapter_id, metrics_list in adapter_metrics.items():
            compliances = sum(1 for m in metrics_list if m["verdict"] == "compliance")
            refusals = sum(1 for m in metrics_list if m["verdict"] == "refusal")
            inconclusive = sum(1 for m in metrics_list if m["verdict"] == "inconclusive")
            asr = compliances / len(metrics_list) if metrics_list else 0.0
            avg_conf = sum(m["confidence"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_agreement = sum(m["judge_agreement"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_time = sum(m["response_time"] for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            avg_tokens = int(sum(m["response_tokens"] for m in metrics_list) / len(metrics_list)) if metrics_list else 0

            adapter_reports[adapter_id] = AdapterReport(
                instance_id=adapter_id,
                model_id=adapter_models.get(adapter_id, adapter_id),
                total_prompts=len(metrics_list),
                compliances=compliances,
                refusals=refusals,
                inconclusive=inconclusive,
                asr=asr,
                avg_confidence=avg_conf,
                avg_judge_agreement=avg_agreement,
                avg_response_time_ms=avg_time,
                avg_response_tokens=avg_tokens,
            )

        # Generate per-judge reports
        judge_reports: Dict[str, JudgePerformanceReport] = {}
        for judge_id, verdicts in judge_metrics.items():
            compliances = sum(1 for v in verdicts if v["label"] == "compliance")
            refusals = sum(1 for v in verdicts if v["label"] == "refusal")
            inconclusive = sum(1 for v in verdicts if v["label"] == "inconclusive")
            avg_conf = sum(v["confidence"] for v in verdicts) / len(verdicts) if verdicts else 0.0
            judge_type = verdicts[0].get("judge_type", "unknown") if verdicts else "unknown"

            judge_reports[judge_id] = JudgePerformanceReport(
                instance_id=judge_id,
                judge_type=judge_type,
                total_verdicts=len(verdicts),
                compliance_verdicts=compliances,
                refusal_verdicts=refusals,
                inconclusive_verdicts=inconclusive,
                avg_confidence=avg_conf,
            )

        # Top examples
        top_compliance_prompts = sorted(
            all_compliances,
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:5]
        top_failure_prompts = sorted(
            all_refusals,
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:5]

        return ExperimentReport(
            experiment_id=experiment_id,
            total_trials=total_trials,
            overall_asr=overall_asr,
            overall_refusal_rate=overall_refusal_rate,
            overall_inconclusive_rate=overall_inconclusive_rate,
            avg_judge_agreement=avg_judge_agreement,
            attack_type_reports=attack_type_reports,
            generator_reports=generator_reports,
            adapter_reports=adapter_reports,
            judge_reports=judge_reports,
            top_compliance_prompts=top_compliance_prompts,
            top_failure_prompts=top_failure_prompts,
            overall_avg_response_time_ms=avg_response_time,
            overall_avg_response_tokens=avg_response_tokens,
            elapsed_seconds=self.elapsed_seconds,
        )


def print_report(report: ExperimentReport) -> None:
    """Pretty-print a comprehensive experiment report to stdout."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT REPORT: {report.experiment_id}")
    print("=" * 80)

    print(f"\nOverall Statistics:")
    print(f"  Total Trials:         {report.total_trials}")
    print(f"  Attack Success Rate:  {report.overall_asr:.2%}")
    print(f"  Refusal Rate:         {report.overall_refusal_rate:.2%}")
    print(f"  Inconclusive Rate:    {report.overall_inconclusive_rate:.2%}")
    print(f"  Avg Judge Agreement:  {report.avg_judge_agreement:.2%}")
    print(f"  Avg Response Time:    {report.overall_avg_response_time_ms:.2f} ms")
    print(f"  Avg Response Tokens:  {report.overall_avg_response_tokens}")
    print(f"  Elapsed Time:         {report.elapsed_seconds:.2f} seconds")

    if report.attack_type_reports:
        print(f"\nAttack Type Breakdown:")
        for atype, atreport in report.attack_type_reports.items():
            print(f"  {atype}:")
            print(f"    - Total:      {atreport.total_prompts}")
            print(f"    - Compliances: {atreport.compliances}")
            print(f"    - ASR:        {atreport.asr:.2%}")
            print(f"    - Avg Conf:   {atreport.avg_confidence:.2%}")

    if report.generator_reports:
        print(f"\nGenerator Breakdown:")
        for generator_id, greport in report.generator_reports.items():
            print(f"  {generator_id}:")
            print(f"    - Name:       {greport.name}")
            print(f"    - Total:      {greport.total_prompts}")
            print(f"    - Compliances: {greport.compliances}")
            print(f"    - ASR:        {greport.asr:.2%}")

    if report.adapter_reports:
        print(f"\nAdapter/Model Breakdown:")
        for instance_id, areport in report.adapter_reports.items():
            print(f"  {instance_id}:")
            print(f"    - Model ID:   {areport.model_id}")
            print(f"    - Total:      {areport.total_prompts}")
            print(f"    - Compliances: {areport.compliances}")
            print(f"    - ASR:        {areport.asr:.2%}")
            print(f"    - Avg Judge Agreement: {areport.avg_judge_agreement:.2%}")
            print(f"    - Avg Response Time:   {areport.avg_response_time_ms:.2f} ms")

    if report.judge_reports:
        print(f"\nJudge Performance:")
        for instance_id, jreport in report.judge_reports.items():
            print(f"  {instance_id}:")
            print(f"    - Judge Type:       {jreport.judge_type}")
            print(f"    - Total Verdicts:    {jreport.total_verdicts}")
            print(f"    - Compliances:       {jreport.compliance_verdicts}")
            print(f"    - Avg Confidence:    {jreport.avg_confidence:.2%}")

    print(f"\n" + "=" * 80)
