"""Report dataclasses returned by :class:`ExperimentAnalyzer`.

The dataclasses are flat by design: their ``to_dict`` output is the
documented JSON shape produced by ``sentinel analyze --output-json``, so
adding nested fields here would be a breaking change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AttackTypeReport:
    """Summary for a specific attack type (e.g. ``single-turn-jailbreak``)."""
    attack_type: str
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
    """Summary for a specific generator instance."""
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
class AdapterReport:
    """Summary for a specific model adapter."""
    instance_id: str
    model_id: str
    total_prompts: int
    compliances: int
    refusals: int
    inconclusive: int
    asr: float
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
class JudgePerformanceReport:
    """Summary for a specific judge instance."""
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
class ExperimentReport:
    """Top-level report for one experiment run."""
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
