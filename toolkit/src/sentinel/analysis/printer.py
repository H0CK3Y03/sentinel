"""Pretty-print an :class:`ExperimentReport` to stdout."""

from __future__ import annotations

from sentinel.analysis.reports import ExperimentReport


_DIVIDER = "=" * 80


def print_report(report: ExperimentReport) -> None:
    """Render the report as a series of human-readable sections."""
    print()
    print(_DIVIDER)
    print(f"EXPERIMENT REPORT: {report.experiment_id}")
    print(_DIVIDER)

    _print_overall(report)
    _print_attack_types(report)
    _print_generators(report)
    _print_adapters(report)
    _print_judges(report)

    print()
    print(_DIVIDER)


def _print_overall(report: ExperimentReport) -> None:
    print("\nOverall Statistics:")
    print(f"  Total Trials:         {report.total_trials}")
    print(f"  Attack Success Rate:  {report.overall_asr:.2%}")
    print(f"  Refusal Rate:         {report.overall_refusal_rate:.2%}")
    print(f"  Inconclusive Rate:    {report.overall_inconclusive_rate:.2%}")
    print(f"  Avg Judge Agreement:  {report.avg_judge_agreement:.2%}")
    print(f"  Avg Response Time:    {report.overall_avg_response_time_ms:.2f} ms")
    print(f"  Avg Response Tokens:  {report.overall_avg_response_tokens}")
    print(f"  Elapsed Time:         {report.elapsed_seconds:.2f} seconds")


def _print_attack_types(report: ExperimentReport) -> None:
    if not report.attack_type_reports:
        return
    print("\nAttack Type Breakdown:")
    for atype, sub in report.attack_type_reports.items():
        print(f"  {atype}:")
        print(f"    - Total:        {sub.total_prompts}")
        print(f"    - Compliances:  {sub.compliances}")
        print(f"    - ASR:          {sub.asr:.2%}")
        print(f"    - Avg Conf:     {sub.avg_confidence:.2%}")


def _print_generators(report: ExperimentReport) -> None:
    if not report.generator_reports:
        return
    print("\nGenerator Breakdown:")
    for instance_id, sub in report.generator_reports.items():
        print(f"  {instance_id}:")
        print(f"    - Name:         {sub.name}")
        print(f"    - Total:        {sub.total_prompts}")
        print(f"    - Compliances:  {sub.compliances}")
        print(f"    - ASR:          {sub.asr:.2%}")


def _print_adapters(report: ExperimentReport) -> None:
    if not report.adapter_reports:
        return
    print("\nAdapter/Model Breakdown:")
    for instance_id, sub in report.adapter_reports.items():
        print(f"  {instance_id}:")
        print(f"    - Model ID:            {sub.model_id}")
        print(f"    - Total:               {sub.total_prompts}")
        print(f"    - Compliances:         {sub.compliances}")
        print(f"    - ASR:                 {sub.asr:.2%}")
        print(f"    - Avg Judge Agreement: {sub.avg_judge_agreement:.2%}")
        print(f"    - Avg Response Time:   {sub.avg_response_time_ms:.2f} ms")


def _print_judges(report: ExperimentReport) -> None:
    if not report.judge_reports:
        return
    print("\nJudge Performance:")
    for instance_id, sub in report.judge_reports.items():
        print(f"  {instance_id}:")
        print(f"    - Judge Type:        {sub.judge_type}")
        print(f"    - Total Verdicts:    {sub.total_verdicts}")
        print(f"    - Compliances:       {sub.compliance_verdicts}")
        print(f"    - Avg Confidence:    {sub.avg_confidence:.2%}")
