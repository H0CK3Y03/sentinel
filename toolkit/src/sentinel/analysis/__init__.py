"""Post-hoc analysis of JSONL experiment logs.

This package keeps the public surface that the CLI imports
(`ExperimentAnalyzer`, `print_report`, and the report dataclasses) on a
single import path while organising the implementation into:

* :mod:`sentinel.analysis.reports` - the dataclasses returned by the analyzer.
* :mod:`sentinel.analysis.analyzer` - the JSONL parser and report builder.
* :mod:`sentinel.analysis.printer` - the human-readable stdout report.
"""

from sentinel.analysis.analyzer import ExperimentAnalyzer
from sentinel.analysis.printer import print_report
from sentinel.analysis.reports import (
    AdapterReport,
    AttackTypeReport,
    ExperimentReport,
    GeneratorReport,
    JudgePerformanceReport,
)

__all__ = [
    "ExperimentAnalyzer",
    "print_report",
    "AdapterReport",
    "AttackTypeReport",
    "ExperimentReport",
    "GeneratorReport",
    "JudgePerformanceReport",
]
