"""Command-line interface for the red-teaming toolkit.

Built with Typer so the toolkit can be invoked as::

    sentinel run manifests/example.yaml
    sentinel validate manifests/example.yaml
    sentinel list-plugins
    sentinel analyze logs/experiment.jsonl
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from sentinel import __version__
from sentinel.analysis import ExperimentAnalyzer, print_report
from sentinel.manifest import Manifest, load_manifest
from sentinel.orchestrator import ExperimentSummary, Orchestrator
from sentinel.plugins import list_adapters, list_generators, list_judges

app = typer.Typer(
    name="sentinel",
    help="LLM Red-Teaming Toolkit - run reproducible adversarial experiments.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    manifest_path: str = typer.Argument(..., help="Path to the experiment manifest (YAML/JSON)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the manifest without executing."),
) -> None:
    """Execute a red-teaming experiment described by a manifest file."""
    manifest = _load_manifest_or_exit(manifest_path)

    if dry_run:
        typer.echo(json.dumps(manifest.to_dict(), indent=2))
        return

    _print_run_plan(manifest)
    summary = asyncio.run(Orchestrator(manifest).run())
    _print_run_summary(summary, manifest)


@app.command()
def validate(
    manifest_path: str = typer.Argument(..., help="Path to the experiment manifest (YAML/JSON)."),
) -> None:
    """Validate a manifest file and print its parsed contents."""
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Manifest is valid.")
    typer.echo(json.dumps(manifest.to_dict(), indent=2))


@app.command("list-plugins")
def list_plugins() -> None:
    """List all registered adapters, generators, and judges."""
    typer.echo("Adapters:")
    for name in list_adapters():
        typer.echo(f"  - {name}")
    typer.echo("Generators:")
    for name in list_generators():
        typer.echo(f"  - {name}")
    typer.echo("Judges:")
    for name in list_judges():
        typer.echo(f"  - {name}")


@app.command()
def version() -> None:
    """Print the toolkit version."""
    typer.echo(f"sentinel {__version__}")


@app.command()
def analyze(
    log_path: str = typer.Argument(
        ..., help="Path to the JSONL experiment log file to analyse."
    ),
    output_json: str = typer.Option(
        None, "--output-json", help="Save report as JSON to this file (optional)."
    ),
) -> None:
    """Analyse an experiment log and generate a detailed report."""
    try:
        analyzer = ExperimentAnalyzer(log_path)
        analyzer.load_log()
        report = analyzer.generate_report()
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Error analysing log: {exc}", err=True)
        raise typer.Exit(code=1)

    print_report(report)

    if output_json:
        _save_report_json(report, output_json)


# ---------------------------------------------------------------------------
# Helpers used by the commands above
# ---------------------------------------------------------------------------

def _load_manifest_or_exit(manifest_path: str) -> Manifest:
    try:
        return load_manifest(manifest_path)
    except Exception as exc:
        typer.echo(f"Error loading manifest: {exc}", err=True)
        raise typer.Exit(code=1)


def _print_run_plan(manifest: Manifest) -> None:
    typer.echo(f"[sentinel] Starting experiment {manifest.experiment_id}")
    typer.echo(
        "  adapters  : "
        + ", ".join(f"{a.adapter}({a.instance_id})" for a in manifest.adapters)
    )
    typer.echo(
        "  generators: "
        + ", ".join(f"{g.name}({g.instance_id})" for g in manifest.generators)
    )
    typer.echo(
        "  judges    : "
        + ", ".join(f"{j.name}({j.instance_id})" for j in manifest.judges)
    )
    total_prompts = (
        len(manifest.adapters)
        * len(manifest.generators)
        * manifest.num_batches
        * manifest.batch_size
    )
    typer.echo(
        f"  batches   : "
        f"{len(manifest.adapters)} * {len(manifest.generators)} * "
        f"{manifest.num_batches} * {manifest.batch_size} = {total_prompts}"
    )
    typer.echo(f"  combo conc: {manifest.max_combo_concurrency}")
    typer.echo(f"  prompt conc: {manifest.max_concurrency}")
    typer.echo(f"  pipeline  : {manifest.pipeline_mode}")
    typer.echo(f"  output    : {manifest.output}")
    typer.echo()


def _print_run_summary(summary: ExperimentSummary, manifest: Manifest) -> None:
    typer.echo("[sentinel] Experiment complete.")
    typer.echo(f"  Total prompts    : {summary.total_prompts}")
    typer.echo(f"  Refusals         : {summary.total_refusals}")
    typer.echo(f"  Compliances      : {summary.total_compliances}")
    typer.echo(f"  Inconclusive     : {summary.total_inconclusive}")
    typer.echo(f"  Errors           : {summary.total_errors}")
    typer.echo(f"  ASR              : {summary.asr:.2%}")
    typer.echo(f"  Elapsed          : {summary.elapsed_seconds:.2f}s")
    typer.echo(f"  Log file         : {manifest.output}")


def _save_report_json(report, output_json: str) -> None:
    try:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        typer.echo(f"\nReport saved to: {output_json}")
    except Exception as exc:
        typer.echo(f"Error saving JSON report: {exc}", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entry-point (also used by pyproject.toml console_scripts)
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
