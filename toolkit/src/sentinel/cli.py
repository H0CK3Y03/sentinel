"""Command-line interface for the red-teaming toolkit.

Built with Typer so that the toolkit can be invoked as:

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
from sentinel.manifest import load_manifest
from sentinel.orchestrator import Orchestrator
from sentinel.plugins import (
    list_adapters,
    list_generators,
    list_judges,
)

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
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        typer.echo(f"Error loading manifest: {exc}", err=True)
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(json.dumps(manifest.to_dict(), indent=2))
        return

    typer.echo(f"[sentinel] Starting experiment {manifest.experiment_id}")
    adapter_names = ", ".join(
        f"{adapter.adapter}({adapter.instance_id})" for adapter in manifest.adapters
    )
    typer.echo(f"  adapters  : {adapter_names}")
    generator_names = ", ".join(
        f"{generator.name}({generator.instance_id})" for generator in manifest.generators
    )
    typer.echo(f"  generators: {generator_names}")
    typer.echo(
        f"  judges    : {', '.join(f'{judge.name}({judge.instance_id})' for judge in manifest.judges)}"
    )
    typer.echo(
        f"  batches   : {len(manifest.adapters)} * {len(manifest.generators)} * {manifest.num_batches} * {manifest.batch_size}"
    )
    typer.echo(f"  output    : {manifest.output}")
    typer.echo()

    orch = Orchestrator(manifest)
    summary = asyncio.run(orch.run())

    typer.echo("[sentinel] Experiment complete.")
    typer.echo(f"  Total prompts    : {summary.total_prompts}")
    typer.echo(f"  Refusals         : {summary.total_refusals}")
    typer.echo(f"  Compliances      : {summary.total_compliances}")
    typer.echo(f"  Inconclusive     : {summary.total_inconclusive}")
    typer.echo(f"  Errors           : {summary.total_errors}")
    typer.echo(f"  ASR              : {summary.asr:.2%}")
    typer.echo(f"  Elapsed          : {summary.elapsed_seconds:.2f}s")
    typer.echo(f"  Log file         : {manifest.output}")


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

    # Print to stdout
    print_report(report)

    # Optionally save to JSON
    if output_json:
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
