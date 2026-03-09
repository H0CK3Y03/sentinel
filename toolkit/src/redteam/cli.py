"""Command-line interface for the red-teaming toolkit.

Built with Typer so that the toolkit can be invoked as::

    redteam run manifests/example.yaml
    redteam validate manifests/example.yaml
    redteam list-plugins
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from redteam import __version__
from redteam.manifest import load_manifest
from redteam.orchestrator import Orchestrator
from redteam.plugins import (
    _ADAPTER_REGISTRY,
    _GENERATOR_REGISTRY,
    _JUDGE_REGISTRY,
)

app = typer.Typer(
    name="redteam",
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

    typer.echo(f"[redteam] Starting experiment {manifest.experiment_id}")
    typer.echo(f"  adapter   : {manifest.model.adapter} ({manifest.model.model_id})")
    typer.echo(f"  generator : {manifest.generator.name}")
    typer.echo(f"  judges    : {[j.name for j in manifest.judges]}")
    typer.echo(f"  batches   : {manifest.num_batches} × {manifest.batch_size}")
    typer.echo(f"  output    : {manifest.output}")
    typer.echo()

    orch = Orchestrator(manifest)
    summary = asyncio.run(orch.run())

    typer.echo("[redteam] Experiment complete.")
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
    for name in sorted(_ADAPTER_REGISTRY):
        typer.echo(f"  - {name}")
    typer.echo("Generators:")
    for name in sorted(_GENERATOR_REGISTRY):
        typer.echo(f"  - {name}")
    typer.echo("Judges:")
    for name in sorted(_JUDGE_REGISTRY):
        typer.echo(f"  - {name}")


@app.command()
def version() -> None:
    """Print the toolkit version."""
    typer.echo(f"redteam {__version__}")


# ---------------------------------------------------------------------------
# Entry-point (also used by pyproject.toml console_scripts)
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
