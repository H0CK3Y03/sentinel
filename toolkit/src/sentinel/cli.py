"""Command-line interface for the red-teaming toolkit.

Built with Typer so the toolkit can be invoked as::

    sentinel run manifests/example.yaml
    sentinel run manifests/example.yaml --quiet
    sentinel validate manifests/example.yaml
    sentinel list-plugins
    sentinel analyze logs/experiment.jsonl
    sentinel interactive manifests/example.yaml
    sentinel interactive --adapter openai-api --model-id my-model --base-url https://...
    sentinel version
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer

from sentinel import __version__
from sentinel.analysis import ExperimentAnalyzer, print_report
from sentinel.cli_display import (
    make_run_callbacks,
    print_health_line,
    print_interactive_trial,
)
from sentinel.manifest import Manifest, load_manifest
from sentinel.models import PromptCandidate
from sentinel.orchestrator import ExperimentSummary, Orchestrator
from sentinel.plugins import (
    create_adapter,
    create_generator,
    create_judge,
    list_adapters,
    list_generators,
    list_judges,
)

app = typer.Typer(
    name="sentinel",
    help="LLM Red-Teaming Toolkit — run reproducible adversarial experiments.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    manifest_path: str = typer.Argument(..., help="Path to the experiment manifest (YAML/JSON)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the manifest without executing."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-trial output; show only the summary."),
) -> None:
    """Execute a red-teaming experiment described by a manifest file."""
    manifest = _load_manifest_or_exit(manifest_path)

    if dry_run:
        typer.echo(json.dumps(manifest.to_dict(), indent=2))
        return

    _print_run_plan(manifest)
    total_expected = _compute_total_expected(manifest)

    if quiet:
        orch = Orchestrator(manifest)
        finalize = None
    else:
        gen_names = [
            create_generator(g.name).get_display_name()
            for g in manifest.generators
        ]
        on_trial, on_gen, on_adp, on_jdg, finalize = make_run_callbacks(
            total_expected, gen_names
        )
        orch = Orchestrator(
            manifest,
            on_trial_complete=on_trial,
            on_generation_start=on_gen,
            on_adapter_start=on_adp,
            on_judge_start=on_jdg,
        )

    summary = asyncio.run(orch.run())
    if finalize is not None:
        finalize()
    _print_run_summary(summary, manifest)


@app.command()
def interactive(
    manifest_path: Optional[str] = typer.Argument(
        None,
        help="Manifest file to load adapter/judge config from (optional).",
    ),
    adapter_name: str = typer.Option(
        "stub", "--adapter", "-a",
        help="Adapter name when no manifest is given (e.g. openai-api, llama-cpp).",
    ),
    model_id: str = typer.Option(
        "model", "--model-id",
        help="Model ID string when no manifest is given.",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url",
        help="API base URL for the openai-api adapter (e.g. https://llm.ai.e-infra.cz/v1).",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        envvar="SENTINEL_API_KEY",
        help="API key (also read from SENTINEL_API_KEY env var).",
    ),
    judge_name: str = typer.Option(
        "heuristic", "--judge", "-j",
        help="Judge to evaluate responses when no manifest is given.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt",
        help="System prompt sent before every user turn.",
    ),
) -> None:
    """Start an interactive manual red-teaming session (REPL).

    Load adapter/judge from a manifest, or configure them via flags.
    Type a prompt, see the model response and verdict, repeat.
    Enter 'exit', 'quit', or press Ctrl-C to end the session.
    """
    if manifest_path:
        adapter, judge = _build_components_from_manifest(manifest_path)
    else:
        adapter, judge = _build_components_from_flags(
            adapter_name=adapter_name,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            judge_name=judge_name,
            system_prompt=system_prompt,
        )

    loop = asyncio.new_event_loop()
    try:
        health = loop.run_until_complete(adapter.health_check())
    except Exception as exc:
        typer.echo(f"[sentinel] Health check failed: {exc}", err=True)
        raise typer.Exit(code=1)

    _print_interactive_banner(adapter.model_id, judge.name, health.value)
    _run_interactive_loop(loop, adapter, judge)
    loop.close()


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
# Interactive helpers
# ---------------------------------------------------------------------------

def _build_components_from_manifest(manifest_path: str):
    """Load the first adapter and judge from *manifest_path*."""
    manifest = _load_manifest_or_exit(manifest_path)
    adapter_cfg = manifest.adapters[0]
    judge_cfg = manifest.judges[0]

    if len(manifest.adapters) > 1:
        typer.secho(
            f"  [note] Manifest has {len(manifest.adapters)} adapters — using the first: "
            f"{adapter_cfg.adapter}({adapter_cfg.instance_id})",
            fg=typer.colors.YELLOW,
        )
    if len(manifest.judges) > 1:
        typer.secho(
            f"  [note] Manifest has {len(manifest.judges)} judges — using the first: "
            f"{judge_cfg.name}({judge_cfg.instance_id})",
            fg=typer.colors.YELLOW,
        )

    adapter = create_adapter(adapter_cfg.adapter, adapter_cfg.model_id, adapter_cfg.config)
    adapter.configure(adapter_cfg.config)
    judge = create_judge(judge_cfg.name)
    judge.configure(judge_cfg.config)
    return adapter, judge


def _build_components_from_flags(
    *,
    adapter_name: str,
    model_id: str,
    base_url: Optional[str],
    api_key: Optional[str],
    judge_name: str,
    system_prompt: Optional[str],
):
    """Build adapter and judge from individual CLI flags (no manifest)."""
    config: dict = {}
    if base_url:
        config["base_url"] = base_url
    if api_key:
        config["api_key"] = api_key
    if system_prompt:
        config["system_prompt"] = system_prompt

    adapter = create_adapter(adapter_name, model_id, config)
    adapter.configure(config)
    judge = create_judge(judge_name)
    judge.configure({})
    return adapter, judge


def _print_interactive_banner(adapter_id: str, judge_name: str, health: str) -> None:
    typer.echo()
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.secho("  sentinel — interactive mode", bold=True)
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"  Adapter : {adapter_id}")
    typer.echo(f"  Judge   : {judge_name}")
    print_health_line("adapter", health)
    typer.echo()
    typer.secho("  Type your prompt and press Enter.", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("  Commands: 'exit' / 'quit' / Ctrl-C to end the session.", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("─" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo()


def _run_interactive_loop(loop, adapter, judge) -> None:
    """Read prompts from stdin, query the adapter, and print verdicts."""
    trial = 0
    while True:
        try:
            raw = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            typer.secho("[sentinel] Session ended.", fg=typer.colors.BRIGHT_BLACK)
            return

        if raw.lower() in ("exit", "quit", "q"):
            typer.secho("[sentinel] Session ended.", fg=typer.colors.BRIGHT_BLACK)
            return
        if not raw:
            continue

        trial += 1
        prompt = PromptCandidate(text=raw)

        typer.secho(f"  [→] Querying {adapter.model_id}…", fg=typer.colors.BRIGHT_BLACK)
        try:
            response = loop.run_until_complete(adapter.generate(raw))
        except Exception as exc:
            typer.secho(f"  [error] {exc}", fg=typer.colors.RED, err=True)
            continue

        try:
            verdict = judge.evaluate(response, prompt)
        except Exception as exc:
            typer.secho(f"  [judge error] {exc}", fg=typer.colors.YELLOW, err=True)
            verdict = None

        print_interactive_trial(trial, response, verdict)


# ---------------------------------------------------------------------------
# Run-command helpers
# ---------------------------------------------------------------------------

def _compute_total_expected(manifest: Manifest) -> int:
    """Estimate total conversations (multi-turn counts as one)."""
    return (
        len(manifest.adapters)
        * manifest.num_batches
        * manifest.batch_size
        * len(manifest.generators)
    )


def _load_manifest_or_exit(manifest_path: str) -> Manifest:
    try:
        return load_manifest(manifest_path)
    except Exception as exc:
        typer.echo(f"Error loading manifest: {exc}", err=True)
        raise typer.Exit(code=1)


def _print_run_plan(manifest: Manifest) -> None:
    total_prompts = _compute_total_expected(manifest)
    typer.echo()
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.secho(f"  sentinel — experiment {manifest.experiment_id}", bold=True)
    if manifest.description:
        typer.echo(f"  {manifest.description}")
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo(
        "  Adapters   : "
        + ", ".join(f"{a.adapter}({a.instance_id})" for a in manifest.adapters)
    )
    typer.echo(
        "  Generators : "
        + ", ".join(f"{g.name}({g.instance_id})" for g in manifest.generators)
    )
    typer.echo(
        "  Judges     : "
        + ", ".join(f"{j.name}({j.instance_id})" for j in manifest.judges)
    )
    typer.echo(
        f"  Prompts    : "
        f"{len(manifest.adapters)} adapters × {manifest.num_batches} batches × "
        f"{manifest.batch_size} × {len(manifest.generators)} generators ≈ {total_prompts} conversations"
    )
    num_combos = len(manifest.adapters) * len(manifest.generators)
    typer.echo(
        f"  Combos     : {num_combos} adapter×generator pair(s), "
        f"{min(manifest.max_combo_concurrency, num_combos)} running in parallel"
    )
    typer.echo(f"  Concurrency: {manifest.max_concurrency} prompt(s) in parallel per combo")
    typer.echo(f"  Pipeline   : {manifest.pipeline_mode}")
    typer.echo(f"  Output     : {manifest.output}")
    typer.secho("─" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo()


def _print_run_summary(summary: ExperimentSummary, manifest: Manifest) -> None:
    typer.echo()
    typer.secho("─" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.secho("  Experiment complete.", bold=True)
    typer.echo(f"  Experiment ID    : {summary.experiment_id}")
    typer.echo(f"  Total prompts    : {summary.total_prompts}")
    typer.secho(f"  Compliances      : {summary.total_compliances}", fg=typer.colors.RED)
    typer.secho(f"  Refusals         : {summary.total_refusals}", fg=typer.colors.GREEN)
    typer.echo(f"  Inconclusive     : {summary.total_inconclusive}")
    typer.echo(f"  Errors           : {summary.total_errors}")
    asr_color = (
        typer.colors.RED if summary.asr > 0.3
        else typer.colors.YELLOW if summary.asr > 0
        else typer.colors.GREEN
    )
    typer.secho(f"  Attack Success   : {summary.asr:.2%}", fg=asr_color, bold=True)
    typer.echo(f"  Total tokens     : {summary.total_tokens:,}")
    typer.echo(f"  Avg latency      : {summary.avg_latency_ms:.0f} ms")
    typer.echo(f"  Elapsed          : {summary.elapsed_seconds:.2f}s")
    typer.echo(f"  Log              : {manifest.output}")
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo()


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

def _load_dotenv() -> None:
    """Load a .env file from the cwd (or any parent directory) without extra deps."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_file = directory / ".env"
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def main() -> None:
    _load_dotenv()
    app()


if __name__ == "__main__":
    main()
