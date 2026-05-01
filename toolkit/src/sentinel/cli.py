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
from typing import Callable, Optional

import typer

from sentinel import __version__
from sentinel.analysis import ExperimentAnalyzer, print_report
from sentinel.manifest import Manifest, load_manifest
from sentinel.models import ModelResponse, PromptCandidate, Verdict
from sentinel.orchestrator import ExperimentSummary, Orchestrator
from sentinel.plugins import create_adapter, create_judge, list_adapters, list_generators, list_judges

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

    total_expected = (
        len(manifest.adapters)
        * len(manifest.generators)
        * manifest.num_batches
        * manifest.batch_size
    )

    callback: Optional[Callable] = None if quiet else _make_trial_callback(total_expected)
    summary = asyncio.run(Orchestrator(manifest, on_trial_complete=callback).run())
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

    Examples
    --------
    Using a manifest::

        sentinel interactive manifests/einfra-api.yaml

    Direct flags::

        sentinel interactive --adapter openai-api --model-id deepseek-r1-distill-qwen-32b \\
            --base-url https://llm.ai.e-infra.cz/v1 --api-key $SENTINEL_API_KEY
    """
    if manifest_path:
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

    else:
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

    # Health-check before starting the loop.
    loop = asyncio.new_event_loop()
    try:
        health = loop.run_until_complete(adapter.health_check())
    except Exception as exc:
        typer.echo(f"[sentinel] Health check failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo()
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.secho("  sentinel — interactive mode", bold=True)
    typer.secho("═" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"  Adapter : {adapter.model_id}")
    typer.echo(f"  Judge   : {judge.name}")
    _print_health_line("adapter", health.value)
    typer.echo()
    typer.secho("  Type your prompt and press Enter.", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("  Commands: 'exit' / 'quit' / Ctrl-C to end the session.", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("─" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.echo()

    trial = 0
    while True:
        try:
            raw = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            typer.secho("[sentinel] Session ended.", fg=typer.colors.BRIGHT_BLACK)
            break

        if raw.lower() in ("exit", "quit", "q"):
            typer.secho("[sentinel] Session ended.", fg=typer.colors.BRIGHT_BLACK)
            break
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

        _print_interactive_trial(trial, response, verdict)

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
# Per-trial output helpers
# ---------------------------------------------------------------------------

def _verdict_label(verdict: Optional[Verdict]) -> str:
    if verdict is None:
        return "ERROR"
    return verdict.labels[0].upper() if verdict.labels else "INCONCLUSIVE"


def _verdict_color(label: str) -> str:
    return {
        "COMPLIANCE": typer.colors.RED,
        "REFUSAL": typer.colors.GREEN,
        "INCONCLUSIVE": typer.colors.YELLOW,
        "ERROR": typer.colors.MAGENTA,
    }.get(label, typer.colors.WHITE)


def _make_trial_callback(total_expected: int) -> Callable:
    """Return a callback that prints one line per completed trial."""

    def callback(
        trial_num: int,
        prompt: PromptCandidate,
        response: Optional[ModelResponse],
        verdict: Optional[Verdict],
    ) -> None:
        label = _verdict_label(verdict)
        color = _verdict_color(label)
        latency = f"{response.latency_ms:.0f}ms" if response else "  —  "
        tokens = f"{response.tokens}tok" if response else ""
        confidence = f"({verdict.confidence:.0%})" if verdict else "      "
        preview = (prompt.text[:52].replace("\n", " ") + "…") if len(prompt.text) > 52 else prompt.text

        typer.secho(f"  [{trial_num:4d}/{total_expected:4d}] ", nl=False)
        typer.secho(f"{label:<13}", fg=color, bold=True, nl=False)
        typer.echo(f" {confidence:<6}  {latency:>8}  {tokens:<7}  {preview}")

    return callback


def _print_interactive_trial(
    trial: int,
    response: ModelResponse,
    verdict: Optional[Verdict],
) -> None:
    label = _verdict_label(verdict)
    color = _verdict_color(label)

    typer.echo()
    typer.secho("─" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.secho(f"  Trial #{trial}", bold=True)
    typer.echo()

    # Response text — wrap at 80 chars
    response_text = response.text or "[empty response]"
    typer.echo("  Response:")
    for line in _wrap(response_text, width=58):
        typer.echo(f"    {line}")

    typer.echo()
    typer.secho(f"  Verdict    : ", nl=False)
    typer.secho(label, fg=color, bold=True)

    if verdict:
        typer.echo(f"  Confidence : {verdict.confidence:.0%}")
        if verdict.explanation:
            typer.echo(f"  Reasoning  : {verdict.explanation[:120]}")

    typer.echo(f"  Latency    : {response.latency_ms:.0f} ms   |   Tokens: {response.tokens}")
    typer.echo()


def _wrap(text: str, width: int) -> list[str]:
    """Very simple word-wrap (no external dependency)."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        while len(paragraph) > width:
            cut = paragraph.rfind(" ", 0, width)
            if cut == -1:
                cut = width
            lines.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        lines.append(paragraph)
    return lines


def _print_health_line(label: str, status: str) -> None:
    color = {
        "ok": typer.colors.GREEN,
        "degraded": typer.colors.YELLOW,
        "unavailable": typer.colors.RED,
    }.get(status, typer.colors.WHITE)
    typer.secho(f"  {label.capitalize()} health: ", nl=False)
    typer.secho(status.upper(), fg=color, bold=True)


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
    total_prompts = (
        len(manifest.adapters)
        * len(manifest.generators)
        * manifest.num_batches
        * manifest.batch_size
    )
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
        f"{len(manifest.adapters)} adapters × {len(manifest.generators)} generators × "
        f"{manifest.num_batches} batches × {manifest.batch_size} = {total_prompts}"
    )
    typer.echo(f"  Concurrency: combos={manifest.max_combo_concurrency}  prompts={manifest.max_concurrency}")
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
    asr_color = typer.colors.RED if summary.asr > 0.3 else typer.colors.YELLOW if summary.asr > 0 else typer.colors.GREEN
    typer.secho(f"  Attack Success   : {summary.asr:.2%}", fg=asr_color, bold=True)
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
    """Load a .env file from the cwd (or parent directories) without extra deps."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_file = directory / ".env"
        if env_file.exists():
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
