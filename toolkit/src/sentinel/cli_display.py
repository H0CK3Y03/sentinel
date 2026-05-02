"""Console display helpers for the run command.

Holds the live status panel and per-trial formatting, so ``cli.py`` can stay
focused on command dispatch rather than rendering.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Tuple

import typer

from sentinel.models import ModelResponse, PromptCandidate, Verdict


# ---------------------------------------------------------------------------
# Column widths (kept here so the panel and trial line stay in sync)
# ---------------------------------------------------------------------------

# Progress bracket "  [XXXX/XXXX] " is 14 chars, not configurable.
_COL_VERDICT = 13   # "COMPLIANCE   "
_COL_CONF    =  6   # "(75%)  "
_COL_LATENCY =  8   # "2500ms" right-aligned
_COL_TOKENS  =  7   # "123tok " left-padded
_COL_ATTACK  = 10   # "jailbreak " left-padded


# ---------------------------------------------------------------------------
# Verdict labels and colours
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    "COMPLIANCE": typer.colors.RED,
    "REFUSAL": typer.colors.GREEN,
    "INCONCLUSIVE": typer.colors.YELLOW,
    "ERROR": typer.colors.MAGENTA,
}


def verdict_label(verdict: Optional[Verdict]) -> str:
    """Return the upper-cased verdict label, or ``ERROR`` / ``INCONCLUSIVE``."""
    if verdict is None:
        return "ERROR"
    return verdict.labels[0].upper() if verdict.labels else "INCONCLUSIVE"


def verdict_color(label: str) -> str:
    return _VERDICT_COLORS.get(label, typer.colors.WHITE)


def print_health_line(label: str, status: str) -> None:
    color = {
        "ok": typer.colors.GREEN,
        "degraded": typer.colors.YELLOW,
        "unavailable": typer.colors.RED,
    }.get(status, typer.colors.WHITE)
    typer.secho(f"  {label.capitalize()} health: ", nl=False)
    typer.secho(status.upper(), fg=color, bold=True)


def wrap_text(text: str, width: int) -> List[str]:
    """Very simple word-wrap, no external dependency."""
    lines: List[str] = []
    for paragraph in text.splitlines():
        while len(paragraph) > width:
            cut = paragraph.rfind(" ", 0, width)
            if cut == -1:
                cut = width
            lines.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        lines.append(paragraph)
    return lines


# ---------------------------------------------------------------------------
# Trial table header
# ---------------------------------------------------------------------------

def print_trial_header() -> None:
    """Print the column-name row and a separator line."""
    typer.secho(
        f"  [{'progress':>9}] "
        f"{'verdict':<{_COL_VERDICT}}"
        f" {'conf':<{_COL_CONF}}"
        f"  {'latency':>{_COL_LATENCY}}"
        f"  {'tokens':<{_COL_TOKENS}}"
        f"  {'attack':<{_COL_ATTACK}}"
        f"  prompt",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho("  " + "─" * 90, fg=typer.colors.BRIGHT_BLACK)


def print_interactive_trial(
    trial: int,
    response: ModelResponse,
    verdict: Optional[Verdict],
) -> None:
    """Render a single trial result for the interactive REPL."""
    label = verdict_label(verdict)
    color = verdict_color(label)

    typer.echo()
    typer.secho("─" * 62, fg=typer.colors.BRIGHT_BLACK)
    typer.secho(f"  Trial #{trial}", bold=True)
    typer.echo()

    response_text = response.text or "[empty response]"
    typer.echo("  Response:")
    for line in wrap_text(response_text, width=58):
        typer.echo(f"    {line}")

    typer.echo()
    typer.secho("  Verdict    : ", nl=False)
    typer.secho(label, fg=color, bold=True)

    if verdict:
        typer.echo(f"  Confidence : {verdict.confidence:.0%}")
        if verdict.explanation:
            typer.echo(f"  Reasoning  : {verdict.explanation[:120]}")

    typer.echo(f"  Latency    : {response.latency_ms:.0f} ms   |   Tokens: {response.tokens}")
    typer.echo()


# ---------------------------------------------------------------------------
# Live status panel
# ---------------------------------------------------------------------------

class LivePanel:
    """Multi-line live status panel: one row per generator, three stage columns.

    Uses ANSI cursor-up (``\\033[NF``) to overwrite the panel in place. Trial
    result lines are inserted above the panel by moving up before printing,
    then redrawing the panel below.

    Stage counts per slot:
        _GEN   — generator is producing a prompt (LLM call in progress)
        _RESP  — adapter is querying the target model
        _JUDGE — judges are evaluating the response
    """

    _GEN, _RESP, _JUDGE = 0, 1, 2
    _STAGE_LABELS        = ("generating", "responding", "judging")
    _STAGE_HEADER_LABELS = ("generator",  "adapter",    "judge")
    _STAGE_COLORS        = (typer.colors.YELLOW, typer.colors.CYAN, typer.colors.MAGENTA)
    _COL_NAME  = 14
    _COL_STAGE = 16

    def __init__(self, slot_names: List[str]) -> None:
        self._names = slot_names
        # [gen_count, resp_count, judge_count] per slot name
        self._state: Dict[str, List[int]] = {n: [0, 0, 0] for n in slot_names}
        # stages that have been active (count > 0) in the current batch
        self._ever_active: Dict[str, set[int]] = {n: set() for n in slot_names}
        # stages currently showing "done" (were active, count fell to 0)
        self._done_stages: Dict[str, set[int]] = {n: set() for n in slot_names}
        self._done: set[str] = set()
        self._initialized = False

    # -- stage transitions ---------------------------------------------------

    def generation_start(self, name: str) -> None:
        # New batch starts — reset per-stage done state for this slot.
        if name in self._done_stages:
            self._done_stages[name].clear()
            self._ever_active[name].clear()
        self._inc(name, self._GEN, +1)

    def adapter_start(self, name: str) -> None:
        self._inc(name, self._GEN, -1)
        self._inc(name, self._RESP, +1)

    def judge_start(self, name: str) -> None:
        self._inc(name, self._RESP, -1)
        self._inc(name, self._JUDGE, +1)

    def trial_done(self, name: str) -> None:
        """Decrement whichever stage is active (judge → resp fallback for errors)."""
        if name not in self._state:
            return
        s = self._state[name]
        if s[self._JUDGE] > 0:
            self._inc(name, self._JUDGE, -1)
        elif s[self._RESP] > 0:
            self._inc(name, self._RESP, -1)

    def _inc(self, name: str, stage: int, delta: int) -> None:
        if name not in self._state:
            return
        s = self._state[name]
        if delta > 0:
            self._ever_active[name].add(stage)
            self._done_stages[name].discard(stage)
        s[stage] = max(0, s[stage] + delta)
        if delta < 0 and s[stage] == 0 and stage in self._ever_active[name]:
            self._done_stages[name].add(stage)
        self._redraw()

    # -- rendering -----------------------------------------------------------

    def _render_stage(self, label: str, count: int, color: str, is_done: bool = False) -> str:
        if is_done:
            return typer.style(f"{'done':<{self._COL_STAGE}}", fg=typer.colors.GREEN)
        if count == 0:
            return typer.style(f"{'--':<{self._COL_STAGE}}", fg=typer.colors.BRIGHT_BLACK)
        suffix = f"({count})" if count > 1 else "..."
        return typer.style(f"{label + suffix:<{self._COL_STAGE}}", fg=color)

    def _render_row(self, name: str) -> str:
        name_col = typer.style(f"{name:<{self._COL_NAME}}", fg=typer.colors.BRIGHT_BLACK)
        if name in self._done:
            done_col = typer.style("complete", fg=typer.colors.GREEN)
            return f"  {name_col}  {done_col}"
        s = self._state[name]
        done_s = self._done_stages.get(name, set())
        stages = "  ".join(
            self._render_stage(lbl, s[i], col, i in done_s)
            for i, (lbl, col) in enumerate(zip(self._STAGE_LABELS, self._STAGE_COLORS))
        )
        return f"  {name_col}  {stages}"

    def _panel_height(self) -> int:
        """Total lines occupied by the panel (header + separator + generator rows)."""
        return len(self._names) + 2

    def draw_panel(self) -> None:
        """Public draw: write the full panel at the current cursor position."""
        header = (
            "  "
            + typer.style(f"{'name':<{self._COL_NAME}}", fg=typer.colors.BRIGHT_BLACK)
            + "  "
            + "  ".join(
                typer.style(f"{lbl:<{self._COL_STAGE}}", fg=col)
                for lbl, col in zip(self._STAGE_HEADER_LABELS, self._STAGE_COLORS)
            )
        )
        sep_len = self._COL_NAME + 2 + (self._COL_STAGE + 2) * 3 - 2
        separator = typer.style("  " + "─" * sep_len, fg=typer.colors.BRIGHT_BLACK)
        sys.stdout.write(f"\r\033[K{header}\n")
        sys.stdout.write(f"\r\033[K{separator}\n")
        for name in self._names:
            sys.stdout.write(f"\r\033[K{self._render_row(name)}\n")
        sys.stdout.flush()
        self._initialized = True

    def _redraw(self) -> None:
        if self._initialized:
            sys.stdout.write(f"\033[{self._panel_height()}F")
        self.draw_panel()

    # -- trial result --------------------------------------------------------

    def print_trial(self, line: str, detail: Optional[str] = None) -> None:
        """Insert *line* (and optional *detail*) above the panel, then redraw."""
        if self._initialized:
            sys.stdout.write(f"\033[{self._panel_height()}F")
        sys.stdout.write(f"\r\033[K{line}\n")
        if detail:
            sys.stdout.write(f"\r\033[K{detail}\n")
        self.draw_panel()

    def finalize(self) -> None:
        """Mark all slots as complete and do a final redraw."""
        self._done = set(self._names)
        self._redraw()


# ---------------------------------------------------------------------------
# Trial-result formatter
# ---------------------------------------------------------------------------

def _format_error_detail(response: Optional[ModelResponse]) -> Optional[str]:
    """Build a console error-detail line, or None if no useful info available."""
    indent = "  " + " " * 15 + "  "  # aligns with verdict column
    if response is None:
        return (
            indent
            + typer.style(
                "↳ adapter call failed (network / timeout) — details in JSONL log",
                fg=typer.colors.RED,
            )
        )
    if response.is_error:
        msg = response.metadata.get("error") or response.text or "unknown error"
        return indent + typer.style(f"↳ adapter error: {msg[:90]}", fg=typer.colors.RED)
    return None


def _format_trial_line(
    conv_counter: int,
    total_expected: int,
    prompt: PromptCandidate,
    response: Optional[ModelResponse],
    verdict: Optional[Verdict],
) -> str:
    """Format a single trial as one line for the live panel."""
    label = verdict_label(verdict)
    color = verdict_color(label)
    latency = f"{response.latency_ms:.0f}ms" if response and not response.is_error else "—"
    tokens  = f"{response.tokens}tok"         if response and not response.is_error else ""
    conf_str = f"({verdict.confidence:.0%})" if verdict else ""
    attack = prompt.metadata.get("display_name") or prompt.metadata.get("attack_type", "")
    attack = attack[:_COL_ATTACK]

    preview_len = 52
    raw_preview = prompt.text.replace("\n", " ")
    preview = (raw_preview[:preview_len] + "…") if len(prompt.text) > preview_len else raw_preview

    bracket = f"[{conv_counter:4d}/{total_expected:4d}] "
    return (
        f"  {bracket}"
        + typer.style(f"{label:<{_COL_VERDICT}}", fg=color, bold=True)
        + f" {conf_str:<{_COL_CONF}}"
        + f"  {latency:>{_COL_LATENCY}}"
        + f"  {tokens:<{_COL_TOKENS}}"
        + f"  {attack:<{_COL_ATTACK}}"
        + f"  {preview}"
    )


# Callback bundle returned by ``make_run_callbacks``.
RunCallbacks = Tuple[Callable, Callable, Callable, Callable, Callable]


def make_run_callbacks(total_expected: int, generator_names: List[str]) -> RunCallbacks:
    """Build the orchestrator callbacks that drive the live panel + trial lines.

    Returns ``(on_trial_complete, on_gen_start, on_adp_start, on_jdg_start, finalize)``.
    """
    print_trial_header()
    panel = LivePanel(generator_names)
    panel.draw_panel()  # show initial idle panel right after the header

    conv_counter = 0  # incremented once per conversation (turn == 0 only)

    def on_trial_complete(
        trial_num: int,
        prompt: PromptCandidate,
        response: Optional[ModelResponse],
        verdict: Optional[Verdict],
    ) -> None:
        nonlocal conv_counter
        name = prompt.metadata.get("display_name", "")
        panel.trial_done(name)

        # Follow-up turns of a multi-turn conversation update stage counts but
        # don't print their own line.
        if prompt.metadata.get("turn", 0) > 0:
            return

        conv_counter += 1
        line = _format_trial_line(conv_counter, total_expected, prompt, response, verdict)
        detail = _format_error_detail(response) if verdict is None else None
        panel.print_trial(line, detail=detail)

    def on_generation_start(name: str) -> None:
        panel.generation_start(name)

    def on_adapter_start(name: str) -> None:
        panel.adapter_start(name)

    def on_judge_start(name: str) -> None:
        panel.judge_start(name)

    return (
        on_trial_complete,
        on_generation_start,
        on_adapter_start,
        on_judge_start,
        panel.finalize,
    )
