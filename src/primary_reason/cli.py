from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from primary_reason import ReasonCauseVerifier, __version__
from primary_reason.core.types import DistanceMetric, InterventionStrategy

app = typer.Typer(
    name="primary-reason",
    help="Davidson primary-reason verifier for LLM chain-of-thought.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed primary-reason version."""
    console.print(f"primary-reason {__version__}")


@app.command()
def verify(
    prompt: str = typer.Argument(..., help="Task prompt presented to the model"),
    cot: str = typer.Option(..., "--cot", help="Chain-of-thought trace text (or @path/to/file)"),
    model: str = typer.Option("claude-opus-4-7", "--model"),
    adapter: str = typer.Option("anthropic", "--adapter"),
    strategies: str = typer.Option("delete,paraphrase", "--strategies"),
    distance_metric: str = typer.Option("lexical", "--distance"),
    swampman: bool = typer.Option(False, "--swampman", help="Also run Swampman Test Battery"),
    out_json: Path | None = typer.Option(None, "--out", help="Write full result JSON to this file"),
) -> None:
    """Run full Davidson primary-reason + counterfactual-faithfulness verification."""
    cot_text = _read_inline_or_file(cot)
    strategy_tuple = tuple(
        _validate_strategy(s.strip()) for s in strategies.split(",") if s.strip()
    )
    verifier = ReasonCauseVerifier(
        model=model,
        adapter=adapter,
        intervention_strategies=strategy_tuple,
        distance_metric=_validate_distance(distance_metric),
    )
    result = verifier.verify(prompt=prompt, cot=cot_text, run_swampman=swampman)
    _render(result)
    if out_json:
        out_json.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        console.print(f"[dim]wrote {out_json}[/]")


@app.command()
def extract(
    prompt: str = typer.Argument(...),
    cot: str = typer.Option(..., "--cot"),
    model: str = typer.Option("claude-opus-4-7", "--model"),
    adapter: str = typer.Option("anthropic", "--adapter"),
) -> None:
    """Run T1 only: extract Davidson primary reasons per CoT step."""
    from primary_reason.adapters import build_adapter
    from primary_reason.core.splitter import split_cot
    from primary_reason.extractors.primary_reason import extract_primary_reasons

    cot_text = _read_inline_or_file(cot)
    a = build_adapter(adapter, model=model)
    steps = split_cot(cot_text)
    reasons = extract_primary_reasons(steps, prompt, a)
    tbl = Table(title="Primary Reasons (T1)")
    tbl.add_column("step")
    tbl.add_column("pro_attitude")
    tbl.add_column("belief")
    tbl.add_column("conf")
    for r in reasons:
        tbl.add_row(str(r.step_index), r.pro_attitude, r.belief, f"{r.confidence:.2f}")
    console.print(tbl)


@app.command()
def stb(
    model: str = typer.Option("claude-opus-4-7", "--model"),
    adapter: str = typer.Option("anthropic", "--adapter"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run the Swampman Test Battery (T1.5) standalone."""
    v = ReasonCauseVerifier(model=model, adapter=adapter, seed=seed)
    score = v.swampman_score()
    console.print(
        Panel.fit(
            f"FPA score: {score.fpa_score:.3f}\n"
            f"95% CI: [{score.bootstrap_ci[0]:.3f}, {score.bootstrap_ci[1]:.3f}]\n"
            f"discriminates: {score.discriminates}\n"
            f"n_trials: {score.n_trials}",
            title="Swampman Test Battery",
        )
    )


_VALID_STRATEGIES = ("delete", "paraphrase", "negate")
_VALID_DISTANCES = ("exact", "lexical", "embedding")


def _validate_strategy(s: str) -> InterventionStrategy:
    if s not in _VALID_STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy {s!r}; expected one of {_VALID_STRATEGIES}")
    return cast(InterventionStrategy, s)


def _validate_distance(s: str) -> DistanceMetric:
    if s not in _VALID_DISTANCES:
        raise typer.BadParameter(f"Unknown distance {s!r}; expected one of {_VALID_DISTANCES}")
    return cast(DistanceMetric, s)


def _read_inline_or_file(arg: str) -> str:
    if arg.startswith("@"):
        return Path(arg[1:]).read_text()
    return arg


def _render(result) -> None:  # type: ignore[no-untyped-def]
    console.print(
        Panel.fit(
            f"model: {result.model}\nadapter: {result.adapter}\n"
            f"faithfulness: {result.faithfulness.score:.3f}\n"
            f"steps: {len(result.steps)} primary_reasons: {len(result.primary_reasons)}",
            title="primary-reason verify",
        )
    )
    tbl = Table(title="Per-step faithfulness")
    tbl.add_column("step")
    tbl.add_column("score")
    for idx, sc in sorted(result.faithfulness.per_step.items()):
        tbl.add_row(str(idx), f"{sc:.3f}")
    console.print(tbl)
    if result.swampman_score is not None:
        s = result.swampman_score
        console.print(
            f"[bold]Swampman FPA[/] {s.fpa_score:.3f}  CI [{s.bootstrap_ci[0]:.3f}, {s.bootstrap_ci[1]:.3f}]"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
