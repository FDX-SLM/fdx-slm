"""CLI: build a leaderboard comparing several evaluation reports.

Collects ``outputs/eval/<run>/report.json`` files and renders a ranked markdown leaderboard
(overall score + pairwise win-rate) plus a per-mode matrix, so you can see which training recipe
— or your SLM vs the parent model — wins on the full eval suite. See docs/BASELINES.md.

Examples::

    uv run python scripts/compare_baselines.py                         # all runs under outputs/eval
    uv run python scripts/compare_baselines.py --report outputs/eval/bl_lora_single/report.json \
        --report outputs/eval/base_qwen/report.json
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.reporting.compare import (
    build_leaderboard,
    find_reports,
    load_reports,
    write_comparison,
    write_comparison_csv,
)
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap

app = typer.Typer(add_completion=False, help="Compare evaluation reports into a leaderboard.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    eval_root: Path = typer.Option(
        Path("outputs/eval"), "--eval-root", help="Directory holding <run>/report.json subdirs."
    ),
    report: list[Path] = typer.Option(
        None, "--report", help="Explicit report.json path(s); repeat. Overrides --eval-root."
    ),
    out: Path = typer.Option(
        Path("outputs/eval/comparison.md"), "--out", help="Where to write the leaderboard markdown."
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Render a leaderboard from evaluation reports."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    paths = list(report) if report else find_reports(eval_root)
    if not paths:
        console.print(f"[red]No report.json found (looked under {eval_root}).[/red]")
        raise typer.Exit(code=1)
    reports = load_reports(paths)
    write_comparison(reports, out)
    csv_out = out.with_suffix(".csv")
    write_comparison_csv(reports, csv_out)
    console.print(build_leaderboard(reports))
    console.print(f"[green]Leaderboard written to {out} (+ {csv_out})[/green]")


if __name__ == "__main__":
    app()
