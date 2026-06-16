"""CLI: full evaluation of a checkpoint (docs/SPEC.md §11, Step 5).

Thin entrypoint — delegates to :func:`slm_coach.eval.runner.run_evaluation`. Generates answers
offline, scores with the 7-criteria rubric + GPT/Gemini judges, computes the per-mode
breakdown (+ optional pairwise win-rate and latency), and writes the report.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_eval_config
from slm_coach.eval.runner import run_evaluation
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Evaluate a checkpoint (rubric + judges + per-mode).")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to the evaluation config YAML."),
    model: Path = typer.Option(..., "--model", help="Checkpoint to evaluate (offline)."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve config + plan only (no GPU/API)."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Offline mock run: canned generation + mock judge (no GPU/API)."
    ),
    run_name: str = typer.Option(
        None, "--run-name", help="Report dir name (outputs/eval/<run-name>/); default timestamp."
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Run the full evaluation pipeline."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cfg = load_eval_config(config)
    set_seed(cfg.seed)
    logger.info(
        "Starting evaluation",
        extra={"model": str(model), "judges": cfg.judges, "gold": cfg.gold, "mock": mock},
    )
    if dry_run:
        console.print_json(cfg.model_dump_json(indent=2))
        console.print(
            f"[green]Dry run OK — would evaluate {model} with judges {cfg.judges}.[/green]"
        )
        raise typer.Exit()
    report_dir = run_evaluation(cfg, model, mock=mock, run_name=run_name)
    console.print(f"[green]Report written to {report_dir}[/green]")


if __name__ == "__main__":
    app()
