"""CLI: T3 alignment — DPO (docs/SPEC.md §11, Step 4).

Thin entrypoint — delegates to :func:`slm_coach.training.align.run_alignment`. DPO requires an
SFT checkpoint as its starting point (``--sft-checkpoint``).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_align_config
from slm_coach.training.align import run_alignment
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Train T3: DPO alignment.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to the alignment config YAML."),
    sft_checkpoint: Path | None = typer.Option(
        None, "--sft-checkpoint", help="SFT start checkpoint (required for DPO)."
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Checkpoint dir to resume from."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve config + plan only (no GPU)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Run DPO alignment."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cfg = load_align_config(config)
    set_seed(cfg.seed)
    start = sft_checkpoint or (Path(cfg.sft_checkpoint) if cfg.sft_checkpoint else None)
    logger.info(
        "Starting alignment",
        extra={"run_name": cfg.run_name, "method": cfg.align.method, "sft_checkpoint": str(start)},
    )
    if start is None:
        raise typer.BadParameter("DPO requires --sft-checkpoint (or sft_checkpoint in the config).")
    if dry_run:
        console.print_json(cfg.model_dump_json(indent=2))
        console.print(f"[green]Dry run OK — method={cfg.align.method}, start={start}[/green]")
        raise typer.Exit()
    best = run_alignment(cfg, sft_checkpoint=start, resume=resume)
    console.print(f"[green]Best checkpoint: {best}[/green]")


if __name__ == "__main__":
    app()
