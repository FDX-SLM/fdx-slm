"""CLI: head-to-head win-rate between two evaluated models (e.g. SLM vs the parent Qwen).

Reuses each run's ``per_sample.csv`` (no re-generation) and asks the judges to pick the better
answer per case. Convention: ``--a`` is the model under test, ``--b`` the baseline/parent — a win
means A beats B. See docs/RUNBOOK.md.

Example::

    uv run python scripts/compare_models.py \
        --a outputs/eval/eval_dpo/per_sample.csv --label-a "SLM (DPO)" \
        --b outputs/eval/eval_base_qwen3_8b/per_sample.csv --label-b "Qwen3-8B (parent)"
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_eval_config
from slm_coach.eval.headtohead import (
    build_headtohead_markdown,
    head_to_head,
    load_per_sample,
    write_headtohead_csv,
)
from slm_coach.eval.judge import MockJudge, build_judges, judge_usage
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Head-to-head win-rate between two models.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    a: Path = typer.Option(..., "--a", help="per_sample.csv of the model under test (e.g. SLM)."),
    b: Path = typer.Option(..., "--b", help="per_sample.csv of the baseline/parent."),
    config: Path = typer.Option(
        Path("configs/eval.yaml"), "--config", help="Eval config (judges/judge_models)."
    ),
    label_a: str = typer.Option("SLM", "--label-a", help="Display name for model A."),
    label_b: str = typer.Option("parent", "--label-b", help="Display name for model B."),
    out: Path = typer.Option(
        Path("outputs/eval/headtohead.md"), "--out", help="Where to write the markdown."
    ),
    mock: bool = typer.Option(False, "--mock", help="Offline mock judge (no API keys)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Compute and write the head-to-head win-rate."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cfg = load_eval_config(config)
    set_seed(cfg.seed)
    rows_a, rows_b = load_per_sample(a), load_per_sample(b)
    judges = [MockJudge()] if mock else build_judges(cfg.judges, cfg.judge_models)
    result = head_to_head(judges, rows_a, rows_b)
    usage = judge_usage(judges)
    markdown = build_headtohead_markdown(
        result, label_a=label_a, label_b=label_b, usage=(usage if usage["calls"] else None)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    csv_out = out.with_suffix(".csv")
    write_headtohead_csv(result, csv_out)
    console.print(markdown)
    console.print(f"[green]Head-to-head written to {out} (+ {csv_out})[/green]")


if __name__ == "__main__":
    app()
