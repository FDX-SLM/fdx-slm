"""CLI: materialize a mode-stratified train/val holdout split.

Reads the delivered training records (``data/sft/*.jsonl`` + ``data/reasoning/*.jsonl``, approved
only), holds out an EVEN number of records per mode so every mode is represented in ``val``
(see :func:`slm_coach.data.split.stratified_holdout`), and writes the disjoint partition to
``<holdout_dir>/{train,val}.jsonl``. The source files are left untouched (single source of truth),
so this is safe to re-run whenever the data team delivers more data.

    uv run python scripts/split_holdout.py --config configs/sft_coach_9b.yaml

Training then reads ``data/holdout/{train,val}.jsonl`` directly (``data.holdout_dir``); the val
records never enter the training set.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_sft_config
from slm_coach.data.loader import discover_files
from slm_coach.data.schema import iter_jsonl, parse_record
from slm_coach.data.split import stratified_holdout
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Materialize a mode-stratified train/val holdout.")
console = Console()
logger = get_logger(__name__)

#: Data types that feed SFT training (preference pairs are split separately for alignment).
#: SFT/reasoning feed one training pool; preference feeds DPO/ORPO — split each independently.
SFT_DATA_TYPES = ("sft", "reasoning")
PREFERENCE_DATA_TYPES = ("preference",)


def _load_raw(data_dir: str, data_types: tuple[str, ...], keep_audit: set[str]) -> list[dict]:
    """Return approved raw JSONL objects for ``data_types`` (validated, originals preserved)."""
    records: list[dict] = []
    for data_type in data_types:
        for path in discover_files(data_dir, data_type):
            for lineno, obj in iter_jsonl(path):
                try:
                    parsed = parse_record(obj)
                except Exception as exc:  # noqa: BLE001 - report and skip malformed lines
                    logger.warning(
                        "Skipping invalid record",
                        extra={"path": path.name, "line": lineno, "error": str(exc)},
                    )
                    continue
                if parsed.audit_status in keep_audit:
                    records.append(obj)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSON Lines (UTF-8, one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _split_and_write(
    *,
    label: str,
    records: list[dict],
    holdout_dir: Path,
    train_name: str,
    val_name: str,
    fraction: float,
    min_total: int,
    seed: int,
) -> bool:
    """Stratify ``records`` into ``{train_name, val_name}`` under ``holdout_dir`` and report.

    Returns True on success; False (with a warning) when the data is too small to hold out a
    mode-covering val of the requested size — the caller then leaves the consumer to fall back.
    """
    if not records:
        console.print(f"[yellow]{label}: no approved records — skipping.[/yellow]")
        return False
    try:
        train, val = stratified_holdout(
            records,
            mode_of=lambda r: str(r.get("mode")),
            fraction=fraction,
            min_total=min_total,
            seed=seed,
        )
    except ValueError as exc:
        console.print(
            f"[yellow]{label}: cannot hold out a stratified val ({exc}) — skipping.[/yellow]"
        )
        return False

    _write_jsonl(holdout_dir / train_name, train)
    _write_jsonl(holdout_dir / val_name, val)
    val_modes = Counter(str(r.get("mode")) for r in val)
    train_modes = Counter(str(r.get("mode")) for r in train)
    console.print(
        f"[green]{label}: wrote {len(train)} train + {len(val)} val "
        f"({train_name} / {val_name})[/green]"
    )
    console.print(f"  val per mode:   {dict(sorted(val_modes.items()))}")
    console.print(f"  train per mode: {dict(sorted(train_modes.items()))}")
    missing = set(train_modes) - set(val_modes)
    if missing:
        console.print(f"[yellow]  Warning: modes absent from val: {sorted(missing)}[/yellow]")
    return True


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to the SFT config YAML."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Split delivered data into stratified holdouts (SFT + preference) under ``holdout_dir``."""
    configure_logging(json_logs=json_logs)
    cfg = load_sft_config(config)
    set_seed(cfg.seed)

    if not cfg.data.holdout_dir:
        console.print("[red]data.holdout_dir is not set in the config — nothing to write.[/red]")
        raise typer.Exit(code=1)

    keep_audit = set(cfg.data.keep_audit_status)
    holdout_dir = Path(cfg.data.holdout_dir)

    sft_ok = _split_and_write(
        label="SFT",
        records=_load_raw(cfg.data.dir, SFT_DATA_TYPES, keep_audit),
        holdout_dir=holdout_dir,
        train_name="train.jsonl",
        val_name="val.jsonl",
        fraction=cfg.data.val_fraction,
        min_total=cfg.data.val_min_total,
        seed=cfg.seed,
    )
    _split_and_write(
        label="Preference (DPO)",
        records=_load_raw(cfg.data.dir, PREFERENCE_DATA_TYPES, keep_audit),
        holdout_dir=holdout_dir,
        train_name="preference_train.jsonl",
        val_name="preference_val.jsonl",
        fraction=cfg.data.pref_val_fraction,
        min_total=cfg.data.pref_val_min_total,
        seed=cfg.seed,
    )

    if not sft_ok:
        console.print(
            "[red]No SFT holdout written — SFT training would fall back to in-memory split.[/red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
