"""Multi-stage QLoRA SFT (T2): run curriculum stages, chaining checkpoints.

Each stage is built by :func:`slm_coach.data.mixture.build_curriculum` and trained from the
previous stage's adapter (true curriculum chaining — the adapter is *continued*, not reset).
The final stage's ``best`` is copied to ``<run>/best`` and ``<run>/last``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from slm_coach.data.loader import load_records
from slm_coach.data.mixture import build_curriculum
from slm_coach.training.sft import load_gold_subset, run_sft_core, split_holdout
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import SFTFileConfig

logger = get_logger(__name__)


def _default_specs() -> list[dict]:
    """Fallback single stage when no curriculum is declared."""
    return [{"name": "full", "include": ["sft", "reasoning"]}]


def run_multistage_training(
    config: SFTFileConfig,
    *,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run the curriculum (e.g. broad sft -> + reasoning) and return the best checkpoint.

    Args:
        config: Validated multi-stage SFT config (declares ``stages``).
        resume: Optional checkpoint to resume the first stage from.
        dry_run: If True, build the stage plan without launching training.

    Returns:
        Path to the final best checkpoint directory (``<run>/best``).
    """
    output_dir = Path(config.output_dir) / config.run_name
    data = load_records(config.data.dir, ("sft", "reasoning"), config.data.keep_audit_status)

    # Hold out a TRUE validation set ONCE (per data type) and exclude it from EVERY stage, so the
    # out-of-sample eval_loss is never leaked by a later stage re-including the same records.
    val_records: list = []
    records_by_type: dict = {}
    for data_type in ("sft", "reasoning"):
        kept, held = split_holdout(list(data[data_type]), config.sft.val_split, config.seed)
        records_by_type[data_type] = kept
        val_records.extend(held)

    specs = [s.model_dump() for s in config.stages] or _default_specs()
    stages = build_curriculum(specs, records_by_type, seed=config.seed)
    gold_records = load_gold_subset(config)

    logger.info(
        "Multi-stage plan",
        extra={
            "run_name": config.run_name,
            "stages": [(s.name, len(s)) for s in stages],
            "n_val_holdout": len(val_records),
        },
    )
    if dry_run:
        logger.info(
            "Dry run: skipping model load + training", extra={"output": str(output_dir / "best")}
        )
        return output_dir / "best"

    init_from = config.model_name
    existing_adapter: str | None = None
    final_best = output_dir / "best"

    for index, stage in enumerate(stages):
        stage_dir = output_dir / f"stage{index}_{stage.name}"
        stage_best = run_sft_core(
            config,
            stage.records,
            gold_records,
            stage_dir,
            init_from=init_from,
            existing_adapter=existing_adapter,
            reasoning_thinking=stage.reasoning_thinking,
            resume=resume if index == 0 else None,
            stage_name=stage.name,
            val_records=val_records,
        )
        existing_adapter = str(stage_best)  # next stage continues this adapter
        final_best = stage_best

    _promote_final(final_best, output_dir)
    logger.info("Multi-stage training complete", extra={"best": str(output_dir / "best")})
    return output_dir / "best"


def _promote_final(stage_best: Path, output_dir: Path) -> None:
    """Copy the final stage's best adapter to ``<run>/best`` and ``<run>/last``."""
    for name in ("best", "last"):
        target = output_dir / name
        if target.resolve() == stage_best.resolve():
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(stage_best, target)
