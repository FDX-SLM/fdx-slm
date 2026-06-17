"""Alignment (T3): DPO via TRL.

DPO continues the SFT adapter and requires an SFT checkpoint as its starting point. The method
is config-driven (``config.align.method``) — never hardcoded. Heavy deps are imported lazily;
``dry_run`` loads the preference data and logs the plan without a GPU.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.data.formatting import to_preference_dataset
from slm_coach.data.loader import load_records
from slm_coach.tracking import init_tracking
from slm_coach.training.callbacks import write_meta_json
from slm_coach.training.model import precision_kwargs, prepare_peft_model, save_checkpoint
from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import AlignFileConfig

logger = get_logger(__name__)


def _valid_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the dataclass ``config_cls`` accepts; log any dropped (TRL API drift).

    TRL renames/removes config fields across versions (e.g. ``DPOConfig`` dropped
    ``max_prompt_length``). Filtering here keeps the trainer construction working across versions
    instead of crashing on an unexpected keyword.
    """
    valid = {f.name for f in dataclasses.fields(config_cls)}
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        logger.warning(
            "Dropping config kwargs unsupported by this TRL version",
            extra={"config": config_cls.__name__, "dropped": dropped},
        )
    return {k: v for k, v in kwargs.items() if k in valid}


def run_alignment(
    config: AlignFileConfig,
    *,
    sft_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run DPO alignment and return the best checkpoint path.

    Args:
        config: Validated alignment config.
        sft_checkpoint: SFT starting point — required for DPO. Overrides ``config.sft_checkpoint``
            when provided.
        resume: Optional checkpoint to resume from.
        dry_run: If True, load preference data and log the plan without launching training.

    Returns:
        Path to the best aligned checkpoint directory.

    Raises:
        ValueError: If no SFT checkpoint is provided.
    """
    start = sft_checkpoint or config.sft_checkpoint
    if not start:
        raise ValueError("DPO requires an SFT checkpoint as its starting point (--sft-checkpoint).")

    output_dir = Path(config.output_dir) / config.run_name
    data = load_records(config.data.dir, ("preference",), config.data.keep_audit_status)
    preferences = data["preference"]

    logger.info(
        "Alignment plan",
        extra={"method": config.align.method, "n_pref": len(preferences), "start": str(start)},
    )
    if dry_run:
        logger.info(
            "Dry run: skipping model load + training", extra={"output": str(output_dir / "best")}
        )
        return output_dir / "best"

    dataset = to_preference_dataset(preferences)
    return _run_align_core(config, dataset, output_dir, start=start, resume=resume)


def _run_align_core(
    config: AlignFileConfig,
    dataset: Any,
    output_dir: Path,
    *,
    start: str | Path | None,
    resume: str | Path | None,
) -> Path:
    """Build and run the DPO trainer, then save best + last + ``meta.json``."""
    require("torch", "train")
    trl = require("trl", "train")
    from transformers import EarlyStoppingCallback

    # DPO continues the SFT adapter.
    loaded = prepare_peft_model(
        config.model_name,
        config.model,
        config.quant,
        config.lora,
        existing_adapter=str(start),
    )
    tracker = init_tracking(config, run_name=config.run_name)

    # Best-checkpoint: hold out a small preference eval split, select best by eval_loss
    # (alignment has no rubric callback, so eval_loss is the selection metric).
    eval_dataset = None
    load_best = config.train.load_best_model_at_end
    if load_best and len(dataset) >= 20:
        split = dataset.train_test_split(test_size=0.1, seed=config.seed)
        dataset, eval_dataset = split["train"], split["test"]
    elif load_best:
        logger.warning(
            "Too few preference pairs for an eval split; disabling load_best_model_at_end",
            extra={"n": len(dataset)},
        )
        load_best = False

    common = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.align.epochs,
        "learning_rate": config.align.lr,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "beta": config.align.beta,
        "max_length": config.align.max_length,
        "max_prompt_length": config.align.max_prompt_length,
        "warmup_ratio": config.train.warmup_ratio,
        "weight_decay": config.train.weight_decay,
        "lr_scheduler_type": config.train.lr_scheduler_type,
        "max_grad_norm": config.train.max_grad_norm,
        "optim": config.train.optim,
        "gradient_checkpointing": config.train.gradient_checkpointing,
        "use_liger_kernel": config.train.use_liger_kernel,
        "logging_steps": config.train.logging_steps,
        "save_strategy": "steps",
        "save_steps": config.train.save_steps,
        "save_total_limit": config.train.save_total_limit,
        "seed": config.seed,
        "report_to": [],
        **precision_kwargs(),
    }
    if load_best and eval_dataset is not None:
        common.update(
            {
                "eval_strategy": "steps",
                "eval_steps": config.train.eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "loss",  # eval_loss (lower is better)
                "greater_is_better": False,
            }
        )
    # DPO-specific loss controls.
    dpo_kwargs = {
        "loss_type": config.align.loss_type,
        "label_smoothing": config.align.label_smoothing,
    }
    if config.align.rpo_alpha is not None:
        dpo_kwargs["rpo_alpha"] = config.align.rpo_alpha
    args = trl.DPOConfig(**_valid_kwargs(trl.DPOConfig, {**common, **dpo_kwargs}))
    trainer = trl.DPOTrainer(
        model=loaded.model,
        ref_model=None,  # PEFT: reference is the adapter-disabled base
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=loaded.tokenizer,
    )
    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=config.train.early_stopping_patience)
    )

    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    save_checkpoint(trainer.model, loaded.tokenizer, best_dir)
    save_checkpoint(trainer.model, loaded.tokenizer, last_dir)
    write_meta_json(
        best_dir, config=config.model_dump(), seed=config.seed, data_version=config.data.lang
    )
    write_meta_json(
        last_dir, config=config.model_dump(), seed=config.seed, data_version=config.data.lang
    )
    rep = config.reporting
    if rep.tables or rep.plots:
        try:
            from slm_coach.reporting import export_training_artifacts

            trainer.save_state()  # writes the cumulative trainer_state.json to output_dir
            export_training_artifacts(output_dir, make_tables=rep.tables, make_plots=rep.plots)
        except Exception as exc:  # noqa: BLE001 - artifacts must never fail a good run
            logger.warning("Could not export training metrics", extra={"error": str(exc)})
    tracker.close()
    logger.info("Alignment complete", extra={"method": config.align.method, "best": str(best_dir)})
    return best_dir
