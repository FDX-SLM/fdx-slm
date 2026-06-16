"""Pydantic config models and the ``base.yaml`` + override loader.

Every hyperparameter and path comes from ``configs/*.yaml`` (never hardcoded). A config file
may declare ``defaults: base.yaml``; :func:`load_raw_config` resolves and deep-merges that
base before validation. ``${ENV_VAR}`` references in string values are expanded from the
environment at load time, so secrets stay in ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Shared / base
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    """Base for all config models: ignore unknown keys so configs can evolve safely."""

    model_config = ConfigDict(extra="ignore")


class TrackingConfig(_Base):
    """Langfuse tracking settings (sample-generation logging).

    Langfuse credentials come from the environment (``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``), never from config. Quantitative metrics are
    handled by :mod:`slm_coach.reporting` (CSV + charts), not by a tracking server.
    """

    langfuse: bool = True


class ReportingConfig(_Base):
    """Metric-artifact settings: CSV tables + PNG charts written at end of train/eval.

    ``tables`` uses stdlib ``csv`` and always works; ``plots`` needs the optional ``viz`` extra
    (matplotlib) and degrades to a no-op (with a hint) when it is not installed.
    """

    tables: bool = True
    plots: bool = True


class DataConfig(_Base):
    """Data directory and audit-filter settings."""

    dir: str = "data"
    keep_audit_status: list[str] = Field(default_factory=lambda: ["approved"])
    lang: str = "vi"


class BaseConfig(_Base):
    """Settings shared by every run (base model, paths, seed, tracking, data)."""

    model_name: str
    output_dir: str = "checkpoints"
    report_dir: str = "outputs"
    seed: int = 42
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


# ---------------------------------------------------------------------------
# Model / LoRA / training control
# ---------------------------------------------------------------------------


class LoRAConfig(_Base):
    """LoRA / QLoRA adapter hyperparameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"


class QuantConfig(_Base):
    """Quantization settings (4-bit QLoRA for T2)."""

    load_in_4bit: bool = False


class ModelRuntimeConfig(_Base):
    """Base-model loading options."""

    max_seq_len: int = 4096
    attn_implementation: str = "sdpa"  # sdpa works everywhere; set flash_attention_2 if installed
    use_unsloth: bool = True
    dtype: str = "bfloat16"


class MixtureConfig(_Base):
    """Single/multi-turn mixing ratio for SFT."""

    multi_turn: float = 0.66
    single: float = 0.34


class TrainControlConfig(_Base):
    """Checkpointing / eval / early-stopping controls shared by SFT and alignment."""

    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rubric_avg"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_seq_len: int = 4096
    max_grad_norm: float = 1.0  # gradient clipping
    optim: str = "adamw_torch"  # HF optimizer id (e.g. adamw_torch, adamw_torch_fused, adamw_8bit)
    gradient_checkpointing: bool = True  # trade compute for memory
    use_liger_kernel: bool = False  # Liger fused kernels (needs the `liger-kernel` package)


class SFTParams(TrainControlConfig):
    """Supervised fine-tuning hyperparameters."""

    epochs: int = 1
    lr: float = 2.0e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_steps: int | None = None  # caps total steps (overrides epochs); used for smoke tests
    train_on_responses_only: bool = True
    multiturn_masking: bool = True
    packing: bool = False
    # Fraction of training records held out as a TRUE out-of-sample eval set (never trained on),
    # so eval_loss is meaningful and overfit is visible. 0.0 keeps the legacy in-sample behavior.
    val_split: float = 0.0
    mixture: MixtureConfig = Field(default_factory=MixtureConfig)


class StageConfig(_Base):
    """One curriculum stage in a multi-stage SFT run."""

    name: str
    include: list[str]
    mix: MixtureConfig | None = None
    reasoning_thinking: bool = False


class EvalDuringTrainingConfig(_Base):
    """Configuration for the in-loop eval callback."""

    enabled: bool = True
    gold_subset: str = "data/gold/gold_test.jsonl"
    subset_size: int = 32
    max_new_tokens: int = 512
    use_judge: bool = False  # when true, score with real judges instead of the fast proxy
    judges: list[str] = Field(default_factory=lambda: ["gpt", "gemini"])
    judge_models: dict[str, str] = Field(
        default_factory=lambda: {"gpt": "gpt-4o", "gemini": "gemini-1.5-pro"}
    )
    rubric_weights: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level file configs
# ---------------------------------------------------------------------------


class SFTFileConfig(BaseConfig):
    """Schema for ``configs/sft_lora.yaml`` (T1) and ``configs/sft_multistage.yaml`` (T2)."""

    run_name: str = "sft"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    model: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)
    sft: SFTParams = Field(default_factory=SFTParams)
    stages: list[StageConfig] = Field(default_factory=list)
    eval_during_training: EvalDuringTrainingConfig = Field(default_factory=EvalDuringTrainingConfig)

    @property
    def is_multistage(self) -> bool:
        """Whether this config declares a curriculum (more than zero stages)."""
        return len(self.stages) > 0


class AlignParams(_Base):
    """Alignment hyperparameters (method selected here, never hardcoded)."""

    method: Literal["dpo", "orpo"] = "dpo"
    beta: float = 0.1  # pref_beta: KL/preference strength
    lr: float = 5.0e-6
    epochs: int = 1
    max_length: int = 2048
    max_prompt_length: int = 1024
    loss_type: str = "sigmoid"  # DPO loss variant: sigmoid (vanilla DPO), ipo, hinge, ...
    label_smoothing: float = 0.0  # DPO label smoothing (cDPO); 0 = off
    rpo_alpha: float | None = None  # pref_ftx: weight of the SFT/NLL term added to DPO; None = off


class AlignFileConfig(BaseConfig):
    """Schema for ``configs/align_orpo.yaml`` / ``configs/align_dpo.yaml`` (T3)."""

    run_name: str = "align"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    model: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)
    align: AlignParams = Field(default_factory=AlignParams)
    train: TrainControlConfig = Field(default_factory=TrainControlConfig)
    batch_size: int = 2
    grad_accum: int = 8
    sft_checkpoint: str | None = None


class LatencyConfig(_Base):
    """Offline generation-latency measurement settings."""

    measure: bool = True
    n_samples: int = 50


class GenerationConfig(_Base):
    """Offline batch-generation decoding settings."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 8


class EvalFileConfig(BaseConfig):
    """Schema for ``configs/eval.yaml``."""

    gold: str = "data/gold/gold_test.jsonl"
    rubric_weights: dict[str, float] = Field(default_factory=dict)
    judges: list[str] = Field(default_factory=lambda: ["gpt", "gemini"])
    judge_models: dict[str, str] = Field(
        default_factory=lambda: {"gpt": "gpt-4o", "gemini": "gemini-1.5-pro"}
    )
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    per_mode_breakdown: bool = True
    pairwise: bool = False  # also compare each answer against the gold reference (A/B win-rate)
    # Production coach system prompt injected into every gold prompt (so eval matches what the
    # deployed model sees). Applied uniformly to the model under test and any baseline/parent.
    system_prompt: str | None = None

    @field_validator("judges")
    @classmethod
    def _no_teacher_judges(cls, judges: list[str]) -> list[str]:
        """Reject Claude/DeepSeek judges (teachers -> circular/self-preference bias)."""
        banned = {"claude", "anthropic", "deepseek"}
        offending = [j for j in judges if j.lower() in banned]
        if offending:
            raise ValueError(
                f"judges must not include teacher models {offending}; use gpt/gemini only"
            )
        return judges


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)

CONFIGS_DIR = Path("configs")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in strings; ``${VAR}`` -> ``None`` if unset."""
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        # An unresolved ``${VAR}`` (env var absent) collapses to None for optional fields.
        if expanded.startswith("${") and expanded.endswith("}"):
            return None
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base`` (override wins; nested dicts merged)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw_config(path: str | Path) -> dict:
    """Read a YAML config, resolve its ``defaults`` base, and expand env vars.

    Args:
        path: Path to the config file.

    Returns:
        The merged, env-expanded config dict (with the ``defaults`` key removed).

    Raises:
        FileNotFoundError: If the config or its declared base does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    defaults = data.pop("defaults", None)
    if defaults:
        base_path = (path.parent / defaults).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"defaults base not found: {base_path}")
        with base_path.open("r", encoding="utf-8") as handle:
            base_data = yaml.safe_load(handle) or {}
        data = _deep_merge(base_data, data)

    return _expand_env(data)


def load_config(path: str | Path, schema: type[T]) -> T:
    """Load a config file and validate it into ``schema``.

    Args:
        path: Path to the YAML config.
        schema: The pydantic model to validate into.

    Returns:
        The validated config instance.
    """
    return schema.model_validate(load_raw_config(path))


def load_base_config(path: str | Path) -> BaseConfig:
    """Load and validate a base config."""
    return load_config(path, BaseConfig)


def load_sft_config(path: str | Path) -> SFTFileConfig:
    """Load and validate an SFT (T1/T2) config."""
    return load_config(path, SFTFileConfig)


def load_align_config(path: str | Path) -> AlignFileConfig:
    """Load and validate an alignment (T3) config."""
    return load_config(path, AlignFileConfig)


def load_eval_config(path: str | Path) -> EvalFileConfig:
    """Load and validate an evaluation config."""
    return load_config(path, EvalFileConfig)
