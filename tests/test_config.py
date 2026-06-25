"""Tests for config models and the base.yaml + override loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_coach.config import (
    EvalFileConfig,
    load_align_config,
    load_eval_config,
    load_sft_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_sft_coach_9b_config_merges_base():
    cfg = load_sft_config(CONFIGS / "sft_coach_9b.yaml")
    assert cfg.model_name == "Qwen/Qwen3.5-9B"
    assert cfg.seed == 1308  # overridden in sft_coach_9b.yaml (base default is 42)
    assert cfg.data.dir == "data"  # from base.yaml
    assert cfg.tracking.langfuse is True
    assert cfg.run_name == "sft_coach_9b"
    assert cfg.sft.epochs == 2
    assert cfg.lora.r > 0  # tunable hyperparameter; just verify the lora section parsed
    assert cfg.quant.load_in_4bit is True  # QLoRA
    assert cfg.data.holdout_dir == "data/holdout"  # from base.yaml
    assert cfg.data.val_min_total == 200


def test_smoke_config_parses():
    cfg = load_sft_config(CONFIGS / "sft_lora_smoke.yaml")
    assert cfg.sft.max_steps == 60  # smoke cap
    assert cfg.quant.load_in_4bit is False


def test_align_coach_dpo_config():
    dpo = load_align_config(CONFIGS / "align_coach_dpo.yaml")
    assert dpo.align.method == "dpo"
    assert dpo.sft_checkpoint == "checkpoints/sft_coach_9b/best"  # DPO needs an SFT start
    assert dpo.align.loss_type == "sigmoid"  # pref_loss
    assert dpo.align.rpo_alpha is None  # pref_ftx: 0 -> off
    assert dpo.align.lr == 5.0e-6  # low LR for DPO
    assert dpo.train.optim == "adamw_torch"  # optimizer


def test_eval_config_values():
    cfg = load_eval_config(CONFIGS / "eval.yaml")
    assert cfg.judges == ["gpt", "gemini"]
    assert cfg.per_mode_breakdown is True
    assert cfg.rubric_weights["factuality"] == 2.0
    assert cfg.pairwise is True


def test_eval_config_rejects_teacher_judges():
    with pytest.raises(ValidationError):
        EvalFileConfig(model_name="x", judges=["gpt", "claude"])
    with pytest.raises(ValidationError):
        EvalFileConfig(model_name="x", judges=["gpt", "deepseek"])


def test_env_expansion_unset_var_becomes_none(monkeypatch):
    from slm_coach.config import _expand_env

    monkeypatch.delenv("SLM_TEST_VAR", raising=False)
    assert _expand_env({"k": "${SLM_TEST_VAR}"}) == {"k": None}


def test_env_expansion_resolves_set_var(monkeypatch):
    from slm_coach.config import _expand_env

    monkeypatch.setenv("SLM_TEST_VAR", "file:./somewhere")
    assert _expand_env({"k": "${SLM_TEST_VAR}"}) == {"k": "file:./somewhere"}
