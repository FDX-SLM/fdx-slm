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


def test_sft_lora_config_merges_base():
    cfg = load_sft_config(CONFIGS / "sft_lora.yaml")
    assert cfg.model_name == "Qwen/Qwen3.5-9B"  # from base.yaml
    assert cfg.seed == 1308  # overridden in sft_lora.yaml (base default is 42)
    assert cfg.data.dir == "data"  # from base.yaml
    assert cfg.tracking.langfuse is True
    assert cfg.run_name == "sft_lora"
    assert cfg.sft.epochs == 1
    assert cfg.lora.r > 0  # tunable hyperparameter; just verify the lora section parsed
    assert cfg.quant.load_in_4bit is False
    assert cfg.is_multistage is False


def test_sft_multistage_curriculum():
    cfg = load_sft_config(CONFIGS / "sft_multistage.yaml")
    assert cfg.is_multistage is True
    assert cfg.quant.load_in_4bit is True  # QLoRA for T2
    assert [s.name for s in cfg.stages] == ["broad", "reasoning"]
    assert cfg.stages[1].reasoning_thinking is True
    assert cfg.stages[0].mix is not None and cfg.stages[0].mix.multi_turn == 0.66


def test_baseline_configs_isolate_axes():
    base = CONFIGS / "baselines"
    ls = load_sft_config(base / "sft_lora_single.yaml")
    qs = load_sft_config(base / "sft_qlora_single.yaml")
    lm = load_sft_config(base / "sft_lora_multi.yaml")
    qm = load_sft_config(base / "sft_qlora_multi.yaml")

    # Quant axis isolated.
    assert ls.quant.load_in_4bit is False and qs.quant.load_in_4bit is True
    assert lm.quant.load_in_4bit is False and qm.quant.load_in_4bit is True
    # Stage axis isolated.
    assert ls.is_multistage is False and qs.is_multistage is False
    assert lm.is_multistage is True and qm.is_multistage is True
    # Hyperparameters matched across all four corners (only the axes differ).
    for cfg in (ls, qs, lm, qm):
        assert cfg.seed == 1308
        assert cfg.sft.batch_size == 8 and cfg.sft.lr == 2.0e-4
        assert cfg.sft.use_liger_kernel is True
        assert cfg.model.attn_implementation == "sdpa"


def test_align_configs_select_method():
    orpo = load_align_config(CONFIGS / "align_orpo.yaml")
    assert orpo.align.method == "orpo"
    assert orpo.sft_checkpoint is None  # monolithic

    dpo = load_align_config(CONFIGS / "align_dpo.yaml")
    assert dpo.align.method == "dpo"
    assert dpo.sft_checkpoint == "checkpoints/sft_lora/best"  # DPO needs an SFT start
    assert dpo.align.loss_type == "sigmoid"  # pref_loss
    assert dpo.align.rpo_alpha is None  # pref_ftx: 0 -> off
    assert dpo.train.optim == "adamw_torch"  # optimizer
    assert dpo.train.use_liger_kernel is True  # liger_kernel


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
