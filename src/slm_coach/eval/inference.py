"""Offline batch generation for evaluation.

Loads a checkpoint directly (transformers) and generates answers for the gold test set. There
is no serving layer / HTTP server / runtime harness. Heavy deps are imported lazily so this
module imports without a GPU; the evaluation runner provides a mock generator for no-GPU runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import GenerationConfig

logger = get_logger(__name__)


def load_for_inference(checkpoint: str | Path, *, dtype: str = "float16") -> tuple[Any, Any]:
    """Load a checkpoint (model + tokenizer) for offline generation.

    Tries to load as a PEFT adapter first (the usual training output), falling back to a plain
    causal LM. Heavy deps (``torch``, ``transformers``, ``peft``) are imported lazily.

    Args:
        checkpoint: Path to a trained checkpoint (adapter or merged model).
        dtype: Torch dtype name for the weights.

    Returns:
        A ``(model, tokenizer)`` tuple in eval mode.
    """
    transformers = require("transformers", "train")
    torch = require("torch", "train")
    path = str(checkpoint)
    torch_dtype = getattr(torch, dtype, torch.float16)
    # Đặt model lên GPU; không có CUDA thì None (CPU) — runner dùng mock cho máy không GPU.
    device_map = "auto" if torch.cuda.is_available() else None

    try:
        peft = require("peft", "train")
        model = peft.AutoPeftModelForCausalLM.from_pretrained(
            path, dtype=torch_dtype, device_map=device_map
        )
        logger.info("Loaded checkpoint as PEFT adapter", extra={"path": path})
    except Exception:  # noqa: BLE001 - fall back to a plain model when not an adapter
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, dtype=torch_dtype, device_map=device_map
        )
        logger.info("Loaded checkpoint as plain model", extra={"path": path})

    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    # Decoder-only + batched generation BẮT BUỘC left-padding, nếu không kết quả batch bị sai.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[list[dict[str, str]]],
    gen_cfg: GenerationConfig,
) -> list[str]:
    """Generate answers for a batch of chat prompts offline.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        prompts: Chat prompts (each a list of role/content messages).
        gen_cfg: Decoding settings.

    Returns:
        Generated answer strings, aligned with ``prompts``.
    """
    torch = require("torch", "train")
    answers: list[str] = []
    for start in range(0, len(prompts), gen_cfg.batch_size):
        batch = list(prompts[start : start + gen_cfg.batch_size])
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.temperature > 0,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for row in generated:
            answers.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip())
    logger.info("Offline generation complete", extra={"n": len(answers)})
    return answers
