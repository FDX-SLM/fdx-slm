"""Streamlit demo UI cho SLM coach bán iPhone.

Chạy:  uv run streamlit run coach_app.py

Nạp adapter SFT/DPO lên base model rồi chat bằng đúng system prompt production.
Đường dẫn/model lấy từ env (mặc định trỏ tới run QLoRA 9B) — KHÔNG hardcode path máy cá nhân:
    COACH_BASE         base model (mặc định Qwen/Qwen3.5-9B)
    COACH_ADAPTER      thư mục adapter (mặc định checkpoints/sft_coach_9b/best)
    COACH_EVAL_CONFIG  config eval để lấy system_prompt (mặc định configs/eval.yaml)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = os.environ.get("COACH_BASE", "Qwen/Qwen3.5-9B")
ADAPTER = os.environ.get("COACH_ADAPTER", "checkpoints/sft_coach_9b/best")
EVAL_CONFIG = os.environ.get("COACH_EVAL_CONFIG", "configs/eval.yaml")

FALLBACK_SYS = (
    "Bạn là một sale senior iPhone nhiều năm kinh nghiệm, đang kèm cặp một nhân viên junior."
)


def load_system_prompt() -> str:
    """Trả về system prompt production từ eval.yaml; fallback chuỗi ngắn nếu không có."""
    cfg = Path(EVAL_CONFIG)
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        if data.get("system_prompt"):
            return str(data["system_prompt"])
    return FALLBACK_SYS


@st.cache_resource
def load() -> tuple[object, object]:
    """Nạp base (4-bit) + adapter nếu có; thiếu adapter thì chạy base. Trả về (model, tokenizer)."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
    adapter = Path(ADAPTER)
    if adapter.exists():
        model = PeftModel.from_pretrained(base, str(adapter)).eval()
        tok = AutoTokenizer.from_pretrained(str(adapter))
    else:
        st.warning(f"Không thấy adapter `{adapter}` — chạy base `{BASE}` (chưa fine-tune).")
        model = base.eval()
        tok = AutoTokenizer.from_pretrained(BASE)
    return model, tok


SYS = load_system_prompt()
model, tok = load()

st.title("iPhone Sales Coach (SLM)")
if msg := st.chat_input("Tình huống khách..."):
    st.chat_message("user").write(msg)
    # transformers 5.x: apply_chat_template trả BatchEncoding (input_ids + attention_mask),
    # KHÔNG phải tensor thuần → phải return_dict=True rồi generate(**enc).
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=384, do_sample=True, temperature=0.7, top_p=0.9)
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    # Thẻ mở <think> nằm trong prompt (template thinking của Qwen) nên phần sinh ra chỉ có </think>.
    # Tách theo </think>: trước = suy luận (bỏ thẻ mở nếu có), sau = câu trả lời cho khách.
    think_text, answer = "", text.strip()
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        think_text = re.sub(r"^\s*<think>\s*", "", pre).strip()
        answer = post.strip()
    with st.chat_message("assistant"):
        st.write(answer)  # Gợi ý trả lời khách + Vì sao
        if think_text:
            with st.expander("🧠 Suy luận của senior (junior không thấy phần này)"):
                st.write(think_text)
