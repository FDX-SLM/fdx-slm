"""Streamlit demo UI cho SLM coach bán iPhone.

Chạy:  uv run streamlit run coach_app.py

Nạp adapter SFT/DPO lên base model rồi chat bằng đúng system prompt production,
sinh chữ theo kiểu STREAM (token-by-token) cho giống chat thật.
Đường dẫn/model lấy từ env (mặc định trỏ tới run QLoRA 9B) — KHÔNG hardcode path máy cá nhân:
    COACH_BASE         base model (mặc định Qwen/Qwen3.5-9B)
    COACH_ADAPTER      thư mục adapter (mặc định checkpoints/sft_coach_9b/best)
    COACH_EVAL_CONFIG  config eval để lấy system_prompt (mặc định configs/eval.yaml)
    COACH_4BIT         nạp base 4-bit? "true"=QLoRA, "false"=LoRA bf16 (mặc định true)
    COACH_LATENCY_FILE file ghi tốc độ gen token (mặc định outputs/coach_latency.txt)
    COACH_PRODUCTS / COACH_IN_STOCK_ONLY: nguồn catalog (xem coach_tools.py)
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import streamlit as st
import torch
import yaml
from coach_tools import load_catalog
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

BASE = os.environ.get("COACH_BASE", "Qwen/Qwen3.5-9B")
ADAPTER = os.environ.get("COACH_ADAPTER", "checkpoints/sft_coach_9b_lora/best")
EVAL_CONFIG = os.environ.get("COACH_EVAL_CONFIG", "configs/eval.yaml")
FOUR_BIT = os.environ.get("COACH_4BIT", "false").strip().lower() not in ("false", "0", "no")
LATENCY_FILE = os.environ.get("COACH_LATENCY_FILE", "outputs/coach_latency.txt")
SCOPE_GUARD = os.environ.get("COACH_SCOPE_GUARD", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)
MAX_NEW_TOKENS = int(os.environ.get("COACH_MAX_NEW_TOKENS", "512"))

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
    """Nạp base + adapter nếu có; thiếu adapter thì chạy base. Trả về (model, tokenizer).

    ``COACH_4BIT=true`` (mặc định) nạp base 4-bit cho adapter QLoRA; ``false`` nạp bf16 để demo
    adapter LoRA đúng như lúc train (LoRA train trên base bf16, không lượng tử hoá).
    """
    if FOUR_BIT:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb, device_map="auto"
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map="auto"
        )
    adapter = Path(ADAPTER)
    if adapter.exists():
        model = PeftModel.from_pretrained(base, str(adapter)).eval()
        tok = AutoTokenizer.from_pretrained(str(adapter))
    else:
        st.warning(f"Không thấy adapter `{adapter}` — chạy base `{BASE}` (chưa fine-tune).")
        model = base.eval()
        tok = AutoTokenizer.from_pretrained(BASE)
    return model, tok


def _stop_ids(tok: object) -> list[int]:
    """ID các token kết-thúc-lượt để generate DỪNG cuối câu trả lời (không bịa lượt kế tiếp).

    Gồm ``eos`` của tokenizer + ``<|im_end|>`` (marker cuối lượt của chat template Qwen). Thiếu
    cái này, generate chạy hết ``max_new_tokens`` và sinh tiếp cả lượt user/assistant giả.
    """
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for token in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    return list(ids)


def _sanitize_answer(text: str) -> str:
    """Bỏ phần model bịa thêm lượt hội thoại sau câu trả lời (lớp phòng hờ cho demo).

    - Cắt tại ranh giới lượt (``<|im_*|>`` nếu special token lọt ra) hoặc một ``<think>`` mới mở
      (= bắt đầu lượt assistant giả tiếp theo).
    - Gỡ mọi khối ``<think>...</think>`` và thẻ lẻ còn dính trong câu trả lời.
    """
    text = re.split(r"<\|im_(?:end|start)\|>", text)[0]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.split(r"<think>", text)[0]
    return text.replace("</think>", "").strip()


def stream_generate(model: object, tok: object, enc: dict, metrics: dict) -> Iterator[str]:
    """Sinh chữ theo luồng: chạy ``model.generate`` ở thread phụ, yield từng mẩu text đã decode.

    Đồng thời đo tốc độ và ghi vào ``metrics``: ``new_tokens`` (đếm chính xác từ output), ``ttft``
    (giây tới token đầu tiên), ``total_s``, ``tok_per_s`` (gồm prefill) và ``decode_tok_per_s``
    (chỉ giai đoạn sinh, sau token đầu).

    Args:
        model: Model đã nạp (base + adapter).
        tok: Tokenizer tương ứng.
        enc: BatchEncoding từ ``apply_chat_template`` đã ``.to(model.device)``.
        metrics: Dict được cập nhật tại chỗ với số liệu tốc độ.

    Yields:
        Các mẩu text mới (đã bỏ prompt, đã decode) ngay khi model sinh ra.
    """
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    eos_ids = _stop_ids(tok)
    prompt_len = int(enc["input_ids"].shape[1])
    kwargs = dict(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        streamer=streamer,
        eos_token_id=eos_ids or None,  # DỪNG ở cuối lượt → không bịa lượt hội thoại tiếp theo
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )

    out_holder: dict[str, object] = {}

    def _run() -> None:
        with torch.no_grad():
            out_holder["seq"] = model.generate(**kwargs)

    t0 = time.perf_counter()
    thread = Thread(target=_run)
    thread.start()
    t_first: float | None = None
    try:
        for piece in streamer:
            if t_first is None:
                t_first = time.perf_counter()
            yield piece
    finally:
        thread.join()
    t_end = time.perf_counter()

    seq = out_holder.get("seq")
    new_tokens = int(seq.shape[1] - prompt_len) if seq is not None else 0  # type: ignore[union-attr]
    decode_s = t_end - (t_first or t0)
    metrics.update(
        {
            "new_tokens": new_tokens,
            "ttft": (t_first - t0) if t_first else 0.0,
            "total_s": t_end - t0,
            "tok_per_s": (new_tokens / (t_end - t0)) if t_end > t0 else 0.0,
            "decode_tok_per_s": (
                ((new_tokens - 1) / decode_s) if new_tokens > 1 and decode_s > 0 else 0.0
            ),
        }
    )


def log_latency(metrics: dict, prompt: str) -> None:
    """Append một dòng tốc độ gen (tab-separated) vào ``LATENCY_FILE`` để phân tích sau."""
    Path(LATENCY_FILE).parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts}\tadapter={ADAPTER}\t4bit={FOUR_BIT}\ttokens={metrics.get('new_tokens', 0)}\t"
        f"ttft={metrics.get('ttft', 0.0):.3f}s\ttok_per_s={metrics.get('tok_per_s', 0.0):.2f}\t"
        f"decode_tok_per_s={metrics.get('decode_tok_per_s', 0.0):.2f}\tprompt={prompt[:60]!r}\n"
    )
    with Path(LATENCY_FILE).open("a", encoding="utf-8") as fh:
        fh.write(line)


def _build_enc(messages: list[dict]) -> dict:
    """apply_chat_template → BatchEncoding trên device."""
    kwargs: dict = {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": True}
    return tok.apply_chat_template(messages, **kwargs).to(model.device)


def _answer_of(full: str) -> str:
    """Tách câu trả lời sạch từ output (bỏ ``<think>``, dọn lượt giả)."""
    body = full.split("</think>", 1)[1] if "</think>" in full else full
    return _sanitize_answer(body)


def _think_of(full: str) -> str:
    """Lấy phần suy luận (bỏ thẻ ``<think>``) để hiển thị thẳng ra UI, không gập."""
    head = full.split("</think>", 1)[0]
    return re.sub(r"^\s*<think>\s*", "", head).strip()


def _render_think(think_ph: object, text: str) -> None:
    """Đổ phần suy luận thẳng ra UI (không gập), tách rõ với câu trả lời bên dưới."""
    if text:
        think_ph.markdown(f"🧠 **Suy luận của senior**\n\n{text}\n\n---")
    else:
        think_ph.empty()


def _stream_turn(
    messages: list[dict],
    think_ph: object,
    answer_ph: object,
    metrics: dict,
) -> str:
    """Stream một lượt generate và hiển thị; trả full text.

    ``<think>`` hiển thị thẳng ra UI (think_ph); phần sau vào answer_ph.
    """
    full = ""
    in_answer = False
    for piece in stream_generate(model, tok, _build_enc(messages), metrics):
        full += piece
        if not in_answer and "</think>" in full:
            _render_think(think_ph, _think_of(full))
            in_answer = True
        if in_answer:
            post = full.split("</think>", 1)[1]
            answer_ph.markdown(_sanitize_answer(post) + " ▌")
        else:
            _render_think(think_ph, re.sub(r"^\s*<think>\s*", "", full).strip())
    return full


USER_AVATAR = "🧑"
BOT_AVATAR = "🍎"

# set_page_config PHẢI là lệnh Streamlit đầu tiên (trước cả load() có thể gọi st.warning).
st.set_page_config(page_title="iPhone Sales Coach", page_icon="🍎", layout="centered")

SYS = load_system_prompt()
CATALOG = load_catalog()
if CATALOG:
    # Grounding: nhồi catalog vào prompt.
    SYS = (
        f"{SYS}\n\n"
        "=== CATALOG iPhone (giá chính hãng FPT Shop) ===\n"
        "BẮT BUỘC: chỉ dùng GIÁ và THÔNG SỐ trong danh sách dưới đây; TUYỆT ĐỐI không bịa hay tự "
        "suy ra số khác. Nếu khách hỏi sản phẩm không có trong danh sách, nói chưa có thông tin và "
        "mời tra catalog, không đoán.\n"
        f"{CATALOG}"
    )
if SCOPE_GUARD:
    # Giới hạn phạm vi: chỉ tư vấn iPhone, ngoài ra từ chối khéo.
    SYS = (
        f"{SYS}\n\n"
        "=== PHẠM VI (BẮT BUỘC) ===\n"
        "Bạn CHỈ tư vấn mua bán điện thoại iPhone. Với MỌI yêu cầu ngoài phạm vi — sản phẩm khác "
        "(tai nghe, sạc, ốp, máy hãng khác...), chuyện ngoài lề, hỏi bạn là ai/là bot — KHÔNG tư "
        'vấn, mà gợi ý trả lời khách đúng câu: "Dạ em chỉ tư vấn bán điện thoại iPhone thôi ạ." '
        "rồi mời khách quay lại nhu cầu iPhone nếu có."
    )
model, tok = load()

st.title("🍎 iPhone Sales Coach")
st.caption("Trợ lý kèm sale — gợi ý câu trả lời cho khách kèm suy luận của senior.")

# Lịch sử hội thoại cho giống chat thật — chỉ lưu trả lời SẠCH (không nhét reasoning vào lượt sau).
if "history" not in st.session_state:
    st.session_state.history = []  # list[{"role": "user"|"assistant", "content": str}]

with st.sidebar:
    st.header("⚙️ Phiên làm việc")
    st.markdown(f"**Adapter** · `{Path(ADAPTER).parent.name}`")
    st.markdown(f"**Precision** · {'4-bit (QLoRA)' if FOUR_BIT else 'bf16 (LoRA)'}")
    st.divider()
    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.caption("Giá & thông số được nhồi từ catalog vào prompt để model bám số liệu thật.")

for turn in st.session_state.history:
    avatar = BOT_AVATAR if turn["role"] == "assistant" else USER_AVATAR
    st.chat_message(turn["role"], avatar=avatar).write(turn["content"])

if not st.session_state.history:
    st.info(
        "👋 Nhập **tình huống khách** vào ô bên dưới — ví dụ: "
        '*"Khách muốn mua iPhone tầm 15 triệu, nên tư vấn sao?"*'
    )

if msg := st.chat_input("Tình huống khách..."):
    st.chat_message("user", avatar=USER_AVATAR).write(msg)
    st.session_state.history.append({"role": "user", "content": msg})

    messages = [{"role": "system", "content": SYS}, *st.session_state.history]

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        # Suy luận hiện thẳng ra UI (không gập trong st.status), rồi tới câu trả lời.
        think_ph = st.empty()
        answer_ph = st.empty()

        metrics: dict = {}
        full = _stream_turn(messages, think_ph, answer_ph, metrics)

        answer = _answer_of(full)
        answer_ph.markdown(answer or "_(không có nội dung)_")

        # Tốc độ gen (của lượt sinh câu trả lời hiển thị) + ghi file.
        if metrics.get("new_tokens"):
            st.caption(
                f"⚡ {metrics['tok_per_s']:.1f} tok/s · {metrics['new_tokens']} tokens · "
                f"TTFT {metrics['ttft']:.2f}s · {'4-bit' if FOUR_BIT else 'bf16'}"
                " · 📦 catalog"
            )
            log_latency(metrics, msg)

    st.session_state.history.append({"role": "assistant", "content": answer})
