"""Demo function-calling (tool-use) tra cứu iPhone bằng Qwen — KHÔNG phải pipeline train/eval.

Minh hoạ model *chủ động gọi tool* ``lookup_iphone`` (xem ``coach_tools.py``) để lấy giá/thông số
CHÍNH XÁC từ ``data/products.jsonl`` thay vì bịa.

    uv run python tool_lookup_demo.py "iPhone 16 256GB giá bao nhiêu"
    COACH_ADAPTER="" uv run python tool_lookup_demo.py "..."   # chạy BASE (tool-call chuẩn nhất)

LƯU Ý: adapter coach SFT trên format think+answer; nó vẫn có thể gọi tool (khả năng base Qwen còn
giữ) nhưng đôi khi phát ở format XML ``<function=...>`` — ``parse_tool_calls`` đã xử lý cả hai.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from coach_tools import TOOL_IMPLS, TOOLS, parse_tool_calls

BASE = os.environ.get("COACH_BASE", "Qwen/Qwen3.5-9B")
ADAPTER = os.environ.get("COACH_ADAPTER", "checkpoints/sft_coach_9b/best")
FOUR_BIT = os.environ.get("COACH_4BIT", "true").strip().lower() not in ("false", "0", "no")

SYSTEM = (
    "Bạn là trợ lý tư vấn iPhone tại FPT Shop. Khi khách hỏi về giá, dung lượng, màu, tồn kho hay "
    "thông số, BẮT BUỘC gọi tool `lookup_iphone` để lấy số liệu thật — TUYỆT ĐỐI không bịa giá. "
    "Chỉ tư vấn sản phẩm tool trả về (đang kinh doanh). Trả lời ngắn gọn, văn nói, xưng 'em' gọi "
    "khách 'anh/chị'."
)


def _generate(model: object, tok: object, messages: list[dict]) -> str:
    """Sinh một lượt assistant (greedy), trả text đã decode (bỏ prompt)."""
    import torch

    text = tok.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    enc = tok(text, return_tensors="pt").to(model.device)
    eos = [
        i
        for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
        if isinstance(i, int) and i >= 0
    ]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=eos or None,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def run(question: str) -> None:
    """Vòng function-calling: hỏi → model gọi tool → thực thi → trả lời cuối (in trace)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    src = ADAPTER if Path(ADAPTER).exists() else BASE
    print(f"[load] base={BASE} adapter={src if src != BASE else '(none, base)'} 4bit={FOUR_BIT}")
    if FOUR_BIT:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=bnb, device_map="auto"
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map="auto"
        )
    if src != BASE:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, src).eval()
        tok = AutoTokenizer.from_pretrained(src)
    else:
        model = base.eval()
        tok = AutoTokenizer.from_pretrained(BASE)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    print(f"\n[user] {question}")

    resp = _generate(model, tok, messages)
    calls = parse_tool_calls(resp)

    if not calls:
        print("\n[assistant — KHÔNG gọi tool]\n" + resp)
        return

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ],
        }
    )
    for c in calls:
        impl = TOOL_IMPLS.get(c["name"])
        result = impl(**c["arguments"]) if impl else {"error": f"unknown tool {c['name']}"}
        n = len(result) if isinstance(result, list) else 1
        print(f"\n[tool-call] {c['name']}({c['arguments']}) → {n} kết quả")
        print("[tool-result] " + json.dumps(result, ensure_ascii=False))
        messages.append(
            {"role": "tool", "name": c["name"], "content": json.dumps(result, ensure_ascii=False)}
        )

    final = _generate(model, tok, messages)
    print("\n[assistant — sau khi tra tool]\n" + final)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Demo Qwen function-calling tra cứu iPhone.")
    parser.add_argument("question", help="Câu hỏi của khách, vd 'iPhone 16 256GB giá bao nhiêu'.")
    args = parser.parse_args()
    run(args.question)


if __name__ == "__main__":
    main()
