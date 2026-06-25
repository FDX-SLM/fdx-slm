"""Công cụ tra cứu sản phẩm dùng chung cho demo (coach_app.py + tool_lookup_demo.py).

Gồm: catalog grounding (nhồi vào prompt), tool ``lookup_iphone`` (function-calling), và parser
tool-call hỗ trợ cả 2 format model có thể phát (JSON và XML ``<function=...>``). KHÔNG thuộc
pipeline train/eval — chỉ phục vụ demo.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

PRODUCTS_FILE = os.environ.get("COACH_PRODUCTS", "data/products.jsonl")
IN_STOCK_ONLY = os.environ.get("COACH_IN_STOCK_ONLY", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)

#: crawl_note chứa một trong các cụm này = ngừng bán → loại khỏi kết quả tool.
DISCONTINUED_MARKERS = ("ngừng kinh doanh", "hết hàng toàn quốc")


def _vnd(amount: object) -> str:
    """Định dạng VND kiểu '11.990.000đ'."""
    try:
        return f"{int(amount):,}".replace(",", ".") + "đ"
    except (TypeError, ValueError):
        return "—"


def _is_active(note: object) -> bool:
    """True nếu sản phẩm còn kinh doanh (crawl_note không chứa cụm ngừng bán)."""
    text = str(note or "").lower()
    return not any(marker in text for marker in DISCONTINUED_MARKERS)


def _iter_products() -> list[dict]:
    """Đọc toàn bộ sản phẩm hợp lệ từ ``PRODUCTS_FILE`` (bỏ dòng hỏng)."""
    path = Path(PRODUCTS_FILE)
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _price_label(prices: list[int]) -> str:
    """Nhãn giá: 1 giá → số; nhiều → khoảng; rỗng → 'giá đang biến động'."""
    if not prices:
        return "giá đang biến động, để em xác nhận lại với anh/chị"
    if len(prices) == 1:
        return _vnd(prices[0])
    return f"{_vnd(prices[0])}–{_vnd(prices[-1])}"


def load_catalog() -> str:
    """Catalog gọn (gộp theo series+dung lượng) để CHÈN VÀO PROMPT làm grounding giá.

    Mặc định chỉ giữ nhóm còn hàng (``IN_STOCK_ONLY``). Trả về chuỗi rỗng nếu không có file.
    """
    groups: dict[tuple[str, object], dict] = {}
    for r in _iter_products():
        key = (str(r.get("series", "")), r.get("storage_gb"))
        g = groups.setdefault(key, {"prices": set(), "colors": [], "available": False, "row": r})
        if r.get("price") is not None:
            g["prices"].add(r["price"])
        if r.get("color") and r["color"] not in g["colors"]:
            g["colors"].append(r["color"])
        if str(r.get("availability", "")).strip().lower().startswith("còn"):
            g["available"] = True
    if not groups:
        return ""

    items = list(groups.items())
    if IN_STOCK_ONLY:
        items = [it for it in items if it[1]["available"]]
    rows = sorted(items, key=lambda it: (min(it[1]["prices"]) if it[1]["prices"] else 0))

    lines = []
    for (series, storage), g in rows:
        r = g["row"]
        specs = ", ".join(
            s
            for s in (
                str(r.get("chip")) if r.get("chip") else "",
                f'{r.get("screen_inch")}"' if r.get("screen_inch") else "",
                f'camera chính {r.get("camera_main_mp")}MP' if r.get("camera_main_mp") else "",
            )
            if s
        )
        colors = "/".join(g["colors"]) if g["colors"] else "—"
        stock = "còn hàng" if g["available"] else "hết/không rõ"
        lines.append(
            f"- {series} {storage}GB — {_price_label(sorted(g['prices']))} — {specs} — "
            f"màu: {colors} — {stock} (BH {r.get('warranty_months', '—')} tháng)"
        )
    return "\n".join(lines)


def lookup_iphone(series: str | None = None, storage_gb: int | None = None) -> list[dict]:
    """Tool: iPhone CÒN KINH DOANH khớp ``series``/``storage_gb`` (loại theo crawl_note).

    Khác ``load_catalog``: lọc theo ``crawl_note`` (ngừng kinh doanh / hết hàng toàn quốc), trả
    JSON cho model. Sản phẩm chưa có giá → ``price`` = 'giá đang biến động'.

    Args:
        series: Lọc theo dòng máy (khớp chuỗi con tên, không phân biệt hoa thường). None = tất cả.
        storage_gb: Lọc theo dung lượng GB. None = mọi dung lượng.

    Returns:
        Danh sách dict ``{series, storage_gb, price, chip, screen_inch, colors, warranty_months}``.
    """
    groups: dict[tuple[str, object], dict] = {}
    for r in _iter_products():
        if not _is_active(r.get("crawl_note")):
            continue
        if series and series.strip().lower() not in str(r.get("name", "")).lower():
            continue
        if storage_gb and r.get("storage_gb") != storage_gb:
            continue
        key = (str(r.get("series", "")), r.get("storage_gb"))
        g = groups.setdefault(key, {"prices": set(), "colors": [], "row": r})
        if r.get("price") is not None:
            g["prices"].add(r["price"])
        if r.get("color") and r["color"] not in g["colors"]:
            g["colors"].append(r["color"])

    results: list[dict] = []
    for (series_name, storage), g in sorted(
        groups.items(), key=lambda it: (min(it[1]["prices"]) if it[1]["prices"] else 0)
    ):
        r = g["row"]
        results.append(
            {
                "series": series_name,
                "storage_gb": storage,
                "price": _price_label(sorted(g["prices"])),
                "chip": r.get("chip"),
                "screen_inch": r.get("screen_inch"),
                "colors": g["colors"],
                "warranty_months": r.get("warranty_months"),
            }
        )
    return results


#: Schema tool cho apply_chat_template(tools=...).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_iphone",
            "description": (
                "Tra cứu iPhone ĐANG KINH DOANH tại FPT Shop (giá, dung lượng, chip, màu). "
                "Tự loại sản phẩm đã ngừng kinh doanh / hết hàng toàn quốc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {
                        "type": "string",
                        "description": "Dòng máy, vd 'iPhone 16', 'iPhone 15 Pro'. Trống = tất cả.",
                    },
                    "storage_gb": {
                        "type": "integer",
                        "description": "Dung lượng GB: 128, 256, 512. Trống = mọi dung lượng.",
                    },
                },
            },
        },
    }
]

#: Tên tool → hàm thực thi.
TOOL_IMPLS = {"lookup_iphone": lookup_iphone}

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _coerce(value: str) -> object:
    """Ép tham số chuỗi về int khi là số nguyên (vd storage_gb '512' → 512)."""
    value = value.strip()
    return int(value) if re.fullmatch(r"-?\d+", value) else value


def parse_tool_calls(text: str) -> list[dict]:
    """Trích tool-call, hỗ trợ cả 2 format model có thể phát.

    - JSON: ``<tool_call>{"name": .., "arguments": {..}}</tool_call>``
    - XML:  ``<tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>``
    """
    calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(text):
        body = match.group(1).strip()
        if body.startswith("{"):
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": obj.get("name"), "arguments": args})
            continue
        fn = _FUNCTION_RE.search(body)
        if fn:
            args = {k: _coerce(v) for k, v in _PARAMETER_RE.findall(fn.group(2))}
            calls.append({"name": fn.group(1), "arguments": args})
    return calls


def looks_like_tool_call(text: str) -> bool:
    """Heuristic phát hiện model đang phát tool-call (để UI đổi sang trạng thái 'tra cứu')."""
    return "<tool_call>" in text or "<function=" in text
