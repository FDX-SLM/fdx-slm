r"""Convert canonical records into TRL-ready formats.

Responsibilities:

* ``sft`` records -> ChatML ``messages`` (chat template applied at train time).
* ``reasoning`` records -> assistant turn with ``<think>...\\n</think>\\n{response}`` folding
  (toggleable so non-thinking examples also exist). The audit-only ``why`` field is **never**
  included.
* ``preference`` records -> ``{prompt, chosen, rejected}`` with an explicit prompt.
* **Multi-turn masking**: train-on-responses-only computes loss on *every* assistant turn,
  not just the last. :func:`iter_assistant_spans` exposes the trainable spans so this can be
  unit-tested offline without a tokenizer; the trainer applies the equivalent token-level mask
  via the tokenizer's chat template at training time.

Metadata (``mode``, ``persona``, ``source`` ...) is deliberately dropped here — it must never
enter the tokenized sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from slm_coach.data.schema import (
    CanonicalRecord,
    PreferenceRecord,
    ReasoningRecord,
    Role,
    SFTRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import Dataset

# ChatML control tokens used by the offline renderer (mirrors Qwen-style templates).
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

Messages = list[dict[str, str]]


def _msg(role: str, content: str) -> dict[str, str]:
    """Build a plain ``{"role", "content"}`` dict (metadata stripped)."""
    return {"role": role, "content": content}


def fold_reasoning(reasoning_steps: Sequence[str], response: str, *, thinking: bool) -> str:
    """Fold chain-of-thought steps into a single assistant turn.

    Args:
        reasoning_steps: Ordered reasoning steps.
        response: The final answer shown to the user.
        thinking: If ``True``, prepend a ``<think>...</think>`` block; otherwise return the
            bare response (a non-thinking example).

    Returns:
        The assistant-turn content string.
    """
    if not thinking:
        return response
    thought = "\n".join(reasoning_steps)
    return f"<think>\n{thought}\n</think>\n{response}"


def sft_to_messages(record: SFTRecord) -> Messages:
    """Project an SFT record onto role/content message dicts (metadata removed)."""
    return [_msg(m.role.value, m.content) for m in record.messages]


def reasoning_to_messages(record: ReasoningRecord, *, thinking: bool) -> Messages:
    """Project a reasoning record onto a user/assistant pair (``why`` excluded).

    Args:
        record: The reasoning record.
        thinking: Whether to fold a ``<think>`` block into the assistant turn.

    Returns:
        A two-message conversation: user ``situation`` then the assistant answer.
    """
    return [
        _msg(Role.user.value, record.situation),
        _msg(
            Role.assistant.value,
            fold_reasoning(record.reasoning, record.response, thinking=thinking),
        ),
    ]


def record_to_messages(record: CanonicalRecord, *, reasoning_thinking: bool = False) -> Messages:
    """Convert any SFT/reasoning record into ChatML messages.

    Args:
        record: A ``sft`` or ``reasoning`` canonical record.
        reasoning_thinking: Whether reasoning records fold a ``<think>`` block.

    Returns:
        The message list (role/content only).

    Raises:
        TypeError: If ``record`` is a preference record (use :func:`to_preference_dataset`).
    """
    if isinstance(record, SFTRecord):
        return sft_to_messages(record)
    if isinstance(record, ReasoningRecord):
        return reasoning_to_messages(record, thinking=reasoning_thinking)
    raise TypeError(f"record_to_messages does not handle {type(record).__name__}")


def render_chatml(messages: Messages) -> str:
    """Render messages to a deterministic ChatML string (offline, no tokenizer).

    Used for offline masking tests and as a fallback when no tokenizer is supplied. At
    training time the real tokenizer's chat template is used instead.

    Args:
        messages: Role/content message dicts.

    Returns:
        The concatenated ChatML string.
    """
    parts = [f"{_IM_START}{m['role']}\n{m['content']}{_IM_END}\n" for m in messages]
    return "".join(parts)


def iter_assistant_spans(messages: Messages) -> list[tuple[int, int]]:
    """Return character spans of every assistant *content* region in the rendered text.

    This is the offline, tokenizer-free expression of train-on-responses-only with
    multi-turn masking: there is one span per assistant turn (not only the last), and
    user/system content is excluded.

    Args:
        messages: Role/content message dicts (same order passed to :func:`render_chatml`).

    Returns:
        A list of ``(start, end)`` character offsets into ``render_chatml(messages)``,
        one per assistant turn, each bounding exactly that turn's content.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        header = f"{_IM_START}{message['role']}\n"
        content_start = cursor + len(header)
        content_end = content_start + len(message["content"])
        if message["role"] == Role.assistant.value:
            spans.append((content_start, content_end))
        # advance past content + closing token + newline
        cursor = content_end + len(_IM_END) + 1
    return spans


def to_sft_dataset(
    records: Sequence[CanonicalRecord],
    *,
    reasoning_thinking: bool = False,
    tokenizer: Any | None = None,
) -> Dataset:
    """Build a TRL-compatible SFT dataset from canonical records.

    Each row carries ``messages`` (role/content list) plus a rendered ``text`` column. When a
    ``tokenizer`` is provided its chat template is applied; otherwise the offline ChatML
    renderer is used. The trainer applies multi-turn assistant masking from ``messages``.

    Args:
        records: SFT and/or reasoning records (preference records are rejected upstream).
        reasoning_thinking: Whether reasoning records fold a ``<think>`` block.
        tokenizer: Optional HF tokenizer whose ``apply_chat_template`` renders ``text``.

    Returns:
        A :class:`datasets.Dataset` with ``messages`` and ``text`` columns.
    """
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for record in records:
        messages = record_to_messages(record, reasoning_thinking=reasoning_thinking)
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = render_chatml(messages)
        rows.append({"messages": messages, "text": text})
    return Dataset.from_list(rows)


def preference_to_example(record: PreferenceRecord) -> dict[str, Any]:
    """Convert one preference record into a TRL DPO example.

    Args:
        record: A preference canonical record.

    Returns:
        A dict with explicit ``prompt`` messages and ``chosen``/``rejected`` assistant text.
    """
    prompt = [_msg(m.role.value, m.content) for m in record.prompt]
    chosen = "\n".join(m.content for m in record.chosen)
    rejected = "\n".join(m.content for m in record.rejected)
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def to_preference_dataset(records: Sequence[PreferenceRecord]) -> Dataset:
    """Build a TRL preference dataset (``prompt``/``chosen``/``rejected``).

    Args:
        records: Preference canonical records.

    Returns:
        A :class:`datasets.Dataset` with ``prompt``, ``chosen``, and ``rejected`` columns.
    """
    from datasets import Dataset

    return Dataset.from_list([preference_to_example(r) for r in records])
