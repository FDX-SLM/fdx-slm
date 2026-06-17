"""Canonical record schema and validation for the delivered JSONL data contract.

The data team delivers three record shapes (``sft``, ``reasoning``, ``preference``) that
share a common metadata block (see ``docs/SPEC.md`` §3). This module validates each record
with pydantic and reports distribution statistics. The repo *consumes* this data; it never
creates it.

``mode`` is metadata only — it must never enter the tokenized training sequence (enforced
downstream in :mod:`slm_coach.data.formatting`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)


class Role(str, Enum):
    """Valid message roles in a conversation turn."""

    system = "system"
    user = "user"
    assistant = "assistant"


class Mode(str, Enum):
    """The canonical 7 conversation modes for this project (see docs/SPEC.md §3).

    Metadata only — never fed into the training sequence. These exact string values
    match the data the data team produces and the distillation prompt library.
    """

    purchase_intent = "purchase_intent"  # khách có ý định mua, cần dẫn tới quyết định
    comparison = "comparison"  # so sánh model/cấu hình
    objection_handling = "objection_handling"  # xử lý phản đối (giá, lo ngại, do dự)
    upsell = "upsell"  # gợi ý nâng cấp/phụ kiện
    after_sales = "after_sales"  # bảo hành, đổi trả, hỗ trợ sau mua
    complex_query = "complex_query"  # câu hỏi kỹ thuật phức tạp
    edge_case = "edge_case"  # tình huống biên (thô lỗ, mơ hồ, thao túng)


class DataType(str, Enum):
    """The three canonical record shapes."""

    sft = "sft"
    reasoning = "reasoning"
    preference = "preference"


class ConversationType(str, Enum):
    """Whether an SFT record is a single exchange or a multi-turn dialogue."""

    single = "single"
    multi_turn = "multi_turn"


class Message(BaseModel):
    """A single chat turn (role + content)."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("message content must be non-empty")
        return value


class _CommonMeta(BaseModel):
    """Metadata shared by every canonical record."""

    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    id: str = Field(min_length=1)
    mode: Mode
    persona: str = Field(min_length=1)
    source: str | None = None
    lang: str = "vi"
    version: str = Field(min_length=1)
    audit_status: str = Field(min_length=1)


class SFTRecord(_CommonMeta):
    """Supervised fine-tuning record (single-turn or multi-turn)."""

    data_type: Literal["sft"]
    conversation_type: ConversationType = ConversationType.multi_turn
    messages: list[Message] = Field(min_length=1)

    @field_validator("conversation_type", mode="before")
    @classmethod
    def _normalize_conversation_type(cls, value: object) -> object:
        """Accept the common ``"single_turn"`` spelling as an alias for ``"single"``."""
        if isinstance(value, str) and value == "single_turn":
            return ConversationType.single.value
        return value

    @field_validator("messages")
    @classmethod
    def _has_assistant_and_user(cls, messages: list[Message]) -> list[Message]:
        roles = {m.role for m in messages}
        if Role.assistant not in roles:
            raise ValueError("sft record must contain at least one assistant turn")
        if Role.user not in roles:
            raise ValueError("sft record must contain at least one user turn")
        return messages

    @property
    def num_assistant_turns(self) -> int:
        """Number of assistant turns (used to classify single vs multi-turn)."""
        return sum(1 for m in self.messages if m.role == Role.assistant)


class ReasoningRecord(_CommonMeta):
    """Chain-of-thought record. ``why`` is audit-only and must never be trained."""

    data_type: Literal["reasoning"]
    situation: str = Field(min_length=1)
    reasoning: list[str] = Field(min_length=1)
    response: str = Field(min_length=1)
    why: str | None = None  # audit-only — NEVER folded into the training sequence

    @field_validator("reasoning")
    @classmethod
    def _steps_non_empty(cls, steps: list[str]) -> list[str]:
        if any(not s or not s.strip() for s in steps):
            raise ValueError("reasoning steps must all be non-empty")
        return steps


class PreferenceRecord(_CommonMeta):
    """Preference pair for DPO alignment (explicit prompt + chosen/rejected)."""

    data_type: Literal["preference"]
    bad_type: str | None = None
    prompt: list[Message] = Field(min_length=1)
    chosen: list[Message] = Field(min_length=1)
    rejected: list[Message] = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def _prompt_has_user(cls, messages: list[Message]) -> list[Message]:
        if not any(m.role == Role.user for m in messages):
            raise ValueError("preference prompt must contain a user turn")
        return messages

    @field_validator("chosen", "rejected")
    @classmethod
    def _completion_is_assistant(cls, messages: list[Message]) -> list[Message]:
        if not all(m.role == Role.assistant for m in messages):
            raise ValueError("chosen/rejected turns must all have the assistant role")
        return messages


class GoldCase(BaseModel):
    """A single gold test case (canonical, pinned shape) consumed by evaluation.

    Canonical delivery shape (one JSON object per line in ``gold_test.jsonl``)::

        {"id": "...", "mode": "comparison",
         "messages": [{"role": "user", "content": "..."}],
         "reference": "<senior/gold answer>", "persona": "P0x"}

    For tolerance, a before-validator also accepts a few common variants: a ``prompt`` /
    ``question`` / ``situation`` field instead of ``messages``; a trailing assistant turn or a
    ``response`` / ``answer`` field as the ``reference``. After validation every case exposes a
    uniform ``messages`` prompt (no assistant turns), a ``reference`` string, and a ``mode``.
    """

    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    id: str = Field(min_length=1)
    mode: Mode
    messages: list[Message] = Field(min_length=1)
    reference: str = ""
    persona: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        """Normalize common gold shapes into ``messages`` (prompt) + ``reference``."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if isinstance(data.get("messages"), list):
            messages = list(data["messages"])
            # The gold answer is the LAST assistant turn. Drop ONLY that turn (and use it as the
            # reference if none was given) — keep any EARLIER assistant turns so multi-turn
            # conversation context survives in the prompt.
            if data.get("reference"):
                if messages and messages[-1].get("role") == "assistant":
                    messages.pop()
            else:
                for index in range(len(messages) - 1, -1, -1):
                    if messages[index].get("role") == "assistant":
                        data["reference"] = messages[index].get("content", "")
                        messages = messages[:index]
                        break
            data["messages"] = messages or data["messages"]
        else:
            for key in ("prompt", "question", "situation", "input"):
                if key in data:
                    value = data[key]
                    data["messages"] = (
                        value
                        if isinstance(value, list)
                        else [{"role": "user", "content": str(value)}]
                    )
                    break
        if not data.get("reference"):
            for key in ("response", "answer", "gold", "expected"):
                if data.get(key):
                    data["reference"] = str(data[key])
                    break
        return data

    @property
    def prompt(self) -> list[dict[str, str]]:
        """The prompt as role/content dicts (ready for a chat template)."""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]


def parse_gold_case(obj: dict) -> GoldCase:
    """Validate a raw dict into a :class:`GoldCase`.

    Raises:
        pydantic.ValidationError: If the object violates the gold schema.
    """
    return GoldCase.model_validate(obj)


CanonicalRecord = Annotated[
    SFTRecord | ReasoningRecord | PreferenceRecord,
    Field(discriminator="data_type"),
]

_RECORD_ADAPTER: TypeAdapter[CanonicalRecord] = TypeAdapter(CanonicalRecord)


def parse_record(obj: dict) -> CanonicalRecord:
    """Validate a raw dict into the correct canonical record.

    Args:
        obj: A decoded JSONL object with a ``data_type`` discriminator.

    Returns:
        The validated :data:`CanonicalRecord` (one of the three concrete models).

    Raises:
        pydantic.ValidationError: If the object violates the schema.
    """
    return _RECORD_ADAPTER.validate_python(obj)


@dataclass
class ValidationStats:
    """Aggregate validation statistics for a file or directory."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    by_data_type: Counter[str] = field(default_factory=Counter)
    by_mode: Counter[str] = field(default_factory=Counter)
    errors: list[tuple[int, str]] = field(default_factory=list)

    def merge(self, other: ValidationStats) -> None:
        """Accumulate another stats object into this one (used across files)."""
        self.total += other.total
        self.valid += other.valid
        self.invalid += other.invalid
        self.by_data_type.update(other.by_data_type)
        self.by_mode.update(other.by_mode)
        self.errors.extend(other.errors)

    def as_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "by_data_type": dict(self.by_data_type),
            "by_mode": dict(self.by_mode),
            "errors": self.errors,
        }


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(line_number, object)`` for each non-blank line of a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.

    Yields:
        Tuples of 1-based line number and the decoded JSON object.

    Raises:
        json.JSONDecodeError: If a non-blank line is not valid JSON.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield lineno, json.loads(line)


def validate_file(path: str | Path) -> ValidationStats:
    """Validate every record in a JSONL file and collect distribution statistics.

    Malformed JSON and schema violations are recorded as errors (with line numbers)
    rather than raising, so a whole file can be inspected in one pass.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A :class:`ValidationStats` summarizing valid/invalid counts and distributions.
    """
    stats = ValidationStats()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats.total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.invalid += 1
                stats.errors.append((lineno, f"invalid JSON: {exc}"))
                continue
            try:
                record = parse_record(obj)
            except ValidationError as exc:
                stats.invalid += 1
                stats.errors.append((lineno, _short_error(exc)))
                continue
            stats.valid += 1
            stats.by_data_type[record.data_type] += 1
            stats.by_mode[record.mode] += 1
    return stats


def _short_error(exc: ValidationError) -> str:
    """Condense a pydantic error into a single readable line."""
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    return f"{loc}: {first.get('msg', 'validation error')}"
