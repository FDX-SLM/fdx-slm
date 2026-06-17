"""Tests for the canonical record schema and file validation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from slm_coach.data.schema import (
    PreferenceRecord,
    ReasoningRecord,
    SFTRecord,
    parse_record,
    validate_file,
)


def _sft(**over) -> dict:
    base = {
        "id": "s1",
        "data_type": "sft",
        "conversation_type": "multi_turn",
        "mode": "comparison",
        "persona": "P01",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý bán hàng."},
            {"role": "user", "content": "iPhone 15 khác gì 14?"},
            {"role": "assistant", "content": "iPhone 15 có cổng USB-C..."},
            {"role": "user", "content": "Giá thế nào?"},
            {"role": "assistant", "content": "Giá khoảng 22 triệu."},
        ],
        "lang": "vi",
        "version": "v1",
        "audit_status": "approved",
    }
    base.update(over)
    return base


def _reasoning(**over) -> dict:
    base = {
        "id": "r1",
        "data_type": "reasoning",
        "mode": "objection_handling",
        "persona": "P02",
        "situation": "Khách chê đắt.",
        "reasoning": ["Xác định ngân sách", "Nhấn mạnh giá trị"],
        "response": "Em hiểu ạ, mình xem trả góp nhé.",
        "why": "audit-only note",
        "lang": "vi",
        "version": "v1",
        "audit_status": "approved",
    }
    base.update(over)
    return base


def _preference(**over) -> dict:
    base = {
        "id": "p1",
        "data_type": "preference",
        "mode": "purchase_intent",
        "persona": "P03",
        "bad_type": "pushy",
        "prompt": [{"role": "user", "content": "Có giảm giá không?"}],
        "chosen": [{"role": "assistant", "content": "Mình có ưu đãi trả góp 0%."}],
        "rejected": [{"role": "assistant", "content": "Mua ngay đi anh, hết hàng giờ!"}],
        "lang": "vi",
        "version": "v1",
        "audit_status": "approved",
    }
    base.update(over)
    return base


def test_parse_sft_record():
    rec = parse_record(_sft())
    assert isinstance(rec, SFTRecord)
    assert rec.mode == "comparison"
    assert rec.data_type == "sft"
    assert rec.num_assistant_turns == 2


def test_sft_accepts_single_turn_alias_and_ignores_extra_fields():
    # Real delivery uses "single_turn" (alias of "single") + extra metadata fields.
    rec = parse_record(
        _sft(
            conversation_type="single_turn",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "<think>x</think>\ny"},
            ],
            use_case="training",
            modes_covered=["comparison", "objection_handling"],
            customer_persona="C04",
            source="synthetic-deepseek-v4-flash",
        )
    )
    assert rec.conversation_type == "single"  # normalized
    assert rec.num_assistant_turns == 1


def test_parse_reasoning_record_keeps_why_field_for_audit():
    rec = parse_record(_reasoning())
    assert isinstance(rec, ReasoningRecord)
    assert rec.why == "audit-only note"  # present in the record, but never trained
    assert rec.reasoning == ["Xác định ngân sách", "Nhấn mạnh giá trị"]


def test_parse_preference_record():
    rec = parse_record(_preference())
    assert isinstance(rec, PreferenceRecord)
    assert rec.bad_type == "pushy"


@pytest.mark.parametrize(
    "mutation",
    [
        {"mode": "not_a_mode"},
        {"messages": [{"role": "user", "content": "hi"}]},  # no assistant turn
        {"messages": [{"role": "assistant", "content": "hi"}]},  # no user turn
        {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]},
    ],
)
def test_invalid_sft_records_raise(mutation):
    with pytest.raises(ValidationError):
        parse_record(_sft(**mutation))


def test_invalid_data_type_raises():
    with pytest.raises(ValidationError):
        parse_record(_sft(data_type="unknown"))


def test_reasoning_requires_non_empty_steps():
    with pytest.raises(ValidationError):
        parse_record(_reasoning(reasoning=["ok", "  "]))


def test_preference_completion_must_be_assistant():
    with pytest.raises(ValidationError):
        parse_record(_preference(chosen=[{"role": "user", "content": "wrong role"}]))


def test_validate_file_collects_stats(tmp_path):
    path = tmp_path / "mixed.jsonl"
    lines = [
        json.dumps(_sft(id="ok1")),
        json.dumps(_reasoning(id="ok2")),
        json.dumps(_sft(id="bad", mode="not_a_mode")),  # invalid
        "{ not json",  # malformed
        "",  # blank, skipped
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    stats = validate_file(path)
    assert stats.total == 4
    assert stats.valid == 2
    assert stats.invalid == 2
    assert stats.by_data_type["sft"] == 1
    assert stats.by_data_type["reasoning"] == 1
    assert stats.by_mode["comparison"] == 1
    assert len(stats.errors) == 2
