"""Offline tests for the eval add-ons: system-prompt injection, judge cost, head-to-head, gold."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from slm_coach.eval.headtohead import build_headtohead_markdown, head_to_head, load_per_sample
from slm_coach.eval.judge import MockJudge, OpenAIJudge, judge_usage
from slm_coach.eval.runner import _with_system

# --- system-prompt injection -----------------------------------------------------------------


def test_with_system_injection():
    prompt = [{"role": "user", "content": "hi"}]
    assert _with_system(prompt, None) == prompt  # no-op when unset
    out = _with_system(prompt, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == prompt[0]
    # Never double-inject when a system turn already exists.
    with_sys = [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]
    assert _with_system(with_sys, "SYS") == with_sys


# --- judge cost / token tracking -------------------------------------------------------------


def test_judge_usage_aggregation():
    judge = OpenAIJudge("gpt-4o")
    judge.n_calls, judge.prompt_tokens, judge.completion_tokens = 3, 1000, 500
    usage = judge_usage([judge])
    assert usage["calls"] == 3
    assert usage["total_tokens"] == 1500
    assert usage["est_usd"] > 0  # gpt-4o is priced
    assert usage["by_judge"]["gpt"]["model"] == "gpt-4o"


def test_judge_usage_zero_for_mock():
    usage = judge_usage([MockJudge()])
    assert usage["calls"] == 0 and usage["est_usd"] == 0.0


# --- head-to-head win-rate -------------------------------------------------------------------


def test_load_per_sample_roundtrip(tmp_path):
    path = tmp_path / "per_sample.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "mode", "prompt", "answer", "reference"])
        writer.writeheader()
        writer.writerow(
            {"id": "c1", "mode": "comparison", "prompt": "p", "answer": "a", "reference": "r"}
        )
    rows = load_per_sample(path)
    assert rows["c1"]["answer"] == "a" and rows["c1"]["mode"] == "comparison"


def test_head_to_head_mock_winrate():
    rows_a = {
        "c1": {"mode": "comparison", "prompt": "p", "answer": "Dạ em xin tư vấn kỹ cho anh ạ"}
    }
    rows_b = {"c1": {"mode": "comparison", "prompt": "p", "answer": "ừ"}}
    result = head_to_head([MockJudge()], rows_a, rows_b)
    assert result["n"] == 1
    assert result["overall"]["win"] == 1.0  # A (longer + polite) beats B
    md = build_headtohead_markdown(result, label_a="SLM", label_b="parent")
    assert "SLM" in md and "comparison" in md and "Headline" in md


# --- gold dataset validity (synthetic file written into data/) --------------------------------


def test_gold_dataset_valid_and_covers_all_modes():
    from slm_coach.data.loader import load_gold_cases
    from slm_coach.data.schema import Mode

    path = Path("data/gold/gold_test.jsonl")
    if not path.is_file():
        pytest.skip("gold_test.jsonl not present")
    cases = load_gold_cases(path)
    assert len(cases) >= 14
    assert {c.mode for c in cases} == {m.value for m in Mode}  # all 7 conversation modes
    assert all(c.reference.strip() for c in cases)  # every case has a gold reference
    assert all(c.prompt and c.prompt[0]["role"] != "assistant" for c in cases)
