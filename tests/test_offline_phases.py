"""Offline (no-GPU/no-API) tests for the Phase 2-4 logic that does not need heavy deps."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slm_coach.config import EvalDuringTrainingConfig, EvalFileConfig
from slm_coach.data.schema import parse_gold_case
from slm_coach.eval.harness_task import TASK_NAME, build_task_config, write_task_yaml
from slm_coach.eval.judge import MockJudge, build_judges, parse_scores, parse_winner
from slm_coach.eval.latency import summarize_latencies
from slm_coach.eval.metrics import SampleScore, aggregate_by_mode
from slm_coach.eval.report import write_report
from slm_coach.eval.rubric import CRITERIA, DEFAULT_WEIGHTS, render_rubric_block
from slm_coach.eval.runner import run_evaluation
from slm_coach.export.quantize import load_calibration_texts
from slm_coach.training.callbacks import (
    EvalDuringTraining,
    overlap_f1,
    proxy_rubric_score,
    write_meta_json,
)
from slm_coach.training.sft import split_holdout

# --- Phase 2: checkpoint meta + eval-during-training callback ---------------------------------


def test_write_meta_json(tmp_path):
    ckpt = tmp_path / "best"
    path = write_meta_json(
        ckpt, config={"model_name": "x"}, seed=42, data_version="v1", metrics={"a": 1.0}
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seed"] == 42
    assert data["data_version"] == "v1"
    assert data["config"]["model_name"] == "x"
    assert data["metrics"]["a"] == 1.0
    assert "git_commit" in data


def test_overlap_f1_and_proxy_scale():
    assert overlap_f1("a b c", "a b c") == pytest.approx(1.0)
    assert overlap_f1("", "x") == 0.0
    assert proxy_rubric_score("a b", "a b") == pytest.approx(5.0)
    assert proxy_rubric_score("x", "y") == pytest.approx(1.0)


def test_eval_during_training_proxy_metric():
    cfg = EvalDuringTrainingConfig(enabled=True, subset_size=4)
    gold = [
        {"prompt": [{"role": "user", "content": "hi"}], "reference": "xin chào", "mode": "greeting"}
    ]
    callback = EvalDuringTraining(
        cfg, gold_records=gold, generate_fn=lambda prompts: ["xin chào"], tracker=None
    )
    metrics: dict[str, float] = {}
    callback.on_evaluate(None, SimpleNamespace(global_step=10), "CONTROL", metrics=metrics)
    assert metrics["eval_rubric_avg"] == pytest.approx(5.0)
    assert callback.history == [(10, pytest.approx(5.0))]


def test_eval_during_training_with_judges():
    cfg = EvalDuringTrainingConfig(enabled=True, subset_size=2, use_judge=True)
    gold = [
        {"prompt": [{"role": "user", "content": "hi"}], "reference": "xin chào", "mode": "greeting"}
    ]
    callback = EvalDuringTraining(
        cfg,
        gold_records=gold,
        generate_fn=lambda prompts: ["Dạ em cảm ơn anh ạ, iPhone rất tốt."],
        judges=[MockJudge()],
        metric_name="rubric_avg",
    )
    metrics: dict[str, float] = {}
    callback.on_evaluate(None, SimpleNamespace(global_step=5), "CONTROL", metrics=metrics)
    assert 1.0 <= metrics["eval_rubric_avg"] <= 5.0


# --- True held-out validation split (no leak) ------------------------------------------------


def test_split_holdout_no_leak_and_deterministic():
    records = list(range(100))
    train, val = split_holdout(records, 0.1, seed=1308)
    assert len(train) == 90 and len(val) == 10
    assert set(train).isdisjoint(set(val))  # held-out is never in train (no leak)
    assert set(train) | set(val) == set(records)  # partition, nothing dropped
    # Deterministic for the same seed; different seed reshuffles which rows are held out.
    assert split_holdout(records, 0.1, seed=1308)[1] == val
    assert split_holdout(records, 0.1, seed=7)[1] != val


def test_split_holdout_disabled_returns_all_train():
    records = list(range(20))
    train, val = split_holdout(records, 0.0, seed=1)
    assert train == records and val == []


def test_split_holdout_tiny_dataset_falls_back():
    # 10 records * 0.05 = 0.5 -> rounds to 0 val -> fallback to no split (smoke-safe).
    train, val = split_holdout(list(range(10)), 0.05, seed=1)
    assert len(train) == 10 and val == []


# --- Phase 3: rubric, judges, pairwise, latency, harness, report ------------------------------


def test_render_rubric_block():
    block = render_rubric_block()
    assert "factuality" in block
    assert "language_quality" in block
    assert block.count("\n") == len(CRITERIA) - 1


def test_mock_judge_score_and_compare():
    score = MockJudge().score(
        prompt="p", answer="Dạ em cảm ơn anh ạ, iPhone 15 rất tốt.", criteria=CRITERIA
    )
    values = score.as_dict()
    assert set(values) == set(CRITERIA)
    assert all(1.0 <= v <= 5.0 for v in values.values())

    judge = MockJudge()
    assert judge.compare(prompt="p", answer_a="Dạ em xin tư vấn kỹ ạ", answer_b="ừ") == "A"
    assert judge.compare(prompt="p", answer_a="x", answer_b="x") == "tie"


def test_build_judges_rejects_teacher_models():
    with pytest.raises(ValueError):
        build_judges(["gpt", "claude"], {})
    assert build_judges(["mock"], {})[0].name == "mock"


def test_parse_scores_and_winner_robust():
    body = json.dumps(dict.fromkeys(CRITERIA, 4))
    assert parse_scores(f"noise {body} end", CRITERIA).factuality == 4.0
    assert parse_scores("{}", CRITERIA).tone == 3.0  # missing -> default 3
    assert parse_winner('{"winner": "A"}') == "A"
    assert parse_winner('the answer is {"winner":"B"}') == "B"
    assert parse_winner("garbage") == "tie"


def test_summarize_latencies():
    stats = summarize_latencies([0.1, 0.2, 0.3, 0.4])
    assert stats.n == 4
    assert stats.p50 == pytest.approx(0.25)
    assert stats.mean == pytest.approx(0.25)
    assert summarize_latencies([]).n == 0


def test_build_and_write_task_config(tmp_path):
    cfg = build_task_config("data/gold/gold_test.jsonl")
    assert cfg["task"] == TASK_NAME
    assert cfg["dataset_kwargs"]["data_files"]["test"] == "data/gold/gold_test.jsonl"
    assert cfg["output_type"] == "generate_until"

    yaml_path = write_task_yaml("data/gold/gold_test.jsonl", tmp_path)
    assert yaml_path.exists()
    assert TASK_NAME in yaml_path.read_text(encoding="utf-8")


def test_load_calibration_texts(tmp_path):
    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "\n".join(
            [
                json.dumps({"text": "câu một"}),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ]
                    }
                ),
                json.dumps({"reference": "tham chiếu"}),
                "dòng thuần",
            ]
        ),
        encoding="utf-8",
    )
    texts = load_calibration_texts(calib)
    assert texts == ["câu một", "u\na", "tham chiếu", "dòng thuần"]


def test_write_report(tmp_path):
    samples = [
        SampleScore("a", "comparison", dict.fromkeys(CRITERIA, 4.0)),
        SampleScore("b", "objection_handling", dict.fromkeys(CRITERIA, 2.0)),
    ]
    breakdown = aggregate_by_mode(samples, DEFAULT_WEIGHTS)
    md, js = write_report(
        tmp_path,
        breakdown=breakdown,
        extras={
            "model": "m",
            "judges": ["mock"],
            "pairwise_vs_reference": {"win": 1.0, "tie": 0.0, "loss": 0.0, "n": 2},
        },
    )
    assert md.exists() and js.exists()
    text = md.read_text(encoding="utf-8")
    assert "Per-mode breakdown" in text
    assert "objection_handling" in text
    assert "Pairwise vs reference" in text
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert set(payload["per_mode"]) == {"comparison", "objection_handling"}
    assert "pii_leak_rate" not in payload  # PII removed


# --- Gold schema -----------------------------------------------------------------------------


def test_gold_case_normalization():
    gc = parse_gold_case(
        {
            "id": "g1",
            "mode": "comparison",
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        }
    )
    assert gc.prompt == [{"role": "user", "content": "Q"}]
    assert gc.reference == "A"
    assert gc.mode == "comparison"

    gc2 = parse_gold_case(
        {"id": "g2", "mode": "upsell", "prompt": "Hỏi gì đó", "response": "Trả lời"}
    )
    assert gc2.prompt == [{"role": "user", "content": "Hỏi gì đó"}]
    assert gc2.reference == "Trả lời"


# --- End-to-end offline mock evaluation ------------------------------------------------------


def test_run_evaluation_mock_end_to_end(tmp_path):
    gold = tmp_path / "gold.jsonl"
    cases = [
        {
            "id": "c1",
            "mode": "comparison",
            "messages": [
                {"role": "user", "content": "iPhone 15 vs 14?"},
                {"role": "assistant", "content": "15 có USB-C"},
            ],
        },
        {
            "id": "c2",
            "mode": "objection_handling",
            "messages": [
                {"role": "user", "content": "Đắt quá"},
                {"role": "assistant", "content": "Dạ mình xem trả góp ạ"},
            ],
        },
    ]
    gold.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")

    cfg = EvalFileConfig(
        model_name="test",
        gold=str(gold),
        report_dir=str(tmp_path / "out"),
        latency={"measure": False},
        pairwise=True,
    )
    report_dir = run_evaluation(cfg, "mock-model", mock=True, run_name="run1")

    assert (report_dir / "report.md").exists()
    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert set(payload["per_mode"]) == {"comparison", "objection_handling"}
    assert payload["overall"]["n"] == 2
    assert "pairwise_vs_reference" in payload["extras"]
