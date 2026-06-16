"""Offline tests for the reporting package (CSV tables + chart degradation)."""

from __future__ import annotations

import csv
import json

from slm_coach.eval.metrics import ModeBreakdown, SampleScore, aggregate_by_mode
from slm_coach.eval.rubric import CRITERIA, DEFAULT_WEIGHTS
from slm_coach.reporting import (
    export_eval_artifacts,
    export_training_artifacts,
    plot_eval,
    plot_training_curves,
)
from slm_coach.reporting.tables import (
    merge_log_history,
    read_trainer_log,
    write_per_mode_table,
    write_per_sample_table,
    write_training_table,
)

# --- trainer_state log parsing ----------------------------------------------------------------

_LOG_HISTORY = [
    {"loss": 2.0, "learning_rate": 1e-4, "grad_norm": 1.5, "epoch": 0.5, "step": 10},
    {"eval_loss": 1.8, "eval_rubric_avg": 3.2, "eval_runtime": 9.9, "epoch": 0.5, "step": 10},
    {"loss": 1.1, "learning_rate": 5e-5, "grad_norm": 1.1, "epoch": 1.0, "step": 20},
    {"eval_loss": 1.4, "eval_rubric_avg": 3.9, "epoch": 1.0, "step": 20},
    {"train_runtime": 75.0, "train_samples_per_second": 4.0, "total_flos": 1.0, "step": 20},
]


def test_merge_log_history_combines_train_and_eval_rows():
    rows = merge_log_history(_LOG_HISTORY)
    assert [r["step"] for r in rows] == [10, 20]
    first = rows[0]
    assert first["loss"] == 2.0
    assert first["eval_loss"] == 1.8
    assert first["eval_rubric_avg"] == 3.2
    # Summary/runtime keys are dropped from the per-step table.
    assert "train_runtime" not in rows[1]
    assert "eval_runtime" not in first
    assert "total_flos" not in rows[1]


def test_read_trainer_log_from_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"log_history": _LOG_HISTORY}), encoding="utf-8"
    )
    rows = read_trainer_log(tmp_path)
    assert len(rows) == 2
    assert rows[1]["eval_rubric_avg"] == 3.9


def test_read_trainer_log_missing_returns_empty(tmp_path):
    assert read_trainer_log(tmp_path) == []


# --- CSV tables -------------------------------------------------------------------------------


def test_write_training_table_columns_and_content(tmp_path):
    rows = merge_log_history(_LOG_HISTORY)
    path = write_training_table(rows, tmp_path)
    parsed = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert parsed[0]["step"] == "10"
    assert parsed[0]["loss"] == "2.0"
    assert parsed[1]["eval_rubric_avg"] == "3.9"
    # Preferred columns lead the header.
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:4] == ["step", "epoch", "loss", "eval_loss"]


def _breakdown() -> ModeBreakdown:
    samples = [
        SampleScore("a", "comparison", dict.fromkeys(CRITERIA, 4.0)),
        SampleScore("b", "objection_handling", dict.fromkeys(CRITERIA, 2.0)),
    ]
    return aggregate_by_mode(samples, DEFAULT_WEIGHTS)


def test_write_per_mode_table_has_overall_and_modes(tmp_path):
    path = write_per_mode_table(_breakdown(), tmp_path)
    parsed = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    modes = {row["mode"] for row in parsed}
    assert modes == {"overall", "comparison", "objection_handling"}
    assert all(c in parsed[0] for c in CRITERIA)


def test_write_per_sample_table_orders_columns(tmp_path):
    rows = [
        {"id": "c1", "mode": "comparison", "score_10": 8.0, "answer": "x", "reference": "y"},
    ]
    path = write_per_sample_table(rows, tmp_path)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:3] == ["id", "mode", "score_10"]
    assert header[-2:] == ["answer", "reference"]


def test_mode_breakdown_round_trip_from_dict():
    original = _breakdown()
    restored = ModeBreakdown.from_dict(original.as_dict())
    assert restored.overall.n == original.overall.n
    assert set(restored.per_mode) == set(original.per_mode)
    assert restored.per_mode["comparison"].weighted_avg_10 == (
        original.per_mode["comparison"].weighted_avg_10
    )


# --- High-level exports + chart degradation ---------------------------------------------------


def test_export_training_artifacts_writes_table(tmp_path):
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"log_history": _LOG_HISTORY}), encoding="utf-8"
    )
    result = export_training_artifacts(tmp_path)
    assert (tmp_path / "metrics" / "training_log.csv").exists()
    assert len(result["tables"]) == 1
    # plots: either rendered (if matplotlib present) or gracefully skipped — never an error.
    assert isinstance(result["plots"], list)
    assert all(p.exists() for p in result["plots"])


def test_export_eval_artifacts_writes_tables(tmp_path):
    result = export_eval_artifacts(
        tmp_path,
        breakdown=_breakdown(),
        extras={"pairwise_vs_reference": {"win": 1.0, "tie": 0.0, "loss": 0.0, "n": 2}},
        sample_rows=[{"id": "c1", "mode": "comparison", "score_10": 8.0}],
    )
    assert (tmp_path / "per_mode.csv").exists()
    assert (tmp_path / "criteria.csv").exists()
    assert (tmp_path / "per_sample.csv").exists()
    assert len(result["tables"]) == 3
    assert isinstance(result["plots"], list)
    assert all(p.exists() for p in result["plots"])


def test_plot_functions_degrade_without_raising(tmp_path):
    # Whether or not matplotlib is installed, these return a list and never raise.
    rows = merge_log_history(_LOG_HISTORY)
    assert isinstance(plot_training_curves(rows, tmp_path), list)
    assert isinstance(plot_eval(_breakdown(), tmp_path), list)


# --- Baseline comparison leaderboard --------------------------------------------------------


def _write_report_json(tmp_path, run, *, model, overall_10, modes, win=None):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    extras = {"model": model}
    if win is not None:
        extras["pairwise_vs_reference"] = {"win": win, "tie": 0.0, "loss": 1 - win, "n": 2}
    payload = {
        "overall": {"n": 2, "weighted_avg_10": overall_10},
        "per_mode": {m: {"weighted_avg_10": s} for m, s in modes.items()},
        "extras": extras,
    }
    (d / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    return d / "report.json"


def test_compare_leaderboard_ranks_and_tabulates(tmp_path):
    from slm_coach.reporting.compare import build_leaderboard, find_reports, load_reports

    _write_report_json(
        tmp_path,
        "bl_qlora_multi",
        model="ckpt/best",
        overall_10=8.1,
        modes={"comparison": 8.0, "objection_handling": 8.2},
        win=0.75,
    )
    _write_report_json(
        tmp_path,
        "base_qwen",
        model="Qwen/Qwen3-8B",
        overall_10=5.4,
        modes={"comparison": 6.0, "objection_handling": 4.8},
    )
    reports = load_reports(find_reports(tmp_path))
    assert set(reports) == {"bl_qlora_multi", "base_qwen"}

    md = build_leaderboard(reports)
    # Higher overall ranks first.
    assert md.index("bl_qlora_multi") < md.index("base_qwen")
    assert "Qwen/Qwen3-8B" in md
    assert "75%" in md  # pairwise win-rate
    assert "objection_handling" in md  # per-mode matrix column


def test_compare_leaderboard_empty():
    from slm_coach.reporting.compare import build_leaderboard

    assert "No reports found" in build_leaderboard({})
