"""Tests for the mode-stratified holdout split."""

from __future__ import annotations

import pytest

from slm_coach.data.split import stratified_holdout

MODES = [
    "purchase_intent",
    "comparison",
    "objection_handling",
    "upsell",
    "complex_query",
    "after_sales",
    "edge_case",
]


def _make_records(per_mode: int) -> list[dict]:
    """Return ``per_mode`` records for each of the 7 modes, each with a unique id."""
    records: list[dict] = []
    for mode in MODES:
        for i in range(per_mode):
            records.append({"id": f"{mode}-{i}", "mode": mode})
    return records


def _mode(record: dict) -> str:
    return record["mode"]


def test_every_mode_present_in_val() -> None:
    records = _make_records(40)  # 280 total
    _, val = stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=200, seed=1)
    assert {r["mode"] for r in val} == set(MODES)


def test_val_meets_minimum_total() -> None:
    records = _make_records(40)  # 280 total; min_total=200 dominates
    train, val = stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=200, seed=1)
    assert len(val) >= 200
    assert len(train) + len(val) == len(records)


def test_val_even_across_modes() -> None:
    records = _make_records(40)
    _, val = stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=200, seed=1)
    counts = [sum(1 for r in val if r["mode"] == m) for m in MODES]
    assert max(counts) - min(counts) <= 1  # evenly distributed (±1 for the remainder)


def test_fraction_dominates_when_larger() -> None:
    records = _make_records(100)  # 700 total; 10% = 70 > min_total=50
    _, val = stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=50, seed=1)
    assert len(val) >= 70


def test_disjoint_and_deterministic() -> None:
    records = _make_records(40)
    train1, val1 = stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=200, seed=7)
    train2, val2 = stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=200, seed=7)
    val_ids = {r["id"] for r in val1}
    train_ids = {r["id"] for r in train1}
    assert val_ids.isdisjoint(train_ids)  # no leakage
    assert [r["id"] for r in val1] == [r["id"] for r in val2]  # reproducible
    assert [r["id"] for r in train1] == [r["id"] for r in train2]


def test_raises_when_target_exceeds_dataset() -> None:
    records = _make_records(5)  # 35 total < min_total
    with pytest.raises(ValueError):
        stratified_holdout(records, mode_of=_mode, fraction=0.10, min_total=200, seed=1)


def test_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        stratified_holdout([], mode_of=_mode, fraction=0.10, min_total=10, seed=1)
