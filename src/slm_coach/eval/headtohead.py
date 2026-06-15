"""Direct model-vs-model win-rate (head-to-head) from saved evaluation answers.

Reuses the per-sample answers each evaluation already writes (``per_sample.csv``), so a head-to-
head (e.g. our SLM vs the parent Qwen) needs **no re-generation** — only judge ``compare`` calls.
Convention: model **A** is the model under test, model **B** the baseline/parent; a "win" means A
was judged better than B. Cases are aligned by ``id`` present in both runs.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slm_coach.eval.metrics import pairwise_winrate
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


def load_per_sample(path: str | Path) -> dict[str, dict[str, str]]:
    """Load a run's ``per_sample.csv`` into ``id -> {mode, prompt, answer, reference}``."""
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = row.get("id")
            if cid:
                rows[cid] = {
                    "mode": row.get("mode", ""),
                    "prompt": row.get("prompt", ""),
                    "answer": row.get("answer", ""),
                    "reference": row.get("reference", ""),
                }
    return rows


def head_to_head(
    judges: Sequence[Any],
    rows_a: dict[str, dict[str, str]],
    rows_b: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Judge model A vs model B per shared case; aggregate win/tie/loss (overall + per mode).

    Args:
        judges: Judge backends (``compare`` is called; majority vote across judges).
        rows_a: Model-under-test per-sample rows (the "A" side; a win means A is better).
        rows_b: Baseline/parent per-sample rows (the "B" side).

    Returns:
        ``{"overall": winrate, "per_mode": {mode: winrate}, "n": int}`` where ``winrate`` is from
        :func:`slm_coach.eval.metrics.pairwise_winrate` (``win`` = A beats B).
    """
    shared = [cid for cid in rows_a if cid in rows_b]
    overall_votes: list[str] = []
    per_mode_votes: dict[str, list[str]] = {}
    for cid in shared:
        a, b = rows_a[cid], rows_b[cid]
        votes = [
            judge.compare(prompt=a["prompt"], answer_a=a["answer"], answer_b=b["answer"])
            for judge in judges
        ]
        wins, losses = votes.count("A"), votes.count("B")
        verdict = "A" if wins > losses else ("B" if losses > wins else "tie")
        overall_votes.append(verdict)
        per_mode_votes.setdefault(a["mode"], []).append(verdict)
    logger.info("Head-to-head complete", extra={"n_pairs": len(shared)})
    return {
        "overall": pairwise_winrate(overall_votes),
        "per_mode": {m: pairwise_winrate(v) for m, v in sorted(per_mode_votes.items())},
        "n": len(shared),
    }


def build_headtohead_markdown(
    result: dict[str, Any], *, label_a: str, label_b: str, usage: dict[str, Any] | None = None
) -> str:
    """Render the head-to-head win-rate (overall + per-mode) as markdown (``win`` = A beats B)."""
    overall = result.get("overall", {})

    def _row(name: str, winrate: dict[str, Any]) -> str:
        return (
            f"| {name} | {winrate.get('win', 0) * 100:.1f}% | {winrate.get('tie', 0) * 100:.1f}% "
            f"| {winrate.get('loss', 0) * 100:.1f}% | {int(winrate.get('n', 0))} |"
        )

    lines = [
        f"# Head-to-head: {label_a} vs {label_b}",
        "",
        f"**A = {label_a} (model under test) · B = {label_b} (baseline). "
        "Win = A judged better than B.**",
        "",
        "| Slice | A win | Tie | B win | n |",
        "| --- | ---: | ---: | ---: | ---: |",
        _row("overall", overall),
    ]
    for mode, winrate in result.get("per_mode", {}).items():
        lines.append(_row(mode, winrate))
    a_win = overall.get("win", 0) * 100
    lines += [
        "",
        f"> **Headline:** {label_a} thắng {label_b} **{a_win:.0f}%** số ca "
        f"(n={int(overall.get('n', 0))}).",
        "",
    ]
    if usage:
        lines += [
            "## Judge API usage & cost (estimate)",
            "",
            f"- Calls: {usage.get('calls', 0)} | tokens: {usage.get('total_tokens', 0):,} "
            f"| est. cost: ${usage.get('est_usd', 0):.4f}",
            "",
        ]
    return "\n".join(lines) + "\n"


def write_headtohead_csv(result: dict[str, Any], out_path: str | Path) -> Path:
    """Write the head-to-head win/tie/loss table as a flat CSV (overall + one row per mode).

    Columns: ``slice, a_win_pct, tie_pct, b_win_pct, n`` (``a_win`` = A judged better than B).
    """

    def _row(slice_name: str, winrate: dict[str, Any]) -> dict[str, Any]:
        return {
            "slice": slice_name,
            "a_win_pct": round(float(winrate.get("win", 0.0)) * 100, 1),
            "tie_pct": round(float(winrate.get("tie", 0.0)) * 100, 1),
            "b_win_pct": round(float(winrate.get("loss", 0.0)) * 100, 1),
            "n": int(winrate.get("n", 0)),
        }

    rows = [_row("overall", result.get("overall", {}))]
    rows += [_row(mode, winrate) for mode, winrate in result.get("per_mode", {}).items()]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["slice", "a_win_pct", "tie_pct", "b_win_pct", "n"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote head-to-head CSV", extra={"path": str(out), "rows": len(rows)})
    return out
