"""Compare multiple evaluation ``report.json`` files into a single leaderboard.

Each baseline/run writes ``outputs/eval/<run>/report.json`` (see :mod:`slm_coach.eval.report`).
This module loads several of them and renders a ranked leaderboard (overall score + pairwise
win-rate) plus a per-mode score matrix, so you can see at a glance which training recipe — or how
your SLM vs the parent model — wins on the full eval suite. Pure stdlib; no GPU/API.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


def load_reports(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Load ``report.json`` files, keyed by their containing run-dir name.

    Args:
        paths: Paths to ``report.json`` files.

    Returns:
        Mapping of run name -> parsed report payload (unreadable files are skipped with a warning).
    """
    reports: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw)
        try:
            reports[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping unreadable report", extra={"path": str(path), "error": str(exc)}
            )
    return reports


def find_reports(eval_root: str | Path) -> list[Path]:
    """Return all ``*/report.json`` files under an eval root (e.g. ``outputs/eval``)."""
    root = Path(eval_root)
    return sorted(root.glob("*/report.json")) if root.is_dir() else []


def _overall(payload: Mapping[str, Any]) -> float:
    """Overall weighted score on the 0-10 scale (0 if absent)."""
    return float(payload.get("overall", {}).get("weighted_avg_10", 0.0))


def _model(payload: Mapping[str, Any]) -> str:
    """The evaluated model path/id recorded in the report."""
    return str(payload.get("extras", {}).get("model", "?"))


def _pairwise_win(payload: Mapping[str, Any]) -> str:
    """Pairwise win-rate vs the gold reference, formatted as a percent (or ``—``)."""
    pairwise = payload.get("extras", {}).get("pairwise_vs_reference")
    if not pairwise:
        return "—"
    return f"{float(pairwise.get('win', 0.0)) * 100:.0f}%"


def build_leaderboard(reports: Mapping[str, dict[str, Any]]) -> str:
    """Render a markdown leaderboard + per-mode matrix from loaded reports.

    Args:
        reports: Mapping of run name -> report payload (from :func:`load_reports`).

    Returns:
        A markdown document ranking the runs by overall score, with a per-mode breakdown.
    """
    if not reports:
        return "# Baseline comparison\n\n_No reports found._\n"

    ranked = sorted(reports.items(), key=lambda kv: _overall(kv[1]), reverse=True)

    lines: list[str] = [
        "# Baseline comparison",
        "",
        "## Leaderboard (sorted by overall /10)",
        "",
        "| Rank | Run | Model | Overall /10 | Pairwise win vs gold | n |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for rank, (name, payload) in enumerate(ranked, start=1):
        n = int(payload.get("overall", {}).get("n", 0))
        lines.append(
            f"| {rank} | {name} | {_model(payload)} | {_overall(payload):.2f} "
            f"| {_pairwise_win(payload)} | {n} |"
        )

    modes = sorted({m for payload in reports.values() for m in payload.get("per_mode", {})})
    if modes:
        lines += [
            "",
            "## Per-mode score /10 (rows = run, columns = mode)",
            "",
            "| Run | " + " | ".join(modes) + " |",
            "| --- |" + " ---: |" * len(modes),
        ]
        for name, payload in ranked:
            per_mode = payload.get("per_mode", {})
            cells = [
                (f"{per_mode[m]['weighted_avg_10']:.2f}" if m in per_mode else "—") for m in modes
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def write_comparison(reports: Mapping[str, dict[str, Any]], out_path: str | Path) -> Path:
    """Write the markdown leaderboard to ``out_path`` and return it."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_leaderboard(reports), encoding="utf-8")
    logger.info("Wrote baseline comparison", extra={"path": str(out), "runs": len(reports)})
    return out


def leaderboard_rows(reports: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten loaded reports into one ranked row per run (overall + per-mode columns).

    Args:
        reports: Mapping of run name -> report payload (from :func:`load_reports`).

    Returns:
        Rows sorted by overall /10 (descending), each with rank, run, model, scores, and one
        ``mode_<name>`` column per conversation mode present in any report.
    """
    ranked = sorted(reports.items(), key=lambda kv: _overall(kv[1]), reverse=True)
    modes = sorted({m for payload in reports.values() for m in payload.get("per_mode", {})})
    rows: list[dict[str, Any]] = []
    for rank, (name, payload) in enumerate(ranked, start=1):
        per_mode = payload.get("per_mode", {})
        row: dict[str, Any] = {
            "rank": rank,
            "run": name,
            "model": _model(payload),
            "overall_10": round(_overall(payload), 4),
            "pairwise_win_vs_gold": _pairwise_win(payload),
            "n": int(payload.get("overall", {}).get("n", 0)),
        }
        for mode in modes:
            row[f"mode_{mode}"] = (
                round(float(per_mode[mode]["weighted_avg_10"]), 4) if mode in per_mode else ""
            )
        rows.append(row)
    return rows


def write_comparison_csv(reports: Mapping[str, dict[str, Any]], out_path: str | Path) -> Path:
    """Write the leaderboard as a flat CSV (one row per run) for spreadsheet use."""
    rows = leaderboard_rows(reports)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["rank", "run", "model", "overall_10", "n"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote baseline comparison CSV", extra={"path": str(out), "runs": len(rows)})
    return out
