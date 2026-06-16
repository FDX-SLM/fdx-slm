"""Evaluation report writer (markdown + json).

Writes ``outputs/eval/<run>/report.md`` and ``report.json`` with the per-mode score table
(the primary output), mean rubric scores, judge agreement, optional pairwise win-rate vs the
reference, latency, and a comparison against a previous baseline when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.eval.metrics import ModeBreakdown

logger = get_logger(__name__)


def _fmt(value: float | None) -> str:
    """Format an optional float to two decimals (``"n/a"`` for ``None``)."""
    return "n/a" if value is None else f"{value:.2f}"


def _per_mode_table(breakdown: ModeBreakdown, baseline: dict[str, Any] | None) -> list[str]:
    """Render the per-mode score table (with baseline deltas when available)."""
    baseline_modes = (baseline or {}).get("per_mode", {})
    lines = ["| Mode | n | Score /10 | Δ vs baseline |", "| --- | ---: | ---: | ---: |"]
    for mode, summary in sorted(breakdown.per_mode.items(), key=lambda kv: kv[1].weighted_avg_10):
        prev = baseline_modes.get(mode, {}).get("weighted_avg_10")
        delta = "—" if prev is None else f"{summary.weighted_avg_10 - prev:+.2f}"
        lines.append(f"| {mode} | {summary.n} | {summary.weighted_avg_10:.2f} | {delta} |")
    return lines


def _criteria_table(breakdown: ModeBreakdown) -> list[str]:
    """Render the overall per-criterion mean table."""
    lines = ["| Criterion | Mean /5 |", "| --- | ---: |"]
    for criterion, mean in breakdown.overall.per_criterion_mean.items():
        lines.append(f"| {criterion} | {mean:.2f} |")
    return lines


def _build_markdown(
    breakdown: ModeBreakdown,
    extras: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> str:
    """Assemble the full markdown report."""
    weakest = breakdown.weakest_modes(3)
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"- **Model:** {extras.get('model', 'n/a')}",
        f"- **Judges:** {', '.join(extras.get('judges', []))}",
        f"- **Cases:** {breakdown.overall.n}",
        f"- **Overall score:** {breakdown.overall.weighted_avg_10:.2f}/10",
        "",
        "## Per-mode breakdown (primary output)",
        "",
        *_per_mode_table(breakdown, baseline),
        "",
        f"**Weakest modes (reinforce these):** "
        f"{', '.join(f'{m} ({s:.2f})' for m, s in weakest) or 'n/a'}",
        "",
        "## Mean rubric criteria",
        "",
        *_criteria_table(breakdown),
        "",
    ]

    pairwise = extras.get("pairwise_vs_reference")
    if pairwise:
        lines += [
            "## Pairwise vs reference (A=model, B=gold)",
            "",
            f"- win: {pairwise.get('win', 0) * 100:.1f}% | tie: {pairwise.get('tie', 0) * 100:.1f}%"
            f" | loss: {pairwise.get('loss', 0) * 100:.1f}% (n={int(pairwise.get('n', 0))})",
            "",
        ]

    latency = extras.get("latency")
    if latency:
        lines += [
            "## Generation latency (offline)",
            "",
            f"- p50: {_fmt(latency.get('p50'))}s | p95: {_fmt(latency.get('p95'))}s "
            f"| mean: {_fmt(latency.get('mean'))}s (n={latency.get('n', 0)})",
            "",
        ]

    agreement = extras.get("judge_disagreement")
    if agreement is not None:
        lines += ["## Judge agreement", "", f"- Mean disagreement (max-min): {agreement:.3f}", ""]

    usage = extras.get("judge_usage")
    if usage:
        lines += [
            "## Judge API usage & cost (estimate)",
            "",
            f"- Calls: {usage.get('calls', 0)} | tokens: {usage.get('total_tokens', 0):,} "
            f"(in {usage.get('prompt_tokens', 0):,} / out {usage.get('completion_tokens', 0):,}) "
            f"| **est. cost: ${usage.get('est_usd', 0):.4f}**",
            "",
            "| Judge | Model | Calls | In tok | Out tok | Est. $ |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for name, info in usage.get("by_judge", {}).items():
            lines.append(
                f"| {name} | {info.get('model', '?')} | {info.get('calls', 0)} "
                f"| {info.get('prompt_tokens', 0):,} | {info.get('completion_tokens', 0):,} "
                f"| {info.get('est_usd', 0):.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report(
    output_dir: str | Path,
    *,
    breakdown: ModeBreakdown,
    extras: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the markdown + json evaluation report.

    Args:
        output_dir: Destination directory (``outputs/eval/<run>/``).
        breakdown: The per-mode score breakdown (primary output).
        extras: Optional extra sections (model, judges, latency, judge agreement, pairwise).
        baseline: Optional previous-run summary for comparison.

    Returns:
        Paths to the written ``report.md`` and ``report.json``.
    """
    extras = extras or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "report.md"
    json_path = out / "report.json"

    md_path.write_text(_build_markdown(breakdown, extras, baseline), encoding="utf-8")
    payload = {
        "overall": breakdown.overall.as_dict(),
        "per_mode": {mode: s.as_dict() for mode, s in breakdown.per_mode.items()},
        "weakest_modes": breakdown.weakest_modes(3),
        "extras": extras,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote eval report", extra={"md": str(md_path), "json": str(json_path)})
    return md_path, json_path
