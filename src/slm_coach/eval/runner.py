"""Evaluation orchestration: load gold test, drive generation, score, and report.

Ties together :mod:`slm_coach.eval.inference`, :mod:`slm_coach.eval.judge`,
:mod:`slm_coach.eval.metrics`, :mod:`slm_coach.eval.latency`, and :mod:`slm_coach.eval.report`.

Two no-GPU paths keep this runnable without a GPU or API keys:

* ``dry_run`` — load the gold set, validate judges, and log the plan without generating.
* ``mock`` — generate canned answers and score them with :class:`~slm_coach.eval.judge.MockJudge`,
  producing a real report fully offline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.data.loader import load_gold_cases
from slm_coach.eval.judge import MockJudge, build_judges, judge_usage
from slm_coach.eval.metrics import (
    SampleScore,
    aggregate_by_mode,
    judge_disagreement,
    pairwise_winrate,
)
from slm_coach.eval.report import write_report
from slm_coach.eval.rubric import CRITERIA, DEFAULT_WEIGHTS, to_ten_scale, weighted_average
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import EvalFileConfig
    from slm_coach.data.schema import GoldCase

logger = get_logger(__name__)


def _mock_answer(reference: str) -> str:
    """Deterministic canned VN answer (offline mock generation)."""
    base = reference.strip() or "Dạ, em sẽ tư vấn giúp anh/chị ạ."
    return f"Dạ, em xin tư vấn ạ. {base}"


def _with_system(prompt: list[dict[str, str]], system_prompt: str | None) -> list[dict[str, str]]:
    """Prepend the production system prompt to a gold prompt (no-op if absent or already present).

    Applied only to the **generation** input so the evaluated model behaves as in production;
    the judges still see the user-facing request, not the coaching instructions.
    """
    if not system_prompt or any(m.get("role") == "system" for m in prompt):
        return prompt
    return [{"role": "system", "content": system_prompt}, *prompt]


def _score_case(judges: list[Any], prompt_text: str, answer: str) -> tuple[dict[str, float], float]:
    """Score one answer across all judges; return mean criteria and mean disagreement."""
    per_criterion: dict[str, list[float]] = {c: [] for c in CRITERIA}
    per_judge_overall: dict[str, float] = {}
    for judge in judges:
        values = judge.score(prompt=prompt_text, answer=answer, criteria=CRITERIA).as_dict()
        for criterion, value in values.items():
            per_criterion[criterion].append(value)
        per_judge_overall[judge.name] = sum(values.values()) / len(values)
    mean_criteria = {c: (sum(v) / len(v) if v else 0.0) for c, v in per_criterion.items()}
    return mean_criteria, judge_disagreement(per_judge_overall)


def _majority_winner(judges: list[Any], prompt_text: str, answer: str, reference: str) -> str:
    """Pairwise verdict (answer=A vs reference=B) by majority vote across judges."""
    votes = [
        judge.compare(prompt=prompt_text, answer_a=answer, answer_b=reference) for judge in judges
    ]
    wins = votes.count("A")
    losses = votes.count("B")
    if wins > losses:
        return "A"
    if losses > wins:
        return "B"
    return "tie"


def _find_baseline(report_root: Path) -> dict[str, Any] | None:
    """Load the most recent prior ``report.json`` under ``report_root`` (or ``None``)."""
    if not report_root.is_dir():
        return None
    candidates = sorted(report_root.glob("*/report.json"))
    if not candidates:
        return None
    import json

    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def run_evaluation(
    config: EvalFileConfig,
    model_path: str | Path,
    *,
    dry_run: bool = False,
    mock: bool = False,
    run_name: str | None = None,
) -> Path:
    """Run the full evaluation pipeline and write the report.

    Args:
        config: Validated evaluation config.
        model_path: Checkpoint to evaluate (offline).
        dry_run: If True, load gold + validate judges and log the plan without generating.
        mock: If True, use canned generation + a mock judge (no GPU/API).
        run_name: Optional report subdirectory name (defaults to a timestamp).

    Returns:
        Path to the written report directory, or the planned path under ``dry_run``.
    """
    report_root = Path(config.report_dir) / "eval"
    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    report_dir = report_root / run_name

    gold_path = Path(config.gold)
    cases: list[GoldCase] = load_gold_cases(gold_path) if gold_path.is_file() else []
    if not cases and not dry_run:
        raise FileNotFoundError(f"Gold test set not found or empty: {gold_path}")

    logger.info(
        "Evaluation plan",
        extra={
            "model": str(model_path),
            "n_cases": len(cases),
            "judges": ["mock"] if mock else config.judges,
            "mock": mock,
            "pairwise": config.pairwise,
        },
    )
    if dry_run:
        logger.info("Dry run: skipping generation + scoring", extra={"report_dir": str(report_dir)})
        return report_dir

    prompts = [c.prompt for c in cases]
    references = [c.reference for c in cases]
    modes = [c.mode for c in cases]
    gen_prompts = [_with_system(p, config.system_prompt) for p in prompts]

    if mock:
        answers = [_mock_answer(r) for r in references]
        judges: list[Any] = [MockJudge()]
    else:
        from slm_coach.eval.inference import batch_generate, load_for_inference

        model, tokenizer = load_for_inference(model_path)
        answers = batch_generate(model, tokenizer, gen_prompts, config.generation)
        judges = build_judges(config.judges, config.judge_models)

    weights = config.rubric_weights or DEFAULT_WEIGHTS
    samples: list[SampleScore] = []
    disagreements: list[float] = []
    pairwise_votes: list[str] = []
    sample_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        prompt_text = "\n".join(m["content"] for m in prompts[index])
        criteria, disagreement = _score_case(judges, prompt_text, answers[index])
        samples.append(SampleScore(sample_id=case.id, mode=modes[index], criteria=criteria))
        disagreements.append(disagreement)
        weighted5 = weighted_average(criteria, weights)
        sample_rows.append(
            {
                "id": case.id,
                "mode": modes[index],
                "score_5": round(weighted5, 4),
                "score_10": round(to_ten_scale(weighted5), 4),
                **{c: round(criteria[c], 4) for c in CRITERIA},
                "prompt": prompt_text,
                "answer": answers[index],
                "reference": references[index],
            }
        )
        if config.pairwise and references[index].strip():
            pairwise_votes.append(
                _majority_winner(judges, prompt_text, answers[index], references[index])
            )

    breakdown = aggregate_by_mode(samples, weights)

    extras: dict[str, Any] = {
        "model": str(model_path),
        "judges": ["mock"] if mock else list(config.judges),
        "judge_disagreement": (sum(disagreements) / len(disagreements) if disagreements else 0.0),
        "generation": config.generation.model_dump(),
    }
    if config.pairwise and pairwise_votes:
        extras["pairwise_vs_reference"] = pairwise_winrate(pairwise_votes)
    if config.system_prompt:
        extras["system_prompt_used"] = True
    usage = judge_usage(judges)
    if usage["calls"]:
        extras["judge_usage"] = usage
    if config.latency.measure and not mock:
        extras["latency"] = _measure_latency(model, tokenizer, gen_prompts, config)

    baseline = _find_baseline(report_root)
    write_report(report_dir, breakdown=breakdown, extras=extras, baseline=baseline)
    _export_artifacts(config, report_dir, breakdown, extras, sample_rows)
    logger.info("Evaluation complete", extra={"report_dir": str(report_dir)})
    return report_dir


def _export_artifacts(
    config: EvalFileConfig,
    report_dir: Path,
    breakdown: Any,
    extras: dict[str, Any],
    sample_rows: list[dict[str, Any]],
) -> None:
    """Write CSV tables + PNG charts beside the report (never blocks a finished eval)."""
    rep = config.reporting
    if not (rep.tables or rep.plots):
        return
    try:
        from slm_coach.reporting import export_eval_artifacts

        export_eval_artifacts(
            report_dir,
            breakdown=breakdown,
            extras=extras,
            sample_rows=sample_rows,
            make_tables=rep.tables,
            make_plots=rep.plots,
        )
    except Exception as exc:  # noqa: BLE001 - artifacts must never fail a completed eval
        logger.warning("Could not export eval artifacts", extra={"error": str(exc)})


def _measure_latency(
    model: Any, tokenizer: Any, prompts: list, config: EvalFileConfig
) -> dict[str, float]:
    """Measure p50/p95 generation latency over a prompt sample (offline timing)."""
    from slm_coach.eval.inference import batch_generate
    from slm_coach.eval.latency import measure_generation_latency

    def _one(prompt: list[dict[str, str]]) -> str:
        return batch_generate(model, tokenizer, [prompt], config.generation)[0]

    stats = measure_generation_latency(_one, prompts, config.latency.n_samples)
    return stats.as_dict()
