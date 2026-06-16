"""Multi-judge LLM-as-judge: GPT + Gemini ONLY.

Claude and DeepSeek are the teacher models that generated the data, so judging with them risks
circular / self-preference bias — they are never used as judges (enforced both here and in
:class:`slm_coach.config.EvalFileConfig`). Each judge supports two modes:

* per-criterion **rubric scoring** (1-5 each, guided by ``CRITERIA_DESCRIPTIONS``), and
* **pairwise (A/B) comparison** of two answers to the same prompt.

API keys come from ``.env``; calls retry on transient errors via tenacity. A :class:`MockJudge`
provides deterministic offline scoring/comparison for no-API runs and tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential

from slm_coach.eval.rubric import SCALE_MAX, SCALE_MIN, RubricScore, render_rubric_block
from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)

#: Judges that must never be used (teacher models -> circular bias).
BANNED_JUDGES = frozenset({"claude", "anthropic", "deepseek"})

#: Pairwise verdict: which answer wins, or a tie.
Winner = Literal["A", "B", "tie"]

_SCORE_TEMPLATE = (
    "Bạn là giám khảo đánh giá một trợ lý bán iPhone bằng tiếng Việt.\n"
    "Chấm điểm CÂU TRẢ LỜI cho YÊU CẦU theo từng tiêu chí, thang điểm {lo}-{hi} (số nguyên).\n"
    "Tiêu chí và cách chấm:\n{rubric}\n\n"
    "Chỉ trả về một đối tượng JSON với đúng các khóa: {criteria}.\n\n"
    "YÊU CẦU:\n{prompt}\n\nCÂU TRẢ LỜI:\n{answer}\n\nJSON:"
)

_PAIRWISE_TEMPLATE = (
    "Bạn là giám khảo đánh giá một trợ lý bán iPhone bằng tiếng Việt.\n"
    "Cùng một YÊU CẦU, hãy chọn câu trả lời nào TỐT HƠN xét theo các tiêu chí:\n{rubric}\n\n"
    "YÊU CẦU:\n{prompt}\n\n"
    "CÂU TRẢ LỜI A:\n{answer_a}\n\nCÂU TRẢ LỜI B:\n{answer_b}\n\n"
    'Chỉ trả về JSON: {{"winner": "A" | "B" | "tie", "reason": "..."}}\nJSON:'
)


@runtime_checkable
class Judge(Protocol):
    """Protocol implemented by each judge backend (GPT, Gemini, Mock)."""

    name: str

    def score(self, *, prompt: str, answer: str, criteria: Sequence[str]) -> RubricScore:
        """Score a single answer against the rubric criteria."""
        ...

    def compare(self, *, prompt: str, answer_a: str, answer_b: str) -> Winner:
        """Decide which of two answers to the same prompt is better."""
        ...


def _render_score_prompt(prompt: str, answer: str, criteria: Sequence[str]) -> str:
    """Render the per-criterion scoring instruction for a (prompt, answer) pair."""
    return _SCORE_TEMPLATE.format(
        lo=SCALE_MIN,
        hi=SCALE_MAX,
        rubric=render_rubric_block(tuple(criteria)),
        criteria=", ".join(criteria),
        prompt=prompt,
        answer=answer,
    )


def _render_pairwise_prompt(prompt: str, answer_a: str, answer_b: str) -> str:
    """Render the pairwise A/B comparison instruction."""
    return _PAIRWISE_TEMPLATE.format(
        rubric=render_rubric_block(), prompt=prompt, answer_a=answer_a, answer_b=answer_b
    )


def _clamp(value: Any) -> float:
    """Coerce a judge-returned value into the valid 1-5 range (default 3 on failure)."""
    try:
        return float(min(SCALE_MAX, max(SCALE_MIN, float(value))))
    except (TypeError, ValueError):
        return 3.0


def parse_scores(text: str, criteria: Sequence[str]) -> RubricScore:
    """Parse a judge's JSON response into a :class:`RubricScore` (robust to noise).

    Args:
        text: Raw judge response (expected to contain a JSON object).
        criteria: Criteria to extract.

    Returns:
        A validated :class:`RubricScore`; missing/invalid criteria default to 3.
    """
    payload = _extract_json(text)
    return RubricScore(**{c: _clamp(payload.get(c, 3)) for c in criteria})


def parse_winner(text: str) -> Winner:
    """Parse a pairwise verdict ('A' / 'B' / 'tie') from a judge response."""
    winner = str(_extract_json(text).get("winner", "")).strip().upper()
    if winner == "A":
        return "A"
    if winner == "B":
        return "B"
    return "tie"


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from noisy judge text (empty dict on failure)."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Judge returned unparseable JSON; using defaults")
    return {}


class OpenAIJudge:
    """GPT judge backed by the OpenAI API."""

    name = "gpt"

    def __init__(self, model: str = "gpt-4o") -> None:
        """Store the OpenAI model id and zero the usage counters."""
        self.model = model
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _client(self) -> Any:
        """Build an OpenAI client (key from ``OPENAI_API_KEY``)."""
        return require("openai", "eval").OpenAI()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
    def _chat(self, content: str) -> str:
        """Send a single JSON-mode chat request and return the raw text (records token usage)."""
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        self.n_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return response.choices[0].message.content or ""

    def score(self, *, prompt: str, answer: str, criteria: Sequence[str]) -> RubricScore:
        """Score one answer via the OpenAI chat completions API."""
        return parse_scores(self._chat(_render_score_prompt(prompt, answer, criteria)), criteria)

    def compare(self, *, prompt: str, answer_a: str, answer_b: str) -> Winner:
        """Pick the better of two answers via the OpenAI API."""
        return parse_winner(self._chat(_render_pairwise_prompt(prompt, answer_a, answer_b)))


class GeminiJudge:
    """Gemini judge backed by the Google GenAI API."""

    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro") -> None:
        """Store the Gemini model id and zero the usage counters."""
        self.model = model
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
    def _generate(self, content: str) -> str:
        """Send a single request to Google GenAI and return the raw text (records token usage)."""
        genai = require("google.genai", "eval")
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        response = client.models.generate_content(model=self.model, contents=content)
        self.n_calls += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
            self.completion_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        return getattr(response, "text", "") or ""

    def score(self, *, prompt: str, answer: str, criteria: Sequence[str]) -> RubricScore:
        """Score one answer via the Google GenAI API."""
        return parse_scores(
            self._generate(_render_score_prompt(prompt, answer, criteria)), criteria
        )

    def compare(self, *, prompt: str, answer_a: str, answer_b: str) -> Winner:
        """Pick the better of two answers via the Google GenAI API."""
        return parse_winner(self._generate(_render_pairwise_prompt(prompt, answer_a, answer_b)))


class MockJudge:
    """Deterministic offline judge (no API) for ``--mock`` runs and tests.

    Scores scale gently with answer length and the presence of polite Vietnamese markers, so
    the full scoring/report pipeline can be exercised without API keys.
    """

    name = "mock"

    _POLITE_MARKERS = ("dạ", "ạ", "anh", "chị", "cảm ơn")

    def __init__(self) -> None:
        """Initialize zero usage counters (the mock makes no API calls)."""
        self.model = "mock"
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _heuristic(self, answer: str) -> float:
        """A length + politeness heuristic mapped onto the 1-5 scale."""
        words = len(answer.split())
        polite = sum(1 for m in self._POLITE_MARKERS if m in answer.lower())
        return _clamp(SCALE_MIN + words / 20.0 + polite * 0.5)

    def score(self, *, prompt: str, answer: str, criteria: Sequence[str]) -> RubricScore:
        """Return deterministic 1-5 scores derived from the answer text."""
        base = self._heuristic(answer)
        polite = sum(1 for m in self._POLITE_MARKERS if m in answer.lower())
        scores = {c: base for c in criteria}
        if "tone" in scores:
            scores["tone"] = _clamp(SCALE_MIN + polite)
        return RubricScore(**{c: _clamp(scores.get(c, 3)) for c in criteria})

    def compare(self, *, prompt: str, answer_a: str, answer_b: str) -> Winner:
        """Pick the higher-heuristic answer deterministically."""
        score_a, score_b = self._heuristic(answer_a), self._heuristic(answer_b)
        if abs(score_a - score_b) < 1e-9:
            return "tie"
        return "A" if score_a > score_b else "B"


def build_judges(judge_names: Sequence[str], judge_models: dict[str, str]) -> list[Judge]:
    """Instantiate the configured judges (GPT/Gemini/Mock), rejecting banned teacher models.

    Args:
        judge_names: Judge identifiers from config (e.g. ``["gpt", "gemini"]``).
        judge_models: Mapping of judge name to concrete model id.

    Returns:
        A list of ready judge backends.

    Raises:
        ValueError: If any banned judge is requested or a name is unknown.
    """
    offending = [j for j in judge_names if j.lower() in BANNED_JUDGES]
    if offending:
        raise ValueError(f"banned judges requested (teacher models): {offending}")

    judges: list[Judge] = []
    for name in judge_names:
        key = name.lower()
        if key == "gpt":
            judges.append(OpenAIJudge(judge_models.get("gpt", "gpt-4o")))
        elif key == "gemini":
            judges.append(GeminiJudge(judge_models.get("gemini", "gemini-1.5-pro")))
        elif key == "mock":
            judges.append(MockJudge())
        else:
            raise ValueError(f"unknown judge '{name}'; supported: gpt, gemini, mock")
    return judges


#: Approximate provider prices in USD per 1M tokens, ``model -> (input, output)``. These move
#: over time — treat the resulting cost as an estimate and update as your contract changes.
JUDGE_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}


def _estimate_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for one judge's token usage (0 if the model isn't priced)."""
    for key, (price_in, price_out) in JUDGE_PRICE_PER_1M.items():
        if key in model:
            return prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out
    return 0.0


def judge_usage(judges: Sequence[Judge]) -> dict[str, Any]:
    """Aggregate API calls + token usage (and an approximate USD cost) across judges.

    Args:
        judges: The judge backends used in a run (after they have been called).

    Returns:
        A dict with totals (``calls``, ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
        ``est_usd``) plus a ``by_judge`` per-judge breakdown.
    """
    by_judge: dict[str, Any] = {}
    total_calls = total_in = total_out = 0
    total_usd = 0.0
    for judge in judges:
        calls = int(getattr(judge, "n_calls", 0))
        in_tok = int(getattr(judge, "prompt_tokens", 0))
        out_tok = int(getattr(judge, "completion_tokens", 0))
        model = str(getattr(judge, "model", judge.name))
        usd = _estimate_usd(model, in_tok, out_tok)
        by_judge[judge.name] = {
            "model": model,
            "calls": calls,
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "est_usd": round(usd, 4),
        }
        total_calls += calls
        total_in += in_tok
        total_out += out_tok
        total_usd += usd
    return {
        "calls": total_calls,
        "prompt_tokens": total_in,
        "completion_tokens": total_out,
        "total_tokens": total_in + total_out,
        "est_usd": round(total_usd, 4),
        "by_judge": by_judge,
    }
