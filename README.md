# SLM Sales Coach — fine-tuning & evaluation pipeline

A complete, runnable pipeline that **fine-tunes and evaluates** a Small Language Model acting as
an **iPhone sales coach in Vietnamese** (base model: `Qwen/Qwen3.5-9B`). The pipeline goes
**data → train → evaluate → export** and **stops at producing the model**.

- **In scope:** data loading/validation, training (T1 LoRA SFT, T2 multi-stage QLoRA SFT, T3
  DPO/ORPO), evaluation (rubric + multi-judge + per-mode + pairwise), and export/quantization
  (merge → FP16 → AWQ INT4 + GGUF Q4_K_M).
- **Out of scope:** data *generation* (another team owns it), any serving / inference-runtime /
  API layer, and there is **no Makefile** — the project is managed entirely with **`uv`**.

> The repo *consumes* data; it never creates it. See [data/README.md](data/README.md) for the
> data contract.

---

## Install

Everything runs through [`uv`](https://docs.astral.sh/uv/). The project targets **Python 3.12**
(pinned in `.python-version`).

```bash
# 1. Install uv (one time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the venv + install the CPU core (no GPU needed)
uv sync

# 3. Run the tests (no GPU, no API keys)
uv run pytest
```

### Dependency extras (GPU / eval / export)

To keep `uv sync` working on any machine, the heavy and platform-specific dependencies live in
**optional extras** (a plain `uv sync` installs only the cross-platform core + dev tools, which
is all the tests and dry-runs need). Install extras on the box that needs them:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `train` | torch, trl, peft, accelerate, bitsandbytes | actual training / merging (GPU) |
| `gpu` | unsloth, flash-attn | Unsloth + FA2 speedups (**Linux + CUDA only**) |
| `eval` | openai, google-genai, lm-eval | LLM judges, harness |
| `export` | autoawq | AWQ INT4 quantization |
| `tracking` | langfuse | qualitative sample-generation logging |
| `viz` | matplotlib | PNG charts for loss/eval curves + per-mode bars (CSV tables need no extra) |

```bash
# On the GPU training/eval box (Linux + CUDA):
uv sync --extra train --extra gpu --extra eval --extra export --extra tracking --extra viz
```

GPU/optional modules import lazily and degrade safely: every CLI runs under `--dry-run` (and the
evaluator under `--mock`) with **no GPU and no API keys**.

---

## Run without a GPU

```bash
# Unit tests (schema, formatting+masking, loader, mixture, per-mode metrics, offline phase logic)
uv run pytest

# Validate a data delivery against the contract
uv run python scripts/validate_data.py --data-dir data/

# Dry-run any training CLI: resolves config + builds the data/plan, no model load
uv run python scripts/train_sft.py --config configs/sft_coach_9b.yaml --dry-run

# Mock evaluation: canned generation + mock judge -> a real report, fully offline
uv run python scripts/evaluate.py --config configs/eval.yaml --model any --mock

# Local 8GB smoke test of the full T1 loop (small base model, ~200 steps) — needs the train extra
uv run python scripts/train_sft.py --config configs/sft_lora_smoke.yaml
```

---

## Project layout

```
configs/        base.yaml + sft_coach_9b / sft_lora_smoke / align_coach_dpo / eval
src/slm_coach/
  config.py     pydantic models + base-merge loader (${ENV} expansion)
  tracking.py   Langfuse facade for sample generations (no-op without the tracking extra)
  reporting/    metric CSV tables + PNG charts (tables.py · plots.py); degrades without matplotlib
  data/         schema · loader · formatting (ChatML, <think>, multi-turn masking) · mixture
  training/     model (Unsloth/FA2, LoRA/QLoRA) · sft · align (DPO) · callbacks
  eval/         inference · runner · rubric · judge (GPT+Gemini, pairwise) · latency · metrics · report · harness_task
  export/       merge (LoRA→FP16) · quantize (AWQ INT4 + GGUF Q4_K_M)
  utils/        logging · seed · deps
scripts/        thin CLIs: validate_data · train_sft · train_align · evaluate · export_model · plot_metrics
tests/          unit tests (no GPU / no API keys)
```

## Conventions

- **Config-driven:** every hyperparameter lives in `configs/*.yaml` (never hardcoded). A config
  declares `defaults: base.yaml` and overrides sections; `${ENV}` values resolve from the
  environment at load time.
- **Secrets in `.env` only** (see `.env.example`); `data/`, `checkpoints/`, `outputs/` are
  gitignored.
- **The 7 conversation modes** (in `src/slm_coach/data/schema.py`): `purchase_intent`,
  `comparison`, `objection_handling`, `upsell`, `after_sales`, `complex_query`, `edge_case`.
  `mode` is metadata — it never enters the training sequence.
- **Judges are GPT + Gemini only** — never Claude/DeepSeek (the teacher models that produced the
  data), to avoid circular / self-preference bias. Enforced in config validation.

---

## Documentation

The README stays high-level on purpose. The detailed guides live in [`docs/`](docs/):

| Doc | What it covers |
| --- | --- |
| **[docs/RUNBOOK.md](docs/RUNBOOK.md)** | The whole pipeline end to end — every command, what it consumes/produces, and *why* the steps run in this order. **Start here to run anything.** |
| **[docs/EVAL_COMPARISON.md](docs/EVAL_COMPARISON.md)** | Comparing several models (LoRA / QLoRA / base / teacher) — judge keys (and what to do with none), eval-per-model, leaderboard + head-to-head. |
| **[docs/METRICS.md](docs/METRICS.md)** | How to read every metric file (which column means what, `score_5` vs `score_10`, where to look to spot overfit). |
| **[docs/SPEC.md](docs/SPEC.md)** | The full design spec (authoritative on conflicts). |
| **[data/README.md](data/README.md)** | The data contract — the exact JSONL shapes the data team delivers. |

## Run guide (happy path)

> Full detail in **[docs/RUNBOOK.md](docs/RUNBOOK.md)**. Add `--dry-run` to any training CLI (or
> `--mock` to `evaluate.py`) to exercise the wiring with no GPU / no API keys.

| # | Step | Command |
| --- | --- | --- |
| 0 | Setup | `uv sync --extra train --extra eval --extra viz --extra tracking` · `cp .env.example .env` |
| 1 | Validate data | `uv run python scripts/validate_data.py --data-dir data/` |
| 2 | SFT (the real model) | `uv run python scripts/train_sft.py --config configs/sft_coach_9b.yaml` |
| 3 | Eval SFT | `uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft_coach_9b/best --run-name eval_sft` |
| 4 | DPO alignment | `uv run python scripts/train_align.py --config configs/align_coach_dpo.yaml --sft-checkpoint checkpoints/sft_coach_9b/best` |
| 5 | Eval aligned | `uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/align_coach_dpo/best --run-name eval_dpo` |
| 6 | Export | `uv run python scripts/export_model.py --checkpoint checkpoints/align_coach_dpo/best --formats awq,gguf` |

A fast local smoke of the full loop (small base model, no real data needed):
`uv run python scripts/train_sft.py --config configs/sft_lora_smoke.yaml`.
