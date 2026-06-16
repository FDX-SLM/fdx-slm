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
uv run python scripts/train_multistage.py --config configs/sft_multistage.yaml --dry-run

# Mock evaluation: canned generation + mock judge -> a real report, fully offline
uv run python scripts/evaluate.py --config configs/eval.yaml --model any --mock

# Local 8GB smoke test of the full T1 loop (small base model, ~200 steps) — needs the train extra
uv run python scripts/train_sft.py --config configs/sft_lora_smoke.yaml
```

---

## Project layout

```
configs/        base.yaml + sft_lora / sft_multistage / align_orpo / align_dpo / eval / sft_lora_smoke
src/slm_coach/
  config.py     pydantic models + base-merge loader (${ENV} expansion)
  tracking.py   Langfuse facade for sample generations (no-op without the tracking extra)
  reporting/    metric CSV tables + PNG charts (tables.py · plots.py); degrades without matplotlib
  data/         schema · loader · formatting (ChatML, <think>, multi-turn masking) · mixture
  training/     model (Unsloth/FA2, LoRA/QLoRA) · sft · multistage · align (DPO/ORPO) · callbacks
  eval/         inference · runner · rubric · judge (GPT+Gemini, pairwise) · latency · metrics · report · harness_task
  export/       merge (LoRA→FP16) · quantize (AWQ INT4 + GGUF Q4_K_M)
  utils/        logging · seed · deps
scripts/        thin CLIs: validate_data · train_sft · train_multistage · train_align · evaluate · export_model · plot_metrics
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

## How to run — step by step

> Everything runs through `uv`. `uv run <cmd>` executes `<cmd>` inside the project's virtual environment. You never need to manually activate a venv.
>
> For a linear, *why-in-this-order* walkthrough of the whole pipeline see **[docs/RUNBOOK.md](docs/RUNBOOK.md)**; for comparing recipes + benchmarking vs the parent model see **[docs/BASELINES.md](docs/BASELINES.md)**; for how to read every metric file (what each column means, score_5 vs score_10) see **[docs/METRICS.md](docs/METRICS.md)**.

### Step 0 — One-time setup

```bash
# Install uv (one time, if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root: create the venv and install all dependencies from pyproject.toml
uv sync

# Create your local secrets file and fill in the judge API keys + Langfuse keys
cp .env.example .env
#   edit .env: OPENAI_API_KEY, GOOGLE_API_KEY (for eval judges), LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST

# Sanity check: run the unit tests (no GPU / no API keys needed)
uv run pytest
```

**What this does:** `uv sync` builds an isolated environment with the exact pinned dependencies. The tests confirm the data schema, formatting/masking, loader, and metrics logic all work before you spend any GPU time. **Check:** all tests pass.

### Step 1 — Validate the data (from the data team)

```bash
uv run python scripts/validate_data.py --data-dir data/
```

**Purpose:** confirm the delivered JSONL matches the data contract (Section 3) before training. **Requires:** `data/{sft,reasoning,preference}/*.jsonl` present. **Produces:** a console/JSON report of valid vs invalid records and the distribution by `mode` and `data_type`. **Check:** invalid count is 0 (or acceptable) and all 7 modes are represented — a skewed distribution here predicts a weak model later.

### Step 2 — (Optional) Train T1: fast LoRA SFT baseline

```bash
uv run python scripts/train_sft.py --config configs/sft_lora.yaml
```

**Purpose:** a quick, cheap baseline to confirm the whole training loop works and to give you an early reference point. **Requires:** valid SFT data, a GPU. **Produces:** checkpoints under `checkpoints/sft_lora/` (keeps `best` and `last`). **Check:** training loss decreases and `checkpoints/sft_lora/best/` exists. You can skip straight to Step 3 if you don't need the baseline.

### Step 3 — Train T2: multi-stage QLoRA SFT (the real SFT)

```bash
uv run python scripts/train_multistage.py --config configs/sft_multistage.yaml
```

**Purpose:** the main supervised model. It runs the curriculum: stage `broad` (sft only, ~2/3 multi-turn) → stage `reasoning` (sft + reasoning, with `<think>` folding). Each stage starts from the previous stage's checkpoint. **Requires:** valid SFT + reasoning data, a GPU. **Produces:** `checkpoints/sft_multistage/best/` and `.../last/`, plus a `meta.json` per checkpoint and `<stage>/metrics/` (loss/eval curves + `training_log.csv`). Eval-during-training runs every `eval_steps`. **Check:** `checkpoints/sft_multistage/best/` exists; the eval-rubric curve in `metrics/eval_metric.png` improved across stages.

### Step 4 — Alignment (Phase 2): ORPO or DPO

Choose based on how many preference pairs you have (the rule of thumb: **≥800 pairs → ORPO**, otherwise **DPO**).

```bash
# Option A — ORPO (monolithic; recommended when you have enough preference pairs)
uv run python scripts/train_align.py --config configs/align_orpo.yaml

# Option B — DPO (needs the SFT checkpoint from Step 3 as its starting point)
uv run python scripts/train_align.py --config configs/align_dpo.yaml \
       --sft-checkpoint checkpoints/sft_multistage/best
```

**Purpose:** sharpen the model toward senior-style answers (chosen) over weak ones (rejected). DPO must start from the SFT checkpoint; ORPO combines the SFT-like and preference objectives in one stage. **Requires:** valid preference data, a GPU. **Produces:** `checkpoints/align_orpo/best/` (or `align_dpo/best/`). **Check:** the aligned checkpoint exists and scores higher than the SFT checkpoint when you run Step 5.

### Step 5 — Evaluate

```bash
uv run python scripts/evaluate.py --config configs/eval.yaml \
       --model checkpoints/align_orpo/best
```

**Purpose:** the full evaluation. It loads the checkpoint, generates answers for the gold test offline, scores them with the 7-criteria rubric + GPT/Gemini judges, computes the **per-mode breakdown**, judge agreement, optional pairwise win-rate vs the reference, and (optional) latency. **Requires:** `data/gold/gold_test.jsonl`, judge API keys in `.env`. **Produces:** `outputs/eval/<run>/report.md` + `report.json`, plus metric tables (`per_mode.csv`, `criteria.csv`, `per_sample.csv`) and charts (`per_mode.png`, `criteria.png`, `pairwise.png` when `pairwise: true`). **Check:** open `report.md` — the per-mode table tells you exactly which modes are weak so the data team can reinforce them. You can run this on any checkpoint (SFT vs aligned) to compare.

> **Metric artifacts.** Every training run writes `<checkpoint-dir>/metrics/training_log.csv` plus `loss_curve.png`, `lr_schedule.png`, and `eval_metric.png` (the eval-rubric curve); every eval run writes the CSVs/PNGs above. PNG charts need the `viz` extra (`uv sync --extra viz`); without it the CSV tables are still written and charts are skipped with a hint. Toggle per run with `reporting: {tables: false, plots: false}` in any config.

### Step 6 — Export / quantize (final deliverable)

```bash
uv run python scripts/export_model.py --checkpoint checkpoints/align_orpo/best --formats awq,gguf
```

**Purpose:** turn the best checkpoint into deployable model files. It merges the LoRA adapter into FP16 safetensors, then quantizes to AWQ INT4 (GPU) and GGUF Q4_K_M (CPU/edge). **Requires:** the chosen best checkpoint. **Produces:** `outputs/exported/awq/` and `outputs/exported/gguf/`. **Check:** both folders contain the quantized model — this is the end state of the pipeline.

### Utilities

```bash
# Resume an interrupted training run from its last checkpoint
uv run python scripts/train_sft.py --config configs/sft_lora.yaml --resume checkpoints/sft_lora/last

# Regenerate metric tables + charts from a finished run (e.g. after installing the viz extra)
uv run python scripts/plot_metrics.py --run-dir checkpoints/sft_lora
uv run python scripts/plot_metrics.py --report outputs/eval/<run>/report.json

# Lint & format
uv run ruff check .
uv run black .

# Add a new dependency later
uv add <package>
```

**Typical full sequence:** Step 0 → 1 → 3 → 4 → 5 → 6 (Step 2 is an optional baseline).
