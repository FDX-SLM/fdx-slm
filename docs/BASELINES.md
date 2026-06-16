# Baselines & benchmarks — how to run each, and pick the best recipe

This guide sets up a **controlled comparison** of training recipes and a **benchmark vs the parent
model**, all scored on the same full eval suite (`configs/eval.yaml`: 7-criteria rubric, per-mode
breakdown, GPT+Gemini judges, pairwise-vs-gold). Each run writes `outputs/eval/<name>/report.json`;
`scripts/compare_baselines.py` turns them into a single leaderboard.

> **Principle:** the baseline configs are **hyperparameter-matched** — every SFT corner uses the
> exact same hyperparameters (lr, batch, seed 1308, …). Only the axis under test changes, so a
> score difference is attributable to that axis, not to noise.

---

## 1. The comparison matrix

### Axis 1 + 2 — LoRA vs QLoRA × single-stage vs multi-stage (4 SFT corners)

| Run name | Config | Quant | Stages | Train with |
| --- | --- | --- | --- | --- |
| `bl_lora_single`  | `configs/baselines/sft_lora_single.yaml`  | LoRA (16-bit) | single | `scripts/train_sft.py` |
| `bl_qlora_single` | `configs/baselines/sft_qlora_single.yaml` | QLoRA (4-bit) | single | `scripts/train_sft.py` |
| `bl_lora_multi`   | `configs/baselines/sft_lora_multi.yaml`   | LoRA (16-bit) | multi  | `scripts/train_multistage.py` |
| `bl_qlora_multi`  | `configs/baselines/sft_qlora_multi.yaml`  | QLoRA (4-bit) | multi  | `scripts/train_multistage.py` |

- **LoRA vs QLoRA** → compare `*_lora_*` vs `*_qlora_*` (same stages).
- **single vs multi** → compare `*_single` vs `*_multi` (same quant).

### Axis 3 — SFT-only vs SFT+DPO

| Run name | Config | Starts from |
| --- | --- | --- |
| `<best SFT corner>` | (one of the 4 above) | — |
| `align_dpo` | `configs/align_dpo.yaml` | the best SFT corner's `best/` |

### Reference benchmarks (no training — eval the model directly)

| Run name | Model | Why |
| --- | --- | --- |
| `base_qwen3_8b`  | `Qwen/Qwen3-8B`  | **parent model, zero-shot** — the "did fine-tuning help?" baseline |
| `base_qwen3_14b` | `Qwen/Qwen3-14B` | bigger parent — "is our 8B fine-tune better than a larger base?" |
| `gold` (built-in) | — | the teacher answers are the upper bound; already shown as *pairwise win vs gold* |

> Other use-case-relevant references you can add the same way: a Vietnamese-tuned base, a
> distilled small model, or the same parent with a hand-written sales-coach **system prompt**
> (prompt-only baseline — see note in §6).

---

## 2. Train each baseline (run separately)

Single-stage corners → `train_sft.py`; multi-stage corners → `train_multistage.py`:

```bash
uv run python scripts/train_sft.py        --config configs/baselines/sft_lora_single.yaml
uv run python scripts/train_sft.py        --config configs/baselines/sft_qlora_single.yaml
uv run python scripts/train_multistage.py --config configs/baselines/sft_lora_multi.yaml
uv run python scripts/train_multistage.py --config configs/baselines/sft_qlora_multi.yaml
```

Each writes `checkpoints/<run_name>/best/` (+ `last/`, `meta.json`, and `metrics/` loss/eval
curves). Add `--dry-run` first to sanity-check the plan with no GPU.

### SFT + DPO (after you know the best SFT corner)

DPO continues the SFT adapter, so point it at that corner's `best/`:

```bash
uv run python scripts/train_align.py --config configs/align_dpo.yaml \
    --sft-checkpoint checkpoints/bl_qlora_multi/best        # <- your best SFT corner
```

Writes `checkpoints/align_dpo/best/`. (ORPO alternative: `configs/align_orpo.yaml`, monolithic, no
`--sft-checkpoint`.)

---

## 3. Evaluate each on the full suite

Same command for every model — just change `--model` and `--run-name` (the report dir):

```bash
# the 4 SFT corners
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/bl_lora_single/best  --run-name eval_bl_lora_single
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/bl_qlora_single/best --run-name eval_bl_qlora_single
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/bl_lora_multi/best   --run-name eval_bl_lora_multi
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/bl_qlora_multi/best  --run-name eval_bl_qlora_multi

# SFT + DPO
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/align_dpo/best --run-name eval_align_dpo

# parent / reference models (no training — pass the HF id directly)
uv run python scripts/evaluate.py --config configs/eval.yaml --model Qwen/Qwen3-8B  --run-name eval_base_qwen3_8b
uv run python scripts/evaluate.py --config configs/eval.yaml --model Qwen/Qwen3-14B --run-name eval_base_qwen3_14b
```

Each writes `outputs/eval/<run-name>/report.md` + `report.json` (per-mode table, rubric criteria,
pairwise-vs-gold, judge agreement, latency) and the CSV/PNG charts.

> **No GPU / no API keys?** Add `--mock` to exercise the whole pipeline offline (canned answers +
> a deterministic mock judge) — useful to validate the wiring before spending GPU/API budget.

---

## 4. Compare → the leaderboard (pick the winner)

```bash
uv run python scripts/compare_baselines.py            # reads every outputs/eval/*/report.json
# or pick specific runs:
uv run python scripts/compare_baselines.py \
    --report outputs/eval/eval_bl_qlora_multi/report.json \
    --report outputs/eval/eval_base_qwen3_8b/report.json
```

Prints and writes `outputs/eval/comparison.md`:
- **Leaderboard** ranked by overall /10, with pairwise-win-rate vs gold.
- **Per-mode matrix** (rows = run, columns = the 7 conversation modes) — shows *which recipe wins on
  which slice*, e.g. multi-stage may help `objection_handling` while QLoRA costs a bit on
  `factuality`.

Read it as: **SFT corners vs each other** → best training recipe; **best recipe vs `base_qwen3_*`**
→ the lift your fine-tune adds over the parent; **SFT-only vs `align_dpo`** → whether alignment paid
off.

---

## 5. Suggested order of operations

1. `--dry-run` all four SFT configs (catch config errors free).
2. Train + eval the 4 SFT corners → `compare_baselines.py` → pick the best corner.
3. Run DPO on that corner → eval `align_dpo`.
4. Eval the parent references (`base_qwen3_8b`, `base_qwen3_14b`).
5. `compare_baselines.py` once more for the full leaderboard (SLM vs parent vs recipes).

---

## 6. Notes & caveats

- **Fairness:** every model is scored on the *same* `data/gold/gold_test.jsonl` with the *same*
  judges. The parent is evaluated **zero-shot** (just the gold user prompts), which is exactly the
  point — it shows the lift fine-tuning adds.
- **Prompt-only baseline:** to benchmark "parent + a good Vietnamese sales-coach **system prompt**"
  (does prompting alone match fine-tuning?), the system prompt must be present in the gold prompts
  (or add an eval-time system-prompt flag). Today the eval feeds gold prompts verbatim, so add the
  coach system message to the gold cases if you want this baseline. *(Not yet a CLI flag.)*
- **Judge cost:** each eval ≈ `n_cases × 2 judges × (1 score + 1 pairwise)` **sequential** API
  calls. With many baselines this adds up — start with a small `gold_test.jsonl`, or `--mock` for
  pipeline checks. (A parallel-judge speedup is a known future improvement.)
- **Reproducibility:** all baselines use `seed: 1308`; each checkpoint's `meta.json` records the git
  commit + data version, so a leaderboard row is traceable to exact code+data.
- **"Best" = your call:** the overall /10 is a weighted mean (`configs/eval.yaml` weights —
  factuality & safety weighted highest). If a slice matters more for your use case (e.g.
  `objection_handling`), rank by that column in the per-mode matrix instead.
```
