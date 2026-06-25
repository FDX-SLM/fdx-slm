# Runbook — chạy pipeline từ đầu tới cuối

Pipeline làm **data → train → evaluate → export** rồi **dừng ở việc tạo ra model** (không serve).
Mỗi bước ăn output của bước trước. Đường đi chính:

```
0 Setup → 1 Validate → 2 Smoke (tùy chọn) → 3 SFT thật → 4 Eval SFT → 5 DPO → 6 Eval DPO → 7 Export
```

**Vì sao theo thứ tự này:**
- **Validate trước train**: train chỉ nạp record `approved`; bắt lỗi data trước khi tốn GPU.
- **SFT trước DPO**: DPO tiếp tục adapter của SFT — thiếu checkpoint SFT thì DPO báo lỗi ngay.
- **Eval trước Export**: chỉ quantize (không đảo ngược được) bản đã đo và chọn là tốt nhất.

> Thêm `--dry-run` vào lệnh train để duyệt kế hoạch **không cần GPU**; `evaluate.py --mock` chạy eval thử **không cần GPU/API key**.
> Mọi lệnh chạy qua `uv run` — **không cần** `source .venv/bin/activate`.

---

## Bước 0 — Setup (một lần)

```bash
uv sync --extra train --extra eval --extra viz --extra tracking
cp .env.example .env        # điền OPENAI_API_KEY, GOOGLE_API_KEY (judge), LANGFUSE_* (tùy chọn)
nvidia-smi                  # xác nhận thấy GPU
```
**Lưu ý GPU (RTX 5090 / Blackwell sm_120):** verify torch là build cu12.8+ và bitsandbytes ≥ 0.45:
```bash
uv run python -c "import torch,bitsandbytes as bnb; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability(),bnb.__version__)"
```
**Kiểm tra:** `uv run pytest` xanh (chạy không cần GPU/key).

---

## Bước 1 — Validate dữ liệu

```bash
uv run python scripts/validate_data.py --data-dir data --report outputs/data_report.json
```
**Tiêu thụ:** `data/{sft,reasoning,preference}/*.jsonl` + `data/gold/gold_test.jsonl`.
**Sản xuất:** bảng valid/invalid theo mode + `outputs/data_report.json`. **Thoát mã 1 nếu có record lỗi.**
**Kiểm tra:** "All records valid." + đủ 7 mode. **Train chỉ dùng record `audit_status == approved`** — nếu data còn `pending`, nó bị bỏ qua âm thầm.

---

## Bước 1b — Tách holdout val (stratified theo mode)

Tách tập validation **cố định, chia đều 7 mode** ra khỏi train. Chạy lại **mỗi khi data thay đổi** (vd. gen thêm lên 2k):
```bash
uv run python scripts/split_holdout.py --config configs/sft_coach_9b.yaml
```
**Tiêu thụ:** `data/{sft,reasoning,preference}/*.jsonl` (approved) — **giữ nguyên, không sửa nguồn**.
**Sản xuất** (disjoint; val chia đều mode, cấu hình ở `base.yaml` `data:`):
- SFT: `data/holdout/train.jsonl` + `val.jsonl` (val ≥ `max(val_min_total, val_fraction × tổng)`).
- DPO: `data/holdout/preference_train.jsonl` + `preference_val.jsonl` (val ≥ `max(pref_val_min_total, pref_val_fraction × tổng)`).
**Kiểm tra:** mỗi nhóm in "wrote N train + M val", val per-mode đều và đủ 7 mode (không cảnh báo "modes absent from val"). Preference quá nhỏ → bỏ qua kèm cảnh báo, DPO tự fallback split ngẫu nhiên.
Train SFT (Bước 2/3) và DPO (Bước 5) tự đọc holdout tương ứng — val **không bao giờ** lọt vào train; chưa chạy bước này thì fallback split in-memory kèm cảnh báo.

---

## Bước 2 — (Tùy chọn) Smoke test

Mồi nhanh trên model nhỏ để chắc cả pipeline chạy trước khi train bản 9B:
```bash
uv run python scripts/train_sft.py --config configs/sft_lora_smoke.yaml
```
**Kiểm tra:** có `checkpoints/sft_lora_smoke/best/`; loss giảm trong `metrics/loss_curve.png`.

---

## Bước 3 — SFT thật (Qwen3.5-9B)

```bash
uv run python scripts/train_sft.py --config configs/sft_coach_9b.yaml
# (giữ log:)  ... 2>&1 | tee outputs/sft_coach_9b.log
# (resume:)   ... --resume checkpoints/sft_coach_9b/last
```
**Tiêu thụ:** `data/sft/*.jsonl` + `data/reasoning/*.jsonl` (đã approved).
**Sản xuất:** `checkpoints/sft_coach_9b/best|last`, `meta.json`, `metrics/` (loss/eval curves + CSV).
**Kiểm tra:** `checkpoints/sft_coach_9b/best/` tồn tại; loss giảm.

> Qwen3.5-9B (`model_type: qwen3_5`) cần **transformers ≥ 5.12** (đã ghim trong `pyproject.toml`). Bản cũ báo `model type qwen3_5 not recognized`.

---

## Bước 4 — Eval model SFT

```bash
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model checkpoints/sft_coach_9b/best --run-name eval_sft
```
**Tiêu thụ:** checkpoint + `data/gold/gold_test.jsonl` + judge GPT/Gemini (key trong `.env`).
**Sản xuất:** `outputs/eval/eval_sft/report.md|json` (per-mode /10, 7 tiêu chí, judge agreement, PII leak, latency) + `per_sample.csv`.
**Kiểm tra:** mở `report.md`, xem bảng per-mode + các mode yếu nhất.

> Eval **inject `system_prompt`** (trong `configs/eval.yaml`) vào mọi prompt gold để model hành xử như lúc bán hàng thật. Sửa `system_prompt` thành prompt production trước khi chạy thật.

---

## Bước 5 — DPO alignment

```bash
uv run python scripts/train_align.py --config configs/align_coach_dpo.yaml \
    --sft-checkpoint checkpoints/sft_coach_9b/best
```
**Tiêu thụ:** `data/preference/*.jsonl` + **checkpoint SFT** làm điểm xuất phát.
**Sản xuất:** `checkpoints/align_coach_dpo/best|last`, `meta.json`, `metrics/`.
**Vì sao cần SFT trước:** DPO tiếp tục adapter SFT; thiếu `--sft-checkpoint` (hoặc `sft_checkpoint` trong config) script báo lỗi ngay.
**Kiểm tra:** `checkpoints/align_coach_dpo/best/` tồn tại.

---

## Bước 6 — Eval model đã align

```bash
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model checkpoints/align_coach_dpo/best --run-name eval_dpo
# So SFT vs DPO thành 1 bảng xếp hạng:
uv run python scripts/compare_baselines.py --report outputs/eval/eval_sft/report.json \
    --report outputs/eval/eval_dpo/report.json
```
**Vì sao:** so `eval_dpo` vs `eval_sft` trên cùng gold + cùng judge → quyết định lấy bản nào đi export.
**Sản xuất:** `outputs/eval/comparison.{md,csv}`. **Kiểm tra:** overall /10 + per-mode của `eval_dpo` vs `eval_sft`.

> **Win-rate vs model mẹ (tùy chọn, "số cho sếp"):** eval model mẹ zero-shot rồi so 1-1 — dùng lại `per_sample.csv`, không generate lại:
> ```bash
> uv run python scripts/evaluate.py --config configs/eval.yaml --model Qwen/Qwen3.5-9B --run-name eval_base_qwen
> uv run python scripts/compare_models.py \
>     --a outputs/eval/eval_dpo/per_sample.csv        --label-a "SLM (DPO)" \
>     --b outputs/eval/eval_base_qwen/per_sample.csv  --label-b "Qwen3.5-9B (mẹ)"
> ```

---

## Bước 7 — Export / Quantize (sản phẩm cuối)

```bash
uv run python scripts/export_model.py \
    --checkpoint checkpoints/align_coach_dpo/best --formats awq,gguf
# (calib AWQ in-domain, tùy chọn:) --calib-data data/sft/your_texts.jsonl
```
**Tiêu thụ:** best checkpoint (bản thắng ở Bước 6). **Sản xuất:** `outputs/exported/fp16/`, `outputs/exported/awq/` (INT4), `outputs/exported/gguf/` (Q4_K_M).
**Vì sao cuối cùng:** merge LoRA→FP16 rồi quantize là **không đảo ngược** và tốn kém — chỉ làm cho **một** model đã chọn.
**Kiểm tra:** cả `awq/` và `gguf/` có file model.

---

## Metrics & biểu đồ

- Mỗi lần train ghi `checkpoints/<run>/metrics/` (`training_log.csv`, `loss_curve.png`, `eval_metric.png`); mỗi lần eval ghi CSV + PNG cạnh `report.md`. Train và eval ghi ở **hai thư mục khác nhau** (`checkpoints/` vs `outputs/eval/`).
- Cách đọc từng con số: **[docs/METRICS.md](METRICS.md)**.
- Vẽ lại từ run đã xong: `uv run python scripts/plot_metrics.py --run-dir checkpoints/align_coach_dpo`.
- **Langfuse** (tùy chọn): bật `tracking.langfuse` + `LANGFUSE_*` trong `.env` → log sample generation lúc eval-during-training.

---

## Đường đi tối thiểu

| Bước | Lệnh gọn | Ra |
| --- | --- | --- |
| 0 | `uv sync --extra train --extra eval --extra viz --extra tracking` | môi trường |
| 1 | `validate_data.py --data-dir data` | data sạch |
| 1b | `split_holdout.py --config configs/sft_coach_9b.yaml` | `data/holdout/{train,val}.jsonl` |
| 2 | `train_sft.py --config configs/sft_lora_smoke.yaml` | smoke OK (tùy chọn) |
| 3 | `train_sft.py --config configs/sft_coach_9b.yaml` | `checkpoints/sft_coach_9b/best` |
| 4 | `evaluate.py --model checkpoints/sft_coach_9b/best --run-name eval_sft` | report SFT |
| 5 | `train_align.py --config configs/align_coach_dpo.yaml --sft-checkpoint checkpoints/sft_coach_9b/best` | `checkpoints/align_coach_dpo/best` |
| 6 | `evaluate.py --model checkpoints/align_coach_dpo/best --run-name eval_dpo` | report DPO |
| 7 | `export_model.py --checkpoint checkpoints/align_coach_dpo/best --formats awq,gguf` | model triển khai |

**Tài liệu liên quan:** [README.md](../README.md) · [docs/SPEC.md](SPEC.md) (thiết kế) · [docs/METRICS.md](METRICS.md) (đọc hiểu số liệu).
