# Runbook — chạy pipeline từ đầu tới cuối (và *vì sao* theo thứ tự này)

Pipeline chỉ làm **data → train → evaluate → export** rồi **dừng ở việc tạo ra model** (không serve).
Mỗi bước **ăn output của bước trước**, nên thứ tự không tùy tiện được. Tài liệu này đi tuyến tính:
mỗi bước có **Nhiệm vụ · Lệnh · Tiêu thụ→Sản xuất · Vì sao ở đây · Cách kiểm tra**.

```
0 Setup ──▶ 1 Validate data ──▶ 2 (T1 SFT nhanh, optional)
                                   │
                                   ▼
                              3 T2 SFT thật ──▶ 4 Eval SFT ──▶ 5 DPO ──▶ 6 Eval aligned ──▶ 8 Export
                                                                  ▲
                                              (5 cần checkpoint SFT làm điểm xuất phát)
                              7 Baselines + benchmark vs model mẹ  → xem docs/BASELINES.md
```

**Chuỗi phụ thuộc (đây là lý do thứ tự):**
- **Validate trước train**: train chỉ nạp record `approved`. Data lỗi/lệch = model lỗi → chặn *trước* khi tốn GPU.
- **SFT trước DPO**: DPO **tiếp tục adapter của SFT** — không có checkpoint SFT thì DPO không có điểm xuất phát (script sẽ báo lỗi).
- **Train trước Eval**: phải có checkpoint mới sinh câu trả lời để chấm.
- **Eval trước Export**: export bản **tốt nhất**, sau khi đã đo và chọn được model thắng. Quantize là không thể đảo ngược.

> Mọi lệnh thêm `--dry-run` để duyệt kế hoạch **không cần GPU**; `evaluate.py --mock` chạy thử eval **không cần GPU/API key**. Dùng để kiểm tra wiring trước khi tốn tài nguyên.

---

## Bước 0 — Setup (một lần)

**Nhiệm vụ:** dựng môi trường + secrets + xác nhận GPU.
```bash
uv sync --extra train --extra eval --extra viz --extra tracking   # + --extra gpu/export khi cần
cp .env.example .env        # điền OPENAI_API_KEY, GOOGLE_API_KEY (judge), LANGFUSE_* (tùy chọn)
nvidia-smi                  # xác nhận thấy GPU
```
**Vì sao đầu tiên:** không có deps/keys thì mọi bước sau fail. `train` = train được; `eval` = judge GPT/Gemini; `viz` = biểu đồ; `tracking` = Langfuse.
**Kiểm tra:** `uv run pytest` xanh (chạy không cần GPU/key).

---

## Bước 1 — Validate dữ liệu

**Nhiệm vụ:** kiểm tra JSONL của team data đúng hợp đồng dữ liệu, đủ 7 mode.
```bash
uv run python scripts/validate_data.py --data-dir data --report outputs/data_report.json
```
**Tiêu thụ:** `data/{sft,reasoning,preference}/*.jsonl` + `data/gold/gold_test.jsonl`.
**Sản xuất:** bảng valid/invalid theo mode + `outputs/data_report.json`. **Thoát mã 1 nếu có record lỗi.**
**Vì sao ở đây:** "rác vào = rác ra". Bắt lỗi/lệch phân phối *trước* khi train. Train chỉ dùng record `approved`.
**Kiểm tra:** "All records valid." + đủ 7 mode (cảnh báo nếu mode nào trống).

---

## Bước 2 — (Tùy chọn) T1: LoRA SFT nhanh

**Nhiệm vụ:** baseline rẻ + smoke-test toàn pipeline trước khi train bản thật.
```bash

**Chạy Smoke để test trước**
uv run python scripts/train_sft.py --config configs/sft_lora_smoke.yaml

uv run python scripts/train_sft.py --config configs/sft_lora.yaml          # thêm --dry-run để duyệt
# Mồi siêu nhỏ trên GPU (model 1.5B, ~60 bước) để chắc pipeline chạy:
# uv run python scripts/train_sft.py --config configs/sft_lora_smoke.yaml
```
**Tiêu thụ:** sft + reasoning (đã approved). **Sản xuất:** `checkpoints/sft_lora/best|last`, `meta.json`, `metrics/` (loss/eval curves + CSV).
**Vì sao ở đây:** nhanh, đơn-stage — nếu hỏng thì lộ vấn đề sớm; cũng là một góc baseline (xem Bước 7).
**Kiểm tra:** có `checkpoints/sft_lora/best/`; loss giảm trong `metrics/loss_curve.png`.

---

## Bước 3 — T2: multi-stage QLoRA SFT (bản SFT thật)

**Nhiệm vụ:** model giám sát chính. Curriculum: stage `broad` (sft) → stage `reasoning` (sft + `<think>`).
```bash
uv run python scripts/train_multistage.py --config configs/sft_multistage.yaml
```
**Tiêu thụ:** sft + reasoning. **Sản xuất:** `checkpoints/sft_multistage/best|last` (best của stage cuối được promote), `meta.json`, `metrics/` mỗi stage.
**Vì sao ở đây:** đây là SFT "thật" để DPO căn chỉnh lên. Curriculum dạy nền rộng trước, rồi mới reasoning.
**Kiểm tra:** `checkpoints/sft_multistage/best/` tồn tại; rubric tăng dần qua các stage (`metrics/eval_metric.png`).

> Thích đơn-stage theo đúng config FPT của bạn? Dùng `configs/sft_lora.yaml` ở Bước 2 làm SFT chính thay cho Bước 3.

---

## Bước 4 — Eval model SFT

**Nhiệm vụ:** đo chất lượng SFT trên gold test, **theo từng mode**.
```bash
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model checkpoints/sft_multistage/best --run-name eval_sft
```
**Tiêu thụ:** checkpoint + `data/gold/gold_test.jsonl` + judge GPT/Gemini (key trong `.env`).
**Sản xuất:** `outputs/eval/eval_sft/report.md|json` (per-mode /10, 7 tiêu chí, pairwise-vs-gold, judge agreement, **chi phí API + token**, latency) + CSV/PNG (`per_sample.csv` chứa câu trả lời từng ca).
**Vì sao ở đây:** biết **mode nào yếu** trước khi alignment; và làm **mốc** để so "DPO có giúp không" ở Bước 6.
**Kiểm tra:** mở `report.md`, xem bảng per-mode + 3 mode yếu nhất + mục *Judge API usage & cost*.

> **System prompt production:** eval **inject sẵn** `system_prompt` (trong `configs/eval.yaml`) vào mọi prompt gold để model hành xử **giống lúc bán hàng thật** — và áp **cùng** prompt đó cho model mẹ/baseline để so công bằng. Sửa `system_prompt` thành đúng prompt bạn ship production trước khi chạy thật. Judge **không** thấy system prompt (chỉ chấm theo yêu cầu của khách).

---

## Bước 5 — T3: DPO alignment

**Nhiệm vụ:** căn chỉnh theo cặp preference (chosen/rejected).
```bash
uv run python scripts/train_align.py --config configs/align_dpo.yaml \
    --sft-checkpoint checkpoints/sft_multistage/best
```
**Tiêu thụ:** `data/preference/*.jsonl` + **checkpoint SFT** làm điểm xuất phát. **Sản xuất:** `checkpoints/align_dpo/best|last`, `meta.json`, `metrics/`.
**Vì sao ở đây / vì sao cần SFT trước:** DPO **tiếp tục adapter SFT**; thiếu `--sft-checkpoint` (hoặc `sft_checkpoint` trong config) script báo lỗi ngay. (ORPO thì monolithic — `configs/align_orpo.yaml`, không cần `--sft-checkpoint`. Quy tắc: ≥800 cặp → ORPO, ít hơn → DPO; bạn đang chọn DPO.)
**Kiểm tra:** `checkpoints/align_dpo/best/` tồn tại.

---

## Bước 6 — Eval model đã align

**Nhiệm vụ:** DPO có cải thiện so với SFT không.
```bash
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model checkpoints/align_dpo/best --run-name eval_dpo
uv run python scripts/compare_baselines.py --report outputs/eval/eval_sft/report.json \
    --report outputs/eval/eval_dpo/report.json
```
**Sản xuất:** `outputs/eval/eval_dpo/...` + leaderboard `outputs/eval/comparison.md`.
**Vì sao ở đây:** so trực tiếp **SFT vs SFT+DPO** trên cùng gold + cùng judge → quyết định lấy bản nào đi export.
**Kiểm tra:** overall /10 và các cột per-mode của `eval_dpo` so với `eval_sft`.

---

## Bước 7 — (Tùy chọn) Baselines + benchmark vs model mẹ

**Nhiệm vụ:** so các công thức train (LoRA vs QLoRA, single vs multi, SFT vs SFT+DPO) và **SLM vs model mẹ** (Qwen).
→ Quy trình đầy đủ + lệnh ở **[docs/BASELINES.md](BASELINES.md)**. Tóm tắt:
```bash
# eval model mẹ zero-shot (không train) để biết fine-tune nâng được bao nhiêu
uv run python scripts/evaluate.py --config configs/eval.yaml --model Qwen/Qwen3.5-9B --run-name eval_base_qwen3_8b
uv run python scripts/compare_baselines.py      # gộp mọi report thành 1 bảng xếp hạng
```
**Vì sao tách riêng:** đây là nghiên cứu so sánh, không nằm trên đường "tạo 1 model". Chạy khi muốn chọn công thức tốt nhất / báo cáo benchmark.

### Bước 7b — Win-rate SLM vs model mẹ (head-to-head) — "số cho sếp"

**Nhiệm vụ:** so **trực tiếp 1-1** câu trả lời SLM vs model mẹ → ra câu "SLM thắng Qwen gốc **X%**".
Dùng lại `per_sample.csv` của 2 lần eval ở trên — **không generate lại**, chỉ judge `compare`:
```bash
uv run python scripts/compare_models.py \
    --a outputs/eval/eval_dpo/per_sample.csv          --label-a "SLM (DPO)" \
    --b outputs/eval/eval_base_qwen3_8b/per_sample.csv --label-b "Qwen3-8B (mẹ)"
```
**Tiêu thụ:** 2 file `per_sample.csv` (từ Bước 6 + Bước 7) + judges. **Sản xuất:** `outputs/eval/headtohead.md` — win/tie/loss **tổng** + **theo mode** + dòng *Headline* "thắng X%" + chi phí judge.
**Vì sao:** đây là con số trực quan nhất để báo cáo. `--mock` để chạy thử offline. (Lưu ý: 2 file `per_sample.csv` phải từ eval **thật**, không phải `--mock`.)

---

## Bước 8 — Export / Quantize (sản phẩm cuối)

**Nhiệm vụ:** biến best checkpoint thành file model triển khai được.
```bash
uv run python scripts/export_model.py \
    --checkpoint checkpoints/align_dpo/best --formats awq,gguf
# (đường dữ liệu hiệu chỉnh AWQ in-domain, tùy chọn:) --calib-data data/sft/your_texts.jsonl
```
**Tiêu thụ:** best checkpoint (bản thắng ở Bước 6/7). **Sản xuất:** `outputs/exported/fp16/`, `outputs/exported/awq/` (INT4), `outputs/exported/gguf/` (Q4_K_M).
**Vì sao cuối cùng:** merge LoRA→FP16 rồi quantize là **không đảo ngược** và tốn kém — chỉ làm cho **một** model đã được chọn. Đây là điểm kết của pipeline.
**Kiểm tra:** cả `awq/` và `gguf/` có file model.

---

## Xuyên suốt — Metrics, biểu đồ, theo dõi

- **Tự động:** mỗi lần train ghi `checkpoints/<run>/metrics/` (`training_log.csv`, `loss_curve.png`, `eval_metric.png`, `lr_schedule.png`); mỗi lần eval ghi CSV + PNG cạnh `report.md`. **Train và eval ghi ở hai thư mục khác nhau** (`checkpoints/` vs `outputs/eval/`) — chạy `train_*.py` *không* sinh `outputs/eval/`.
- **Cách đọc từng con số** (cột nào nghĩa gì, score_5 vs score_10, nhìn đâu để kết luận): **[docs/METRICS.md](METRICS.md)**.
- **Vẽ lại** từ một run đã xong: `uv run python scripts/plot_metrics.py --run-dir checkpoints/align_dpo` (hoặc `--report outputs/eval/<run>/report.json`).
- **Langfuse** (tùy chọn): bật `tracking.langfuse` + `LANGFUSE_*` trong `.env` → log sample generation lúc eval-during-training.

---

## Đường đi tối thiểu (happy path)

> **0 → 1 → 3 → 4 → 5 → 6 → 8** (Bước 2 và 7 là tùy chọn).

| Bước | Lệnh gọn | Ra |
| --- | --- | --- |
| 0 | `uv sync --extra train --extra eval --extra viz --extra tracking` | môi trường |
| 1 | `validate_data.py --data-dir data` | data sạch |
| 3 | `train_multistage.py --config configs/sft_multistage.yaml` | `checkpoints/sft_multistage/best` |
| 4 | `evaluate.py --model checkpoints/sft_multistage/best --run-name eval_sft` | report SFT |
| 5 | `train_align.py --config configs/align_dpo.yaml --sft-checkpoint checkpoints/sft_multistage/best` | `checkpoints/align_dpo/best` |
| 6 | `evaluate.py --model checkpoints/align_dpo/best --run-name eval_dpo` | report DPO |
| 8 | `export_model.py --checkpoint checkpoints/align_dpo/best --formats awq,gguf` | model triển khai |

**Tài liệu liên quan:** [README.md](../README.md) (cài đặt nhanh) · [docs/SPEC.md](SPEC.md) (thiết kế) · [docs/BASELINES.md](BASELINES.md) (so sánh recipe + benchmark mẹ) · [docs/METRICS.md](METRICS.md) (đọc hiểu số liệu).
