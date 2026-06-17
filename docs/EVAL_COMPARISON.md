# So sánh nhiều model — eval + leaderboard

Hướng dẫn so chất lượng giữa **các bản fine-tune (LoRA / QLoRA)**, **base model zero-shot**, và
**teacher**. Bổ trợ cho [RUNBOOK.md](RUNBOOK.md) (Bước 4–6) và [METRICS.md](METRICS.md) (đọc số).

Nguyên tắc: **train → eval từng model (cùng điều kiện) → compare**. Mỗi model eval ra một
`outputs/eval/<run-name>/` riêng; hai script `compare_*` gộp chúng lại.

---

## 0. Cần KEY gì? (bạn đang chưa có)

Eval thật là **LLM-as-judge** → cần **GPT hoặc Gemini**. **Cấm Claude/DeepSeek** làm judge (chúng là
teacher sinh ra data → thiên vị, code chặn luôn). Ba lựa chọn:

| Tình huống | Cấu hình | Có dùng để kết luận chất lượng? |
|---|---|---|
| **Có key Gemini** (Google AI Studio có free tier) ✅ khuyến nghị | `judges: ["gemini"]` trong `configs/eval.yaml` + `GOOGLE_API_KEY` trong `.env` | **Có** — 1 judge đủ ra điểm dùng được |
| Có cả 2 key | `judges: ["gpt", "gemini"]` + cả `OPENAI_API_KEY` lẫn `GOOGLE_API_KEY` | **Có** — robust nhất, thêm judge-agreement |
| **Không key nào** | `evaluate.py ... --mock` | **KHÔNG** — điểm là giả (canned), chỉ để test pipeline chạy thông |

- Chỉ cần **1 key Gemini** là so được tất cả (judge OpenAI chỉ khởi tạo khi `judges` có `"gpt"`).
- `--mock` ra `report.md` + `per_sample.csv` đầy đủ định dạng nhưng **điểm vô nghĩa** → đừng dùng để
  chọn bản thắng; chỉ để verify `evaluate.py` + `compare_*` chạy trước khi tốn API.

> Lấy key Gemini free → điền vào `.env`:
> ```
> GOOGLE_API_KEY=AIza...
> ```
> rồi đổi `configs/eval.yaml`: `judges: ["gemini"]`.

---

## 1. So được những gì

| Model | Eval kiểu gì | `--model` |
|---|---|---|
| **QLoRA** (4-bit) | checkpoint local | `checkpoints/sft_coach_9b/best` |
| **LoRA** (16-bit) | checkpoint local | `checkpoints/sft_coach_9b_lora/best` |
| **Base zero-shot** (chưa train) | nạp HF id | `Qwen/Qwen3.5-9B` |
| **Teacher** (Claude/DeepSeek) | ❌ KHÔNG — là API, không phải model local; lại bị cấm làm judge | — |

**Teacher so kiểu gì?** Câu mẫu trong `data/gold/gold_test.jsonl` **chính là chuẩn teacher**. Vì
`eval.yaml` để `pairwise: true`, mỗi model đã được judge so 1-1 với gold → cột `pairwise_win_vs_gold`
chính là "thắng teacher bao nhiêu %". Không cần (và không thể) chạy teacher qua pipeline offline này.

---

## 2. Eval từng model (CÙNG điều kiện, khác `--run-name`)

Dùng chung một `configs/eval.yaml` (cùng gold, cùng judge, cùng `system_prompt`, cùng seed) — chỉ đổi
`--model` và `--run-name`:

```bash
# QLoRA
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model checkpoints/sft_coach_9b/best       --run-name eval_qlora
# LoRA
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model checkpoints/sft_coach_9b_lora/best  --run-name eval_lora
# Base zero-shot (mốc "fine-tune nâng được bao nhiêu")
uv run python scripts/evaluate.py --config configs/eval.yaml \
    --model Qwen/Qwen3.5-9B                      --run-name eval_base
```
Mỗi lệnh ra `outputs/eval/<run-name>/report.json` + `per_sample.csv` (+ PNG nếu có `--extra viz`).

> Chưa có key → thêm `--mock` vào mỗi lệnh để chạy thử thông pipeline (điểm giả, đừng kết luận).

---

## 3. So — 2 góc nhìn

```bash
# (a) Bảng xếp hạng tất cả: điểm /10 tổng + per-mode đặt cạnh nhau
uv run python scripts/compare_baselines.py \
    --report outputs/eval/eval_qlora/report.json \
    --report outputs/eval/eval_lora/report.json \
    --report outputs/eval/eval_base/report.json
#   → outputs/eval/comparison.{md,csv}
# (Chạy KHÔNG kèm --report thì nó tự gom MỌI outputs/eval/*/report.json)

# (b) Win-rate 1-1 giữa 2 bản bất kỳ (vd LoRA vs QLoRA, hoặc SLM vs base)
uv run python scripts/compare_models.py \
    --a outputs/eval/eval_lora/per_sample.csv  --label-a "LoRA" \
    --b outputs/eval/eval_qlora/per_sample.csv --label-b "QLoRA"
#   → outputs/eval/headtohead.{md,csv}
```

---

## 4. Đọc kết quả để KẾT LUẬN

**`outputs/eval/comparison.md`** (từ 3a):
- **`overall_10`** — điểm trung bình /10 trên cả gold. Cao hơn = tốt hơn → số quyết định chính.
- **`pairwise_win_vs_gold`** — % thắng so với câu mẫu teacher (gold).
- **Ma trận per-mode** — mạnh/yếu từng mode (`objection_handling`, `upsell`, `after_sales`...). Có thể
  một bản thắng tổng nhưng thua ở vài mode → biết slice nào cần data thêm.

**`outputs/eval/headtohead.md`** (từ 3b): dòng *Headline* "A thắng B **X%**" + tách theo mode. Dùng khi
hai bản điểm /10 sàn sàn nhau.

**Quy tắc chọn:** lấy bản `overall_10` cao nhất; chênh < ~0.2 thì nhìn `headtohead` + per-mode để chốt →
bản thắng đem **export** (RUNBOOK Bước 7).

---

## 5. Để so SẠCH (fairness)

- **Cùng mọi thứ ngoài trục đang so**: cùng base, cùng `seed: 1308`, cùng `max_seq_len`, cùng gold,
  cùng `eval.yaml` (judge + `system_prompt`). LoRA vs QLoRA chỉ nên khác `load_in_4bit`.
- **Bẫy seq_len**: nếu phải hạ `max_seq_len` của LoRA xuống 1024 vì OOM, thì hạ QLoRA xuống 1024 luôn
  (hoặc ghi rõ trong báo cáo) — không thì bảng so đang lẫn 2 biến.
- **Cùng số judge**: đừng eval bản này bằng 2 judge, bản kia bằng 1 judge.
- **Reproducible**: mỗi report ghi lại checkpoint + data version + seed; giữ nguyên để chạy lại ra cùng số.
