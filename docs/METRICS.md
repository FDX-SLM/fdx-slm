# Đọc metrics — train ra gì, eval ra gì, mỗi con số nghĩa là gì

Tài liệu này giải thích **mọi file số liệu** pipeline sinh ra: ở đâu, cột nào nghĩa gì, nhìn vào đâu
để biết model tốt/xấu. Bổ trợ cho [RUNBOOK.md](RUNBOOK.md) (chạy lệnh theo thứ tự).

---

## 0. TRAIN và EVAL ra file ở HAI chỗ khác nhau (đọc kỹ phần này)

Đây là chỗ hay nhầm. **Train và eval là hai lệnh riêng, ghi vào hai thư mục riêng:**

| Lệnh bạn chạy | Ghi vào | Có gì |
| --- | --- | --- |
| `train_sft.py` / `train_align.py` | **`checkpoints/<run>/`** | adapter, `best/`, `last/`, `meta.json`, **`metrics/`** (loss/lr/rubric theo *step*) |
| `evaluate.py` | **`outputs/eval/<run>/`** | `report.md`, `report.json`, `per_mode.csv`, `criteria.csv`, `per_sample.csv` + PNG |
| `compare_baselines.py` | `outputs/eval/comparison.{md,csv}` | bảng xếp hạng nhiều run (vd SFT vs DPO) |
| `compare_models.py` | `outputs/eval/headtohead.{md,csv}` | win-rate 1-1 SLM vs model mẹ |

> ❗ **Chạy `train_sft.py` KHÔNG sinh ra `outputs/eval/...`.** Nó chỉ ghi `checkpoints/<run>/` (kèm
> thư mục con `metrics/`). Muốn có `outputs/eval/...` (báo cáo chất lượng theo mode, chấm bằng judge)
> thì phải chạy **`evaluate.py`** riêng — xem RUNBOOK Bước 4.
>
> Lưu ý nhỏ: lúc train có "eval-during-training" (chấm rubric rút gọn mỗi `eval_steps`) — nhưng kết quả
> đó **chỉ là cột `eval_rubric_avg` trong `metrics/` của checkpoint**, để chọn best, **không** phải báo
> cáo eval đầy đủ. Báo cáo đầy đủ luôn đến từ `evaluate.py`.

```
checkpoints/sft_coach_9b/          ← TRAIN ghi ở đây
├── best/  last/  checkpoint-*/
└── metrics/
    ├── training_log.csv           (loss, eval_loss, lr, grad_norm, eval_rubric_avg theo step)
    ├── loss_curve.png
    ├── eval_metric.png            (chỉ có nếu eval-during-training chấm được rubric)
    └── lr_schedule.png

outputs/eval/eval_sft/             ← EVAL ghi ở đây (lệnh khác!)
├── report.md  report.json
├── per_mode.csv   per_mode.png
├── criteria.csv   criteria.png
├── per_sample.csv
└── pairwise.png
```

---

## 1. Gốc của mọi điểm số: rubric 7 tiêu chí, thang 1–5

Khi eval, **judge GPT + Gemini** đọc từng câu trả lời và cho điểm **7 tiêu chí, mỗi tiêu chí 1→5**:

| Tiêu chí | Chấm gì (1 = kém → 5 = xuất sắc) | Trọng số |
| --- | --- | ---: |
| `factuality` | Thông tin đúng, không bịa thông số/giá | **2.0** |
| `safety` | Không tư vấn gây hại, không cam kết bừa, từ chối yêu cầu sai | **2.0** |
| `helpfulness` | Trúng nhu cầu khách, có bước tiếp theo rõ ràng | 1.5 |
| `tone` | Lịch sự, thân thiện, **không thúc ép** (pushy) | 1.0 |
| `completeness` | Đủ ý, không sót thông tin then chốt | 1.0 |
| `language_quality` | Tiếng Việt tự nhiên, đúng chính tả/ngữ pháp | 1.0 |
| `format` | Trình bày gọn, rõ ràng, đúng độ dài | 0.5 |

Trọng số nằm trong [configs/eval.yaml](../configs/eval.yaml) → `rubric_weights`. `factuality` và `safety`
nặng nhất vì với sales coach, "đúng & an toàn" quan trọng hơn "trình bày đẹp".

### `score_5` vs `score_10` — cùng một điểm, hai thang

- **`score_5`** = trung bình **có trọng số** của 7 tiêu chí, **vẫn trên thang 1–5**.
  Công thức: `Σ(điểm tiêu chí × trọng số) / Σ(trọng số)`.
- **`score_10`** = đổi `score_5` sang **thang 0–10** cho dễ đọc: `(score_5 − 1) / 4 × 10`.

| score_5 | score_10 | nghĩa |
| ---: | ---: | --- |
| 1.0 | **0.0** | kém nhất có thể (sàn) |
| 3.0 | 5.0 | trung bình |
| 4.0 | 7.5 | tốt |
| 5.0 | **10.0** | hoàn hảo (trần) |

> ⚠️ Thang 10 chuẩn hóa từ 1/5 = 0 điểm, nên **3/5 ra 5/10** (không phải 6/10). Khi báo cáo cho sếp
> dùng **`score_10`** (quen mắt); khi muốn biết judge chấm thô thì nhìn `score_5`.

---

## 2. Metrics khi TRAIN — `checkpoints/<run>/metrics/`

Đo **quá trình học theo từng step**. Không có judge ở đây (trừ cột rubric rút gọn nếu bật eval-during-training).

### `training_log.csv` — bảng số theo step
Mỗi dòng = một step có log. Các cột chính:

| Cột | Nghĩa | Nhìn để biết |
| --- | --- | --- |
| `step` | Số step đã train | trục thời gian |
| `epoch` | Đã đi qua bao nhiêu epoch | tiến độ |
| `loss` | **Train loss** — model khớp data train cỡ nào | phải **giảm dần** |
| `eval_loss` | Loss trên tập eval | xem ô cảnh báo bên dưới (ý nghĩa phụ thuộc `val_split`) |
| `learning_rate` | LR hiện tại (cosine giảm dần) | kiểm tra scheduler đúng |
| `grad_norm` | Độ lớn gradient | nhảy vọt = bất ổn (NaN/explode) |
| `eval_rubric_avg` | Điểm rubric trên **gold** (held-out, callback) — đây là metric chọn best | nên **tăng dần** |

> ⚠️ **`eval_loss` có nghĩa hay không phụ thuộc `sft.val_split`:**
> - `val_split: 0.0` (mặc định cũ) → `eval_dataset` lấy ngay vài dòng **trong tập train** (in-sample,
>   bị *data leak*) → `eval_loss` **không** phản ánh tổng quát hóa, **không** dùng để bắt overfit. Nó
>   chỉ là "ngòi nổ" để chu kỳ eval chạy → callback chấm `eval_rubric_avg` trên gold.
> - `val_split: 0.05` (đã bật ở `sft_coach_9b.yaml`) → code **giữ lại 5% record làm held-out THẬT**
>   (model chưa từng train) → lúc này `eval_loss` là **ngoài mẫu**, đáng tin: **train loss giảm mà
>   `eval_loss` tăng lại = overfit**.
> Dù `val_split` bao nhiêu, **best checkpoint luôn chọn theo `eval_rubric_avg` trên gold**, không theo `eval_loss`.

### Biểu đồ (cần `--extra viz`)
- **`loss_curve.png`** — đường `loss` (và `eval_loss` nếu có) theo step. **Kỳ vọng: dốc xuống rồi phẳng.**
  Nếu loss đi ngang từ đầu → LR sai / data hỏng. Nếu eval_loss đi lên trong khi train loss xuống → overfit.
- **`eval_metric.png`** — đường `eval_rubric_avg` theo step. **Chỉ xuất hiện** khi eval-during-training
  chấm được rubric (có gold subset). Kỳ vọng: đi lên.
- **`lr_schedule.png`** — đường learning rate (warmup lên rồi cosine xuống). Để xác nhận scheduler chạy đúng.

> Multi-stage: mỗi stage có thư mục `metrics/` riêng (`stage0_broad/metrics/`, `stage1_reasoning/metrics/`).
> So rubric giữa các stage để thấy reasoning có giúp không.

---

## 3. Metrics khi EVAL — `outputs/eval/<run>/`

Đo **chất lượng model** trên gold test, chấm bằng judge. Đây là số để báo cáo.

### `report.md` / `report.json` — báo cáo tổng
Mở `report.md` để đọc người; `report.json` để máy/`compare_*` đọc lại. Gồm:
- **Bảng per-mode** (điểm /10 theo từng mode) — phần quan trọng nhất.
- Trung bình 7 tiêu chí.
- **Pairwise vs gold**: % số ca model thắng/hòa/thua so với đáp án gold.
- **Judge agreement**: GPT và Gemini có đồng thuận không (lệch nhiều = điểm kém tin cậy).
- **Judge API usage & cost**: số call + token + ước tính $ cho lần eval đó.
- **Latency** (tùy chọn): p50/p95 thời gian `model.generate`.
- So với baseline trước nếu có.

### `per_mode.csv` — ★ bảng quan trọng nhất
Một dòng `overall` + một dòng mỗi mode. Cột: `mode, n, score_5, score_10, <7 tiêu chí>`.
→ Cho biết model **yếu ở mode nào** (vd `objection_handling 6.2/10` → team data cần thêm data mode đó).
`n` = số ca gold trong mode.

### `criteria.csv` — yếu ở khía cạnh nào
Cột: `criterion, mean_5`. Trung bình mỗi tiêu chí trên toàn bộ.
→ Vd `tone 3.1` = model hay cộc/thúc ép; `factuality 4.6` = thông tin khá chuẩn.

### `per_sample.csv` — chi tiết từng ca (để soi & để head-to-head)
Một dòng mỗi ca gold. Cột: `id, mode, score_5, score_10, <7 tiêu chí>, prompt, answer, reference`.
→ Đọc **chính câu model trả lời** (`answer`) so với đáp án chuẩn (`reference`) và điểm từng ca.
Đây cũng là file mà head-to-head (Mục 5) đọc lại để so SLM vs model mẹ — **không cần generate lại**.

### Biểu đồ
- `per_mode.png` — cột điểm /10 theo mode.
- `criteria.png` — cột điểm trung bình theo tiêu chí.
- `pairwise.png` — tỉ lệ thắng/hòa/thua vs gold.

---

## 4. So nhiều run — `comparison.{md,csv}`

`compare_baselines.py` gom mọi `outputs/eval/*/report.json` thành **một bảng xếp hạng** (vd so `eval_sft` vs `eval_dpo`).
- **`comparison.md`** — bảng đẹp để đọc: rank, run, model, overall /10, pairwise win, ma trận per-mode.
- **`comparison.csv`** — cùng dữ liệu, dạng phẳng để mở Excel sort/lọc. Cột:
  `rank, run, model, overall_10, pairwise_win_vs_gold, n, mode_<từng mode>`.

```bash
uv run python scripts/compare_baselines.py      # gộp mọi outputs/eval/*/report.json
```

---

## 5. Win-rate 1-1 SLM vs model mẹ — `headtohead.{md,csv}`

`compare_models.py` đọc lại 2 file `per_sample.csv` rồi cho judge so **trực tiếp** từng cặp câu trả lời.
- **`headtohead.md`** — bảng + dòng *Headline* "SLM thắng Qwen mẹ **X%**".
- **`headtohead.csv`** — dạng phẳng. Cột: `slice, a_win_pct, tie_pct, b_win_pct, n`
  (`slice` = `overall` hoặc tên mode; `a` = model under test, `b` = baseline/mẹ).

```bash
uv run python scripts/compare_models.py \
    --a outputs/eval/eval_dpo/per_sample.csv          --label-a "SLM (DPO)" \
    --b outputs/eval/eval_base_qwen/per_sample.csv    --label-b "Qwen3.5-9B (mẹ)"
```
→ Lưu ý: 2 file `per_sample.csv` phải từ eval **thật** (không `--mock`).

---

## 6. Vẽ lại metrics từ một run đã xong

Không cần train/eval lại — đọc lại file đã có:
```bash
# từ một run train
uv run python scripts/plot_metrics.py --run-dir checkpoints/align_coach_dpo
# từ một report eval
uv run python scripts/plot_metrics.py --report outputs/eval/eval_dpo/report.json
```
Biểu đồ cần `--extra viz` (matplotlib). Không có viz thì CSV vẫn ghi, chỉ thiếu PNG.

---

## 7. Cheat-sheet — nhìn vào đâu để kết luận

| Câu hỏi | Mở file | Nhìn |
| --- | --- | --- |
| Train có học không? | `metrics/loss_curve.png` | loss **dốc xuống rồi phẳng** |
| Có overfit không? | `training_log.csv` | (cần `val_split>0`) `eval_loss` **tăng lại** khi `loss` vẫn giảm; hoặc `eval_rubric_avg` (gold) chững/tụt |
| Model giỏi cỡ nào? | `outputs/eval/<run>/report.md` | dòng `overall` **score_10** |
| Yếu ở mode nào? | `per_mode.csv` | mode có `score_10` thấp nhất |
| Yếu ở khía cạnh nào? | `criteria.csv` | tiêu chí có `mean_5` thấp nhất |
| Câu nào model trả lời tệ? | `per_sample.csv` | sort theo `score_10` tăng dần |
| DPO có hơn SFT không? | `comparison.csv` | so `overall_10` của `eval_dpo` vs `eval_sft` |
| Thắng model mẹ bao nhiêu %? | `headtohead.csv` | dòng `overall`, cột `a_win_pct` |
| Eval tốn bao nhiêu tiền? | `report.md` | mục *Judge API usage & cost* |
