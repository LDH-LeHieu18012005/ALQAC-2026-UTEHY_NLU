# ALQAC 2026 — Dự đoán kết quả tranh chấp dân sự Việt Nam

Pipeline dự đoán nhãn phán quyết (`A_WIN` / `B_WIN` / `PARTIAL_A_WIN` / `PARTIAL_B_WIN`)
cho vụ án dân sự, từ tóm tắt yêu cầu khởi kiện + chứng cứ trích từ hồ sơ + điều luật truy xuất.
Toàn bộ LLM chạy remote trên **Modal GPU** (vLLM); notebook chỉ đóng vai trò client.

- **Tập public**: 50 case, có `verdict_label` → dùng để đánh giá.
- **Tập private**: 60 case, chỉ có `case_id` + `case_query` → là mục tiêu nộp bài.
- **Corpus luật**: 18 văn bản, 3.352 điều.

---

## 1. Kết quả hiện tại (public 50 case, seed 2026)

| Cấu hình | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| **`fullchunks/pado`** ← chủ lực | Qwen3.5-9B | **0.540** | **0.508** |
| `fullchunks/base` | Qwen3.5-9B | 0.540 | 0.428 |
| `fullchunks/base` | DeepSeek-R1-Distill-8B | 0.460 | 0.436 |
| `fullchunks/base` | Qwen3-4B | 0.420 | 0.333 |
| `only_query/pado` | Qwen3-4B | 0.400 | 0.251 |
| `only_query/pado` | Qwen2.5-7B | 0.400 | 0.222 |
| `only_query/pado` | Qwen3.5-9B | 0.340 | 0.133 |
| `only_query/base` | Qwen3.5-9B | 0.300 | 0.226 |
| `fullchunks/base` | Qwen2.5-7B | 0.300 | 0.184 |
| `only_query/base` | Qwen3-4B | 0.280 | 0.176 |

**Mốc so sánh**: majority baseline (đoán tất cả `PARTIAL_A_WIN`) = **0.380**.

Chi tiết run chủ lực (top-14, đã chạy):

```
accuracy 0.540 | macro F1 0.508 | weighted F1 0.537
coverage 1.0   | invalid_output_rate 0.0 | 0 fallback (100% judge_ensemble)
avg input 39.155 tok | avg output 25.105 tok | 32.2 s/case
```

| Lớp | P | R | F1 | n |
|---|---:|---:|---:|---:|
| A_WIN | 0.58 | 0.44 | 0.50 | 16 |
| B_WIN | 0.80 | 0.80 | 0.80 | 10 |
| PARTIAL_A_WIN | 0.46 | 0.58 | 0.51 | 19 |
| PARTIAL_B_WIN | 0.25 | 0.20 | 0.22 | 5 |

---

## 2. Cấu trúc repo

```
data/
  ALQAC2026_public_test.json      50 case, có verdict_label (nguồn đánh giá)
  corpus_law_pub.json             18 văn bản / 3.352 điều
  agent_v4_results.json           chứng cứ trích từ case_fact (evidence agent v4)
  retrieval_top14_ours.json       ĐIỀU LUẬT top-14/case — nguồn retrieval hiện dùng

private/
  ALQAC_private_test.json         60 case, chỉ case_id + case_query
  agent_v4_private_results.json   evidence cho private
  private_test_60_cases_extracted_corpus.json
  embed-model-retrieval-bge-m3.ipynb

pre_retrieval/                    sinh ra agent_v4_results.json
  call_api/call_api_agent_v4.py   agent retrieval chứng cứ (adaptive + LLM 3 điểm)
  chunk_denoise.py                lọc nhiễu thủ tục khỏi chunk
  legal_query.py                  rewrite mô tả vụ án -> truy vấn pháp lý
  data_io.py, llm.py

embedding/                        dựng & đánh giá collection Qdrant BGE-M3
  embed-model-retrieval-bge-m3.ipynb
  evaluate-retrieval-bge-m3.ipynb
  retrieval_eval_bge_m3/*.csv
  retrieval_top10_bge_m3.json

only_query/   base/ pado/         chỉ dùng case_query (không evidence)
fullchunks/   base/ pado/         case_query + evidence agent_v4
  pado/alqac-e2e-bge-m3-qwen35-9b-pado.ipynb   ← notebook chủ lực
  pado/predictions_qwen3-5-9b_seed-2026.json   ← output mới nhất
```

**Bốn nhánh thí nghiệm** = tích của hai trục:

|  | `base` | `pado` |
|---|---|---|
| **`only_query`** | 1 call phân loại, chỉ case_query | issue_mapper + 3 judge vote, chỉ case_query |
| **`fullchunks`** | 1 call phân loại, + evidence | issue_mapper + 3 judge vote, + evidence |

`base` = phân loại một lần (`BENCHMARK_VERSION = 'v2'`).
`pado` = pipeline nhiều stage (`legal-pado-v12-divisible-types-balanced-judge`).

---

## 3. Luồng dữ liệu

```
ALQAC2026_public_test.json ──┬─> case_query ──────────────────────────┐
                             │                                        │
                             └─> case_fact ─> [agent v4 retrieval] ─> evidence (≤8.000 ký tự)
                                              agent_v4_results.json    │
                                                                       ▼
retrieval_top14_ours.json ─────────────> 14 điều luật/case ──> ISSUE MAPPER (1 call, no-thinking)
                                                    │                  │
                                                    │                  ▼ issue_map (claims của A)
                                                    └──────────> LEGAL JUDGE × 3 vote (thinking)
                                                                       │
                                                                       ▼
                                                          aggregate_votes (majority 4 lớp)
                                                                       │
                                                                       ▼
                                                       prediction ──> submission (case_id, prediction)
```

`direct_fallback` chỉ chạy khi cả 3 vote hỏng; `hard_fallback` đảm bảo luôn đủ 50 dòng.
Run mới nhất: **0 case cần fallback**.

### Nội dung prompt đưa vào Judge

| Thành phần | ký tự TB | Ghi chú |
|---|---:|---|
| 14 điều luật | 24.290 | 59% ký tự là luật **tố tụng** (BLTTDS, án phí) |
| Evidence (chunk từ `case_fact`) | 6.022 | cắt ở `MAX_EVIDENCE_CHARS` = 8.000 |
| CASE_QUERY | 343 | tóm tắt yêu cầu của nguyên đơn |
| ISSUE MAP | 269 | chỉ liệt kê yêu cầu của **A** |

---

## 4. Cấu hình notebook chủ lực

`fullchunks/pado/alqac-e2e-bge-m3-qwen35-9b-pado.ipynb`

```python
MODEL              = 'Qwen/Qwen3.5-9B'      # vLLM trên Modal, bfloat16, không lượng tử hoá
GPU_TYPE           = 'A100-40GB'            # env MODAL_GPU
RETRIEVAL_SOURCE   = 'file:retrieval_top14_ours.json'
TOP_K              = 14                     # env ALQAC_TOP_K
MAX_ARTICLE_CHARS  = 6000                   # env ALQAC_MAX_ARTICLE_CHARS
MAX_INPUT_TOKENS   = 24000
MAX_MODEL_LEN      = 40960                  # env VLLM_MAX_MODEL_LEN
MAX_NEW_TOKENS_ISSUE / JUDGE / DIRECT = 4096 / 16384 / 2048
JUDGE_VOTES        = 3
USE_STRUCTURED_OUTPUTS = False              # JSON tự do + parse robust
EVAL_SEEDS         = [2026]
```

Retrieval **không còn dùng Qdrant/BGE-M3 tại runtime** — cell 4 đọc thẳng
`retrieval_top14_ours.json` và chuẩn hoá về schema
`rank/score/law_id/aid/article_no/content_Article`. Không cần `QDRANT_URL`/`QDRANT_API_KEY`.

`MAX_MODEL_LEN = 40960` (không phải 32768) vì top-14 dài gấp ~2,6 lần top-10:
prompt judge worst-case ~16,3k token + `max_new` 16.384 sẽ vượt 32768.
Cell 5 có **preflight đếm token bằng tokenizer local** trước khi tốn GPU; nếu engine
từ chối 40960 thì đặt `VLLM_MAX_MODEL_LEN=32768` và `ALQAC_MAX_ARTICLE_CHARS=4000`.

---

## 5. Cách chạy

**Yêu cầu**: Modal Secret `HF_TOKEN` (Modal Notebooks đã tự xác thực `MODAL_TOKEN_*`).

Upload cạnh notebook (hoặc vào `data/`):
`ALQAC2026_public_test.json`, `agent_v4_results.json`, `retrieval_top14_ours.json`.

Chạy Run All. Output vào `outputs_alqac_e2e/v12_legal_pado_divisible_types_balanced_judge/`:
`predictions_*.json`, `submission_*.json`, `benchmark_manifest_*.json`, `case_predictions_*.csv`.

### Biến môi trường

| Biến | Mặc định | Công dụng |
|---|---|---|
| `ALQAC_RETRIEVAL_PATH` | (tự dò) | trỏ thẳng file điều luật |
| `ALQAC_RETRIEVAL_FILE_NAME` | `retrieval_top14_ours.json` | đổi tên file |
| `ALQAC_TOP_K` | `14` | phải khớp số điều/case trong file |
| `ALQAC_MAX_ARTICLE_CHARS` | `6000` | hạ xuống nếu thiếu context |
| `ALQAC_MAX_EVIDENCE_CHARS` | `8000` | độ dài evidence |
| `VLLM_MAX_MODEL_LEN` | `40960` | trần context engine |
| `MODAL_GPU` | `A100-40GB` | loại GPU |
| `ALQAC_BASE_PREDICTIONS_PATH` | — | bật paired comparison ở cell 8 |

### Cache và chạy lại

`cache_key` = hash(model, case, **điều luật**, seed). Đổi file retrieval → mọi case tự
tính lại, không dùng nhầm kết quả cũ. Ngược lại, chạy lại cùng cấu hình sẽ tái dùng cache.

> **Cảnh báo**: run mới **ghi đè** `predictions_*.json` cùng model/seed.
> Back up trước nếu muốn so sánh paired với run cũ.

---

## 6. Phân tích: tại sao score dừng ở 0.54

Tách nhãn 4 lớp thành hai trục độc lập cho thấy bức tranh rõ hơn accuracy thô
(accuracy thô gây hiểu lầm vì trục A/B lệch 70/30):

| Trục | Balanced acc | φ | p (χ²) |
|---|---:|---:|---:|
| **SIDE** (ai thắng) | 0.729 | 0.467 | 0.001 |
| **DEGREE** (toàn bộ / một phần) | 0.643 | 0.287 | 0.042 |

Cả hai trục đều mang tín hiệu thật nhưng **yếu**. Phân phối dự đoán khớp gold gần như
hoàn hảo (side 37A/13B vs gold 35/15; degree 24/26 vs 26/24) → **không phải lỗi lệch nhãn**,
mà là thiếu chính xác.

### Nguyên nhân đã đo được

**① Kênh trích dẫn luật chết hoàn toàn.** `applied_laws` rỗng **50/50** case —
judge không trích được điều luật nào làm căn cứ. `legal_basis_ranks` chỉ dùng để audit,
không tham gia quyết định nhãn. Vì `USE_STRUCTURED_OUTPUTS = False` nên schema không
được ép, model tự do bỏ trường này.

**② Retrieval vẫn còn xa.** So với gold `related_law_provisions` (chỉ tính luật nội dung):

| | recall@k | case có ≥1 điều đúng | điều nội dung/case | case có 0 điều nội dung |
|---|---:|---:|---:|---:|
| Qdrant top-10 (cũ) | 0.079 | 15/46 | 5,3 | 4 |
| `retrieval_top14_ours` | 0.121 | 19/46 | 6,0 | **0** |

Và luật *có* tác dụng: ở run cũ, case retrieval trúng đạt acc 0.667 vs trượt 0.486.

**③ Lập luận bị đơn có trong prompt nhưng không được yêu cầu xử lý.**
Evidence đã chứa material của B ở phần lớn case ("không đồng ý" 29/50, "bị đơn" 32/50,
"vắng mặt" 22/50, "thừa nhận" 12/50). Nhưng `ISSUE_MAPPER_SYSTEM_PROMPT` cấm tường minh
(*"Không tạo legal issue, defense hay counterclaim thành claim"*), nên issue map chỉ có
yêu cầu của A và judge không có cấu trúc nào để cân nhắc phản bác.

Keyword thô trên `case_fact` cho thấy tín hiệu này có giá trị:

| Đặc trưng | n | P(A thắng \| có) | P(A thắng \| không) | Δ |
|---|---:|---:|---:|---:|
| B "không đồng ý" | 35 | 0.63 | 0.87 | −0.24 |
| B "thừa nhận" | 14 | 0.86 | 0.64 | +0.22 |
| B "vắng mặt" | 19 | 0.79 | 0.65 | +0.14 |

**④ Trần thông tin.** Input = yêu cầu của A + lời trình bày các bên; gold sinh từ
`court_verdict`. Phân biệt `A_WIN` vs `PARTIAL_A_WIN` đòi hỏi biết toà có cắt khoản nào —
gần như không suy ra được. 12/23 case sai là **chỉ sai ở trục mức độ**.

> `court_reasoning` (~9.700 ký tự/case) có sẵn trong dataset nhưng chứa chính lập luận
> dẫn tới phán quyết → dùng là **leakage**. Cần đối chiếu luật thi trước khi động vào.

### Những thứ KHÔNG phải nguyên nhân

- **Không phải model nhỏ.** `fullchunks/base` Qwen3.5-9B cũng đúng 0.540. Side accuracy
  đứng 0.70–0.74 qua mọi model 4B→9B, mọi cấu hình.
- **Không phải thiếu vote/seed/thinking token.** 25k output token/case đã là quá nhiều.
- **Ensemble qua model không cứu được.** Gộp majority 12 run có sẵn cho 0.40–0.54,
  không cái nào vượt run đơn — vì mọi model lệch cùng hướng `PARTIAL_A_WIN` nên
  majority khuếch đại bias (12-run ensemble dự đoán 42/50 là `PARTIAL_A_WIN`).

---

## 7. Phát hiện chưa áp dụng: gộp phiếu 2 trục

Ở run top-14, **mỗi vote riêng lẻ (0.60 / 0.62 / 0.53) đều tốt hơn ensemble 3 vote (0.54)**.
Cách gộp đang làm *giảm* chất lượng: `aggregate_votes` gộp trên nhãn 4 lớp rồi tie-break
về `PARTIAL_A_WIN`, vứt bỏ sự đồng thuận tồn tại ở từng trục con. 24 case split có
accuracy chỉ 0.417.

Gộp lại theo **hai trục riêng** (vote side, vote degree, rồi ghép):

| Cách gộp | Accuracy | Macro F1 |
|---|---:|---:|
| majority 4 lớp (hiện tại) | 0.540 | 0.508 |
| chỉ dùng vote #0 | 0.600 | 0.587 |
| **gộp 2 trục** | **0.640** | **0.644** |

Paired vs hiện tại: 9 case đổi, **6 sai→đúng / 1 đúng→sai**. `PARTIAL_B_WIN` F1 nhảy
0.22 → 0.60. Không cần chạy lại GPU — `derived_label_votes` đã lưu đủ trong file
predictions, tính lại offline được.

> **Chưa kết luận chắc**: n = 50, chỉ 7 cặp bất đồng, McNemar p = 0.125.
> Cần xác nhận trên private test hoặc seed thứ hai.


