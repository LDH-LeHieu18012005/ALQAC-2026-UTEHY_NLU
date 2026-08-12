# ALQAC 2026 — Civil Dispute Outcome Prediction for Vietnamese Law

Predicts the verdict label (`A_WIN` / `B_WIN` / `PARTIAL_A_WIN` / `PARTIAL_B_WIN`) of a
first-instance civil case, from the plaintiff's claim summary + evidence extracted from the
case file + retrieved statutory articles. All LLMs run remotely on **Modal GPU** (vLLM);
the notebooks are clients only.

- **Public**: 50 cases with `verdict_label` → used for evaluation.
- **Private**: 60 cases with only `case_id` + `case_query` → the submission target.
- **Legal corpus**: 18 documents, 3,352 articles.


---

## 1. Experiment space

Three binary axes, giving **8 configurations × 4 models = 32 runs**:

| Axis | Options |
|---|---|
| **Input** | `only_query` (case_query only) — `fullchunks` (case_query + agent_v4 evidence) |
| **Retrieval** | `bge-m3` (dense only, Qdrant) — `B_hyb` (hybrid dense+BM25 over a legal_query rewrite) |
| **Prompting** | `Baseline` (single classification call) — `3 call` / **CCWD** (issue_mapper → 3 judge votes → aggregation) |

**CCWD — Claim-Consensus Weighted Decision** is the proposed architecture.
The name maps directly onto its three stages:

| | Stage | Detail |
|---|---|---|
| **Claim** | Decompose the request | `issue_mapper` splits `case_query` into independently adjudicable claims, each tagged `MATERIAL` / `SECONDARY` |
| **Consensus** | Agreement | The Judge rules on **each claim** across 3 independent votes: `FULL_ACCEPT` / `PARTIAL_A_LEAN` / `PARTIAL_B_LEAN` / `REJECT` |
| **Weighted Decision** | Weighted label derivation | Compute an A-share score, then threshold at 0.85 / 0.55 / 0.30 into the 4-class label |

Stage 3 formula — `s = Σ(wᵢ · aᵢ) / Σwᵢ`, where `w` is the claim's importance weight and
`a` is the outcome score:

| `w` | | `a` | |
|---|---:|---|---:|
| `MATERIAL` | 2.0 | `FULL_ACCEPT` | 1.00 |
| `SECONDARY` | 1.0 | `PARTIAL_A_LEAN` | 0.65 |
| | | `PARTIAL_B_LEAN` | 0.35 |
| | | `REJECT` | 0.00 |

`s ≥ 0.85` → `A_WIN` · `≥ 0.55` → `PARTIAL_A_WIN` · `≥ 0.30` → `PARTIAL_B_WIN` · otherwise → `B_WIN`.

---

## 2. Results — Full chunks + query (50 public cases, seed 2026)

### Retrieval `B_hyb. legal_query` (hybrid dense+BM25)

| No. | Model | strict_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---:|---:|---:|---:|
| Baseline | Qwen3-4B | 0.32 | 0.2556 | 0.2658 | 0.2602 |
| Baseline | Qwen2.5-7B-Instruct | 0.30 | 0.1813 | 0.2367 | 0.1840 |
| Baseline | DeepSeek-R1-Distill-Llama-8B | 0.46 | 0.5944 | 0.4396 | 0.4358 |
| Baseline | Qwen3.5-9B | _0.52_ | _0.3988_ | _0.4477_ | _0.4165_ |
| 3 call | Qwen3-4B | 0.32 | 0.2953 | 0.2740 | 0.2695 |
| 3 call | Qwen2.5-7B-Instruct | 0.30 | 0.1833 | 0.2122 | 0.1946 |
| 3 call | DeepSeek-R1-Distill-Llama-8B | 0.40 | 0.4636 | 0.3115 | 0.3175 |
| **3 call** | **Qwen3.5-9B**| **0.58** | **0.5793** | **0.6340** | **0.5995** |

### Retrieval `bge-m3`

| No. | Model | strict_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---:|---:|---:|---:|
| Baseline | Qwen3-4B | 0.42 | 0.3321 | 0.3577 | 0.3432 |
| Baseline | Qwen2.5-7B-Instruct | 0.32 | 0.1687 | 0.2155 | 0.1621 |
| Baseline | DeepSeek-R1-Distill-Llama-8B | 0.40 | 0.3134 | 0.3515 | 0.3255 |
| Baseline | **Qwen3.5-9B** | **0.58** | **0.4435** | **0.5015** | **0.4574** |
| 3 call | Qwen3-4B | 0.36 | 0.3199 | 0.2704 | 0.2671 |
| 3 call | Qwen2.5-7B-Instruct | 0.36 | 0.2569 | 0.2786 | 0.2210 |
| 3 call | DeepSeek-R1-Distill-Llama-8B | 0.50 | 0.4007 | 0.4079 | 0.3984 |
| 3 call | Qwen3.5-9B | _0.50_ | _0.5268_ | _0.4762_ | _0.4426_ |

## 3. Results — Only query

### Retrieval `bge-m3`

| No. | Model | strict_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---:|---:|---:|---:|
| Baseline | Qwen3-4B | 0.30 | 0.1855 | 0.2388 | 0.1870 |
| Baseline | Qwen2.5-7B-Instruct | 0.32 | 0.1720 | 0.2204 | 0.1812 |
| Baseline | **DeepSeek-R1-Distill-Llama-8B** | **0.42** | **0.3244** | **0.3508** | **0.3320** |
| Baseline | Qwen3.5-9B | 0.32 | 0.2402 | 0.2564 | 0.2366 |
| 3 call | Qwen3-4B | 0.40 | 0.2150 | 0.2755 | 0.2265 |
| 3 call | Qwen2.5-7B-Instruct | _0.40_ | _0.2696_ | _0.3123_ | _0.2769_ |
| 3 call | DeepSeek-R1-Distill-Llama-8B | 0.38 | 0.1922 | 0.2648 | 0.2188 |
| 3 call | Qwen3.5-9B | 0.34 | 0.1810 | 0.2355 | 0.1718 |

### Retrieval `B_hyb. legal_query` (hybrid dense+BM25)

| No. | Model | strict_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---:|---:|---:|---:|
| Baseline | Qwen3-4B | 0.26 | 0.1010 | 0.2007 | 0.1231 |
| Baseline | Qwen2.5-7B-Instruct | 0.36 | 0.1964 | 0.2442 | 0.1882 |
| Baseline | DeepSeek-R1-Distill-Llama-8B | 0.28 | 0.2139 | 0.2227 | 0.2170 |
| Baseline | Qwen3.5-9B | 0.28 | 0.2356 | 0.2276 | 0.2144 |
| **3 call** | **Qwen3-4B** | **0.44** | **0.2713** | **0.2993** | **0.2361** |
| 3 call | Qwen2.5-7B-Instruct | 0.38 | 0.2406 | 0.2967 | 0.2526 |
| 3 call | DeepSeek-R1-Distill-Llama-8B | 0.34 | 0.3648 | 0.2798 | 0.2787 |
| 3 call | Qwen3.5-9B | _0.40_ | _0.1866_ | _0.2750_ | _0.1846_ |

## 4. Statute retrieval quality (%)

| Input | Embedding | Recall@10 | Hit Rate@10 | MRR@10 | Precision@10 |
|---|---|---:|---:|---:|---:|
| only-query | bge-m3 | 4.05 | 38 | 15.76 | 5.2 |
| only-query | **B_hyb. legal_query** | **26.62** | **88** | **34.21** | **29.6** |
| full-chunks | bge-m3 | 11.37 | 38 | 18.28 | 5.6 |
| full-chunks | **B_hyb. legal_query** | **26.52** | **90** | **31.54** | **29.2** |

Hybrid dense+BM25 over queries rewritten into legal terminology **outperforms pure dense
retrieval by a wide margin** (Hit@10 88–90 vs 38). This is the single largest improvement
in the project.


## 5. Repository layout

```
data/
  ALQAC2026_public_test.json   50 cases with verdict_label
  corpus_law_pub.json          18 documents / 3,352 articles
  agent_v4_results.json        evidence extracted from case_fact
  retrieval_top14_ours.json    top-14 statutes per case
  retrieval_top14_only_query.json

private/
  ALQAC_private_test.json          60 cases
  agent_v4_private_results.json
  retrieval_top14_private.json
  predictions_qwen3-5-9b_seed-2026.json
  [submission] UTEHY_NLU.json      ← submitted file
  alqac-e2e-bge-m3-qwen35-9b-pado.ipynb

pre_retrieval/                 produces agent_v4_results.json
  call_api/call_api_agent_v4.py  evidence retrieval agent (adaptive + LLM at 3 points)
  chunk_denoise.py               strips procedural noise from chunks
  legal_query.py                 rewrites the case description into a legal query
  data_io.py, llm.py
  eval-retrieval-bge.ipynb

embedding/                     builds & evaluates the BGE-M3 Qdrant collection
  embed-model-retrieval-bge-m3.ipynb
  evaluate-retrieval-bge-m3.ipynb
  retrieval_eval_bge_m3/*.csv

only_query/   bge-m3/{base,ccwd}   B_hyb. legal_query (hybrid dense+BM25)/{base,ccwd}
fullchunks/   bge-m3/{base,ccwd}   B_hyb. legal_query (hybrid dense+BM25)/{base,ccwd}
```

Each leaf directory holds 4 notebooks (one per model) + 4 `predictions_*_seed-2026.json` files.

> Filenames and code identifiers still carry the legacy `pado` prefix (`*-pado.ipynb`,
> `PADO_PIPELINE_VERSION`) — these refer to the same CCWD architecture. Do not change the
> **value** of the `PADO_PIPELINE_VERSION` string: it is part of `cache_key`, so editing it
> forces a full re-run on GPU.

---

## 6. Running

**Requirements**: Modal Secret `HF_TOKEN`; the `bge-m3` branch additionally needs
`QDRANT_URL` + `QDRANT_API_KEY` (the `B_hyb` branch reads statutes from file, no Qdrant needed).

Upload next to the notebook (or into `data/`): `ALQAC2026_public_test.json`,
`agent_v4_results.json`, `retrieval_top14_ours.json`.

Run All. Output goes to `outputs_alqac_e2e/<BENCHMARK_VERSION>/`:
`predictions_*.json`, `submission_*.json`, `benchmark_manifest_*.json`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ALQAC_RETRIEVAL_PATH` | (auto-detect) | point directly at the statute file |
| `ALQAC_TOP_K` | `14` | must match the number of articles per case in the file |
| `ALQAC_MAX_ARTICLE_CHARS` | `6000` | lower it if the context overflows |
| `ALQAC_MAX_EVIDENCE_CHARS` | `8000` | evidence length |
| `VLLM_MAX_MODEL_LEN` | `40960` | engine context ceiling |
| `MODAL_GPU` | `A100-40GB` | GPU type |

`MAX_MODEL_LEN = 40960` because top-14 is roughly 2.6× longer than top-10: a worst-case
judge prompt of ~16.3k tokens plus `max_new` of 16,384 would exceed 32,768. If the engine
rejects it, set `VLLM_MAX_MODEL_LEN=32768` and `ALQAC_MAX_ARTICLE_CHARS=4000`.
