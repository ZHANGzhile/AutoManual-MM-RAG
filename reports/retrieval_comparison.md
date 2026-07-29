# BM25 / Dense LSA / RRF retrieval comparison

Generated from the same 12,236 evidence-preserving chunks and the same
30-question Gold Evidence set. Metrics use the 26 answerable questions; the
four no-answer cases remain diagnostic and are excluded from Recall/MRR.

## Fixed configuration

- BM25: SQLite FTS5 with Porter tokenization.
- Dense: 2,048 signed hashed TF-IDF features projected to 128 dimensions with
  deterministic randomized LSA (`seed=2026`, `oversamples=16`).
- Dense is an offline latent-semantic baseline, **not** a neural embedding
  model (`neural_embedding=false`). No model was downloaded.
- RRF: BM25 and Dense top-50 candidates, `k=60`, equal weights `1.0/1.0`.
- Every backend receives the same brand/model/year/region/language/manual type
  hard filters before ranking. RRF parameters were not tuned on the Gold set.

## Results

| Backend | Recall@5 | Recall@10 | MRR@10 | Metadata violations |
|---|---:|---:|---:|---:|
| BM25-only | **0.8846** | **0.9615** | **0.7991** | 0 |
| Dense LSA-only | 0.7692 | 0.7692 | 0.5833 | 0 |
| BM25 + Dense RRF | 0.8077 | 0.8846 | 0.7475 | 0 |

RRF did not improve this corpus baseline. It improved image-grounded MRR from
0.8000 (BM25) to 1.0000, but the Dense ranking failed both safety questions
within the top 10. Under equal-weight fusion, those two BM25 rank-2 safety hits
fell outside the RRF top 10. Procedure Recall@10 was equal at 0.9091, but
procedure MRR fell from 0.7071 to 0.5851. Total Recall/MRR declined.

Notable answerable cases:

| Question | BM25 rank | Dense rank | RRF rank |
|---|---:|---:|---:|
| `bronco_proc_001` | 2 | miss | 5 |
| `f150_safety_008` | 2 | miss | miss |
| `f150_proc_010` | 9 | miss | 9 |
| `f150_proc_011` | miss | miss | miss |
| `mache_safety_014` | 2 | miss | miss |
| `mache_proc_017` | 6 | miss | 8 |

All three backends returned candidates for all four no-answer questions.
Abstention is not configured, so this is expected and is not counted as
retrieval recall.

## Integrity and provenance audit

The generated Dense index is 8,689,863 bytes and loaded successfully twice.
The audit verified:

- embedding shape `[12236, 128]`;
- 12,236 chunk records across four documents;
- zero chunk/metadata alignment mismatches;
- zero non-finite vectors, zero zero-length embeddings, and zero normalization
  failures;
- zero hard-filter violations across document/model/year/region checks;
- 300 RRF results across all 30 questions;
- zero source-rank, source-score, RRF-formula, or fused metadata violations.

Each RRF result retains `bm25_rank`, `bm25_score`, `dense_rank`,
`dense_score`, and `rrf_score`.

## Reproduce

Install only the lightweight retrieval dependency if the active Python does
not already provide it:

```powershell
python -m pip install -r .\requirements-retrieval.txt
```

Then rebuild Dense, evaluate all three backends, and run the audit with one
command:

```powershell
python .\scripts\run_retrieval_comparison.py
```

Machine-readable outputs:

- `outputs/metrics/bm25_baseline.json`
- `outputs/metrics/dense_lsa_baseline.json`
- `outputs/metrics/hybrid_rrf.json`
- `outputs/metrics/retrieval_comparison.json`
- `outputs/metrics/retrieval_audit.json`

The local index remains under ignored `data/indexes/dense_lsa.npz`.
