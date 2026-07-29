# Filtered BM25 baseline

Date: 2026-07-29

## Scope

This baseline covers the first retrieval stage only:

1. build section- and page-aware text chunks;
2. curate a 30-question set tied to real normalized element IDs;
3. build and evaluate SQLite FTS5 BM25 with metadata hard filters.

It does not include dense embeddings, RRF, reranking, table-row retrieval,
visual embeddings, answer generation, or a no-answer threshold.

## Chunking

The chunk builder consumes the normalized `elements.jsonl` files and emits
stable `chunks.jsonl` records. It:

- never crosses a physical PDF page or `section_path`;
- groups consecutive numbered steps without changing their order;
- keeps Warning, Caution, and Note blocks standalone;
- carries all vehicle metadata, one-based page numbers, source element IDs, and
  previous/next chunk IDs;
- stores titles in `section_path` rather than duplicating them in content;
- preserves an oversized source element rather than splitting it and losing its
  evidence boundary.

| Type | Chunks |
|---|---:|
| Regular text | 6,086 |
| Ordered steps | 1,400 |
| Warning | 1,648 |
| Caution | 2 |
| Note | 3,100 |
| **Total** | **12,236** |

The chunks contain 22,974 original text elements. There are 118 source elements
longer than the 1,200-character grouping target; these remain intact by design.

## Evaluation set

`data/eval/questions.jsonl` contains 30 manually reviewed questions:

- 26 answerable questions with one or more real `element_id` Gold Evidence
  anchors;
- four no-answer diagnostics with empty Gold Evidence;
- ordinary facts, safety guidance, ordered procedures, image-associated
  explanations, and four identical steering-wheel questions used to verify
  model isolation.

Every Gold element was checked against the current chunk index for matching
document and physical PDF page before evaluation.

## Retrieval configuration

- Backend: Python standard-library `sqlite3`, SQLite FTS5.
- Ranking: FTS5 BM25.
- Tokenizer: Porter stemming over Unicode word tokens.
- Query construction: unique non-stopword terms combined with OR.
- Indexed fields: chunk content, section path, and document/model aliases.
- Hard filters: `doc_id`, brand, model, year, region, language, manual type, and
  optional chunk type.
- Evaluation depth: 10.
- A hit occurs when a returned chunk contains any Gold `element_id`.

## Results

| Metric | Result |
|---|---:|
| Answerable questions | 26 |
| Recall@5 | 0.8846 |
| Recall@10 | 0.9615 |
| MRR@10 | 0.7991 |
| Metadata filter violations | 0 |

Category results:

| Category | Questions | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Fact | 4 | 1.0000 | 1.0000 | 1.0000 |
| Image-associated text | 5 | 1.0000 | 1.0000 | 0.8000 |
| Metadata isolation | 4 | 1.0000 | 1.0000 | 1.0000 |
| Procedure | 11 | 0.7273 | 0.9091 | 0.7071 |
| Safety | 2 | 1.0000 | 1.0000 | 0.5000 |

The main weak point is procedure retrieval. One F-150 Lightning power-tailgate
question misses the Gold chunk at depth 10, while the child-restraint and
Mach-E no-power-frunk procedures first appear at ranks 9 and 6. These are real
baseline failures and have not been hidden by rewriting the Gold labels.

All four no-answer diagnostics return some lexical candidate. This is expected:
BM25 ranks matches but does not decide whether evidence is sufficient. The
cases are excluded from recall/MRR until a calibrated abstention rule exists.

Detailed ranks and top results are stored in
`outputs/metrics/bm25_baseline.json`.

## Reproduction

```powershell
python .\scripts\import_mineru_output.py --strict
python .\scripts\build_chunks.py
python .\scripts\build_bm25_index.py
python .\scripts\evaluate_bm25.py
python -m unittest discover -s tests -v
```
