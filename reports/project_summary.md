# AutoManual-MM-RAG project summary

## Delivered MVP

- Four official Ford 2026 North American English owner manuals.
- 30,131 normalized elements, including 3,287 images and 516 table crops.
- Page/section/vehicle provenance and metadata hard filters throughout.
- BM25, dense LSA, RRF hybrid, traditional visual, table-crop, and curated
  table-row retrieval.
- Evidence-constrained extractive answers with citations, ordered-step and
  Warning preservation, refusal behavior, and a three-tab Gradio demo.
- An optional Responses-compatible LLM/VLM generation layer with labeled text
  and image evidence, citation validation, and offline fallback.
- A dedicated Alibaba Cloud Qwen Chat Completions adapter, verified with a
  paid `qwen3-vl-flash` image-grounded call in Germany (Frankfurt).
- A FastAPI text/image/table service plus a read-only-data Docker deployment.
- A staged public-PDF-to-index rebuild command with integrity locks, isolated
  MinerU dependencies, audit reports, and restart support.
- A deterministic 29,797-node / 125,699-edge manual graph, graph-path
  retrieval, and an explicit-state Agentic GraphRAG workflow.
- 66 passing automated tests and reproducible offline evaluation artifacts.

## Measured results

| Evaluation | Result |
|---|---:|
| BM25 Recall@10 | 0.9615 |
| Visual + text-hint Recall@1 | 1.0000 |
| Text answer/refusal decision accuracy | 0.9667 |
| Text no-answer accuracy | 1.0000 |
| Table-crop Recall@1 | 1.0000 |
| Verified-row exact-value coverage | 1.0000 |
| Verified-row no-answer accuracy | 1.0000 |
| Metadata filter violations | 0 |
| Live Qwen visual generation | Passed |
| Agentic route accuracy (12-query dev set) | 1.0000 |
| Agentic multi-hop evidence recall | 0.9167 |
| Agentic Gold path accuracy | 0.5714 |
| Agentic refusal accuracy | 1.0000 |

The visual and table figures are development-set results. The table-value
benchmark covers 23 manually verified rows from nine selected source tables;
it is not a claim of general OCR over all 516 crops.

The Agentic figures are also development-set results. GraphRAG did not improve
evidence recall over the unchanged baseline (`0.9167` for both). Standalone
GraphRAG refused none of four no-answer questions, while the Agentic Evidence
Critic preserved `1.0000` refusal accuracy at higher latency.

## Portfolio-ready description

- Built a traceable multimodal RAG system over four automotive owner manuals,
  normalizing 30K+ text, image, and table elements with vehicle-level hard
  filters and physical-page citations.
- Implemented BM25/dense/RRF and image retrieval plus evidence-constrained
  answering; achieved 0.9615 BM25 Recall@10, 1.0000 image+text Recall@1, and
  zero cross-vehicle metadata violations on the development evaluations.
- Added verified table-row answering with source-image hashes, applicability
  guards, explicit refusal behavior, a Gradio demo, optional grounded LLM/VLM
  generation, a FastAPI/Docker deployment path, and 66 passing tests.
- Added a provider-isolated Qwen3-VL adapter and verified the complete
  upload-to-retrieval-to-cited-generation path against the Frankfurt API
  without committing credentials or workspace identifiers.
- Built an evidence-provenance automotive graph and explicit-state Agentic
  workflow with conditional routing, concurrent specialist retrieval, one
  bounded replan, citation/metadata guards, independent CLI/API entrypoints,
  execution traces, and a three-system multi-hop evaluation.
