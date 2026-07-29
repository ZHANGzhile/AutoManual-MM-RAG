# Data directory

This directory separates source documents, MinerU outputs, retrieval
evaluation data, and temporary transfer artifacts.

```text
data/
├── manifests/  # Corpus metadata and source URLs; tracked by Git
├── raw/        # Downloaded source manuals
├── parsed/     # MinerU Markdown, JSON, tables, and images
├── processed/  # Normalized JSONL and import summary; generated and ignored
├── indexes/    # Generated local retrieval indexes; ignored
├── eval/       # Curated evaluation questions and small fixtures
└── incoming/   # Temporary archives copied from parsing environments
```

## Storage policy

- Commit manifests, evaluation JSON/JSONL, and small hand-selected fixtures.
- Only commit original owner-manual PDFs when redistribution is permitted.
- Keep model files and generated vector indexes outside regular Git history.
- Rebuild `processed/` with `python scripts/import_mineru_output.py --strict`.
- Track curated Gold Evidence under `eval/`, but rebuild `indexes/` locally.
- `eval/visual_questions.jsonl` and its 12 small `visual_queries/` images are
  reproducible, perturbed derivatives of official image evidence for the
  visual MVP; source and query hashes plus transformation parameters are
  recorded in every row.
- GitHub browser uploads are limited to 25 MiB per file and regular Git blocks
  files above 100 MiB. Use a private GitHub Release or Git LFS for larger files.
