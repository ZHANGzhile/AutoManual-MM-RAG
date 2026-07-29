# Data directory

This directory separates source documents, MinerU outputs, retrieval
evaluation data, and temporary transfer artifacts.

```text
data/
├── manifests/  # Corpus metadata and source URLs; tracked by Git
├── raw/        # Downloaded source manuals
├── parsed/     # MinerU Markdown, JSON, tables, and images
├── eval/       # Curated evaluation questions and small fixtures
└── incoming/   # Temporary archives copied from parsing environments
```

## Storage policy

- Commit manifests, evaluation JSON/JSONL, and small hand-selected fixtures.
- Only commit original owner-manual PDFs when redistribution is permitted.
- Keep model files and generated vector indexes outside regular Git history.
- GitHub browser uploads are limited to 25 MiB per file and regular Git blocks
  files above 100 MiB. Use a private GitHub Release or Git LFS for larger files.
