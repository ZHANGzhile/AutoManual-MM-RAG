# Curated table-row exact answering

The full MinerU parse contains 516 traceable table crops but no structured
cell HTML. To complete a safe exact-value demo without claiming broad OCR,
this stage manually verifies 23 rows from nine selected source tables.

## Coverage

| Vehicle | Curated content |
|---|---|
| Ford Bronco 2026 | Dynamic/static roof loads, 37-inch-tire roof load, M12/M14 wheel torque |
| Ford F-150 Lightning 2026 | Charging-equipment specifications and ground-leakage indicator state |
| Ford Mustang Mach-E 2026 | Three complete charge-speed limitation status rows |
| Ford Maverick 2026 | Washer-fluid quantity, non-hybrid fuel capacity, M14 wheel torque |

Every row stores:

- source table element ID and physical PDF page;
- ordered cell headers and values;
- vehicle metadata and section path;
- original table-crop path and SHA-256;
- `manual_visual_verification` method and verification date.

The index build fails if a source crop is missing or its SHA-256 differs from
the curated record.

## Development evaluation

The 17-question set contains 13 exact-value questions represented by the
curated rows and four deliberately uncovered or incompatible questions.

| Metric | Result |
|---|---:|
| Row Recall@1 | 1.0000 |
| Row Recall@5 | 1.0000 |
| Decision accuracy | 1.0000 |
| No-answer accuracy | 1.0000 |
| Answer rate | 0.7647 |
| Expected-value coverage | 1.0000 |
| Metadata filter violations | 0 |

Expected-value coverage verifies that all required strings and units appear
in the returned answer. Answers also include model, year, section, physical
PDF page, and the source image hash. The refusal cases cover a value outside
the curated rows, weak unrelated retrieval, and a hybrid/non-hybrid
applicability conflict.

This is a deliberately narrow development benchmark over selected rows. It is
not evidence that arbitrary tables or unseen cell layouts can be read.
Uncovered questions fall back to table-crop retrieval instead of fabricating a
numeric answer.

## Reproduce

```powershell
python .\scripts\build_table_row_index.py
python .\scripts\evaluate_table_rows.py
```

Tracked artifacts:

- `data/curated/table_rows.jsonl`
- `data/eval/table_row_questions.jsonl`
- `outputs/metrics/table_row_answering.json`

The generated `data/indexes/table_rows.sqlite3` remains local and ignored.
