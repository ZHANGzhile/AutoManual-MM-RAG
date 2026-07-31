# Table-crop evidence retrieval baseline

This stage makes the 516 normalized table elements searchable without
pretending that unavailable cell structure exists.

## Resource boundary

| Resource | Count |
|---|---:|
| Table elements | 516 |
| Valid traceable table crops | 516 |
| Documents | 4 |
| Structured cell tables | 0 |

Every indexed record preserves the element ID, vehicle metadata, one-based
physical PDF page, section path, contextual text, source locator, and original
crop path. The index searches section titles, captions when available,
adjacent text, and document metadata using SQLite FTS5.

It does not index table rows, cell values, or units. Any numeric answer must
be verified against the displayed crop until selected tables are re-parsed or
OCRed.

## Evaluation

The 12-query development set contains three table-localization questions for
each vehicle:

- Bronco: roof-rack loads, exterior bulbs, wheel-nut torque;
- F-150 Lightning: charging overview, ground-leakage fault, trailer TPMS;
- Mustang Mach-E: charging overview, charge-speed limitation, TPMS
  troubleshooting;
- Maverick: washer fluid, fuel-tank capacity, wheel nuts.

| Metric | Result |
|---|---:|
| Recall@1 | 1.0000 |
| Recall@5 | 1.0000 |
| Recall@10 | 1.0000 |
| MRR@10 | 1.0000 |
| Metadata filter violations | 0 |

The questions intentionally use recognizable section topics and therefore
make this an index-wiring benchmark, not an independent semantic test. The
result proves that the correct table crop can be localized under vehicle hard
filters; it does not prove row-value question answering.

## Reproduce

```powershell
python .\scripts\build_table_index.py
python .\scripts\evaluate_tables.py
```

Machine-readable artifacts:

- `data/eval/table_questions.jsonl`
- `outputs/metrics/table_retrieval_baseline.json`

The generated `data/indexes/tables.sqlite3` remains local and ignored by Git.
