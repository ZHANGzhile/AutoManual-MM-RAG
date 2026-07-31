# Complete rebuild pipeline

`scripts/run_full_pipeline.py` is the canonical end-to-end entrypoint.

## Stages

1. `download`: fetch the four official HTTPS sources and verify the locked
   byte size and SHA-256.
2. `mineru`: parse each PDF in the isolated `.venv-mineru` environment and
   validate Markdown plus content-list outputs.
3. `import`: normalize MinerU elements with strict page, asset, and anomaly
   checks.
4. `chunks`: preserve page/section boundaries, ordered procedures, and safety
   blocks.
5. `indexes`: build BM25, dense LSA, table-crop, curated table-row, and visual
   indexes.
6. `evaluate`: rebuild visual Gold data, run text/visual/table/answer metrics,
   and execute retrieval integrity audits.
7. `test`: run the complete unit and local integration suite.

Each successful MinerU document has an ignored `_SUCCESS.json` marker, so an
interrupted run resumes without reparsing completed manuals. `--from-stage`
and `--to-stage` provide narrower restarts.

## Verified local run

The post-MinerU path was rebuilt from the four existing successful parses:

| Artifact | Count |
|---|---:|
| Normalized elements | 30,131 |
| Text chunks | 12,236 |
| Image elements | 3,287 |
| Table crops | 516 |
| Curated verified rows | 23 |
| Runtime indexes | 5 |
| Import anomalies | 0 |
| Metadata filter violations | 0 |

The run completed import, chunking, indexes, evaluation, audits, and tests in
about 35 seconds on the development machine. MinerU parsing time and model
download time are intentionally excluded because completed parse markers were
reused.

The machine-readable record is
`outputs/metrics/full_pipeline_run.json`.

The final automated suite contains 56 passing tests. A real localhost API
smoke run also exercised all three endpoints: text returned three cited
evidence items, the verified table route returned Bronco physical PDF page
294, and image-plus-text retrieval returned Bronco physical PDF page 112.

The Compose YAML was parsed successfully. Docker itself is not installed on
the development machine, so the image build was not executed locally. The
optional remote Responses call was validated with a mocked endpoint; no paid
API request was made and the default runtime remains offline.
