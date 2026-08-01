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
5. `indexes`: build BM25, dense LSA, table-crop, curated table-row, visual,
   and deterministic graph indexes.
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
| Runtime indexes in recorded baseline run | 5 |
| Import anomalies | 0 |
| Metadata filter violations | 0 |

The run completed import, chunking, indexes, evaluation, audits, and tests in
about 35 seconds on the development machine. MinerU parsing time and model
download time are intentionally excluded because completed parse markers were
reused.

The machine-readable record is
`outputs/metrics/full_pipeline_run.json`.

The recorded baseline suite contains 58 passing tests. A real localhost API
smoke run also exercised all three endpoints: text returned three cited
evidence items, the verified table route returned Bronco physical PDF page
294, and image-plus-text retrieval returned Bronco physical PDF page 112.

The Compose YAML was parsed successfully. The Docker CLI is not available in
the current development terminal, so the image build has not been executed
locally. The OpenAI Responses adapter remains mock-tested. A separate Qwen
Chat Completions adapter was validated with both an HTTP 200 authentication
smoke and a paid multimodal `qwen3-vl-flash` request in the Germany
(Frankfurt) region. That request retrieved five Bronco-only evidence records,
generated the correct seatbelt-reminder interpretation, and cited physical
PDF page 29. Credentials and the workspace-specific hostname remain only in
the ignored local `.env`.

## Agentic GraphRAG extension verification

The pipeline definition now adds `build_graph_index.py` to `indexes` and
`evaluate_agentic_graphrag.py` to `evaluate`, for six combined runtime
indexes. In the independent worktree, the graph stage read the main project's
ignored local data through explicit paths and built 29,797 nodes / 125,699
edges; the Agentic comparison and the complete 66-test suite passed. The
original `full_pipeline_run.json` was not overwritten because it records the
earlier baseline rebuild.
